#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "Guillermo Torres, Debora Gil and Pau Cano"
__license__ = "GPLv3"
__email__ = "gtorres,debora,pau@cvc.uab.cat"
__year__ = "2023"
"""
import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from public.NiftyIO import readNifty

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convertir el tensor a una imagen PIL
    transforms.Resize((53, 64)),  # Redimensionar la imagen a 224x224 píxeles
    transforms.ToTensor(),  # Convertir la imagen a un tensor de PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar el tensor
])

# Load a pre-trained VGG16 or VGG19 model
model = models.vgg16(pretrained=True)
# model = models.vgg19(pretrained=True)

#### Parts of the VGG model ####
vgg_features = model.features
vgg_avgpool = model.avgpool
# ":-2" includes the classifier layers of the VGG up to the penultimate layer
vgg_classifier = nn.Sequential(*list(model.classifier.children())[:-2])

def get_file_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

db_path = r'../resources/VOIs'
imageDirectory = 'image'
maskDirectory = 'nodule_mask'

image_names = os.listdir(os.path.join(db_path, imageDirectory))
df_mv = pd.read_excel(r'D:\Project\PythonProject\data_analysis_work\resources\MetadatabyNoduleMaxVoting.xlsx',
                      sheet_name='ML4PM_MetadatabyNoduleMaxVoting',
                      engine='openpyxl'
                      )

# Ensure data is consistent
features = 'radiomics_features.xlsx'
df = pd.read_excel(features)

patient_id = df['patient_id'].astype(str)
nodule_id = df['nodule_id'].astype(str)

full_names = patient_id + '_R_' + nodule_id + '.nii.gz'

unique_full_names = full_names.drop_duplicates().tolist()

choosed_data_paths = [os.path.join(db_path,'image', v) for v in unique_full_names]


# Initialize features and labels
all_features = []
y = []
# patient_id = patient_id.drop_duplicates().tolist()
# nodule_id = nodule_id.drop_duplicates().tolist()

for i in range(len(choosed_data_paths)):
    image_slices = []
    image, _ = readNifty(choosed_data_paths[i], CoordinateOrder='xyz')
    patient_id = re.search(r'LIDC-IDRI-(\d+)', unique_full_names[i]).group(0)
    nodule_id = int(re.search(r'_(\d+)\.', unique_full_names[i]).group(1))

    diagnosis = \
        df_mv[(df_mv.patient_id == patient_id) & (df_mv.nodule_id == nodule_id)].Diagnosis_value.values[0]

    for j in range(image.shape[2]):
        image_slices.append(image[:, :, j])

    for image_slice in image_slices:
        X = image_slice.astype(np.float64)
        # Replicate the array in three channels
        X = np.stack([X] * 3, axis=2)  # (224, 224, 3)

        # Transpose the axis: (3, 224, 224)
        X = X.transpose((2, 0, 1))

        # Convert from numpy array to a tensor of PyTorch
        tensor = torch.from_numpy(X)

        # Apply the transform to the tensor
        tensor = transform(tensor)

        # Expand one dimension
        tensor = tensor.unsqueeze(0)

        # Extract features using the VGG model
        with torch.no_grad():
            out = model.features(tensor)
            out = model.avgpool(out)
            out = out.view(1, -1)
            out = vgg_classifier(out)

        array = out.numpy()
        all_features.append(array)
        y.append(diagnosis)

# Stack features vertically
all_features = np.vstack(all_features)
# print(all_features)
all_features_data_frame = pd.DataFrame(all_features)
all_features_data_frame.to_excel('vgg_features.xlsx',index=False)

y_data_frame = pd.DataFrame({'y': y})
y_data_frame.to_excel('vgg_y_features.xlsx', index=False)

