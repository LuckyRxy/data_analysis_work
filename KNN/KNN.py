import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create a transformation for processing the image
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

# ----------------------------------------------------


#### Parts of the VGG model ####
vgg_features = model.features
vgg_avgpool = model.avgpool
# ":-2" includes the classifier layers of the VGG up to the penultimate layer
vgg_classifier = nn.Sequential(*list(model.classifier.children())[:-2])


def get_file_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


db_path = r'D:\Project\PythonProject\data_analysis_work\resources\VOIs'
imageDirectory = 'image'
maskDirectory = 'nodule_mask'
image_paths = get_file_paths(os.path.join(db_path, imageDirectory))
mask_paths = get_file_paths(os.path.join(db_path, maskDirectory))

image_names = os.listdir(os.path.join(db_path, imageDirectory))

df_mv = pd.read_excel(r'D:\Project\PythonProject\data_analysis_work\resources\MetadatabyNoduleMaxVoting.xlsx',
                      sheet_name='ML4PM_MetadatabyNoduleMaxVoting',
                      engine='openpyxl'
                      )

# Initialize features and labels
all_features = []
y = []
for i in range(min(500, len(image_paths))):
    image_slices = []
    image, _ = readNifty(image_paths[i], CoordinateOrder='xyz')
    patient_id = re.search(r'([A-Z0-9-]+)_R_\d+', image_names[i]).group(1)
    nodule_id = int(re.search(r'R_(\d+)', image_names[i]).group(1))

    diagnosis = \
        df_mv.loc[(df_mv['patient_id'] == patient_id) & (df_mv['nodule_id'] == nodule_id), 'Diagnosis_value'].values[0]

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

print('---------------------------------------KNN-------------------------------------------')

# 1.slice train_data and test_data
X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=42)

# 2.create KNN model
knn = KNeighborsClassifier(n_neighbors=3)  # 选择合适的k值

# 3.train model
knn.fit(X_train, y_train)

# 4.predict
y_pred = knn.predict(X_test)

# 5.evaluate model
accuracy = accuracy_score(y_test, y_pred)

# 6.get prediction report
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=['benign', 'malign'],
    sample_weight=None,
    digits=3,
    output_dict=False,
    zero_division=0
)

print(report)
print(f"Accuracy: {accuracy}")