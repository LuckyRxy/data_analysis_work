"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

import os
import re

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def BoxPlotFeatures(X, y, feature_names, output_dir):
    # enumerate()：同时获取索引和元素
    print("X:", X)
    print("y:", y)
    print("feature_names:", feature_names)
    for c, fn in enumerate(feature_names):
        idx_Malignant = np.nonzero(y == 'Malignant')[0]
        idx_Benign = np.nonzero(y == 'Benign')[0]
        group = [X[idx_Malignant, c], X[idx_Benign, c]]
        print("group:", group)
        plt.figure()
        plt.boxplot(group, labels=['Malignant', 'Benign'])
        plt.title(str(c) + ' ' + fn)
        plt.savefig(os.path.join(output_dir, fn + '.png'))
        plt.close()


def ttest(slice_meta, slice_features):
    x = []

    idx = np.nonzero(slice_meta[:, 3] == 'Benign')[0]
    x.append(slice_features[idx, :])

    idx = np.nonzero(slice_meta[:, 3] == 'Malignant')[0]
    x.append(slice_features[idx, :])

    idx = np.nonzero(slice_meta[:, 3] == 'NoNod')[0]
    x.append(slice_features[idx, :])

    p_val = []

    for i in np.arange(x[0].shape[1]):
        aux = stats.ttest_ind(x[0][:, i], x[1][:, i])
        p_val.append(aux[1])

    p_val = np.array(p_val)
    ranking_idx = np.argsort(p_val)
    pval_sort = p_val[ranking_idx]

    return p_val, ranking_idx, pval_sort


######################################

excel_file = pd.ExcelFile('./features.xlsx')
df = excel_file.parse('Sheet1')

features_columns = [
    'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence',
    'original_glcm_ClusterShade', 'original_glcm_ClusterTendency',
    'original_glcm_Contrast', 'original_glcm_Correlation',
    'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy',
    'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm',
    'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1',
    'original_glcm_Imc2', 'original_glcm_InverseVariance',
    'original_glcm_JointAverage', 'original_glcm_JointEnergy',
    'original_glcm_JointEntropy', 'original_glcm_MCC',
    'original_glcm_MaximumProbability', 'original_glcm_SumAverage',
    'original_glcm_SumEntropy', 'original_glcm_SumSquares'
]
slice_features = df[features_columns]

slice_features = slice_features.iloc[1:].to_numpy()

meta_columns = [
    'patient_id','nodule_id','diagnosis'
]

slice_meta = df[meta_columns]

patient_id = slice_meta['patient_id'].to_numpy()

def extract_number(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    else:
        return None

id = [extract_number(i) for i in patient_id]

def set_val(value):
    return 'Malignant' if value == 1 else 'Benign'

slice_meta.insert(1, 'id', id)

slice_meta['diagnosis'] = slice_meta['diagnosis'].apply(set_val)


slice_meta = slice_meta.iloc[1:].to_numpy()


p_val, ranking_idx, pval_sort = ttest(slice_meta, slice_features)
print(f"p_val:{p_val}\nranking_idx:{ranking_idx}\npval_sort;{pval_sort}")

output_dir = '../resources/boxplots_images'
BoxPlotFeatures(
    X=slice_features,
    y=slice_meta[:, 3],
    feature_names=features_columns,
    output_dir=output_dir
)
