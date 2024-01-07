import os
import numpy as np
import pandas as pd
import random

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn import svm

from Model_Train.model_interface import svm_model, knn_model, logistic_regression_model


def make_positive_negative_lists(filename):
    # Make two list from the filename, which is an .xlsx file.
    # And return two list with positive and negative samples each one.
    # Positive = 1 and Negative = 0.
    df = pd.read_excel(filename,
                       sheet_name='Sheet1',
                       engine='openpyxl'
                       )
    print("Reading the filename: {}".format(filename))

    # Positive (malign) = 1, Negative (benign) = 0.
    positive_samples = df.loc[df['diagnosis'] == 1]
    negative_samples = df.loc[df['diagnosis'] == 0]

    print("Positive samples: {}".format(len(positive_samples)))
    print("Negative samples: {}".format(len(negative_samples)))

    return positive_samples, negative_samples


############ MAIN PROGRAM ########################

print("Begining ...")

################### Parameters to be configured ###################################
path = '../Features_Extraction'
excel_features = os.path.join(path, 'radiomics_features.xlsx')
proportion = 0.8  # define the training samples proportion

all_positive_samples, all_negative_samples = make_positive_negative_lists(excel_features)

all_positive_samples = shuffle(all_positive_samples)
all_negative_samples = shuffle(all_negative_samples)

# Select manually the features
selection = ['original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy',
             'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis',
             'original_firstorder_Maximum',
             'original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy',
             'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis',
             'original_firstorder_Maximum'
             ]

# applying the feature selection (it removes any column that is not included in the 'selection' list)
positive_samples = all_positive_samples.loc[:, selection]
negative_samples = all_negative_samples.loc[:, selection]

# print(f"positive_samples:{positive_samples}\n negative_samples:{negative_samples}")

y_positive_samples = all_positive_samples.loc[:, 'diagnosis']
y_negative_samples = all_negative_samples.loc[:, 'diagnosis']

###################### RANDOM SELECTION OF THE TRAIN AND TEST SET  ##########################################
#### It is necessary to select a proportion of samples from positive and negative cases.

# Random selection of the train and test set from POSITIVE samples
num_positive_samples = len(positive_samples)
num_training_positive_samples = int(num_positive_samples * proportion)
all_positive_indexes = range(0, num_positive_samples)

training_positive_indexes = random.sample(all_positive_indexes, num_training_positive_samples)
testing_positive_indexes = [x for x in all_positive_indexes if x not in training_positive_indexes]

training_positive_samples = positive_samples.iloc[training_positive_indexes, :]
y_training_positive_samples = y_positive_samples.iloc[training_positive_indexes]

testing_positive_samples = positive_samples.iloc[testing_positive_indexes, :]
y_testing_positive_samples = y_positive_samples.iloc[testing_positive_indexes]

print("Training positive samples: {}.".format(len(training_positive_samples)))
print("Testing positive samples: {}.\n".format(len(testing_positive_samples)))

# Random selection of the train and test set from NEGATIVE samples
num_negative_samples = len(negative_samples)
num_training_negative_samples = int(num_negative_samples * proportion)
all_negative_indexes = range(0, num_negative_samples)

training_negative_indexes = random.sample(all_negative_indexes, int(num_negative_samples * proportion))
testing_negative_indexes = [x for x in all_negative_indexes if x not in training_negative_indexes]

training_negative_samples = negative_samples.iloc[training_negative_indexes, :]
y_training_negative_samples = y_negative_samples.iloc[training_negative_indexes]

testing_negative_samples = negative_samples.iloc[testing_negative_indexes, :]
y_testing_negative_samples = y_negative_samples.iloc[testing_negative_indexes]

print("Training negative samples: {}.".format(len(training_negative_samples)))
print("Testing negative samples: {}.".format(len(testing_negative_samples)))

# train and test sets
X_training_samples = training_positive_samples._append(training_negative_samples)
y_training_samples = y_training_positive_samples._append(y_training_negative_samples)

X_test_samples = testing_positive_samples._append(testing_negative_samples)
y_test_samples = y_testing_positive_samples._append(y_testing_negative_samples)

#############################################################################################################

print()
X_training_samples, y_training_samples = shuffle(X_training_samples, y_training_samples, random_state=42)

X_test_samples, y_test_samples = shuffle(X_test_samples, y_test_samples, random_state=2)

# from dataFrame to numpy array
X_training_samples_array = X_training_samples.values
y_training_samples_array = y_training_samples.values
# print("train:____________", y_training_samples_array)

X_test_samples_array = X_test_samples.values
y_test_samples_array = y_test_samples.values
# print("test:____________", y_test_samples_array)

# train and predict model
print('---------------------------------------SVM---------------------------------------------')

model = svm.SVC(gamma='scale')

model.fit(X_training_samples_array, y_training_samples_array)

calibrated_classifier = CalibratedClassifierCV(model, n_jobs=-1)
calibrated_classifier.fit(X_training_samples_array, y_training_samples_array)

# Predict all the test samples
y_prediction = calibrated_classifier.predict(X_test_samples_array)
# 6.get prediction report
report = classification_report(
    y_test_samples_array,
    y_prediction,
    labels=[0, 1],
    target_names=['benign', 'malign'],
    sample_weight=None,
    digits=3,
    output_dict=False,
    zero_division=0
)
accuracy = (np.sum(y_prediction == y_test_samples_array) / y_test_samples_array.size) * 100
print(report)
print(f"Accuracy: {accuracy}")

print('---------------------------------------KNN---------------------------------------------')

knn = KNeighborsClassifier(n_neighbors=4)  # 选择合适的k值

knn.fit(X_training_samples_array, y_training_samples_array)

y_pred = knn.predict(X_test_samples_array)

accuracy = accuracy_score(y_test_samples_array, y_pred)*100

report = classification_report(
    y_test_samples_array,
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

print('--------------------------------Logistic Regression---------------------------------')

logreg_model = LogisticRegression(max_iter=500)

logreg_model.fit(X_training_samples_array, y_training_samples_array)

y_pred = logreg_model.predict(X_test_samples_array)

accuracy = accuracy_score(y_test_samples_array, y_pred)*100

report = classification_report(
    y_test_samples_array,
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