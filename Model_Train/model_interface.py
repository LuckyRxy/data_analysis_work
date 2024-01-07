import numpy as np
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



def svm_model(all_features,y,test_size,random_state):
    print("-------------------------------------SVM---------------------------------------")
    # # DATA SPLITTING
    X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=42)

    # Create and training a Radiomics_SVM_KNN_LogisticRegression classifier
    clf2 = svm.SVC(probability=True, class_weight='balanced')
    clf2.fit(X_train, y_train)

    # Use the probabilities to calibrate a new model
    calibrated_classifier = CalibratedClassifierCV(clf2, n_jobs=-1)
    calibrated_classifier.fit(X_train, y_train)

    y_pred_calib = calibrated_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_calib)*100
    train_report_dict = classification_report(
        y_test,
        y_pred_calib,
        labels=[0, 1],
        target_names=['benign', 'malign'],
        sample_weight=None,
        digits=3,
        output_dict=False,
        zero_division=0
    )

    print(train_report_dict)
    print(f"Accuracy: {accuracy}")

def knn_model(all_features,y,test_size,random_state):
    print('---------------------------------------KNN-------------------------------------------')

    # 1.slice train_data and test_data
    X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=42)

    # 2.create KNN model
    knn = KNeighborsClassifier(n_neighbors=3)

    # 3.train model
    knn.fit(X_train, y_train)

    # 4.predict
    y_pred = knn.predict(X_test)

    # 5.evaluate model
    accuracy = accuracy_score(y_test, y_pred)*100

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

def logistic_regression_model(all_features,y,test_size,random_state):
    print('-----------------------------------Logistic Regression------------------------------------')

    X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.2, random_state=42)

    logreg_model = LogisticRegression(max_iter=500)

    logreg_model.fit(X_train, y_train)

    y_pred = logreg_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)*100

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