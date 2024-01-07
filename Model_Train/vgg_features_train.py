import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from Model_Train.model_interface import svm_model, knn_model, logistic_regression_model

# 从Excel读取数据
df_read = pd.read_excel('../Features_Extraction/vgg_features.xlsx')
y_df_read = pd.read_excel('../Features_Extraction/vgg_y_features.xlsx')

# parameter settings
all_features = df_read.values.tolist()
y = y_df_read['y'].tolist()
test_size=0.3
random_state=42

# train and predict model
svm_model(all_features,y,test_size,random_state)

knn_model(all_features,y,test_size,random_state)

logistic_regression_model(all_features,y,test_size,random_state)