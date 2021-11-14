import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from extra_keras_metrics import get_minimal_multiclass_metrics
from sklearn.metrics import average_precision_score,accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#import user defined libraries
from constants import PROCESSED_TRAIN_PATH,PROCESSED_TEST_PATH,RAW_TEST_PATH,PREDICTION_FILE_PATH

#set seed
np.random.seed(2021)

train_df  = pd.read_csv(PROCESSED_TRAIN_PATH)

feature_cols = ['age', 'vintage', 'is_active', 'current_P16', 'current_P20', 'current_P11', 'current_P8','current_P13',
       'current_P17', 'current_P21', 'current_P12', 'current_P10',
       'current_P19', 'current_P2', 'current_P00', 'current_P18',
       'current_P15', 'current_P6', 'current_P9', 'current_P7', 'current_P3',
       'current_P5', 'current_P4', 'current_P1', 'current_P14']

target_cols = ['future_P8', 'future_P3', 'future_P00', 'future_P6', 'future_P12', 'future_P16',
       'future_P1', 'future_P9', 'future_P10', 'future_P13', 'future_P4',
       'future_P5', 'future_P7', 'future_P11', 'future_P2', 'future_P15',
       'future_P17', 'future_P14', 'future_P20', 'future_P18']

x_train,x_val,y_train,y_val = train_test_split(train_df[feature_cols],train_df[target_cols],test_size = 0.2)

def get_model_mlknn(x_train,y_train):
    model = MLkNN(k=10)
    # Note that this classifier can throw up errors when handling sparse matrices.
    #x_train = lil_matrix(x_train).toarray()
    #y_train = lil_matrix(y_train).toarray()
    # train
    model.fit(np.array(x_train), np.array(y_train))
    return model

def get_model_xgb(x_train,y_train):
    # create XGBoost instance with default hyper-parameters
    xgb_estimator = xgb.XGBClassifier(objective='binary:logistic')
    # create MultiOutputClassifier instance with XGBoost model inside
    model = MultiOutputClassifier(xgb_estimator)
    # fit the model
    model.fit(x_train, y_train)
    return model

def get_model_dl(input_shape,output_shape,x_train,y_train):
    model = tf.keras.Sequential()
    model.add(layers.Input(input_shape))
    model.add(layers.Dense(256,activation = 'relu'))
    model.add(layers.Dense(128,activation = 'relu'))
    model.add(layers.Dense(64,activation = 'relu'))
    model.add(layers.Dense(output_shape,activation = 'sigmoid'))
    model.compile(optimizer = 'nadam',loss = 'binary_crossentropy', metrics = get_minimal_multiclass_metrics())
    history = model.fit(x_train, y_train,
                        batch_size = 50, nb_epoch= 10,
                        verbose=1, validation_data=(x_val, y_val),
                        shuffle = True)
    return model

def get_ovr_model(x_train,y_train):
    estimator = LogisticRegression(solver = 'newton-cg')
    model = OneVsRestClassifier(estimator)
    model.fit(x_train, y_train)
    return model


#get predictions on validation dataset

xgb_model = get_model_xgb(x_train,y_train)
accuracy_score(y_val,xgb_model.predict(x_val))

dl_model = get_model_dl(len(feature_cols),len(target_cols),x_train,y_train)
accuracy_score(y_val,dl_model.predict(x_val).round())

ovr_model = get_ovr_model(x_train,y_train)
accuracy_score(y_train,ovr_model.predict(x_train).round())
accuracy_score(y_val,ovr_model.predict(x_val).round())

#get ensemble predictions on validation dataset

val_result_1 = xgb_model.predict_proba(x_val)
preds_cal_0 = []
for item in val_result_1:
    preds_cal_0.append([local_item[1] for local_item in item] )

val_result_2 = ovr_model.predict_proba(x_val)


ensemble_preds_val = (0.8*np.array(preds_cal_0).T + 0.1*np.array(dl_model.predict(x_val)) + 0.1*val_result_2)
ensemble_preds_val = np.where(ensemble_preds_val >= 0.33, 1, 0)
accuracy_score(y_val,ensemble_preds_val)

#get predictions on test dataset
x_test = pd.read_csv(PROCESSED_TEST_PATH)

result_0 = dl_model.predict(x_test[feature_cols])

result_1 = xgb_model.predict_proba(x_test[feature_cols])
preds_cal_0 = []
for item in result_1:
    preds_cal_0.append([local_item[1] for local_item in item] )

result_2 = ovr_model.predict_proba(x_test[feature_cols])

#ensemble predictions where we weight predictions from each model.
ensemble_preds = (0.8*result_0 + 0.1*np.array(preds_cal_0).T + 0.1*result_2)
ensemble_preds_val = np.where(ensemble_preds_val >= 0.33, 1, ensemble_preds_val)
idxs = np.argsort(-1*ensemble_preds)

#convert predicted probabilities into label sets per predicted row
preds = []
for row in idxs:
    prods = []
    for idx in row[:3]:
        prods.append(target_cols[idx].replace('future_',''))
    preds.append(prods) 

#load customer ids and concat with predicted labels. finally save to disk
customer_ids = pd.read_csv(RAW_TEST_PATH)
out_df = pd.concat([customer_ids['Customer_ID'],pd.Series(preds,name='Product_Holding_B2')],axis = 1)
out_df.to_csv(PREDICTION_FILE_PATH,index = False)

