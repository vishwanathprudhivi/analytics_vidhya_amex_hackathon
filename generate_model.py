import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from extra_keras_metrics import get_minimal_multiclass_metrics
from sklearn.metrics import average_precision_score,accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

#import user defined libraries
from constants import PROCESSED_TRAIN_PATH,PROCESSED_TEST_PATH,RAW_TEST_PATH,PREDICTION_FILE_PATH

train_df  = pd.read_csv(PROCESSED_TRAIN_PATH)
#'gender_Male', 'city_category_C2',
       'customer_category_S2'
feature_cols = ['age', 'vintage', 'is_active', , 'current_P16',
       'current_P13', 'current_P20', 'current_P11', 'current_P8',
       'current_P17', 'current_P21', 'current_P12', 'current_P10',
       'current_P19', 'current_P2', 'current_P00', 'current_P18',
       'current_P15', 'current_P6', 'current_P9', 'current_P7', 'current_P3',
       'current_P5', 'current_P4', 'current_P1', 'current_P14']

target_cols = ['future_P8', 'future_P3', 'future_P00', 'future_P6', 'future_P12', 'future_P16',
       'future_P1', 'future_P9', 'future_P10', 'future_P13', 'future_P4',
       'future_P5', 'future_P7', 'future_P11', 'future_P2', 'future_P15',
       'future_P17', 'future_P14', 'future_P20', 'future_P18']

x_train,x_val,y_train,y_val = train_test_split(train_df[feature_cols],train_df[target_cols],test_size = 0.2)

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
    model.add(layers.Dense(128,activation = tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(layers.Dense(128,activation = tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(layers.Dense(64,activation = tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(layers.Dense(32,activation = tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(layers.Dense(output_shape,activation = 'sigmoid'))
    model.compile(optimizer = 'nadam',loss = 'binary_crossentropy', metrics = get_minimal_multiclass_metrics())
    history = model.fit(x_train, y_train,
                        batch_size = 100, nb_epoch= 10,
                        verbose=1, validation_data=(x_val, y_val),
                        shuffle = True)
    return model

#model = get_model_xgb(x_train,y_train)
#accuracy_score(y_val,model.predict(x_val))

dl_model = get_model_dl(len(feature_cols),len(target_cols),x_train,y_train)
accuracy_score(y_val,dl_model.predict(x_val).round())

x_test = pd.read_csv(PROCESSED_TEST_PATH)
result = dl_model.predict(x_test[feature_cols])

idxs = np.argsort(-1*result)
preds = []
for row in idxs:
    prods = []
    for idx in row[:3]:
        prods.append(target_cols[idx].replace('future_',''))
    preds.append(prods)

customer_ids = pd.read_csv(RAW_TEST_PATH)
out_df = pd.concat([customer_ids['Customer_ID'],pd.Series(preds,name='Product_Holding_B2')],axis = 1)
out_df.to_csv(PREDICTION_FILE_PATH,index = False)