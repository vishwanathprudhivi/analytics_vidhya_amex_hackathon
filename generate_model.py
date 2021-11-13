import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from extra_keras_metrics import get_minimal_multiclass_metrics
from sklearn.metrics import average_precision_score,accuracy_score

#import user defined libraries
from constants import PROCESSED_TRAIN_PATH,PROCESSED_TEST_PATH,RAW_TEST_PATH,PREDICTION_FILE_PATH

train_df  = pd.read_csv(PROCESSED_TRAIN_PATH)

feature_cols = ['age', 'vintage', 'is_active', 'gender_Male', 'city_category_C2',
       'customer_category_S2', 'customer_category_S3', 'current_P16',
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

def get_model(input_shape,output_shape):
    model = tf.keras.Sequential()
    model.add(layers.Input(input_shape))
    model.add(layers.Dense(128,activation = 'relu'))
    model.add(layers.Dense(128,activation = 'relu'))
    model.add(layers.Dense(64,activation = 'relu'))
    model.add(layers.Dense(32,activation = 'relu'))
    model.add(layers.Dense(output_shape,activation = 'sigmoid'))
    model.compile(optimizer = 'nadam',loss = 'binary_crossentropy', metrics = get_minimal_multiclass_metrics())
    return model

model = get_model(len(feature_cols),len(target_cols))

history = model.fit(x_train, y_train,
                    batch_size = 100, nb_epoch= 10,
                    verbose=1, validation_data=(x_val, y_val),
                    shuffle = True)


x_test = pd.read_csv(PROCESSED_TEST_PATH)
result = model.predict(x_test)
result = result.round()

preds = []
for row in result:
    if np.sum(row) > 0:
        prods = []
        for i,v in enumerate(row):
            if v==1:
                prods.append(target_cols[i].replace('future_',''))
        if len(prods)>3:
            preds.append(prods[:3])
        else:
            preds.append(str(prods))
    else:
        preds.append(np.nan)

customer_ids = pd.read_csv(RAW_TEST_PATH)

out_df = pd.concat([customer_ids['Customer_ID'],pd.Series(preds,name='Product_Holding_B2')],axis = 1)
out_df.to_csv(PREDICTION_FILE_PATH,index = False)