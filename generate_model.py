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
import catboost

#import user defined libraries
from constants import PROCESSED_TRAIN_PATH,PROCESSED_TEST_PATH,RAW_TEST_PATH,PREDICTION_FILE_PATH

#set seed
np.random.seed(2021)

train_df  = pd.read_csv(PROCESSED_TRAIN_PATH)

feature_cols = ['age', 'vintage', 'is_active', 'current_P16',
       'current_P13', 'current_P20', 'current_P11', 'current_P8',
       'current_P17', 'current_P21', 'current_P12', 'current_P10',
       'current_P19', 'current_P2', 'current_P00', 'current_P18',
       'current_P15', 'current_P6', 'current_P9', 'current_P7', 'current_P3',
       'current_P5', 'current_P4', 'current_P1', 'current_P14']

target_cols = ['future_P8', 'future_P3', 'future_P00', 'future_P6', 'future_P12', 'future_P16',
       'future_P1', 'future_P9', 'future_P10', 'future_P13', 'future_P4',
       'future_P5', 'future_P7', 'future_P11', 'future_P2', 'future_P15',
       'future_P17', 'future_P14', 'future_P20', 'future_P18']

#remove low event rates
#target_cols = [col for col in target_cols if col not in ['future_P11','future_P2','future_P15','future_P17','future_P14','future_P20','future_P18']]

x_train,x_val,y_train,y_val = train_test_split(train_df[feature_cols],train_df[target_cols],test_size = 0.2)

def create_dataset(n_sample=1000):
    ''' 
    Create a unevenly distributed sample data set multilabel  
    classification using make_classification function
    
    args
    nsample: int, Number of sample to be created
    
    return
    X: pandas.DataFrame, feature vector dataframe with 10 features 
    y: pandas.DataFrame, target vector dataframe with 5 labels
    '''
    X, y = make_classification(n_classes=5, class_sep=2, 
                           weights=[0.1,0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                           n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
    y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), y

def get_tail_label(df):
    """
    Give tail label colums of the given target dataframe
    
    args
    df: pandas.DataFrame, target label df whose tail label has to identified
    
    return
    tail_label: list, a list containing column name of all the tail label
    """
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl)/irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label

def get_index(df):
  """
  give the index of all tail_label rows
  args
  df: pandas.DataFrame, target label df from which index for tail label has to identified
    
  return
  index: list, a list containing index number of all the tail label
  """
  tail_labels = get_tail_label(df)
  index = set()
  for tail_label in tail_labels:
    sub_index = set(df[df[tail_label]==1].index)
    index = index.union(sub_index)
  return list(index)

def get_minority_instace(X, y):
    """
    Give minority dataframe containing all the tail labels
    
    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe
    
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub

def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    if X.shape[0]>=5:
        nbs=NearestNeighbors(n_neighbors=5,metric='euclidean',algorithm='kd_tree').fit(X)
    else:
        nbs=NearestNeighbors(n_neighbors=3,metric='euclidean',algorithm='kd_tree').fit(X)
    euclidean,indices= nbs.kneighbors(X)
    return indices

def MLSMOTE(X,y, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0,n-1)
        neighbour = random.choice(indices2[reference,1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val>2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbour,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target

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
    model.add(layers.Dense(128,activation = 'relu'))
    model.add(layers.Dense(128,activation = 'relu'))
    model.add(layers.Dense(64,activation = 'relu'))
    model.add(layers.Dense(32,activation = 'relu'))
    model.add(layers.Dense(output_shape,activation = 'sigmoid'))
    model.compile(optimizer = 'nadam',loss = 'binary_crossentropy', metrics = get_minimal_multiclass_metrics())
    history = model.fit(x_train, y_train,
                        batch_size = 100, nb_epoch= 10,
                        verbose=1, validation_data=(x_val, y_val),
                        shuffle = True)
    return model

#get minority class data
x_sub, y_sub = get_minority_instace(x_train, y_train)           
x_res,y_res = MLSMOTE(x_sub, y_sub, 500)

mlknn_model = get_model_mlknn(x_train,y_train)
accuracy_score(y_val,mlknn_model.predict(lil_matrix(x_val).toarray()))

xgb_model = get_model_xgb(x_train,y_train)
accuracy_score(y_val,xgb_model.predict(x_val))

dl_model = get_model_dl(len(feature_cols),len(target_cols),x_train,y_train)
accuracy_score(y_val,dl_model.predict(x_val).round())

x_test = pd.read_csv(PROCESSED_TEST_PATH)
result = dl_model.predict(x_test[feature_cols])

result_2 = xgb_model.predict_proba(x_test[feature_cols])

preds_cal = []
for item in result_2:
    preds_cal.append( [local_item[1] for local_item in item] )


idxs = np.argsort(-1*(0.8*result+0.2*np.array(preds_cal).T))

preds = []
for row in idxs:
    prods = []
    for idx in row[:3]:
        prods.append(target_cols[idx].replace('future_',''))
    preds.append(prods)

customer_ids = pd.read_csv(RAW_TEST_PATH)
out_df = pd.concat([customer_ids['Customer_ID'],pd.Series(preds,name='Product_Holding_B2')],axis = 1)
out_df.to_csv(PREDICTION_FILE_PATH,index = False)

