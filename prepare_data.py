#import standard libraries
import pandas as pd
import numpy as np
from pandas.core.arrays import categorical
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import joblib
import argparse

#import user defined libraries
from constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def prepare_data(data_type = 'train',
                 input_data_path = '',
                 output_data_path = '',
                 artifacts_path = '',
                 categorical_features = [],
                 numerical_features = [],
                 ):
    #load data
    df = pd.read_csv(input_data_path)

    #standardize column names
    df.rename(columns = {col:col.lower() for col in df.columns},inplace = True)

    #unpack product holding columns - product_holding_b1, product_holding_b2
    current_products = pd.DataFrame([{'current_'+prod:1 for prod in eval(row['product_holding_b1'])}  for i,row in df.iterrows() for prod in row]).fillna(0)
 
    if data_type == 'train':
        #this wont be present in our validation / test datasets
        future_products = pd.DataFrame([{'future_'+prod:1 for prod in eval(row['product_holding_b2'])}  for i,row in df.iterrows() for prod in row]).fillna(0)

        #get dummies for categorical features
        ohe_encoder = OneHotEncoder(handle_unknown='ignore',sparse = False).fit(df[categorical_features])
        categorical_features_df = pd.DataFrame(ohe_encoder.transform(df[categorical_features]),columns = ohe_encoder.get_feature_names_out(categorical_features))

        #get scaled values of numeric features
        standard_scaler = StandardScaler().fit(df[numerical_features])
        numerical_features_df = pd.DataFrame(standard_scaler.transform(df[numerical_features]),columns = numerical_features)

        #save our data transformers
        joblib.dump(ohe_encoder,artifacts_path+'ohe_obj.pkl')
        joblib.dump(standard_scaler,artifacts_path+'standard_scaler.pkl')
    
    elif data_type == 'test':
        #load our data transformers and call the transform function on the test data
        ohe_encoder = joblib.load(artifacts_path+'ohe_obj.pkl')
        standard_scaler = joblib.load(artifacts_path+'standard_scaler.pkl')

        #call transform
        categorical_features_df = pd.DataFrame(ohe_encoder.transform(df[categorical_features]),columns = ohe_encoder.get_feature_names_out(categorical_features))
        numerical_features_df = pd.DataFrame(standard_scaler.transform(df[numerical_features]),columns = numerical_features)

    #combine all individual datasets together
    out_df = pd.concat([numerical_features_df,categorical_features_df,current_products],axis = 1)

    if data_type == 'train':
        out_df = pd.concat([out_df,future_products],axis = 1)

    #write out the file to disk
    out_df.to_csv(output_data_path,index = False)

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Prep code')
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--input_data_path', type=str)
    parser.add_argument('--output_data_path', type=str)
    parser.add_argument('--artifact_path', type=str)
    args = parser.parse_args()

    '''
    python3 prepare_data.py --data_type=train \
                            --input_data_path=/home/code/data/train_go05W65.csv \
                            --output_data_path=/home/code/data/processed_train.csv\
                            --artifact_path=/home/code/artifacts/
    
    python3 prepare_data.py --data_type=test \
                            --input_data_path=/home/code/data/test_VkM91FT.csv \
                            --output_data_path=/home/code/data/processed_test.csv\
                            --artifact_path=/home/code/artifacts/
    '''
    prepare_data(data_type = args.data_type,
                 input_data_path = args.input_data_path,
                 output_data_path = args.output_data_path,
                 artifacts_path = args.artifact_path,
                 categorical_features = CATEGORICAL_FEATURES,
                 numerical_features = NUMERICAL_FEATURES,
                 )