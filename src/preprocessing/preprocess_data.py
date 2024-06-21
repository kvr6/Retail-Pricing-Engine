import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import yaml

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

TARGET_RECORDS = config['TARGET_RECORDS']
BATCH_SIZE = config['BATCH_SIZE']

def preprocess_data(input_filename, output_dir):
    print("Starting preprocessing...")
    
    numeric_features = ['base_price', 'cost', 'inventory', 'age', 'competitor_price', 'prev_purchases', 'loyalty_score']
    categorical_features = ['category', 'gender', 'location', 'device', 'weather', 'season']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    print("Fitting preprocessor on a data sample...")
    sample_size = 1_000_000
    df_sample = pd.read_csv(input_filename, nrows=sample_size)
    preprocessor.fit(df_sample[numeric_features + categorical_features])

    # Create 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(preprocessor, os.path.join('models', 'preprocessor.joblib'))
    print("Preprocessor saved.")

    chunksize = BATCH_SIZE
    total_processed = 0
    for chunk in pd.read_csv(input_filename, chunksize=chunksize):
        X = preprocessor.transform(chunk[numeric_features + categorical_features])
        
        output_filename = os.path.join(output_dir, f'preprocessed_data_{total_processed}.csv')
        pd.DataFrame(X).to_csv(output_filename, index=False)
        
        total_processed += len(chunk)
        print(f"Processed {total_processed} records")

        if total_processed >= TARGET_RECORDS:
            break

    print("Preprocessing completed.")

if __name__ == "__main__":
    input_filename = os.path.join('data', 'raw', 'retail_data.csv')
    output_dir = os.path.join('data', 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)
    preprocess_data(input_filename, output_dir)