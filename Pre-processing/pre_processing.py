import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from concurrent.futures import ProcessPoolExecutor
import psutil

# Set the target number of records
TARGET_RECORDS = 50_000_000

# Determine the number of CPU cores
NUM_CORES = psutil.cpu_count(logical=False)

# Set batch size
BATCH_SIZE = 1_000_000

print(f"Generating {TARGET_RECORDS} records in batches of {BATCH_SIZE}")

def generate_batch(batch_size, batch_number):
    # Create a fixed date range for one year
    start_date = datetime(2023, 1, 1)
    end_date = start_date + timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Calculate how many times to repeat the date range
    repeat_count = -(-batch_size // len(date_range))  # Ceiling division
    
    # Create the timestamp column by repeating the date range
    timestamps = pd.Series(np.tile(date_range.values, repeat_count)[:batch_size])

    data = {
        'timestamp': timestamps,
        'product_id': np.random.randint(1, 100001, size=batch_size),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books', 'Toys', 'Grocery', 'Automotive', 'Garden'], size=batch_size),
        'base_price': np.random.uniform(1, 10000, size=batch_size),
        'cost': np.random.uniform(0.5, 8000, size=batch_size),
        'inventory': np.random.randint(0, 10000, size=batch_size),
        'customer_id': np.random.randint(1, 1000001, size=batch_size),
        'age': np.random.randint(18, 100, size=batch_size),
        'gender': np.random.choice(['M', 'F', 'Other'], size=batch_size),
        'location': np.random.choice(['US', 'Japan', 'Australia', 'Singapore', 'South Korea', 'UK', 'Germany', 'France', 'Canada', 'Brazil'], size=batch_size),
        'device': np.random.choice(['Mobile', 'Desktop', 'Tablet', 'Smart TV', 'Wearable'], size=batch_size),
        'day_of_week': timestamps.dt.dayofweek,
        'hour': timestamps.dt.hour,
        'is_holiday': np.random.choice([0, 1], size=batch_size, p=[0.97, 0.03]),
        'competitor_price': np.random.uniform(0.8, 12000, size=batch_size),
        'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Snowy'], size=batch_size),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], size=batch_size),
        'prev_purchases': np.random.poisson(lam=5, size=batch_size),
        'loyalty_score': np.random.uniform(0, 100, size=batch_size),
        'discount_eligibility': np.random.choice([0, 1], size=batch_size, p=[0.7, 0.3]),
    }
    
    df = pd.DataFrame(data)
    
    # Add complex patterns
    df['base_price'] = np.where(df['category'] == 'Electronics', df['base_price'] * 1.2, df['base_price'])
    df['base_price'] = np.where(df['location'] == 'US', df['base_price'] * 1.1, df['base_price'])
    df['base_price'] = np.where(df['is_holiday'] == 1, df['base_price'] * 0.9, df['base_price'])
    df['base_price'] = np.where(df['weather'] == 'Rainy', df['base_price'] * 1.05, df['base_price'])
    df['base_price'] = np.where(df['loyalty_score'] > 80, df['base_price'] * 0.95, df['base_price'])
    df['base_price'] = np.where(df['prev_purchases'] > 10, df['base_price'] * 0.97, df['base_price'])
    
    return df

def save_to_csv(df, filename, mode='a'):
    df.to_csv(filename, mode=mode, header=(mode=='w'), index=False)

def generate_and_save_data(filename):
    mode = 'w'
    total_records = 0

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        batch_number = 0
        while total_records < TARGET_RECORDS:
            futures = []
            for _ in range(NUM_CORES):
                if total_records + BATCH_SIZE > TARGET_RECORDS:
                    current_batch_size = TARGET_RECORDS - total_records
                else:
                    current_batch_size = BATCH_SIZE
                
                if current_batch_size > 0:
                    futures.append(executor.submit(generate_batch, current_batch_size, batch_number))
                    total_records += current_batch_size
                    batch_number += 1

                if total_records >= TARGET_RECORDS:
                    break

            for future in futures:
                df = future.result()
                save_to_csv(df, filename, mode)
                mode = 'a'  # Switch to append mode after first write

            print(f"Generated {total_records} records")

    print(f"Finished generating {total_records} records")

def preprocess_data(filename):
    print("Starting preprocessing...")
    
    # Define features
    numeric_features = ['base_price', 'cost', 'inventory', 'age', 'competitor_price', 'prev_purchases', 'loyalty_score']
    categorical_features = ['category', 'gender', 'location', 'device', 'weather', 'season']

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit preprocessor on a sample of the data
    print("Fitting preprocessor on a data sample...")
    sample_size = 1_000_000  # Use 1 million records for fitting
    df_sample = pd.read_csv(filename, nrows=sample_size)
    preprocessor.fit(df_sample[numeric_features + categorical_features])

    # Save the preprocessor
    joblib.dump(preprocessor, 'preprocessor.joblib')
    print("Preprocessor saved.")

    # Process data in chunks
    chunksize = BATCH_SIZE
    total_processed = 0
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        # Preprocess the chunk
        X = preprocessor.transform(chunk[numeric_features + categorical_features])
        
        # Save the preprocessed chunk
        pd.DataFrame(X).to_csv(f'preprocessed_data_{total_processed}.csv', index=False)
        
        total_processed += len(chunk)
        print(f"Processed {total_processed} records")

        if total_processed >= TARGET_RECORDS:
            break

    print("Preprocessing completed.")

def engineer_features(filename):
    print("Starting feature engineering...")
    chunksize = BATCH_SIZE
    total_processed = 0

    for chunk in pd.read_csv(filename, chunksize=chunksize):
        chunk['is_weekend'] = chunk['day_of_week'].isin([5, 6]).astype(int)
        chunk['is_evening'] = chunk['hour'].between(18, 23).astype(int)
        chunk['price_diff'] = chunk['base_price'] - chunk['competitor_price']
        chunk['price_ratio'] = chunk['base_price'] / chunk['competitor_price']
        chunk['customer_segment'] = pd.qcut(chunk['loyalty_score'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        chunk['low_stock'] = (chunk['inventory'] < 50).astype(int)
        chunk['high_demand'] = ((chunk['prev_purchases'] > chunk['prev_purchases'].quantile(0.75)) & 
                                (chunk['inventory'] < chunk['inventory'].quantile(0.25))).astype(int)
        chunk['seasonal_factor'] = chunk.apply(lambda row: 1.1 if (row['season'] == 'Winter' and row['category'] in ['Clothing', 'Sports']) or 
                                               (row['season'] == 'Summer' and row['category'] in ['Garden', 'Sports']) else 1, axis=1)

        # Save the engineered chunk
        chunk.to_csv(f'engineered_data_{total_processed}.csv', index=False)
        
        total_processed += len(chunk)
        print(f"Engineered features for {total_processed} records")

        if total_processed >= TARGET_RECORDS:
            break

    print("Feature engineering completed.")

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    
    filename = 'retail_data.csv'
    
    generate_and_save_data(filename)
    preprocess_data(filename)
    engineer_features(filename)