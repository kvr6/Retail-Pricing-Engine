import pandas as pd
import numpy as np
import os
import yaml

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

TARGET_RECORDS = config['TARGET_RECORDS']
BATCH_SIZE = config['BATCH_SIZE']

def engineer_features(input_filename, output_dir):
    print("Starting feature engineering...")
    chunksize = BATCH_SIZE
    total_processed = 0

    for chunk in pd.read_csv(input_filename, chunksize=chunksize):
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

        output_filename = os.path.join(output_dir, f'engineered_data_{total_processed}.csv')
        chunk.to_csv(output_filename, index=False)
        
        total_processed += len(chunk)
        print(f"Engineered features for {total_processed} records")

        if total_processed >= TARGET_RECORDS:
            break

    print("Feature engineering completed.")

if __name__ == "__main__":
    input_filename = os.path.join('data', 'raw', 'retail_data.csv')
    output_dir = os.path.join('data', 'engineered')
    os.makedirs(output_dir, exist_ok=True)
    engineer_features(input_filename, output_dir)