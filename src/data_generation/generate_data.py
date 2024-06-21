import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import os
import yaml

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

TARGET_RECORDS = config['TARGET_RECORDS']
BATCH_SIZE = config['BATCH_SIZE']
NUM_CORES = config['NUM_CORES']

def generate_batch(batch_size, batch_number):
    start_date = datetime(2023, 1, 1)
    end_date = start_date + timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    repeat_count = -(-batch_size // len(date_range))
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
    
    df['base_price'] = np.where(df['category'] == 'Electronics', df['base_price'] * 1.2, df['base_price'])
    df['base_price'] = np.where(df['location'] == 'US', df['base_price'] * 1.1, df['base_price'])
    df['base_price'] = np.where(df['is_holiday'] == 1, df['base_price'] * 0.9, df['base_price'])
    df['base_price'] = np.where(df['weather'] == 'Rainy', df['base_price'] * 1.05, df['base_price'])
    df['base_price'] = np.where(df['loyalty_score'] > 80, df['base_price'] * 0.95, df['base_price'])
    df['base_price'] = np.where(df['prev_purchases'] > 10, df['base_price'] * 0.97, df['base_price'])
    
    return df

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
                df.to_csv(filename, mode=mode, header=(mode=='w'), index=False)
                mode = 'a'

            print(f"Generated {total_records} records")

    print(f"Finished generating {total_records} records")

if __name__ == "__main__":
    np.random.seed(42)
    filename = os.path.join('data', 'raw', 'retail_data.csv')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    generate_and_save_data(filename)