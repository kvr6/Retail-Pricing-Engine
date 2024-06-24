import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from faker import Faker

fake = Faker()

def load_data(data_dir='data/raw'):
    data = {}
    for file in Path(data_dir).glob('*.csv'):
        data[file.stem] = pd.read_csv(file)
    return data

def handle_missing_values(df):
    # Fill numeric columns with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Fill categorical columns with mode
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
    
    return df

def handle_outliers(df, columns, n_std=3):
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = df[column].clip(lower=mean - n_std*std, upper=mean + n_std*std)
    return df

def normalize_and_standardize(df, columns_to_standardize):
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    return df

def anonymize_data(df, columns_to_anonymize):
    for column in columns_to_anonymize:
        if column == 'name':
            df[column] = [fake.name() for _ in range(len(df))]
        elif column == 'email':
            df[column] = [fake.email() for _ in range(len(df))]
        elif column == 'city':
            df[column] = [fake.city() for _ in range(len(df))]
    return df

def preprocess_sales_data(df):
    df = handle_missing_values(df)
    df = handle_outliers(df, ['quantity', 'price'])
    
    # Convert date to datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate total sale amount
    df['total_amount'] = df['quantity'] * df['price']
    
    # Standardize numeric columns
    df = normalize_and_standardize(df, ['quantity', 'price', 'total_amount'])
    
    return df

def preprocess_product_data(df):
    df = handle_missing_values(df)
    df = handle_outliers(df, ['base_price', 'cost'])
    
    # Calculate profit margin
    df['profit_margin'] = (df['base_price'] - df['cost']) / df['base_price']
    
    # Standardize numeric columns
    df = normalize_and_standardize(df, ['base_price', 'cost', 'profit_margin'])
    
    return df

def preprocess_customer_data(df):
    df = handle_missing_values(df)
    df = handle_outliers(df, ['age'])
    
    # Convert to categorical
    df['gender'] = df['gender'].astype('category')
    df['membership_level'] = df['membership_level'].astype('category')
    
    # Standardize age
    df = normalize_and_standardize(df, ['age'])
    
    # Anonymize personal data
    df = anonymize_data(df, ['name', 'email', 'city'])
    
    return df

def preprocess_store_data(df):
    df = handle_missing_values(df)
    
    # Convert to categorical
    df['size'] = df['size'].astype('category')
    
    # Anonymize city
    df = anonymize_data(df, ['city'])
    
    return df

def preprocess_external_data(df):
    df = handle_missing_values(df)
    df = handle_outliers(df, ['competitor_price'])
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Convert market_trend to categorical
    df['market_trend'] = df['market_trend'].astype('category')
    
    # Standardize competitor_price
    df = normalize_and_standardize(df, ['competitor_price'])
    
    return df

def align_time_series(data):
    # Ensure all datasets cover the same time period
    min_date = max(data['sales_data']['date'].min(), data['external_data']['date'].min())
    max_date = min(data['sales_data']['date'].max(), data['external_data']['date'].max())
    
    data['sales_data'] = data['sales_data'][(data['sales_data']['date'] >= min_date) & (data['sales_data']['date'] <= max_date)]
    data['external_data'] = data['external_data'][(data['external_data']['date'] >= min_date) & (data['external_data']['date'] <= max_date)]
    
    return data

def main():
    data = load_data()
    
    # Preprocess each dataset
    data['sales_data'] = preprocess_sales_data(data['sales_data'])
    data['product_data'] = preprocess_product_data(data['product_data'])
    data['customer_data'] = preprocess_customer_data(data['customer_data'])
    data['store_data'] = preprocess_store_data(data['store_data'])
    data['external_data'] = preprocess_external_data(data['external_data'])
    
    # Align time series data
    data = align_time_series(data)
    
    # Save preprocessed data
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    
    for name, df in data.items():
        df.to_csv(output_dir / f"{name}_processed.csv", index=False)
        print(f"Saved processed {name} to {output_dir}/{name}_processed.csv")

if __name__ == "__main__":
    main()