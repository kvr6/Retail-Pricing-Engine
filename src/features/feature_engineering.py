import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

def load_processed_data(data_dir='data/processed'):
    data = {}
    for file in Path(data_dir).glob('*_processed.csv'):
        data[file.stem.replace('_processed', '')] = pd.read_csv(file)
    return data

def create_time_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def create_lag_features(df, lag_periods=[1, 7, 30]):
    for lag in lag_periods:
        df[f'sales_lag_{lag}'] = df.groupby('product_id')['total_amount'].shift(lag)
    return df

def calculate_price_elasticity(df):
    df['price_change'] = df.groupby('product_id')['price'].pct_change()
    df['demand_change'] = df.groupby('product_id')['quantity'].pct_change()
    df['price_elasticity'] = df['demand_change'] / df['price_change']
    df['price_elasticity'] = df['price_elasticity'].replace([np.inf, -np.inf], np.nan)
    return df

def create_customer_segments(customer_df, n_clusters=4):
    features = ['age']
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_df['customer_segment'] = kmeans.fit_predict(customer_df[features])
    return customer_df

def create_product_features(product_df):
    # One-hot encode the category
    category_dummies = pd.get_dummies(product_df['category'], prefix='category')
    product_df = pd.concat([product_df, category_dummies], axis=1)
    
    # Create price range categories
    product_df['price_range'] = pd.qcut(product_df['base_price'], q=4, labels=['low', 'medium-low', 'medium-high', 'high'])
    
    return product_df

def main():
    data = load_processed_data()
    
    # Feature engineering for sales data
    sales_df = data['sales_data']
    sales_df = create_time_features(sales_df)
    sales_df = create_lag_features(sales_df)
    sales_df = calculate_price_elasticity(sales_df)
    
    # Feature engineering for customer data
    customer_df = data['customer_data']
    customer_df = create_customer_segments(customer_df)
    
    # Feature engineering for product data
    product_df = data['product_data']
    product_df = create_product_features(product_df)
    
    # Merge additional features into sales data
    sales_df = sales_df.merge(customer_df[['customer_id', 'customer_segment']], on='customer_id', how='left')
    sales_df = sales_df.merge(product_df.drop(['base_price', 'cost'], axis=1), on='product_id', how='left')
    
    # Save the feature-engineered data
    output_dir = Path('data/feature_engineered')
    output_dir.mkdir(exist_ok=True)
    
    sales_df.to_csv(output_dir / "sales_features.csv", index=False)
    customer_df.to_csv(output_dir / "customer_features.csv", index=False)
    product_df.to_csv(output_dir / "product_features.csv", index=False)
    
    print("Feature engineering complete. Files saved in data/feature_engineered/")

if __name__ == "__main__":
    main()