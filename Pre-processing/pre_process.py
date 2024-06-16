import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import json
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define data folder path relative to the script directory
data_folder = os.path.join(script_dir, '..', 'Data')

# Load Data from 'Data' folder
sales_df = pd.read_csv(os.path.join(data_folder, 'historical_sales_data.csv'))
customer_df = pd.read_csv(os.path.join(data_folder, 'customer_data.csv'))
product_df = pd.read_csv(os.path.join(data_folder, 'product_data.csv'))
competitor_pricing_df = pd.read_csv(os.path.join(data_folder, 'competitor_pricing_data.csv'))
seasonal_trends_df = pd.read_csv(os.path.join(data_folder, 'seasonal_trends_data.csv'))

# Convert TransactionDate to datetime
sales_df['TransactionDate'] = pd.to_datetime(sales_df['TransactionDate'])

# Extract date part from TransactionDate
sales_df['TransactionDate_Date'] = sales_df['TransactionDate'].dt.date.astype(str)

# Merge DataFrames
data = sales_df.merge(customer_df, on='CustomerID', how='left')
data = data.merge(product_df, on='ProductID', how='left')
data = data.merge(competitor_pricing_df, on=['ProductID'], how='left', suffixes=('', '_competitor'))
data = data.merge(seasonal_trends_df, left_on='TransactionDate_Date', right_on='Date', how='left')

# Drop unnecessary columns
data.drop(columns=['Date', 'PurchaseHistory', 'TransactionDate_Date'], inplace=True)

# Handle missing values for specific columns before applying global imputation
data['Season'].fillna('Unknown', inplace=True)
data['Holiday'].fillna('None', inplace=True)
data['SalesTrend'].fillna(1.0, inplace=True)

# Handle Missing Values
imputer = SimpleImputer(strategy='most_frequent')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Convert JSON-like strings to dictionaries and normalize
def parse_attributes(attributes):
    if pd.isnull(attributes):
        return {}
    try:
        return json.loads(attributes.replace("'", "\""))
    except json.JSONDecodeError:
        return {}

# Parse and expand product attributes
attributes_expanded = data_filled['Attributes'].apply(parse_attributes).apply(pd.Series)
data_filled = data_filled.join(attributes_expanded).drop(columns=['Attributes'])

# Convert categorical data to numerical
categorical_features = ['Category', 'Brand', 'Gender', 'Location', 'CompetitorName', 'Season', 'Holiday']
numerical_features = ['Age', 'Quantity', 'Price', 'Discount', 'CompetitorPrice', 'StockLevel', 'SalesTrend']

# Define preprocessing for categorical and numerical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit and transform the data
data_processed = preprocessor.fit_transform(data_filled)

# Convert to DataFrame
processed_columns = (numerical_features + 
                     list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)))
data_processed_df = pd.DataFrame(data_processed, columns=processed_columns)

# Display the first few rows of the processed data
print("Processed Data:")
print(data_processed_df.head())

# Split data into train and test sets
X = data_processed_df.drop(columns=['TotalAmount'])
y = data_filled['TotalAmount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
X_train.to_csv(os.path.join(script_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(script_dir, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(script_dir, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(script_dir, 'y_test.csv'), index=False)
