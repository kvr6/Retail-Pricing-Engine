import pandas as pd
import numpy as np
import random
from faker import Faker
import os

# Initialize Faker and random seed
fake = Faker()
np.random.seed(42)

# Constants
NUM_RECORDS = 1000
NUM_CUSTOMERS = 200
NUM_PRODUCTS = 50
COMPETITORS = ['BestBuy', 'Target', 'Walmart', 'Costco']
SEASONS = ['Spring', 'Summer', 'Fall', 'Winter']
HOLIDAYS = ['None', 'Christmas', 'Thanksgiving', 'Valentine\'s Day', 'Black Friday', 'Cyber Monday']

# Helper functions
def random_product_category():
    return random.choice(['Electronics', 'Computers', 'Appliances', 'Mobile', 'Accessories'])

def random_brand():
    return random.choice(['Sony', 'Apple', 'Samsung', 'Dell', 'LG', 'HP', 'Lenovo'])

def random_attributes(category):
    attributes = {
        'Electronics': {"size": f"{random.randint(24, 65)} inches", "resolution": random.choice(["4K", "1080p"]), "smart": random.choice(["Yes", "No"])},
        'Computers': {"processor": random.choice(["Intel i5", "Intel i7", "AMD Ryzen 5", "AMD Ryzen 7"]), "RAM": f"{random.randint(8, 32)}GB", "storage": f"{random.randint(256, 1000)}GB SSD"},
        'Appliances': {"capacity": f"{random.randint(100, 500)}L", "type": random.choice(["Single Door", "Double Door"]), "color": random.choice(["Silver", "Black", "White"])},
        'Mobile': {"model": random.choice(["iPhone", "Galaxy", "Pixel"]), "storage": f"{random.randint(64, 512)}GB", "color": random.choice(["Black", "White", "Blue", "Red"])},
        'Accessories': {"type": random.choice(["Charger", "Headphones", "Case"]), "compatibility": random.choice(["iOS", "Android", "Universal"])}
    }
    return attributes.get(category, {})

def generate_date_sequence(start_date, end_date, n):
    return pd.date_range(start=start_date, end=end_date, periods=n)

# Generate Historical Sales Data
sales_data = []
dates = generate_date_sequence('2024-01-01', '2024-06-15', NUM_RECORDS)

for i in range(NUM_RECORDS):
    transaction_id = f"T{i+1:05d}"
    customer_id = f"C{random.randint(1, NUM_CUSTOMERS):03d}"
    product_id = f"P{random.randint(1, NUM_PRODUCTS):03d}"
    date = dates[i]
    quantity = random.randint(1, 5)
    price = round(random.uniform(50, 2000), 2)
    discount = round(random.uniform(0, 0.3), 2)
    total_amount = round(quantity * price * (1 - discount), 2)
    sales_data.append([transaction_id, customer_id, product_id, date, quantity, price, discount, total_amount])

sales_df = pd.DataFrame(sales_data, columns=["TransactionID", "CustomerID", "ProductID", "TransactionDate", "Quantity", "Price", "Discount", "TotalAmount"])

# Generate Customer Data
customer_data = []

for i in range(NUM_CUSTOMERS):
    customer_id = f"C{i+1:03d}"
    name = fake.name()
    email = fake.email()
    age = random.randint(18, 70)
    gender = random.choice(['M', 'F'])
    location = fake.city()
    purchase_history = random.sample([f"T{random.randint(1, NUM_RECORDS):05d}" for _ in range(10)], k=random.randint(1, 5))
    customer_data.append([customer_id, name, email, age, gender, location, purchase_history])

customer_df = pd.DataFrame(customer_data, columns=["CustomerID", "Name", "Email", "Age", "Gender", "Location", "PurchaseHistory"])

# Generate Product Data
product_data = []

for i in range(NUM_PRODUCTS):
    product_id = f"P{i+1:03d}"
    product_name = fake.word().capitalize() + " " + random.choice(["TV", "Laptop", "Smartphone", "Refrigerator", "Headphones"])
    category = random_product_category()
    brand = random_brand()
    stock_level = random.randint(0, 200)
    price = round(random.uniform(50, 2000), 2)
    attributes = random_attributes(category)
    product_data.append([product_id, product_name, category, brand, stock_level, price, attributes])

product_df = pd.DataFrame(product_data, columns=["ProductID", "ProductName", "Category", "Brand", "StockLevel", "Price", "Attributes"])

# Generate Competitor Pricing Data
competitor_pricing_data = []

for i in range(NUM_RECORDS):
    competitor_name = random.choice(COMPETITORS)
    product_id = f"P{random.randint(1, NUM_PRODUCTS):03d}"
    competitor_price = round(random.uniform(50, 2000), 2)
    date_checked = random.choice(dates)
    competitor_pricing_data.append([competitor_name, product_id, competitor_price, date_checked])

competitor_pricing_df = pd.DataFrame(competitor_pricing_data, columns=["CompetitorName", "ProductID", "CompetitorPrice", "DateChecked"])

# Generate Seasonal Trends Data
seasonal_trends_data = []

for date in dates:
    season = random.choice(SEASONS)
    holiday = random.choice(HOLIDAYS)
    sales_trend = round(random.uniform(0.8, 2.0), 2) if holiday != 'None' else round(random.uniform(0.8, 1.2), 2)
    seasonal_trends_data.append([date, season, holiday, sales_trend])

seasonal_trends_df = pd.DataFrame(seasonal_trends_data, columns=["Date", "Season", "Holiday", "SalesTrend"])

# Ensure the Data directory exists
data_dir = os.path.join(os.path.dirname(__file__), 'Data')
os.makedirs(data_dir, exist_ok=True)

# Save to CSV
sales_df.to_csv(os.path.join(data_dir, 'historical_sales_data.csv'), index=False)
customer_df.to_csv(os.path.join(data_dir, 'customer_data.csv'), index=False)
product_df.to_csv(os.path.join(data_dir, 'product_data.csv'), index=False)
competitor_pricing_df.to_csv(os.path.join(data_dir, 'competitor_pricing_data.csv'), index=False)
seasonal_trends_df.to_csv(os.path.join(data_dir, 'seasonal_trends_data.csv'), index=False)

print("Mock data generated and saved in 'Data' folder.")
