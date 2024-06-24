import csv
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker()

def generate_sales_data(num_records=1000000, output_dir='data/raw'):
    filename = os.path.join(output_dir, "sales_data.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['transaction_id', 'date', 'store_id', 'product_id', 'customer_id', 'quantity', 'price'])
        
        for i in range(num_records):
            writer.writerow([
                fake.uuid4(),
                fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S'),
                random.randint(1, 25),  # Assuming 25 stores
                random.randint(1, 1000),  # Assuming 1000 products
                random.randint(1, 100000),  # Assuming 100,000 customers
                random.randint(1, 10),
                round(random.uniform(5.0, 500.0), 2)
            ])
    print(f"Generated {filename}")

def generate_product_data(num_records=1000, output_dir='data/raw'):
    filename = os.path.join(output_dir, "product_data.csv")
    os.makedirs(output_dir, exist_ok=True)
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys', 'Food', 'Beauty']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['product_id', 'name', 'category', 'base_price', 'cost'])
        
        for i in range(num_records):
            base_price = round(random.uniform(10.0, 1000.0), 2)
            writer.writerow([
                i + 1,
                fake.product_name(),
                random.choice(categories),
                base_price,
                round(base_price * random.uniform(0.4, 0.7), 2)  # Cost is 40-70% of base price
            ])
    print(f"Generated {filename}")

def generate_customer_data(num_records=100000, output_dir='data/raw'):
    filename = os.path.join(output_dir, "customer_data.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['customer_id', 'name', 'email', 'age', 'gender', 'city', 'membership_level'])
        
        for i in range(num_records):
            writer.writerow([
                i + 1,
                fake.name(),
                fake.email(),
                random.randint(18, 80),
                random.choice(['M', 'F']),
                fake.city(),
                random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'])
            ])
    print(f"Generated {filename}")

def generate_store_data(num_records=25, output_dir='data/raw'):
    filename = os.path.join(output_dir, "store_data.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['store_id', 'name', 'city', 'state', 'country', 'size'])
        
        for i in range(num_records):
            writer.writerow([
                i + 1,
                f"Store {i+1}",
                fake.city(),
                fake.state(),
                fake.country(),
                random.choice(['Small', 'Medium', 'Large'])
            ])
    print(f"Generated {filename}")

def generate_external_data(num_records=1000000, output_dir='data/raw'):
    filename = os.path.join(output_dir, "external_data.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'product_id', 'competitor_price', 'market_trend'])
        
        start_date = datetime.now() - timedelta(days=365)
        for i in range(num_records):
            date = start_date + timedelta(days=random.randint(0, 365))
            writer.writerow([
                date.strftime('%Y-%m-%d'),
                random.randint(1, 1000),  # Assuming 1000 products
                round(random.uniform(5.0, 500.0), 2),
                random.choice(['Rising', 'Falling', 'Stable'])
            ])
    print(f"Generated {filename}")

if __name__ == "__main__":
    output_dir = 'data/raw'
    generate_sales_data(output_dir=output_dir)
    generate_product_data(output_dir=output_dir)
    generate_customer_data(output_dir=output_dir)
    generate_store_data(output_dir=output_dir)
    generate_external_data(output_dir=output_dir)