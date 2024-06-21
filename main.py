import os
from src.data_generation.generate_data import generate_and_save_data
from src.preprocessing.preprocess_data import preprocess_data
from src.feature_engineering.engineer_features import engineer_features

def main():
    # Generate data
    raw_data_file = os.path.join('data', 'raw', 'retail_data.csv')
    os.makedirs(os.path.dirname(raw_data_file), exist_ok=True)
    generate_and_save_data(raw_data_file)

    # Preprocess data
    preprocessed_data_dir = os.path.join('data', 'preprocessed')
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    preprocess_data(raw_data_file, preprocessed_data_dir)

    # Engineer features
    engineered_data_dir = os.path.join('data', 'engineered')
    os.makedirs(engineered_data_dir, exist_ok=True)
    engineer_features(raw_data_file, engineered_data_dir)

if __name__ == "__main__":
    main()