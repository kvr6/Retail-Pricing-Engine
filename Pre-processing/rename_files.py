import os
import glob

def rename_files(pattern, new_pattern):
    files = sorted(glob.glob(pattern))
    for i, file in enumerate(files):
        if i >= 50:
            break
        new_name = new_pattern.format(i)
        os.rename(file, new_name)
        print(f"Renamed {file} to {new_name}")

if __name__ == "__main__":
    # Rename preprocessed data files
    rename_files("preprocessed_data_*.csv", "preprocessed_data_{}.csv")
    
    # Rename engineered data files
    rename_files("engineered_data_*.csv", "engineered_data_{}.csv")

    print("File renaming completed.")