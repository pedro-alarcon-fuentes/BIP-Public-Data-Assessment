# data_processing.py

"""
Module for processing company data.

Its responsibility is to load, sample, and perform BASIC cleaning of the dataset.
It delivers a DataFrame ready for more complex analysis, keeping the original
address columns intact.

It also provides utility functions such as `normalize_address`.
"""


import pandas as pd
import os

def normalize_address(address_str):
    """
    Takes an address string and converts it into a normalized key.
    This function is a 'utility' to be imported and used wherever needed.
    """
    if not isinstance(address_str, str):
        return ""
        
    address_str = address_str.lower().replace('.', '').replace(',', '')
    address_str = address_str.replace('co ', 'county ')
    tokens = address_str.split()
    stop_words = {'ireland', 'irl', 'romania', 'ro'}
    unique_tokens = set(tokens)
    cleaned_tokens = [token for token in unique_tokens if token not in stop_words]
    cleaned_tokens.sort()
    return " ".join(cleaned_tokens)

def load_and_prepare_companies(original_path, sample_size=None, clean_csv_name="companies_clean_sampled.csv"):
    """
    Loads the companies dataset, samples it, performs basic cleaning, and saves it.
    """

    # Smart caching/reprocessing logic
    if os.path.exists(clean_csv_name):
        df_check = pd.read_csv(clean_csv_name)
        if sample_size is not None and len(df_check) == sample_size:
            print(f"Found processed file '{clean_csv_name}' with matching size. Loading directly.")
            return df_check
        else:
            print(f"The processed file size does not match the requested size. Reprocessing...")

    # Load from original
    print(f"Loading data from original file '{original_path}'...")
    df = pd.read_csv(original_path)

    # Sampling
    if sample_size and len(df) > sample_size:
        print(f"Reducing to a random sample of {sample_size} rows.")
        df = df.sample(n=sample_size, random_state=2409)
    
    # --- CLEANING SECTION ---
    print("Starting basic data cleaning...")
    
    # 1. Clean address columns: Only fill null values.
    address_cols = ['company_address_1', 'company_address_2', 'company_address_3', 'company_address_4']
    for col in address_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # 2. Clean company name
    if 'company_name' in df.columns:
        df['company_name'] = df['company_name'].str.strip()

    print("Basic cleaning completed.")

    # Saving
    print(f"Saving processed DataFrame to '{clean_csv_name}'...")
    df.to_csv(clean_csv_name, index=False)
    
    return df