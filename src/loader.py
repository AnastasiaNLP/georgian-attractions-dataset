# -*- coding: utf-8 -*-
"""
loader.py — module for reading and validating CSV source files.
"""

import os
import pandas as pd
from config import CSV_ENCODING


def load_data(en_csv_path: str, ru_csv_path: str):
    """
    Load English and Russian CSV files, check that they exist and readable.
    """
    # Checking for the presence of files
    if not os.path.exists(en_csv_path):
        raise FileNotFoundError(f"English CSV not found: {en_csv_path}")
    if not os.path.exists(ru_csv_path):
        raise FileNotFoundError(f"Russian CSV not found: {ru_csv_path}")

    # Uploading CSV
    df_en = pd.read_csv(en_csv_path, encoding=CSV_ENCODING)
    df_ru = pd.read_csv(ru_csv_path, encoding=CSV_ENCODING)

    print(f"Loaded {len(df_en)} English rows")
    print(f"Loaded {len(df_ru)} Russian rows")

    return df_en, df_ru


def preview_dataframe(df: pd.DataFrame, n: int = 3):
    """
    Print a short preview of a DataFrame.
    """
    print(f"\nPreview ({n} rows):")
    print(df.head(n).to_markdown(index=False))
