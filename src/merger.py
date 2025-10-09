# -*- coding: utf-8 -*-
"""
merger.py — merge English and Russian CSV data into a unified schema.
"""

import pandas as pd


def unify_schema(df_en: pd.DataFrame, df_ru: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns, unify schema, and merge English + Russian datasets.
    """

    # Renaming columns for EN
    df_en.rename(columns={
        "name_en": "name",
        "description_en": "description",
        "location_en": "location",
        "category_en": "category",
        "ner_en": "ner",
        "photo_filename_en": "original_image_filename",
    }, inplace=True)
    df_en["language"] = "en"

    # Renaming columns for RU
    df_ru.rename(columns={
        "name_ru": "name",
        "description_ru": "description",
        "location_ru": "location",
        "category_ru": "category",
        "ner_ru": "ner",
        "photo_filename_ru": "original_image_filename",
    }, inplace=True)
    df_ru["language"] = "ru"

    #Unification
    df = pd.concat([df_en, df_ru], ignore_index=True)

    #Removing empty lines
    df.dropna(subset=["name"], inplace=True)

    # Adding ID
    df.insert(0, "id", [f"attraction_{i:04d}" for i in range(len(df))])

    print(f" Unified dataset created: {len(df)} total rows")
    return df


def show_language_stats(df: pd.DataFrame):
    """
    Print the number of EN/RU records.
    """
    lang_stats = df["language"].value_counts().to_dict()
    print(f"Language distribution: {lang_stats}")
