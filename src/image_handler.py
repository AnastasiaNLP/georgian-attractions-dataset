# -*- coding: utf-8 -*-
"""
image_handler.py — handle image copying, renaming, and statistics.
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm


def attach_images(df: pd.DataFrame, images_en: str, images_ru: str, output_dir: str) -> pd.DataFrame:
    """
    Copy and link images for each record in the unified dataset.

    Args:
        df: unified dataframe
        images_en: path to English images
        images_ru: path to Russian images
        output_dir: output directory (images will be copied there)

    Returns:
        df with new column 'image' pointing to the copied file
    """

    images_out = os.path.join(output_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    stats = {"images_copied": 0, "images_missing": 0}
    new_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        rec = row.to_dict()
        rec["image"] = None

        filename = rec.get("original_image_filename")
        if pd.notna(filename):
            source_dir = images_en if rec["language"] == "en" else images_ru
            src_path = os.path.join(source_dir, filename)

            if os.path.exists(src_path):
                ext = os.path.splitext(filename)[1]
                new_name = f"{rec['id']}{ext}"
                dst_path = os.path.join(images_out, new_name)
                shutil.copy2(src_path, dst_path)
                rec["image"] = new_name
                stats["images_copied"] += 1
            else:
                stats["images_missing"] += 1

        new_records.append(rec)

    df_out = pd.DataFrame(new_records)

    print(f"\n Images copied: {stats['images_copied']}")
    print(f" Missing images: {stats['images_missing']}")
    print(f" Output directory: {images_out}")

    return df_out
