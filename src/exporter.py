# -*- coding: utf-8 -*-
"""
exporter.py — save dataset files (CSV, JSONL, metadata).
"""

import os
import json
import pandas as pd


def save_csv(df: pd.DataFrame, output_dir: str):
    """Save unified dataset as CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "attractions.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f" Saved CSV → {csv_path}")


def save_jsonl(df: pd.DataFrame, output_dir: str):
    """Export dataset to JSONL (for Hugging Face)."""
    jsonl_path = os.path.join(output_dir, "dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f" Saved JSONL → {jsonl_path}")


def save_metadata(df: pd.DataFrame, output_dir: str):
    """Generate basic metadata with simple statistics."""
    metadata = {
        "records_total": len(df),
        "languages": df["language"].value_counts().to_dict(),
        "fields": list(df.columns),
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved → {meta_path}")


def export_all(df: pd.DataFrame, output_dir: str):
    """High-level function: save CSV, JSONL and metadata."""
    save_csv(df, output_dir)
    save_jsonl(df, output_dir)
    save_metadata(df, output_dir)
    print(f"Export complete: {len(df)} records")



