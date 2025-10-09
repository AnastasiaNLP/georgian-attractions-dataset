# -*- coding: utf-8 -*-
"""
Configuration for the Georgian Attractions Dataset Builder.
"""

import os

# Local paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # project root folder
DATA_DIR = os.path.join(BASE_DIR, "examples")           # CSV files and images
OUTPUT_DIR = os.path.join(BASE_DIR, "output_example")   # final dataset

IMAGES_EN = os.path.join(DATA_DIR, "images_en")
IMAGES_RU = os.path.join(DATA_DIR, "images_ru")

# ---------- Hugging Face ----------
HF_REPO = "AIAnastasia/Georgian-attractions"
HF_PRIVATE = False
HF_COMMIT_MSG = "Dataset update from local builder"

# File settings
CSV_ENCODING = "utf-8"
SEED = 42
