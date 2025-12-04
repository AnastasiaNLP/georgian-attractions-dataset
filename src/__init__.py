"""Georgian Attractions Dataset Creator

Tools for creating a bilingual HuggingFace dataset of Georgian tourist attractions
with images, descriptions in Russian and English.
"""

__version__ = "1.0.0"
__author__ = "AIAnastasia"

from .dataset_creator import GeorgianAttractionsDataset

__all__ = ['GeorgianAttractionsDataset']