"""
Georgian Attractions Dataset Creator

This module creates a HuggingFace dataset from CSV data and images.
Based on the actual Colab notebook workflow.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datasets import Dataset, Features, Value, Image as HFImage


class GeorgianAttractionsDataset:
    """
    Creates a HuggingFace dataset of Georgian attractions with images.

    This class handles:
    - Loading CSV data
    - Cleaning data (removing empty rows, reindexing)
    - Verifying image files
    - Creating HuggingFace dataset
    - Saving to disk
    """

    def __init__(self, csv_path: str, photos_folder: str):
        """
        Initialize the dataset creator.

        Args:
            csv_path: Path to CSV file with attraction data
            photos_folder: Path to folder containing images
        """
        self.csv_path = csv_path
        self.photos_folder = photos_folder
        self.df = None
        self.dataset = None

    def load_csv(self) -> pd.DataFrame:
        """
        Load CSV file with attraction data.

        Returns:
            Loaded DataFrame
        """
        print(f"📊 Loading CSV from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(self.df)} records")
        print(f"Columns: {list(self.df.columns)}")
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """
        Clean data by removing empty rows and reindexing IDs.

        This matches the exact process from the Colab notebook:
        1. Remove completely empty rows
        2. Reset index
        3. Create new sequential IDs from 1 to N

        Returns:
            Cleaned DataFrame
        """
        print("\n Cleaning data...")

        # Remove completely empty rows
        df_cleaned = self.df.dropna(how='all').copy()

        # Reset index and create new IDs
        df_cleaned.reset_index(drop=True, inplace=True)
        df_cleaned['id'] = range(1, len(df_cleaned) + 1)

        removed_rows = len(self.df) - len(df_cleaned)
        print(f" Removed {removed_rows} empty rows")
        print(f" Reindexed IDs from 1 to {len(df_cleaned)}")

        self.df = df_cleaned
        return self.df

    def verify_images(self) -> Tuple[int, List[Tuple]]:
        """
        Verify that images referenced in CSV exist in the photos folder.

        Returns:
            Tuple of (found_count, missing_photos_list)
            missing_photos_list contains tuples of (id, photo_name, place_name)
        """
        print(f"\n Verifying images in: {self.photos_folder}")

        if not os.path.exists(self.photos_folder):
            print(f" Photos folder not found: {self.photos_folder}")
            return 0, []

        # Get all files in photos folder
        photos_in_folder = set(os.listdir(self.photos_folder))
        print(f"Found {len(photos_in_folder)} files in folder")

        missing_photos = []
        existing_photos = []

        # Check each photo referenced in CSV
        for idx, row in self.df.iterrows():
            photo_name = row['photo_name']

            if pd.isna(photo_name) or photo_name == 'nan':
                missing_photos.append((row['id'], 'NaN', row['name']))
            elif photo_name not in photos_in_folder:
                missing_photos.append((row['id'], photo_name, row['name']))
            else:
                existing_photos.append(photo_name)

        print(f"Found images: {len(existing_photos)}")
        print(f"  Missing images: {len(missing_photos)}")

        if missing_photos and len(missing_photos) <= 10:
            print("\nMissing photos:")
            for photo_id, photo_name, name in missing_photos:
                print(f"  ID {photo_id}: {name} - {photo_name}")
        elif missing_photos:
            print(f"\n  {len(missing_photos)} photos missing (showing first 5):")
            for photo_id, photo_name, name in missing_photos[:5]:
                print(f"  ID {photo_id}: {name} - {photo_name}")

        return len(existing_photos), missing_photos

    def create_dataset(self) -> Dataset:
        """
        Create HuggingFace dataset from the cleaned data.

        This creates the dataset with proper features including images.

        Returns:
            Created HuggingFace Dataset
        """
        print("\nCreating HuggingFace dataset...")

        # Prepare dataset records
        dataset_records = []
        images_found = 0
        images_missing = 0

        print("Processing records...")
        for idx, row in self.df.iterrows():
            record = {
                'id': int(row['id']),
                'name': str(row['name']) if pd.notna(row['name']) else '',
                'description': str(row['description']) if pd.notna(row['description']) else '',
                'category': str(row['category']) if pd.notna(row['category']) else '',
                'location': str(row['location']) if pd.notna(row['location']) else '',
                'tags': str(row['tags']) if pd.notna(row['tags']) else '',
                'language': str(row['language']) if pd.notna(row['language']) else '',
                'photo_name': str(row['photo_name']) if pd.notna(row['photo_name']) else '',
                'license': str(row['license']) if pd.notna(row['license']) else '',
                'photo_author': str(row['photo_author']) if pd.notna(row['photo_author']) else ''
            }

            # Add image path if exists
            if pd.notna(row['photo_name']) and row['photo_name'] and row['photo_name'] != 'nan':
                photo_path = os.path.join(self.photos_folder, row['photo_name'])
                if os.path.exists(photo_path):
                    record['image'] = photo_path
                    images_found += 1
                else:
                    record['image'] = None
                    images_missing += 1
            else:
                record['image'] = None

            dataset_records.append(record)

            # Progress indicator
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(self.df)} records")

        print(f"\n Prepared {len(dataset_records)} records")
        print(f"   With images: {images_found}")
        print(f"   Without images: {images_missing}")

        # Define dataset features
        features = Features({
            'id': Value('int64'),
            'name': Value('string'),
            'description': Value('string'),
            'category': Value('string'),
            'location': Value('string'),
            'tags': Value('string'),
            'language': Value('string'),
            'photo_name': Value('string'),
            'license': Value('string'),
            'photo_author': Value('string'),
            'image': HFImage()
        })

        # Create dataset
        print("\nCreating HuggingFace Dataset object...")
        self.dataset = Dataset.from_list(dataset_records, features=features)

        print(f" Dataset created: {len(self.dataset)} records")

        return self.dataset

    def save_dataset(self, output_path: str) -> None:
        """
        Save the dataset to disk.

        Args:
            output_path: Path where to save the dataset
        """
        if self.dataset is None:
            raise ValueError("Dataset not created yet. Call create_dataset() first.")

        print(f"\n Saving dataset to: {output_path}")

        # Create output directory if needed
        os.makedirs(output_path, exist_ok=True)

        # Save dataset
        self.dataset.save_to_disk(output_path)

        print(" Dataset saved successfully!")

        # Calculate size
        total_size = sum(
            f.stat().st_size
            for f in Path(output_path).rglob('*')
            if f.is_file()
        ) / (1024 * 1024)

        print(f"📊 Dataset size: {total_size:.2f} MB")

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        if self.dataset is None:
            raise ValueError("Dataset not created yet. Call create_dataset() first.")

        images_count = sum(
            1 for i in range(len(self.dataset))
            if self.dataset[i]['image'] is not None
        )

        # Language distribution
        from collections import Counter
        languages = Counter(self.dataset['language'])

        stats = {
            'total_records': len(self.dataset),
            'with_images': images_count,
            'without_images': len(self.dataset) - images_count,
            'languages': dict(languages)
        }

        return stats

    def process(self, output_path: str) -> Dataset:
        """
        Complete pipeline: load → clean → verify → create → save.

        Args:
            output_path: Path where to save the dataset

        Returns:
            Created HuggingFace Dataset
        """
        print("="*70)
        print("  Georgian Attractions Dataset Creator")
        print("="*70)

        # Step 1: Load CSV
        self.load_csv()

        # Step 2: Clean data
        self.clean_data()

        # Step 3: Verify images
        self.verify_images()

        # Step 4: Create dataset
        self.create_dataset()

        # Step 5: Save dataset
        self.save_dataset(output_path)

        # Show statistics
        print("\n" + "="*70)
        print("  DATASET STATISTICS")
        print("="*70)
        stats = self.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")

        print("\n✨ Dataset creation complete!")

        return self.dataset


# Example usage
if __name__ == "__main__":
    # Create dataset
    creator = GeorgianAttractionsDataset(
        csv_path="dataset.csv",
        photos_folder="photo_dataset"
    )

    dataset = creator.process("georgian_attractions_dataset")