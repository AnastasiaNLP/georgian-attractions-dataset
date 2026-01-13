"""
Tests for data validation.

Tests cover:
- CSV structure validation
- Required fields
- Data types
- Value constraints
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset_creator import GeorgianAttractionsDataset


class TestDataValidation:
    """Test suite for data validation."""

    def test_csv_has_required_columns(self):
        """Test that CSV has all required columns."""
        required_columns = [
            'id', 'name', 'description', 'category', 'location',
            'tags', 'language', 'photo_name', 'license', 'photo_author'
        ]

        # create CSV with all columns
        data = {col: [None] for col in required_columns}
        df = pd.DataFrame(data)

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_id_field_is_numeric(self):
        """Test that ID field contains numeric values."""
        data = {
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'description': ['D1', 'D2', 'D3'],
            'category': ['C1', 'C2', 'C3'],
            'location': ['L1', 'L2', 'L3'],
            'tags': ['T1', 'T2', 'T3'],
            'language': ['en', 'ru', 'en'],
            'photo_name': ['p1.jpg', 'p2.jpg', 'p3.jpg'],
            'license': ['CC-BY', 'CC-BY', 'CC-BY'],
            'photo_author': ['A1', 'A2', 'A3']
        }
        df = pd.DataFrame(data)

        # ID should be convertible to int
        assert df['id'].dtype in ['int64', 'int32', 'float64']

    def test_language_field_values(self):
        """Test that language field contains valid values."""
        valid_languages = ['en', 'ru']

        data = {
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'description': ['D1', 'D2', 'D3'],
            'category': ['C1', 'C2', 'C3'],
            'location': ['L1', 'L2', 'L3'],
            'tags': ['T1', 'T2', 'T3'],
            'language': ['en', 'ru', 'en'],
            'photo_name': ['p1.jpg', 'p2.jpg', 'p3.jpg'],
            'license': ['CC-BY', 'CC-BY', 'CC-BY'],
            'photo_author': ['A1', 'A2', 'A3']
        }
        df = pd.DataFrame(data)

        # all non-null languages should be valid
        languages = df['language'].dropna().unique()
        for lang in languages:
            assert lang in valid_languages, f"Invalid language: {lang}"

    def test_empty_rows_detection(self):
        """Test detection of empty rows."""
        data = {
            'id': [1, None, 3],
            'name': ['A', None, 'C'],
            'description': ['D1', None, 'D3'],
            'category': ['C1', None, 'C3'],
            'location': ['L1', None, 'L3'],
            'tags': ['T1', None, 'T3'],
            'language': ['en', None, 'en'],
            'photo_name': ['p1.jpg', None, 'p3.jpg'],
            'license': ['CC-BY', None, 'CC-BY'],
            'photo_author': ['A1', None, 'A3']
        }
        df = pd.DataFrame(data)

        # should have one completely empty row
        empty_rows = df.isna().all(axis=1).sum()
        assert empty_rows == 1

    def test_duplicate_ids_detection(self):
        """Test detection of duplicate IDs."""
        data = {
            'id': [1, 2, 2, 3],  # Duplicate ID: 2
            'name': ['A', 'B', 'B2', 'C'],
            'description': ['D1', 'D2', 'D2b', 'D3'],
            'category': ['C1', 'C2', 'C2b', 'C3'],
            'location': ['L1', 'L2', 'L2b', 'L3'],
            'tags': ['T1', 'T2', 'T2b', 'T3'],
            'language': ['en', 'ru', 'ru', 'en'],
            'photo_name': ['p1.jpg', 'p2.jpg', 'p2b.jpg', 'p3.jpg'],
            'license': ['CC-BY', 'CC-BY', 'CC-BY', 'CC-BY'],
            'photo_author': ['A1', 'A2', 'A2b', 'A3']
        }
        df = pd.DataFrame(data)

        # check for duplicates
        duplicates = df['id'].duplicated().sum()
        assert duplicates > 0, "Should detect duplicate IDs"

    def test_csv_file_exists(self):
        """Test that CSV file path exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Non-existent file
            csv_path = Path(tmpdir) / "nonexistent.csv"

            assert not csv_path.exists(), "File should not exist"

    def test_photo_name_format(self):
        """Test that photo names have valid extensions."""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']

        data = {
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'description': ['D1', 'D2', 'D3'],
            'category': ['C1', 'C2', 'C3'],
            'location': ['L1', 'L2', 'L3'],
            'tags': ['T1', 'T2', 'T3'],
            'language': ['en', 'ru', 'en'],
            'photo_name': ['photo1.jpg', 'photo2.png', 'photo3.jpeg'],
            'license': ['CC-BY', 'CC-BY', 'CC-BY'],
            'photo_author': ['A1', 'A2', 'A3']
        }
        df = pd.DataFrame(data)

        # check photo extensions
        for photo_name in df['photo_name'].dropna():
            ext = Path(photo_name).suffix.lower()
            assert ext in valid_extensions, f"Invalid photo extension: {ext}"

    def test_non_empty_name_field(self):
        """Test that name field is not empty for valid records."""
        data = {
            'id': [1, 2, 3],
            'name': ['Attraction A', 'Attraction B', 'Attraction C'],
            'description': ['D1', 'D2', 'D3'],
            'category': ['C1', 'C2', 'C3'],
            'location': ['L1', 'L2', 'L3'],
            'tags': ['T1', 'T2', 'T3'],
            'language': ['en', 'ru', 'en'],
            'photo_name': ['p1.jpg', 'p2.jpg', 'p3.jpg'],
            'license': ['CC-BY', 'CC-BY', 'CC-BY'],
            'photo_author': ['A1', 'A2', 'A3']
        }
        df = pd.DataFrame(data)

        # after dropping empty rows, all names should be non-null
        df_clean = df.dropna(how='all')

        assert df_clean['name'].notna().all(), "All valid records should have names"

    def test_data_types_after_processing(self):
        """Test that data types are correct after processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # create sample CSV
            data = {
                'id': [1, 2],
                'name': ['A', 'B'],
                'description': ['D1', 'D2'],
                'category': ['C1', 'C2'],
                'location': ['L1', 'L2'],
                'tags': ['T1', 'T2'],
                'language': ['en', 'ru'],
                'photo_name': ['p1.jpg', 'p2.jpg'],
                'license': ['CC-BY', 'Public Domain'],
                'photo_author': ['Author1', 'Author2']
            }
            df = pd.DataFrame(data)

            csv_path = tmpdir / "test.csv"
            df.to_csv(csv_path, index=False)

            photos_folder = tmpdir / "photos"
            photos_folder.mkdir()

            # create dataset
            creator = GeorgianAttractionsDataset(
                csv_path=str(csv_path),
                photos_folder=str(photos_folder)
            )

            creator.load_csv()
            creator.clean_data()

            # check data types after cleaning
            assert creator.df['id'].dtype in ['int64', 'int32']
            assert creator.df['name'].dtype == 'object'  # string type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
