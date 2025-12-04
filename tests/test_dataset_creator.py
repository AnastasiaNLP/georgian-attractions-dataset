"""
Tests for GeorgianAttractionsDataset class.

Tests cover:
- CSV loading
- Data cleaning
- Image verification
- Dataset creation
"""

import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset_creator import GeorgianAttractionsDataset


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    data = {
        'id': [1, 2, 3, 4],
        'name': ['Attraction 1', 'Attraction 2', 'Attraction 3', None],
        'description': ['Description 1', 'Description 2', 'Description 3', None],
        'category': ['Museum', 'Park', 'Fortress', None],
        'location': ['Tbilisi', 'Batumi', 'Mtskheta', None],
        'tags': ['history, museum', 'nature, park', 'medieval, fortress', None],
        'language': ['en', 'ru', 'en', None],
        'photo_name': ['photo1.jpg', 'photo2.jpg', None, None],
        'license': ['CC-BY', 'Public Domain', None, None],
        'photo_author': ['Author 1', 'Author 2', None, None]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_files(sample_csv_data):
    """Create temporary CSV and photos folder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create CSV
        csv_path = tmpdir / "test_dataset.csv"
        sample_csv_data.to_csv(csv_path, index=False)

        # Create photos folder with dummy images
        photos_folder = tmpdir / "photos"
        photos_folder.mkdir()

        # Create dummy image files
        (photos_folder / "photo1.jpg").touch()
        (photos_folder / "photo2.jpg").touch()

        yield {
            'csv_path': str(csv_path),
            'photos_folder': str(photos_folder),
            'tmpdir': tmpdir
        }


class TestGeorgianAttractionsDataset:
    """Test suite for GeorgianAttractionsDataset."""

    def test_initialization(self, temp_files):
        """Test that dataset creator initializes correctly."""
        creator = GeorgianAttractionsDataset(
            csv_path=temp_files['csv_path'],
            photos_folder=temp_files['photos_folder']
        )

        assert creator.csv_path == temp_files['csv_path']
        assert creator.photos_folder == temp_files['photos_folder']
        assert creator.df is None
        assert creator.dataset is None

    def test_load_csv(self, temp_files):
        """Test CSV loading."""
        creator = GeorgianAttractionsDataset(
            csv_path=temp_files['csv_path'],
            photos_folder=temp_files['photos_folder']
        )

        df = creator.load_csv()

        assert df is not None
        assert len(df) == 4
        assert 'name' in df.columns
        assert 'description' in df.columns
        assert creator.df is not None

    def test_clean_data(self, temp_files):
        """Test data cleaning removes empty rows and reindexes."""
        creator = GeorgianAttractionsDataset(
            csv_path=temp_files['csv_path'],
            photos_folder=temp_files['photos_folder']
        )

        creator.load_csv()
        df_cleaned = creator.clean_data()

        # Should remove the row with all NaN values
        assert len(df_cleaned) == 3  # One row should be removed

        # IDs should be reindexed from 1 to N
        assert list(df_cleaned['id']) == [1, 2, 3]

    def test_verify_images(self, temp_files):
        """Test image verification."""
        creator = GeorgianAttractionsDataset(
            csv_path=temp_files['csv_path'],
            photos_folder=temp_files['photos_folder']
        )

        creator.load_csv()
        creator.clean_data()

        found_count, missing = creator.verify_images()

        # We have 2 image files, 1 row with no photo_name
        assert found_count == 2
        assert len(missing) == 1  # One row without image

    def test_verify_images_missing_folder(self, temp_files):
        """Test image verification with missing photos folder."""
        creator = GeorgianAttractionsDataset(
            csv_path=temp_files['csv_path'],
            photos_folder="/nonexistent/folder"
        )

        creator.load_csv()
        creator.clean_data()

        found_count, missing = creator.verify_images()

        assert found_count == 0
        assert len(missing) == 0  # Returns empty list when folder doesn't exist

    def test_get_statistics(self, temp_files):
        """Test statistics generation."""
        creator = GeorgianAttractionsDataset(
            csv_path=temp_files['csv_path'],
            photos_folder=temp_files['photos_folder']
        )

        creator.load_csv()
        creator.clean_data()
        creator.create_dataset()

        stats = creator.get_statistics()

        assert 'total_records' in stats
        assert 'with_images' in stats
        assert 'without_images' in stats
        assert 'languages' in stats

        assert stats['total_records'] == 3
        assert stats['with_images'] == 2
        assert stats['without_images'] == 1

    def test_save_dataset(self, temp_files):
        """Test dataset saving."""
        creator = GeorgianAttractionsDataset(
            csv_path=temp_files['csv_path'],
            photos_folder=temp_files['photos_folder']
        )

        creator.load_csv()
        creator.clean_data()
        creator.create_dataset()

        output_path = temp_files['tmpdir'] / "output_dataset"
        creator.save_dataset(str(output_path))

        # Check that dataset folder was created
        assert output_path.exists()
        assert (output_path / "dataset_info.json").exists()

    def test_process_complete_pipeline(self, temp_files):
        """Test complete processing pipeline."""
        creator = GeorgianAttractionsDataset(
            csv_path=temp_files['csv_path'],
            photos_folder=temp_files['photos_folder']
        )

        output_path = temp_files['tmpdir'] / "output_dataset"
        dataset = creator.process(str(output_path))

        # Check dataset was created
        assert dataset is not None
        assert len(dataset) == 3

        # Check output folder exists
        assert output_path.exists()

        # Check dataset has correct fields
        assert 'id' in dataset.column_names
        assert 'name' in dataset.column_names
        assert 'image' in dataset.column_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])