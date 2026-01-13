"""
Upload Georgian Attractions Dataset to HuggingFace Hub

This module handles uploading the dataset to HuggingFace Hub
with proper README documentation.
"""

import os
from pathlib import Path
from typing import Optional
from datasets import load_from_disk
from huggingface_hub import HfApi, login


def create_readme(repo_id: str) -> str:
    """
    Create comprehensive README for the dataset.

    Args:
        repo_id: HuggingFace repository ID (username/dataset-name)

    Returns:
        README content as string
    """
    readme_content = f"""---
language:
- en
- ru
license: cc-by-4.0
size_categories:
- 1K<n<10K
task_categories:
- image-classification
- text-generation
- object-detection
pretty_name: Georgian Attractions Dataset
tags:
- georgia
- tourism
- attractions
- travel
- cultural-heritage
- geography
- landmarks
- multilingual
---

# Georgian Attractions Dataset 🇬🇪

A comprehensive bilingual dataset featuring 1,715 Georgian tourist attractions with 1,522 high-quality images, descriptions in Russian and English, and detailed metadata including location, category, and licensing information.

## Dataset Description

This dataset provides extensive information about tourist attractions, landmarks, and points of interest across Georgia. It includes national parks, museums, fortresses, monasteries, natural landmarks, and historical sites. Each entry contains bilingual descriptions, geographic location data, categorization, searchable tags, and properly licensed images with attribution.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Records | 1,715 |
| Records with Images | 1,522 (89%) |
| Records without Images | 193 (11%) |
| Languages | Russian, English |

## Dataset Structure

### Data Fields

- **id** (`int64`): Unique identifier for each attraction
- **name** (`string`): Name of the attraction
- **description** (`string`): Detailed description
- **category** (`string`): Type of attraction (e.g., "National Park", "Museum")
- **location** (`string`): Geographic location within Georgia
- **tags** (`string`): Comma-separated searchable keywords
- **language** (`string`): Language code ("ru" or "en")
- **photo_name** (`string`): Filename of the image
- **license** (`string`): Image license information
- **photo_author** (`string`): Photographer name
- **image** (`Image`): PIL Image object

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access records
train_data = dataset['train']

# View example
example = train_data[0]
print(f"Name: {{example['name']}}")
print(f"Category: {{example['category']}}")
if example['image']:
    example['image'].show()
```

### Filtering Examples

```python
# Filter by language
russian = train_data.filter(lambda x: x['language'] == 'ru')
english = train_data.filter(lambda x: x['language'] == 'en')

# Filter by category
museums = train_data.filter(lambda x: 'Museum' in x['category'])

# Get only records with images
with_images = train_data.filter(lambda x: x['image'] is not None)
```

## Use Cases

- **Tourism Applications**: Travel guides, recommendation systems
- **Computer Vision**: Image classification, object detection
- **NLP**: Multilingual text processing, translation
- **Research**: Cultural heritage, GIS studies

## Image Licenses

All images are properly licensed and attributed:
- Creative Commons (various versions)
- Public Domain

License and author information is provided in the `license` and `photo_author` fields.

## Citation

```bibtex
@dataset{{georgian_attractions_2024,
  title={{Georgian Attractions Dataset}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```

---

**Dataset Version**: 1.0  
**Last Updated**: December 2024
"""
    return readme_content


class DatasetUploader:
    """
    Handles uploading dataset to HuggingFace Hub.
    """

    def __init__(self, dataset_path: str, repo_id: str, token: Optional[str] = None):
        """
        Initialize uploader.

        Args:
            dataset_path: Path to saved dataset
            repo_id: HuggingFace repository ID (username/dataset-name)
            token: HuggingFace API token (optional, will prompt if not provided)
        """
        self.dataset_path = dataset_path
        self.repo_id = repo_id
        self.token = token
        self.dataset = None

    def authenticate(self) -> None:
        """Authenticate with HuggingFace."""
        print(" Authenticating with HuggingFace...")

        if self.token:
            login(token=self.token)
        else:
            print("Get your token at: https://huggingface.co/settings/tokens")
            login()

        print(" Authentication successful")

    def load_dataset(self) -> None:
        """Load dataset from disk."""
        print(f"\n Loading dataset from: {self.dataset_path}")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        self.dataset = load_from_disk(self.dataset_path)

        # show statistics
        images_count = sum(
            1 for i in range(len(self.dataset))
            if self.dataset[i]['image'] is not None
        )

        print(f" Dataset loaded")
        print(f"   Total records: {len(self.dataset)}")
        print(f"   With images: {images_count}")
        print(f"   Without images: {len(self.dataset) - images_count}")

    def upload_dataset(self, private: bool = False) -> None:
        """
        Upload dataset to HuggingFace Hub.

        Args:
            private: Whether to make the dataset private
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print(f"\nUploading dataset to: {self.repo_id}")
        print(f"   Private: {private}")
        print("    This may take 15-30 minutes...")

        images_count = sum(
            1 for i in range(len(self.dataset))
            if self.dataset[i]['image'] is not None
        )

        # Upload dataset
        self.dataset.push_to_hub(
            repo_id=self.repo_id,
            private=private,
            commit_message=f"Upload Georgian Attractions Dataset with {images_count} images"
        )

        print(f"\n Dataset uploaded successfully!")
        print(f" View at: https://huggingface.co/datasets/{self.repo_id}")

    def upload_readme(self) -> None:
        """Upload README to the dataset repository."""
        print("\n Uploading README...")

        # generate README
        readme_content = create_readme(self.repo_id)

        # save locally
        readme_path = Path("README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        #upload to Hub
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Add comprehensive documentation"
        )

        print(" README uploaded")

        # clean up local file
        readme_path.unlink()

    def process(self, private: bool = False) -> None:
        """
        Complete upload pipeline: authenticate -> load -> upload -> README.

        Args:
            private: Whether to make the dataset private
        """
        print("  Georgian Attractions Dataset Upload")

        # Step 1: Authenticate
        self.authenticate()

        # Step 2: Load dataset
        self.load_dataset()

        # Step 3: Upload dataset
        self.upload_dataset(private=private)

        # Step 4: Upload README
        self.upload_readme()

        print("  Upload complete")
        print(f" Dataset URL: https://huggingface.co/datasets/{self.repo_id}")
        print("\n Wait 10-30 minutes for the dataset viewer to process images")
        print(" Upload complete!")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Upload Georgian Attractions Dataset to HuggingFace Hub'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='georgian_attractions_dataset',
        help='Path to the saved dataset'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='HuggingFace repository ID (username/dataset-name)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace API token (optional)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the dataset private'
    )

    args = parser.parse_args()

    # Upload dataset
    uploader = DatasetUploader(
        dataset_path=args.dataset_path,
        repo_id=args.repo_id,
        token=args.token
    )

    uploader.process(private=args.private)
