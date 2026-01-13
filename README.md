# Georgian Attractions Dataset Creator 🇬🇪

Tools for creating a bilingual HuggingFace dataset of Georgian tourist attractions with images and descriptions in Russian and English.

## Dataset

The Georgian Attractions Dataset is a comprehensive bilingual collection featuring:
- **1,715 records** of tourist attractions across Georgia
- **1,522 high-quality images** with proper licensing
- **Bilingual descriptions** in Russian and English
- **Rich metadata**: locations, categories, tags, and image attributions

** View the dataset**: [AIAnastasia/georgian-attractions](https://huggingface.co/datasets/AIAnastasia/georgian-attractions)

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AnastasiaNLP/georgian-attractions-dataset.git
cd georgian-attractions-dataset

# Install dependencies
pip install -r requirements.txt
```

### Create Dataset

```python
from src import GeorgianAttractionsDataset

# Create dataset
creator = GeorgianAttractionsDataset(
    csv_path="dataset.csv",
    photos_folder="photo_dataset"
)

# Run complete pipeline
dataset = creator.process("georgian_attractions_dataset")
```

### Upload to HuggingFace Hub

```python
from src.upload_to_hub import DatasetUploader

# Upload
uploader = DatasetUploader(
    dataset_path="georgian_attractions_dataset",
    repo_id="YOUR_USERNAME/georgian-attractions"
)

uploader.process()
```

Or use command line:

```bash
python src/upload_to_hub.py --repo-id YOUR_USERNAME/georgian-attractions
```

##  Project Structure

```
georgian-attractions-dataset/
├── src/
│   ├── __init__.py
│   ├── dataset_creator.py      # Main dataset creation class
│   └── upload_to_hub.py        # HuggingFace Hub uploader
│
├── examples/
│   ├── basic_usage.py          # Basic usage example
│   ├── filter_by_language.py   # Filter by language
│   └── load_and_display.py     # Load and display data
│
├── tests/
│   ├── test_dataset_creator.py # Dataset creation tests
│   └── test_data_validation.py # Data validation tests
│
├── notebooks/
│   └── dataset_creation.ipynb  # Colab notebook
│
├── requirements.txt            # Dependencies
├── requirements-dev.txt        # Dev dependencies
├── README.md
├── .gitignore
└── LICENSE
```

##  Data Format

### Input Requirements

**CSV File** (`dataset.csv`):
- `id`: Unique identifier
- `name`: Attraction name
- `description`: Detailed description
- `category`: Category type
- `location`: Geographic location
- `tags`: Searchable keywords
- `language`: Language code (ru/en)
- `photo_name`: Image filename
- `license`: Image license
- `photo_author`: Photographer name

**Images Folder** (`photo_dataset/`):
- Contains all referenced image files
- Supported formats: JPG, JPEG, PNG

##  Usage Examples

### Example 1: Basic Dataset Creation

```python
from src import GeorgianAttractionsDataset

creator = GeorgianAttractionsDataset(
    csv_path="dataset.csv",
    photos_folder="photo_dataset"
)

# Load and clean data
creator.load_csv()
creator.clean_data()

# Verify images
found, missing = creator.verify_images()
print(f"Found {found} images, missing {len(missing)}")

# Create dataset
dataset = creator.create_dataset()

# Save
creator.save_dataset("output_folder")
```

### Example 2: Load and Filter Dataset

```python
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk("georgian_attractions_dataset")

# Filter by language
russian = dataset.filter(lambda x: x['language'] == 'ru')
english = dataset.filter(lambda x: x['language'] == 'en')

# Filter by category
museums = dataset.filter(lambda x: 'Museum' in x['category'])
```

### Example 3: Access Images

```python
# Get record with image
example = dataset[0]

if example['image']:
    # Display image
    example['image'].show()
    
    # Get image size
    print(f"Size: {example['image'].size}")
```

##  Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_dataset_creator.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

##  Dataset Statistics

- **Total Records**: 1,715
- **With Images**: 1,522 (89%)
- **Without Images**: 193 (11%)
- **Languages**: Russian (ru), English (en)
- **Categories**: 20+ types of attractions

##  Use Cases

- **Tourism Applications**: Travel guides, recommendation systems
- **Computer Vision**: Image classification, object detection
- **NLP**: Multilingual text processing, translation
- **Research**: Cultural heritage, GIS studies

##  Documentation

### Main Classes

#### `GeorgianAttractionsDataset`

Main class for dataset creation.

**Methods:**
- `load_csv()` - Load CSV data
- `clean_data()` - Remove empty rows and reindex
- `verify_images()` - Verify image files exist
- `create_dataset()` - Create HuggingFace dataset
- `save_dataset(path)` - Save to disk
- `get_statistics()` - Get dataset statistics
- `process(output_path)` - Run complete pipeline

#### `DatasetUploader`

Class for uploading to HuggingFace Hub.

**Methods:**
- `authenticate()` - Login to HuggingFace
- `load_dataset()` - Load dataset from disk
- `upload_dataset()` - Upload to Hub
- `upload_readme()` - Upload README
- `process()` - Run complete upload pipeline

##  Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

**Dataset License**: CC-BY-4.0 (dataset metadata); individual images retain their original licenses as specified.

##  Acknowledgments

- All photographers and content contributors
- HuggingFace for the datasets library
- The Georgian tourism community
---

**Created**: December 2024  
**Maintained by**: AIAnastasia  
**Dataset URL**: https://huggingface.co/datasets/AIAnastasia/georgian-attractions
