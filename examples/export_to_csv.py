"""
Export Georgian Attractions dataset to CSV format.

This example demonstrates:
- Loading dataset from HuggingFace Hub or local disk
- Exporting full dataset to CSV
- Exporting filtered subsets to CSV
- Exporting with custom columns
- Handling image data (saving paths or skipping)
"""

from datasets import load_dataset, load_from_disk
import pandas as pd
from pathlib import Path
from typing import List, Optional


def export_to_csv(
    dataset,
    output_path: str,
    columns: Optional[List[str]] = None,
    include_image_paths: bool = False,
    filter_language: Optional[str] = None,
    filter_category: Optional[str] = None
):
    """
    Export dataset to CSV file.

    Args:
        dataset: HuggingFace Dataset object
        output_path: Path to save CSV file
        columns: List of columns to export (None = all text columns)
        include_image_paths: If True, include 'has_image' column
        filter_language: Export only specific language (None = all)
        filter_category: Export only specific category (None = all)

    Returns:
        Path to saved CSV file
    """
    # filter dataset if needed
    filtered_dataset = dataset

    if filter_language:
        filtered_dataset = filtered_dataset.filter(lambda x: x['language'] == filter_language)

    if filter_category:
        filtered_dataset = filtered_dataset.filter(lambda x: x['category'] == filter_category)

    # prepare data for export
    data_to_export = []

    for record in filtered_dataset:
        row = {}

        # default columns (all text fields)
        if columns is None:
            columns = ['id', 'name', 'description', 'category', 'location',
                      'tags', 'language', 'photo_name', 'license', 'photo_author']

        # extract specified columns
        for col in columns:
            if col in record:
                row[col] = record[col]

        # add image information if requested
        if include_image_paths:
            row['has_image'] = 'Yes' if record.get('image') is not None else 'No'

        data_to_export.append(row)

    # create DataFrame and save
    df = pd.DataFrame(data_to_export)
    df.to_csv(output_path, index=False, encoding='utf-8')

    return output_path


def export_statistics_csv(dataset, output_path: str):
    """
    Export dataset statistics to CSV.

    Args:
        dataset: HuggingFace Dataset object
        output_path: Path to save statistics CSV
    """
    from collections import Counter

    stats_data = []

    # overall statistics
    total_records = len(dataset)
    with_images = sum(1 for r in dataset if r['image'] is not None)

    stats_data.append({
        'Metric': 'Total Records',
        'Value': total_records
    })

    stats_data.append({
        'Metric': 'Records with Images',
        'Value': with_images
    })

    stats_data.append({
        'Metric': 'Records without Images',
        'Value': total_records - with_images
    })

    stats_data.append({
        'Metric': 'Image Coverage %',
        'Value': f"{with_images/total_records*100:.2f}"
    })

    # language distribution
    languages = Counter(r['language'] for r in dataset)
    for lang, count in languages.items():
        stats_data.append({
            'Metric': f'Language: {lang}',
            'Value': count
        })

    # top categories
    categories = Counter(r['category'] for r in dataset)
    for cat, count in categories.most_common(10):
        stats_data.append({
            'Metric': f'Category: {cat}',
            'Value': count
        })

    # save to CSV
    df = pd.DataFrame(stats_data)
    df.to_csv(output_path, index=False, encoding='utf-8')

    return output_path


def export_by_category(dataset, output_dir: str):
    """
    Export dataset split by categories into separate CSV files.

    Args:
        dataset: HuggingFace Dataset object
        output_dir: Directory to save category CSV files
    """
    from collections import Counter

    # create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # get all categories
    categories = Counter(r['category'] for r in dataset)

    print(f"Exporting {len(categories)} categories to separate files...")

    for category in categories.keys():
        # filter by category
        category_data = [r for r in dataset if r['category'] == category]

        # prepare data
        data_to_export = []
        for record in category_data:
            row = {
                'id': record['id'],
                'name': record['name'],
                'description': record['description'],
                'category': record['category'],
                'location': record['location'],
                'tags': record['tags'],
                'language': record['language'],
                'photo_name': record['photo_name'],
                'license': record['license'],
                'photo_author': record['photo_author'],
                'has_image': 'Yes' if record.get('image') is not None else 'No'
            }
            data_to_export.append(row)

        # save to CSV
        safe_category_name = category.replace('/', '_').replace('\\', '_')
        csv_path = output_path / f"{safe_category_name}.csv"

        df = pd.DataFrame(data_to_export)
        df.to_csv(csv_path, index=False, encoding='utf-8')

        print(f" Save {category}: {len(category_data)} records -> {csv_path.name}")

    return output_dir


def export_split_by_language(dataset, output_dir: str):
    """
    Export dataset split by language into separate CSV files.

    Args:
        dataset: HuggingFace Dataset object
        output_dir: Directory to save language CSV files
    """
    # create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # get unique languages
    languages = set(r['language'] for r in dataset)

    print(f"Exporting {len(languages)} languages to separate files...")

    for language in languages:
        csv_path = output_path / f"dataset_{language}.csv"
        export_to_csv(
            dataset,
            str(csv_path),
            filter_language=language,
            include_image_paths=True
        )

        lang_count = len([r for r in dataset if r['language'] == language])
        print(f" Save {language}: {lang_count} records -> {csv_path.name}")

    return output_dir


def main():
    print("Export to csv examples")

    # load dataset
    print("\n Loading dataset from HuggingFace Hub...")
    dataset = load_dataset('AIAnastasia/georgian-attractions', split='train')
    print(f"Loaded {len(dataset)} records")

    # create output directory
    output_dir = Path('./exported_csv')
    output_dir.mkdir(exist_ok=True)

    # Example 1: Export full dataset
    print("Example 1: Export full dataset to CSV")

    full_csv = output_dir / 'georgian_attractions_full.csv'
    export_to_csv(dataset, str(full_csv), include_image_paths=True)
    print(f"Exported to: {full_csv}")
    print(f"   Size: {full_csv.stat().st_size / 1024:.2f} KB")

    # Example 2: Export only Russian records
    print("Example 2: Export only Russian language records")

    ru_csv = output_dir / 'georgian_attractions_ru.csv'
    export_to_csv(dataset, str(ru_csv), filter_language='ru', include_image_paths=True)

    ru_count = len([r for r in dataset if r['language'] == 'ru'])
    print(f"Exported {ru_count} Russian records to: {ru_csv}")

    #Example 3: Export only English records
    print("Example 3: Export only English language records")

    en_csv = output_dir / 'georgian_attractions_en.csv'
    export_to_csv(dataset, str(en_csv), filter_language='en', include_image_paths=True)

    en_count = len([r for r in dataset if r['language'] == 'en'])
    print(f"Exported {en_count} English records to: {en_csv}")

    # Example 4: Export specific category
    print("Example 4: Export only museums")

    museums_csv = output_dir / 'museums.csv'
    export_to_csv(dataset, str(museums_csv), filter_category='Музей')

    museums_count = len([r for r in dataset if r['category'] == 'Музей'])
    print(f"Exported {museums_count} museums to: {museums_csv}")

    # Example 5: Export with custom columns
    print("Example 5: Export with custom columns (minimal info)")

    minimal_csv = output_dir / 'georgian_attractions_minimal.csv'
    export_to_csv(
        dataset,
        str(minimal_csv),
        columns=['id', 'name', 'category', 'language'],
        include_image_paths=True
    )
    print(f"Exported minimal version to: {minimal_csv}")

    # Example 6: Export statistics
    print("Example 6: Export dataset statistics")

    stats_csv = output_dir / 'dataset_statistics.csv'
    export_statistics_csv(dataset, str(stats_csv))
    print(f"Statistics exported to: {stats_csv}")

    # Example 7: Split by language
    print("Example 7: Export split by language")

    language_dir = output_dir / 'by_language'
    export_split_by_language(dataset, str(language_dir))

    # Example 8: Split by category
    print("Example 8: Export split by category (first 5 categories)")

    from collections import Counter
    categories = Counter(r['category'] for r in dataset)
    top_categories = [cat for cat, _ in categories.most_common(5)]

    category_dir = output_dir / 'by_category'

    for category in top_categories:
        category_csv = Path(category_dir) / f"{category.replace('/', '_')}.csv"
        category_csv.parent.mkdir(parents=True, exist_ok=True)

        export_to_csv(dataset, str(category_csv), filter_category=category, include_image_paths=True)
        cat_count = len([r for r in dataset if r['category'] == category])
        print(f"   {category}: {cat_count} records")

    # summary
    print(f"\n All files exported to: {output_dir.absolute()}")
    print(f"\nExported files:")

    for csv_file in sorted(output_dir.rglob('*.csv')):
        size_kb = csv_file.stat().st_size / 1024
        relative_path = csv_file.relative_to(output_dir)
        print(f"  • {relative_path} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
