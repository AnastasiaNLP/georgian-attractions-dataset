"""
Filter by Language Example

This example demonstrates how to filter the dataset by language
and analyze language distribution.
"""

from datasets import load_from_disk
from collections import Counter


def main():
    """Filter and analyze dataset by language."""

    print("="*70)
    print("  Example: Filter by Language")
    print("="*70)

    # Load dataset
    dataset_path = "georgian_attractions_dataset"

    print(f"\nLoading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    print(f" Loaded {len(dataset)} records")

    # Language distribution
    print("\n" + "="*70)
    print("  Language Distribution")
    print("="*70)

    languages = Counter(dataset['language'])

    for lang, count in languages.items():
        percentage = (count / len(dataset)) * 100
        print(f"{lang}: {count} records ({percentage:.1f}%)")

    # Filter by Russian
    print("\n" + "="*70)
    print("  Russian Attractions")
    print("="*70)

    russian_data = dataset.filter(lambda x: x['language'] == 'ru')
    print(f"Found {len(russian_data)} Russian records")

    # Show examples
    print("\nExamples (first 3):")
    for i in range(min(3, len(russian_data))):
        example = russian_data[i]
        print(f"\n{i+1}. {example['name']}")
        print(f"   Category: {example['category']}")
        print(f"   Location: {example['location'][:50]}...")

    # Filter by English
    print("\n" + "="*70)
    print("  English Attractions")
    print("="*70)

    english_data = dataset.filter(lambda x: x['language'] == 'en')
    print(f"Found {len(english_data)} English records")

    # Show examples
    print("\nExamples (first 3):")
    for i in range(min(3, len(english_data))):
        example = english_data[i]
        print(f"\n{i+1}. {example['name']}")
        print(f"   Category: {example['category']}")
        print(f"   Location: {example['location'][:50]}...")

    # Filter by category and language
    print("\n" + "="*70)
    print("  Museums in Russian")
    print("="*70)

    russian_museums = dataset.filter(
        lambda x: x['language'] == 'ru' and 'Музей' in x['category']
    )

    print(f"Found {len(russian_museums)} Russian museums")

    if len(russian_museums) > 0:
        print("\nExamples:")
        for i in range(min(3, len(russian_museums))):
            example = russian_museums[i]
            print(f"\n{i+1}. {example['name']}")
            print(f"   Location: {example['location']}")

    print("\n✨ Done!")


if __name__ == "__main__":
    main()