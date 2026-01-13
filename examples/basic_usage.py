"""
Basic Usage Example

This example demonstrates how to create a Georgian Attractions dataset
from CSV and images using the dataset creator.
"""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset_creator import GeorgianAttractionsDataset


def main():
    """Create dataset using basic workflow."""
    # configuration
    csv_path = "dataset.csv"
    photos_folder = "photo_dataset"
    output_path = "georgian_attractions_dataset"

    print(f"\nConfiguration:")
    print(f"  CSV: {csv_path}")
    print(f"  Photos: {photos_folder}")
    print(f"  Output: {output_path}")

    # create dataset creator
    creator = GeorgianAttractionsDataset(
        csv_path=csv_path,
        photos_folder=photos_folder
    )

    # run complete pipeline
    dataset = creator.process(output_path)

    # show example record
    print("  Example Record")

    # find first record with image
    for i in range(min(10, len(dataset))):
        if dataset[i]['image'] is not None:
            example = dataset[i]
            print(f"\nID: {example['id']}")
            print(f"Name: {example['name']}")
            print(f"Category: {example['category']}")
            print(f"Language: {example['language']}")
            print(f"Location: {example['location'][:50]}...")
            print(f"Has image: Yes ({example['photo_name']})")
            break

    print("\n Done! Dataset saved to:", output_path)


if __name__ == "__main__":
    main()
