"""
Load and Display Example

This example demonstrates how to load the dataset and display
images with their information.
"""

from datasets import load_from_disk, load_dataset


def display_local_dataset():
    """Load and display from local disk."""

    print("="*70)
    print("  Loading from Local Disk")
    print("="*70)

    dataset_path = "georgian_attractions_dataset"

    print(f"\nLoading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    print(f" Loaded {len(dataset)} records")

    # Count images
    images_count = sum(1 for i in range(len(dataset)) if dataset[i]['image'] is not None)
    print(f" Records with images: {images_count}")

    # Display examples with images
    print("\n" + "="*70)
    print("  Example Attractions (with images)")
    print("="*70)

    shown = 0
    for i in range(len(dataset)):
        if dataset[i]['image'] is not None and shown < 3:
            example = dataset[i]

            print(f"\n{shown + 1}. {example['name']}")
            print(f"   Category: {example['category']}")
            print(f"   Language: {example['language']}")
            print(f"   Location: {example['location'][:60]}...")
            print(f"   Tags: {example['tags'][:60]}...")
            print(f"   License: {example['license']}")
            print(f"   Author: {example['photo_author']}")

            # Display image
            img = example['image']
            print(f"   Image size: {img.size}")
            print(f"   Format: {img.format}")

            # Note: To actually display the image in a window:
            # img.show()

            shown += 1

    print("\n To display images, uncomment img.show() in the code")


def display_from_hub(repo_id: str = "AIAnastasia/georgian-attractions"):
    """Load and display from HuggingFace Hub."""

    print("\n" + "="*70)
    print("  Loading from HuggingFace Hub")
    print("="*70)

    print(f"\nLoading dataset from: {repo_id}")

    try:
        dataset = load_dataset(repo_id)
        train_data = dataset['train']

        print(f" Loaded {len(train_data)} records")

        # Show example
        example = train_data[0]
        print(f"\nExample:")
        print(f"  Name: {example['name']}")
        print(f"  Category: {example['category']}")
        print(f"  Language: {example['language']}")

        if example['image']:
            print(f"  Image: {example['image'].size}")

    except Exception as e:
        print(f"  Could not load from Hub: {e}")
        print("   Make sure the dataset is uploaded to HuggingFace")


def filter_and_display():
    """Filter dataset and display specific attractions."""

    print("\n" + "="*70)
    print("  Filter and Display")
    print("="*70)

    dataset_path = "georgian_attractions_dataset"
    dataset = load_from_disk(dataset_path)

    # Filter: National Parks with images
    print("\nFiltering: National Parks with images")

    national_parks = dataset.filter(
        lambda x: (
            ('National Park' in x['category'] or 'Национальный парк' in x['category'])
            and x['image'] is not None
        )
    )

    print(f"Found {len(national_parks)} national parks with images")

    # Display
    for i in range(min(3, len(national_parks))):
        example = national_parks[i]
        print(f"\n{i+1}. {example['name']}")
        print(f"   Location: {example['location']}")
        print(f"   Description: {example['description'][:100]}...")
        print(f"   Image: {example['photo_name']}")


def main():
    """Run all examples."""

    # Example 1: Load from local disk
    display_local_dataset()

    # Example 2: Load from HuggingFace Hub (if uploaded)
    # Uncomment to try:
    # display_from_hub()

    # Example 3: Filter and display
    filter_and_display()

    print("\n✨ Done!")


if __name__ == "__main__":
    main()