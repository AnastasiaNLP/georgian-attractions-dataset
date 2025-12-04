"""
Query and filter the Georgian Attractions dataset.

This example demonstrates:
- Loading the dataset from HuggingFace Hub or local disk
- Filtering by various criteria (category, language, location, etc.)
- Searching by keywords in name or description
- Combining multiple filters
"""

from datasets import load_dataset, load_from_disk
from typing import Optional, List


def filter_dataset(
    dataset,
    category: Optional[str] = None,
    language: Optional[str] = None,
    has_image: Optional[bool] = None,
    location_contains: Optional[str] = None,
    name_contains: Optional[str] = None,
    description_contains: Optional[str] = None
):
    """
    Filter dataset by multiple criteria.

    Args:
        dataset: HuggingFace Dataset object
        category: Filter by category (exact match)
        language: Filter by language ('ru' or 'en')
        has_image: Filter records with/without images
        location_contains: Filter by location (substring search)
        name_contains: Filter by name (substring search, case-insensitive)
        description_contains: Filter by description (substring search, case-insensitive)

    Returns:
        Filtered dataset
    """

    def filter_fn(example):
        # Category filter
        if category and example['category'] != category:
            return False

        # Language filter
        if language and example['language'] != language:
            return False

        # Image filter
        if has_image is not None:
            has_img = example['image'] is not None
            if has_image != has_img:
                return False

        # Location filter
        if location_contains:
            if not example['location'] or location_contains.lower() not in example['location'].lower():
                return False

        # Name filter
        if name_contains:
            if not example['name'] or name_contains.lower() not in example['name'].lower():
                return False

        # Description filter
        if description_contains:
            if not example['description'] or description_contains.lower() not in example['description'].lower():
                return False

        return True

    return dataset.filter(filter_fn)


def search_by_keywords(dataset, keywords: List[str], fields: List[str] = ['name', 'description', 'tags']):
    """
    Search for records containing any of the keywords in specified fields.

    Args:
        dataset: HuggingFace Dataset object
        keywords: List of keywords to search for
        fields: Fields to search in (default: name, description, tags)

    Returns:
        Filtered dataset
    """
    keywords_lower = [k.lower() for k in keywords]

    def search_fn(example):
        for field in fields:
            if field in example and example[field]:
                text = str(example[field]).lower()
                for keyword in keywords_lower:
                    if keyword in text:
                        return True
        return False

    return dataset.filter(search_fn)


def get_unique_values(dataset, field: str):
    """Get all unique values for a field."""
    return sorted(set(example[field] for example in dataset if example[field]))


def main():
    print("=" * 70)
    print("  QUERY DATASET EXAMPLES")
    print("=" * 70)

    # Load dataset
    print("\n📥 Loading dataset from HuggingFace Hub...")
    dataset = load_dataset('AIAnastasia/georgian-attractions', split='train')
    print(f"✅ Loaded {len(dataset)} records")

    # Example 1: Filter by category
    print("\n" + "=" * 70)
    print("Example 1: Find all museums")
    print("=" * 70)

    museums = filter_dataset(dataset, category='Музей')
    print(f"Found {len(museums)} museums")

    if len(museums) > 0:
        print("\nFirst 3 museums:")
        for i, record in enumerate(museums.select(range(min(3, len(museums))))):
            print(f"\n{i+1}. {record['name']}")
            print(f"   Language: {record['language']}")
            print(f"   Location: {record['location'][:60]}...")

    # Example 2: Filter by language and image presence
    print("\n" + "=" * 70)
    print("Example 2: English records with images")
    print("=" * 70)

    en_with_images = filter_dataset(dataset, language='en', has_image=True)
    print(f"Found {len(en_with_images)} English records with images")

    # Example 3: Search by location
    print("\n" + "=" * 70)
    print("Example 3: Attractions in Tbilisi")
    print("=" * 70)

    tbilisi = filter_dataset(dataset, location_contains='Tbilisi')
    print(f"Found {len(tbilisi)} attractions in Tbilisi")

    # Example 4: Complex filter
    print("\n" + "=" * 70)
    print("Example 4: Russian museums in Tbilisi with images")
    print("=" * 70)

    complex_filter = filter_dataset(
        dataset,
        category='Музей',
        language='ru',
        location_contains='Тбилиси',
        has_image=True
    )
    print(f"Found {len(complex_filter)} records")

    if len(complex_filter) > 0:
        print("\nRecords:")
        for i, record in enumerate(complex_filter.select(range(min(5, len(complex_filter))))):
            print(f"\n{i+1}. {record['name']}")
            print(f"   Location: {record['location']}")
            print(f"   Tags: {record['tags'][:50]}...")

    # Example 5: Keyword search
    print("\n" + "=" * 70)
    print("Example 5: Search for 'church' or 'cathedral'")
    print("=" * 70)

    churches = search_by_keywords(dataset, ['church', 'cathedral', 'храм', 'собор'])
    print(f"Found {len(churches)} records")

    if len(churches) > 0:
        print("\nFirst 3 results:")
        for i, record in enumerate(churches.select(range(min(3, len(churches))))):
            print(f"\n{i+1}. {record['name']}")
            print(f"   Category: {record['category']}")
            print(f"   Language: {record['language']}")

    # Example 6: Get unique categories
    print("\n" + "=" * 70)
    print("Example 6: All unique categories")
    print("=" * 70)

    categories = get_unique_values(dataset, 'category')
    print(f"Found {len(categories)} unique categories:")
    for cat in categories[:15]:
        count = len(filter_dataset(dataset, category=cat))
        print(f"  • {cat}: {count} records")
    if len(categories) > 15:
        print(f"  ... and {len(categories) - 15} more categories")

    # Example 7: Search in name only
    print("\n" + "=" * 70)
    print("Example 7: Find attractions with 'National' in name")
    print("=" * 70)

    national = filter_dataset(dataset, name_contains='National')
    print(f"Found {len(national)} attractions")

    if len(national) > 0:
        print("\nResults:")
        for record in national.select(range(min(5, len(national)))):
            print(f"  • {record['name']} ({record['language']})")


if __name__ == '__main__':
    main()