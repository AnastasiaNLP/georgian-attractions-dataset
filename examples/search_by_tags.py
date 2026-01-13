"""
Search Georgian Attractions dataset by tags.

This example demonstrates:
- Searching for attractions by specific tags
- Finding attractions with multiple tags (AND/OR logic)
- Tag-based recommendations
- Analyzing tag relationships
"""

from datasets import load_dataset
from collections import Counter, defaultdict
from typing import List, Set
import re


def parse_tags(tag_string: str) -> Set[str]:
    """Parse tags from a string into a set of lowercase tags."""
    if not tag_string:
        return set()

    # split by common separators
    tags = re.split(r'[,;|]', tag_string)
    return {tag.strip().lower() for tag in tags if tag.strip()}


def search_by_tag(dataset, tag: str, exact_match: bool = False):
    """
    Search for records containing a specific tag.

    Args:
        dataset: HuggingFace Dataset object
        tag: Tag to search for
        exact_match: If True, match tag exactly; if False, match substring

    Returns:
        Filtered dataset
    """
    tag_lower = tag.lower()

    def filter_fn(example):
        record_tags = parse_tags(example.get('tags', ''))

        if exact_match:
            return tag_lower in record_tags
        else:
            return any(tag_lower in t for t in record_tags)

    return dataset.filter(filter_fn)


def search_by_tags_and(dataset, tags: List[str]):
    """
    Search for records containing ALL specified tags (AND logic).

    Args:
        dataset: HuggingFace Dataset object
        tags: List of tags (all must be present)

    Returns:
        Filtered dataset
    """
    tags_lower = {tag.lower() for tag in tags}

    def filter_fn(example):
        record_tags = parse_tags(example.get('tags', ''))
        return tags_lower.issubset(record_tags)

    return dataset.filter(filter_fn)


def search_by_tags_or(dataset, tags: List[str]):
    """
    Search for records containing ANY of the specified tags (OR logic).

    Args:
        dataset: HuggingFace Dataset object
        tags: List of tags (any can be present)

    Returns:
        Filtered dataset
    """
    tags_lower = {tag.lower() for tag in tags}

    def filter_fn(example):
        record_tags = parse_tags(example.get('tags', ''))
        return bool(tags_lower & record_tags)

    return dataset.filter(filter_fn)


def get_all_tags(dataset) -> Counter:
    """Get all tags with their frequencies."""
    all_tags = []

    for record in dataset:
        tags = parse_tags(record.get('tags', ''))
        all_tags.extend(tags)

    return Counter(all_tags)


def find_related_tags(dataset, tag: str, top_n: int = 10):
    """
    Find tags that commonly appear together with the specified tag.

    Args:
        dataset: HuggingFace Dataset object
        tag: Tag to find related tags for
        top_n: Number of related tags to return

    Returns:
        List of (tag, co-occurrence_count) tuples
    """
    tag_lower = tag.lower()
    related_tags = Counter()

    for record in dataset:
        record_tags = parse_tags(record.get('tags', ''))

        if tag_lower in record_tags:
            # Count all other tags that appear with this tag
            for other_tag in record_tags:
                if other_tag != tag_lower:
                    related_tags[other_tag] += 1

    return related_tags.most_common(top_n)


def recommend_by_tags(dataset, example_id: int, top_n: int = 5):
    """
    Recommend similar attractions based on tag similarity.

    Args:
        dataset: HuggingFace Dataset object
        example_id: ID of the example record
        top_n: Number of recommendations to return

    Returns:
        List of recommended records
    """
    # find the example record
    example = None
    for record in dataset:
        if record['id'] == example_id:
            example = record
            break

    if not example:
        return []

    example_tags = parse_tags(example.get('tags', ''))

    if not example_tags:
        return []

    # calculate similarity scores for all other records
    similarities = []

    for record in dataset:
        if record['id'] == example_id:
            continue

        record_tags = parse_tags(record.get('tags', ''))

        if not record_tags:
            continue

        # jaccard similarity: intersection / union
        intersection = len(example_tags & record_tags)
        union = len(example_tags | record_tags)

        if union > 0:
            similarity = intersection / union
            similarities.append((record, similarity))

    # sort by similarity and return top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [record for record, _ in similarities[:top_n]]


def main():
    print("Search by tags examples")

    # load dataset
    print("\n Loading dataset from HuggingFace Hub...")
    dataset = load_dataset('AIAnastasia/georgian-attractions', split='train')
    print(f" Loaded {len(dataset)} records")

    # Example 1: Get all tags
    print("Example 1: Most common tags")

    all_tags = get_all_tags(dataset)
    print(f"\nTotal unique tags: {len(all_tags)}")
    print("\nTop 20 most common tags:")

    for tag, count in all_tags.most_common(20):
        print(f"  • {tag}: {count}")

    print("Example 2: Find attractions tagged with 'unesco'")

    unesco_sites = search_by_tag(dataset, 'unesco', exact_match=False)
    print(f"Found {len(unesco_sites)} UNESCO-related attractions")

    if len(unesco_sites) > 0:
        print("\nFirst 5 results:")
        for i, record in enumerate(unesco_sites.select(range(min(5, len(unesco_sites))))):
            print(f"\n{i+1}. {record['name']}")
            print(f"   Category: {record['category']}")
            print(f"   Tags: {record['tags'][:80]}...")

    # Example 3: Search with AND logic
    print("Example 3: Find attractions with BOTH 'church' AND 'medieval' tags")

    medieval_churches = search_by_tags_and(dataset, ['church', 'medieval'])
    print(f"Found {len(medieval_churches)} medieval churches")

    if len(medieval_churches) > 0:
        print("\nResults:")
        for record in medieval_churches.select(range(min(3, len(medieval_churches)))):
            print(f"  • {record['name']} ({record['language']})")

    # Example 4: Search with OR logic
    
    print("Example 4: Find attractions with 'fortress' OR 'castle' OR 'fortification'")

    fortifications = search_by_tags_or(dataset, ['fortress', 'castle', 'fortification'])
    print(f"Found {len(fortifications)} fortifications")

    if len(fortifications) > 0:
        print("\nFirst 5 results:")
        for i, record in enumerate(fortifications.select(range(min(5, len(fortifications))))):
            print(f"\n{i+1}. {record['name']}")
            print(f"   Tags: {record['tags'][:60]}...")

    # Example 5: Find related tags
    print("Example 5: Find tags related to 'church'")

    related = find_related_tags(dataset, 'church', top_n=15)
    print(f"\nTags that often appear with 'church':")

    for tag, count in related:
        print(f"  • {tag}: appears together {count} times")

    # Example 6: Tag-based recommendations
    print("Example 6: Recommend similar attractions based on tags")

    # find a record with tags
    example_record = None
    for record in dataset:
        if record['tags'] and len(parse_tags(record['tags'])) >= 3:
            example_record = record
            break

    if example_record:
        print(f"\nBased on: {example_record['name']}")
        print(f"Tags: {example_record['tags'][:80]}...")

        recommendations = recommend_by_tags(dataset, example_record['id'], top_n=5)

        print(f"\nRecommended similar attractions:")
        for i, rec in enumerate(recommendations):
            print(f"\n{i+1}. {rec['name']}")
            print(f"   Category: {rec['category']}")
            print(f"   Tags: {rec['tags'][:60]}...")

    # Example 7: Category-specific tag analysis
    print("Example 7: Most common tags for museums")

    museums = [r for r in dataset if r['category'] == 'Музей' or r['category'] == 'Museum']
    museum_tags = Counter()

    for museum in museums:
        tags = parse_tags(museum.get('tags', ''))
        museum_tags.update(tags)

    print(f"\nAnalyzed {len(museums)} museums")
    print("\nTop 10 tags for museums:")

    for tag, count in museum_tags.most_common(10):
        print(f"  • {tag}: {count}")


if __name__ == '__main__':
    main()
