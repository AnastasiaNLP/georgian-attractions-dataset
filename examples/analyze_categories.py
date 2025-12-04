"""
Analyze and visualize Georgian Attractions dataset statistics.

This example demonstrates:
- Category distribution and statistics
- Language distribution
- Image coverage analysis
- Geographic distribution
- Tag analysis
- License information
"""

from datasets import load_dataset
from collections import Counter
from typing import Dict, List
import re


def get_category_stats(dataset) -> Dict:
    """Get detailed statistics by category."""
    categories = Counter(record['category'] for record in dataset)

    stats = {}
    for category, count in categories.items():
        category_records = [r for r in dataset if r['category'] == category]

        # Count images
        with_images = sum(1 for r in category_records if r['image'] is not None)

        # Count by language
        languages = Counter(r['language'] for r in category_records)

        stats[category] = {
            'total': count,
            'with_images': with_images,
            'without_images': count - with_images,
            'image_coverage': (with_images / count * 100) if count > 0 else 0,
            'languages': dict(languages)
        }

    return stats


def get_language_stats(dataset) -> Dict:
    """Get language distribution statistics."""
    languages = Counter(record['language'] for record in dataset)

    stats = {}
    for lang, count in languages.items():
        lang_records = [r for r in dataset if r['language'] == lang]

        with_images = sum(1 for r in lang_records if r['image'] is not None)
        categories = Counter(r['category'] for r in lang_records)

        stats[lang] = {
            'total': count,
            'with_images': with_images,
            'without_images': count - with_images,
            'image_coverage': (with_images / count * 100) if count > 0 else 0,
            'top_categories': categories.most_common(5)
        }

    return stats


def analyze_locations(dataset) -> Dict:
    """Analyze geographic distribution."""
    # Extract cities/regions from location field
    locations = Counter()

    for record in dataset:
        location = record.get('location', '')
        if location:
            # Try to extract main city/region
            # Format usually: "City, Region, Country" or variations
            parts = location.split(',')
            if parts:
                main_location = parts[0].strip()
                locations[main_location] += 1

    return {
        'total_unique_locations': len(locations),
        'top_locations': locations.most_common(15)
    }


def analyze_tags(dataset) -> Dict:
    """Analyze most common tags."""
    all_tags = []

    for record in dataset:
        tags = record.get('tags', '')
        if tags:
            # Split by common separators
            tag_list = re.split(r'[,;|]', tags)
            all_tags.extend([t.strip().lower() for t in tag_list if t.strip()])

    tag_counter = Counter(all_tags)

    return {
        'total_tags': len(all_tags),
        'unique_tags': len(tag_counter),
        'top_tags': tag_counter.most_common(20)
    }


def get_license_stats(dataset) -> Dict:
    """Analyze license distribution."""
    licenses = Counter(record.get('license', 'Not specified') for record in dataset)

    stats = {}
    for license_type, count in licenses.items():
        # Check if authors are specified for licenses that require attribution
        if 'CC' in license_type.upper() or 'BY' in license_type.upper():
            license_records = [r for r in dataset if r.get('license') == license_type]
            with_author = sum(1 for r in license_records if r.get('photo_author'))
            stats[license_type] = {
                'count': count,
                'with_author': with_author,
                'without_author': count - with_author
            }
        else:
            stats[license_type] = {'count': count}

    return stats


def print_category_report(stats: Dict):
    """Print detailed category statistics."""
    print("\n📊 CATEGORY STATISTICS")
    print("=" * 70)

    # Sort by total count
    sorted_categories = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)

    print(f"\n{'Category':<30} {'Total':>8} {'Images':>8} {'Coverage':>10}")
    print("-" * 70)

    for category, data in sorted_categories[:20]:
        print(f"{category:<30} {data['total']:>8} {data['with_images']:>8} {data['image_coverage']:>9.1f}%")

    if len(sorted_categories) > 20:
        remaining = len(sorted_categories) - 20
        remaining_total = sum(data['total'] for _, data in sorted_categories[20:])
        print(f"... and {remaining} more categories ({remaining_total} records)")


def print_language_report(stats: Dict):
    """Print language statistics."""
    print("\n🌍 LANGUAGE STATISTICS")
    print("=" * 70)

    for lang, data in stats.items():
        print(f"\n{lang.upper()}:")
        print(f"  Total records: {data['total']}")
        print(f"  With images: {data['with_images']} ({data['image_coverage']:.1f}%)")
        print(f"  Without images: {data['without_images']}")

        print(f"\n  Top 5 categories:")
        for cat, count in data['top_categories']:
            print(f"    • {cat}: {count}")


def print_location_report(stats: Dict):
    """Print location statistics."""
    print("\n📍 LOCATION STATISTICS")
    print("=" * 70)

    print(f"\nTotal unique locations: {stats['total_unique_locations']}")
    print(f"\nTop 15 locations:")

    for location, count in stats['top_locations']:
        print(f"  • {location}: {count} attractions")


def print_tags_report(stats: Dict):
    """Print tags statistics."""
    print("\n🏷️  TAG STATISTICS")
    print("=" * 70)

    print(f"\nTotal tags used: {stats['total_tags']}")
    print(f"Unique tags: {stats['unique_tags']}")
    print(f"\nTop 20 most common tags:")

    for tag, count in stats['top_tags']:
        print(f"  • {tag}: {count}")


def print_license_report(stats: Dict):
    """Print license statistics."""
    print("\n⚖️  LICENSE STATISTICS")
    print("=" * 70)

    for license_type, data in stats.items():
        print(f"\n{license_type}:")
        print(f"  Count: {data['count']}")
        if 'with_author' in data:
            print(f"  With author: {data['with_author']}")
            print(f"  Without author: {data['without_author']}")
            if data['without_author'] > 0:
                print(f"  ⚠️  Warning: {data['without_author']} records missing author attribution")


def main():
    print("=" * 70)
    print("  GEORGIAN ATTRACTIONS DATASET - COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    # Load dataset
    print("\n📥 Loading dataset from HuggingFace Hub...")
    dataset = load_dataset('AIAnastasia/georgian-attractions', split='train')
    print(f"✅ Loaded {len(dataset)} records")

    # Overall statistics
    print("\n📈 OVERALL STATISTICS")
    print("=" * 70)

    total_records = len(dataset)
    with_images = sum(1 for r in dataset if r['image'] is not None)
    without_images = total_records - with_images

    print(f"\nTotal records: {total_records}")
    print(f"Records with images: {with_images} ({with_images/total_records*100:.1f}%)")
    print(f"Records without images: {without_images} ({without_images/total_records*100:.1f}%)")

    # Category analysis
    print("\n🔄 Analyzing categories...")
    category_stats = get_category_stats(dataset)
    print_category_report(category_stats)

    # Language analysis
    print("\n🔄 Analyzing languages...")
    language_stats = get_language_stats(dataset)
    print_language_report(language_stats)

    # Location analysis
    print("\n🔄 Analyzing locations...")
    location_stats = analyze_locations(dataset)
    print_location_report(location_stats)

    # Tags analysis
    print("\n🔄 Analyzing tags...")
    tag_stats = analyze_tags(dataset)
    print_tags_report(tag_stats)

    # License analysis
    print("\n Analyzing licenses...")
    license_stats = get_license_stats(dataset)
    print_license_report(license_stats)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n Dataset contains {total_records} attractions")
    print(f" {len(category_stats)} unique categories")
    print(f" {len(language_stats)} languages (Russian and English)")
    print(f" {location_stats['total_unique_locations']} unique locations")
    print(f" {tag_stats['unique_tags']} unique tags")
    print(f" {with_images} images ({with_images/total_records*100:.1f}% coverage)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()