"""
Quick script to display preprocessing statistics from metadata.json
"""
import json
from pathlib import Path

metadata_file = Path("data/processed/FaceForensics++_AllTypes/metadata.json")

with open(metadata_file, 'r') as f:
    data = json.load(f)

stats = data['stats']

print("FACEFORENSICS++ ALL TYPES - PREPROCESSING STATISTICS")
print("="*60)

print(f"\n[*] STATISTICS:")
print(f"  Total videos: {stats['total_videos']}")
print(f"  Successfully processed: {stats['processed_videos']}")
print(f"  Failed: {stats['failed_videos']}")
print(f"  Total faces extracted: {stats['total_faces']:,}")

print(f"\n[*] PER-CLASS BREAKDOWN:")
for class_name, class_stats in stats['per_class'].items():
    print(f"  {class_name}:")
    print(f"    Videos: {class_stats['processed']}/{class_stats['total']}")
    print(f"    Faces: {class_stats['faces']:,}")

print("\n" + "="*60)
