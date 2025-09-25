#!/usr/bin/env python3
"""
Script to merge JSON chunks back into the original file.
Used during deployment to reconstruct the full dataset.
"""

import json
import os

def merge_json_chunks(data_dir, output_file="hseg_final_dataset.json"):
    """
    Merge JSON chunks back into a single file.

    Args:
        data_dir (str): Directory containing the JSON chunks
        output_file (str): Output file path for merged data
    """
    metadata_file = os.path.join(data_dir, "metadata.json")

    # Check if metadata exists
    if not os.path.exists(metadata_file):
        print(f"Error: metadata.json not found in {data_dir}")
        return False

    # Load metadata
    print("Loading metadata...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"Original file: {metadata['original_file']}")
    print(f"Original size: {metadata['original_size_mb']:.1f} MB")
    print(f"Total chunks: {metadata['total_chunks']}")
    print(f"Data type: {metadata['data_type']}")

    # Load and merge chunks
    merged_data = [] if metadata['data_type'] == 'list' else {}

    for chunk_file in metadata['chunk_files']:
        chunk_path = os.path.join(data_dir, chunk_file)

        if not os.path.exists(chunk_path):
            print(f"Error: Chunk file {chunk_file} not found!")
            return False

        print(f"Loading {chunk_file}...")
        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)

        # Merge based on data type
        if metadata['data_type'] == 'list':
            merged_data.extend(chunk_data)
        else:  # dict
            merged_data.update(chunk_data)

    # Save merged file
    print(f"Saving merged data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, separators=(',', ':'))

    # Verify file size
    merged_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Merge complete! Created {output_file} ({merged_size_mb:.1f} MB)")

    # Verify data integrity
    if metadata['data_type'] == 'list':
        print(f"Verified: {len(merged_data)} total items")
    else:
        print(f"Verified: {len(merged_data)} total keys")

    return True

if __name__ == "__main__":
    data_dir = "data"

    if os.path.exists(data_dir):
        success = merge_json_chunks(data_dir)
        if success:
            print("✅ JSON merge completed successfully!")
        else:
            print("❌ JSON merge failed!")
    else:
        print(f"Error: Data directory '{data_dir}' not found!")