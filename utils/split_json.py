#!/usr/bin/env python3
"""
Script to split large JSON file into smaller chunks for GitHub compatibility.
Splits the hseg_final_dataset.json file into 20MB chunks.
"""

import json
import os
import math

def split_json_file(input_file, output_dir, chunk_size_mb=20):
    """
    Split a large JSON file into smaller chunks.

    Args:
        input_file (str): Path to the input JSON file
        output_dir (str): Directory to save the chunks
        chunk_size_mb (int): Maximum size per chunk in MB
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON data
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Calculate chunk size in bytes
    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    # If data is a list, split by items
    if isinstance(data, list):
        total_items = len(data)
        print(f"Total items: {total_items}")

        # Estimate items per chunk
        total_size = os.path.getsize(input_file)
        items_per_chunk = max(1, int((chunk_size_bytes / total_size) * total_items))

        # Split into chunks
        chunks = []
        for i in range(0, total_items, items_per_chunk):
            chunk = data[i:i + items_per_chunk]
            chunks.append(chunk)

        print(f"Created {len(chunks)} chunks with approximately {items_per_chunk} items each")

    # If data is a dict, split by keys
    elif isinstance(data, dict):
        keys = list(data.keys())
        total_keys = len(keys)
        print(f"Total keys: {total_keys}")

        # Estimate keys per chunk
        total_size = os.path.getsize(input_file)
        keys_per_chunk = max(1, int((chunk_size_bytes / total_size) * total_keys))

        # Split into chunks
        chunks = []
        for i in range(0, total_keys, keys_per_chunk):
            chunk_keys = keys[i:i + keys_per_chunk]
            chunk = {key: data[key] for key in chunk_keys}
            chunks.append(chunk)

        print(f"Created {len(chunks)} chunks with approximately {keys_per_chunk} keys each")

    else:
        print("Error: JSON data must be either a list or dictionary to split")
        return

    # Save chunks
    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_dir, f"hseg_data_part_{i+1:02d}.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, separators=(',', ':'))  # Compact format

        # Check file size
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Created {output_file} ({file_size_mb:.1f} MB)")

    # Create metadata file
    metadata = {
        "original_file": input_file,
        "original_size_mb": os.path.getsize(input_file) / (1024 * 1024),
        "total_chunks": len(chunks),
        "data_type": "list" if isinstance(data, list) else "dict",
        "chunk_files": [f"hseg_data_part_{i+1:02d}.json" for i in range(len(chunks))]
    }

    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSplit complete! Created {len(chunks)} chunks in {output_dir}")
    print(f"Metadata saved to {metadata_file}")

if __name__ == "__main__":
    input_file = "hseg_final_dataset.json"
    output_dir = "data"

    if os.path.exists(input_file):
        split_json_file(input_file, output_dir)
    else:
        print(f"Error: {input_file} not found!")