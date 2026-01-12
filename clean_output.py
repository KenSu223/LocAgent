#!/usr/bin/env python3
"""
Clean output files with failed results.

This script scans all *.json files in the output folder and removes
files that have "success": false.

Usage:
    python -m clean_output
"""

import json
import os
from pathlib import Path


def clean_failed_outputs():
    """Scan output folder and remove all JSON files with success: false."""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    # Find all JSON files recursively
    json_files = list(output_dir.rglob("*.json"))

    if not json_files:
        print("No JSON files found in output folder")
        return

    print(f"Scanning {len(json_files)} JSON files in {output_dir}...")
    print("=" * 60)

    removed_count = 0
    skipped_count = 0
    error_count = 0

    for json_file in json_files:
        # Skip results.json summary files
        if json_file.name == "results.json":
            skipped_count += 1
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check if success is False
            if isinstance(data, dict) and data.get("success") is False:
                rel_path = json_file.relative_to(output_dir)
                print(f"[REMOVE] {rel_path}")
                json_file.unlink()
                removed_count += 1

        except json.JSONDecodeError:
            print(f"[ERROR] Invalid JSON: {json_file.name}")
            error_count += 1
        except Exception as e:
            print(f"[ERROR] {json_file.name}: {e}")
            error_count += 1

    print("=" * 60)
    print(f"Summary:")
    print(f"  Removed: {removed_count} failed result files")
    print(f"  Skipped: {skipped_count} summary files (results.json)")
    print(f"  Errors:  {error_count}")


if __name__ == "__main__":
    clean_failed_outputs()
