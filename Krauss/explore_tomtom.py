"""
Explore TomTom JSON file structure
Figures out what data is available so we can extract what we need

Usage: python explore_tomtom.py <path_to_json>
   or: python explore_tomtom.py   (will ask for path)
"""

import json
import sys
from pathlib import Path
from collections import Counter


def explore_value(value, prefix="", depth=0, max_depth=5):
    """Recursively explore a JSON value and describe its structure"""
    indent = "  " * depth

    if depth > max_depth:
        print(f"{indent}{prefix}... (max depth reached)")
        return

    if isinstance(value, dict):
        print(f"{indent}{prefix}DICT with {len(value)} keys:")
        for i, (k, v) in enumerate(value.items()):
            if i < 15:  # Show first 15 keys
                explore_value(v, prefix=f"['{k}'] → ", depth=depth + 1, max_depth=max_depth)
            elif i == 15:
                print(f"{indent}  ... and {len(value) - 15} more keys")
                break

    elif isinstance(value, list):
        print(f"{indent}{prefix}LIST with {len(value)} items")
        if len(value) > 0:
            # Show first item structure
            print(f"{indent}  First item:")
            explore_value(value[0], prefix="[0] → ", depth=depth + 1, max_depth=max_depth)
            if len(value) > 1:
                print(f"{indent}  Second item:")
                explore_value(value[1], prefix="[1] → ", depth=depth + 1, max_depth=max_depth)
            if len(value) > 2:
                print(f"{indent}  ... ({len(value)} items total)")

    elif isinstance(value, (int, float)):
        print(f"{indent}{prefix}NUMBER: {value}")

    elif isinstance(value, str):
        if len(value) > 100:
            print(f"{indent}{prefix}STRING: '{value[:100]}...' (len={len(value)})")
        else:
            print(f"{indent}{prefix}STRING: '{value}'")

    elif isinstance(value, bool):
        print(f"{indent}{prefix}BOOL: {value}")

    elif value is None:
        print(f"{indent}{prefix}NULL")

    else:
        print(f"{indent}{prefix}TYPE={type(value).__name__}: {str(value)[:100]}")


def find_speed_fields(data, prefix="", results=None, depth=0, max_depth=6):
    """Search for any fields that might contain speed data"""
    if results is None:
        results = []
    if depth > max_depth:
        return results

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else key
            key_lower = key.lower()

            # Check if this key looks speed/traffic related
            if any(word in key_lower for word in [
                "speed", "velocity", "travel", "time", "flow",
                "density", "congestion", "segment", "road",
                "freeflow", "current", "jam", "confidence",
                "length", "frc", "distance"
            ]):
                if isinstance(value, (int, float)):
                    results.append((current_path, f"NUMBER: {value}"))
                elif isinstance(value, str):
                    results.append((current_path, f"STRING: '{value[:80]}'"))
                elif isinstance(value, list):
                    results.append((current_path, f"LIST[{len(value)}]"))
                elif isinstance(value, dict):
                    results.append((current_path, f"DICT[{len(value)} keys]"))

            # Recurse
            if isinstance(value, (dict, list)):
                find_speed_fields(value, current_path, results, depth + 1, max_depth)

    elif isinstance(data, list) and len(data) > 0:
        find_speed_fields(data[0], f"{prefix}[0]", results, depth + 1, max_depth)

    return results


def main():
    # Get file path
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = input("Enter path to TomTom JSON file: ").strip().strip('"')

    path = Path(json_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    file_size_mb = path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print(f"EXPLORING TOMTOM JSON: {path.name}")
    print(f"File size: {file_size_mb:.1f} MB")
    print("=" * 70)

    # Load JSON
    print("\nLoading JSON (this may take a moment for large files)...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 1: TOP-LEVEL STRUCTURE")
    print("=" * 70)
    explore_value(data, max_depth=3)

    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 2: SEARCHING FOR SPEED/TRAFFIC FIELDS")
    print("=" * 70)
    speed_fields = find_speed_fields(data)
    if speed_fields:
        print(f"\nFound {len(speed_fields)} potentially relevant fields:\n")
        for path_str, desc in speed_fields[:50]:
            print(f"  {path_str:<60s} {desc}")
        if len(speed_fields) > 50:
            print(f"\n  ... and {len(speed_fields) - 50} more")
    else:
        print("\nNo obvious speed/traffic fields found by name.")
        print("Showing deeper structure...")
        explore_value(data, max_depth=5)

    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 3: TRYING COMMON TOMTOM FORMATS")
    print("=" * 70)

    # TomTom Move / Traffic Stats API format
    if isinstance(data, dict):
        # Check for common TomTom keys
        for key in ["results", "data", "segments", "flowSegmentData",
                     "routes", "network", "features", "records"]:
            if key in data:
                print(f"\nFound key '{key}':")
                explore_value(data[key], max_depth=4)
                break

    # GeoJSON format
    if isinstance(data, dict) and "type" in data:
        print(f"\nGeoJSON type: {data.get('type')}")
        if "features" in data:
            print(f"Number of features: {len(data['features'])}")
            if len(data['features']) > 0:
                print("\nFirst feature:")
                explore_value(data['features'][0], max_depth=4)

    # List of records format
    if isinstance(data, list):
        print(f"\nTop-level is a LIST with {len(data)} items")
        if len(data) > 0:
            print("\nFirst record:")
            explore_value(data[0], max_depth=4)
            
            # Check all keys across first 100 records
            if isinstance(data[0], dict):
                all_keys = Counter()
                for record in data[:100]:
                    if isinstance(record, dict):
                        all_keys.update(record.keys())
                print(f"\nAll keys across first 100 records:")
                for key, count in all_keys.most_common():
                    print(f"  {key:<40s} (appears in {count}/100 records)")

    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 4: SAMPLE DATA VALUES")
    print("=" * 70)

    # Try to find and print actual speed values
    def extract_sample_records(data, n=5):
        """Try to get a few sample data records"""
        if isinstance(data, list) and len(data) > 0:
            return data[:n]
        elif isinstance(data, dict):
            for key in ["results", "data", "segments", "features",
                        "records", "flowSegmentData"]:
                if key in data:
                    return extract_sample_records(data[key], n)
        return None

    samples = extract_sample_records(data)
    if samples:
        print(f"\nFirst {len(samples)} data records (full detail):\n")
        for i, record in enumerate(samples):
            print(f"--- Record {i} ---")
            print(json.dumps(record, indent=2, default=str)[:2000])
            print()
    else:
        print("\nCouldn't extract sample records automatically.")
        print("Showing raw first 3000 characters of JSON:")
        with open(path, "r", encoding="utf-8") as f:
            print(f.read(3000))

    # ============================================================
    print("\n" + "=" * 70)
    print("SECTION 5: QUICK STATS")
    print("=" * 70)

    # Try to count time-related entries
    def count_entries(data):
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            for key in ["results", "data", "segments", "features", "records"]:
                if key in data and isinstance(data[key], list):
                    return len(data[key])
        return "unknown"

    print(f"\n  Total entries: {count_entries(data)}")
    print(f"  File size: {file_size_mb:.1f} MB")

    print()
    print("=" * 70)
    print("NEXT: Copy the output above and share it with me.")
    print("I'll write the exact extraction script for your data format.")
    print("=" * 70)


if __name__ == "__main__":
    main()