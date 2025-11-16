"""
Data Validation Script.

This script validates the consistency of processed JSON data.
Migrated from check.py with enhanced error reporting.

Usage:
    python validate_data.py --input data.json
    python validate_data.py  # defaults to ../processed_data/data.json
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from config import settings


def check_consistency(json_file_path: Path, verbose: bool = False) -> bool:
    """
    Check consistency of receiver coordinates in JSON data.

    For each timestamp, all pow_rx_tx entries should have the same
    receiver coordinates (2nd and 3rd elements should match).

    Args:
        json_file_path: Path to JSON file
        verbose: Print detailed information

    Returns:
        True if data is consistent, False otherwise
    """
    print(f"Loading data from {json_file_path}...")

    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"ERROR: File not found: {json_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON format: {e}")
        return False

    print(f"Validating {len(data)} timestamps...")

    inconsistent_count = 0
    consistent_count = 0

    for timestamp, values in data.items():
        pow_rx_tx = values.get('pow_rx_tx', [])

        if not pow_rx_tx:
            if verbose:
                print(f"WARNING: No pow_rx_tx data at timestamp {timestamp}")
            continue

        # Get reference receiver coordinates from first entry
        reference_lat = pow_rx_tx[0][1]  # 2nd element (latitude)
        reference_lon = pow_rx_tx[0][2]  # 3rd element (longitude)

        # Check all entries have matching receiver coordinates
        inconsistent = False
        for i, entry in enumerate(pow_rx_tx):
            if len(entry) < 3:
                print(f"ERROR: Invalid entry at timestamp {timestamp}, index {i}: {entry}")
                inconsistent_count += 1
                inconsistent = True
                break

            if entry[1] != reference_lat or entry[2] != reference_lon:
                print(f"ERROR: Inconsistent receiver coordinates at timestamp {timestamp}")
                print(f"  Expected: lat={reference_lat}, lon={reference_lon}")
                print(f"  Found at index {i}: lat={entry[1]}, lon={entry[2]}")
                inconsistent_count += 1
                inconsistent = True
                break

        if not inconsistent:
            consistent_count += 1
            if verbose:
                print(f"✓ Timestamp {timestamp}: {len(pow_rx_tx)} entries, all consistent")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total timestamps: {len(data)}")
    print(f"Consistent timestamps: {consistent_count}")
    print(f"Inconsistent timestamps: {inconsistent_count}")

    if inconsistent_count == 0:
        print("\n✓ All pow_rx_tx entries have consistent receiver coordinates.")
        print("Data validation PASSED!")
        return True
    else:
        print(f"\n✗ Found {inconsistent_count} timestamps with inconsistent data.")
        print("Data validation FAILED!")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate consistency of RF measurement JSON data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.PROCESSED_DATA_DIR / "data.json",
        help="Path to JSON file (default: ../processed_data/data.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation information"
    )

    args = parser.parse_args()

    # Run validation
    is_valid = check_consistency(args.input, args.verbose)

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
