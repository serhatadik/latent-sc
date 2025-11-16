#!/usr/bin/env python
"""
Basic Function Testing Script

Tests core mathematical functions without importing heavy dependencies.
This script can run with minimal requirements (just numpy).
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResults:
    """Track test results."""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []

    def record_pass(self, test_name):
        """Record a passed test."""
        self.tests_run += 1
        self.tests_passed += 1
        print(f"  [PASS] {test_name}")

    def record_fail(self, test_name, reason):
        """Record a failed test."""
        self.tests_run += 1
        self.tests_failed += 1
        self.failures.append((test_name, reason))
        print(f"  [FAIL] {test_name}: {reason}")

    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"  Total tests: {self.tests_run}")
        print(f"  Passed: {self.tests_passed}")
        print(f"  Failed: {self.tests_failed}")

        if self.tests_failed > 0:
            print(f"\nFailed tests:")
            for test_name, reason in self.failures:
                print(f"  - {test_name}: {reason}")

        print(f"\n{'='*60}")
        if self.tests_failed == 0:
            print("[SUCCESS] ALL TESTS PASSED")
        else:
            print(f"[FAILURE] {self.tests_failed} TESTS FAILED")
        print(f"{'='*60}\n")


# Import individual functions to test
try:
    from src.utils.conversions import dB_to_lin, lin_to_dB
    CONVERSIONS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import conversions: {e}")
    CONVERSIONS_AVAILABLE = False

try:
    from src.utils.coordinates import euclidean_distance, serialize_index, deserialize_index
    COORDINATES_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import coordinates: {e}")
    COORDINATES_AVAILABLE = False

try:
    from src.localization.path_loss import log_distance_path_loss
    PATH_LOSS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import path_loss: {e}")
    PATH_LOSS_AVAILABLE = False

try:
    from src.interpolation.idw import idw_weights
    IDW_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import idw: {e}")
    IDW_AVAILABLE = False

try:
    from src.interpolation.confidence import calculate_confidence_level
    CONFIDENCE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import confidence: {e}")
    CONFIDENCE_AVAILABLE = False


def test_conversions(results):
    """Test dB conversion functions."""
    if not CONVERSIONS_AVAILABLE:
        print("\nSkipping conversion tests (module not available)")
        return

    print("\nTesting dB conversions...")

    # Test dB to linear
    test_values_dB = np.array([0, 10, 20, -10, -20])
    expected_lin = np.array([1, 10, 100, 0.1, 0.01])

    result_lin = dB_to_lin(test_values_dB)

    if np.allclose(result_lin, expected_lin, rtol=1e-10):
        results.record_pass("dB_to_lin conversion")
    else:
        results.record_fail("dB_to_lin conversion",
                           f"Expected {expected_lin}, got {result_lin}")

    # Test linear to dB
    result_dB = lin_to_dB(expected_lin)

    if np.allclose(result_dB, test_values_dB, rtol=1e-10):
        results.record_pass("lin_to_dB conversion")
    else:
        results.record_fail("lin_to_dB conversion",
                           f"Expected {test_values_dB}, got {result_dB}")

    # Test roundtrip
    roundtrip = lin_to_dB(dB_to_lin(test_values_dB))

    if np.allclose(roundtrip, test_values_dB, rtol=1e-10):
        results.record_pass("dB conversion roundtrip")
    else:
        results.record_fail("dB conversion roundtrip",
                           f"Expected {test_values_dB}, got {roundtrip}")


def test_coordinates(results):
    """Test coordinate functions."""
    if not COORDINATES_AVAILABLE:
        print("\nSkipping coordinate tests (module not available)")
        return

    print("\nTesting coordinate functions...")

    # Test serialize/deserialize
    row, col = 5, 10
    num_cols = 50
    serialized = serialize_index(row, col, num_cols)
    expected_serial = row * num_cols + col

    if serialized == expected_serial:
        results.record_pass("serialize_index")
    else:
        results.record_fail("serialize_index",
                           f"Expected {expected_serial}, got {serialized}")

    row_back, col_back = deserialize_index(serialized, num_cols)

    if row_back == row and col_back == col:
        results.record_pass("deserialize_index")
    else:
        results.record_fail("deserialize_index",
                           f"Expected ({row}, {col}), got ({row_back}, {col_back})")

    # Test Euclidean distance
    point1 = np.array([0, 0])
    point2 = np.array([3, 4])
    distance = euclidean_distance(point1, point2, scale=1.0)
    expected_distance = 5.0

    if np.isclose(distance, expected_distance):
        results.record_pass("euclidean_distance (3-4-5 triangle)")
    else:
        results.record_fail("euclidean_distance (3-4-5 triangle)",
                           f"Expected {expected_distance}, got {distance}")


def test_path_loss(results):
    """Test path loss model."""
    if not PATH_LOSS_AVAILABLE:
        print("\nSkipping path loss tests (module not available)")
        return

    print("\nTesting path loss model...")

    # Test basic path loss (Equation 1)
    ti = 50  # Transmit power in dB
    distances = np.array([1, 10, 100, 1000])
    pi0 = 0  # Reference path loss
    np_exponent = 2  # Path loss exponent
    di0 = 1  # Reference distance

    received_powers = log_distance_path_loss(ti, distances, pi0, np_exponent, di0)

    # Expected: ti - 10*np*log10(d/di0)
    expected = ti - 10 * np_exponent * np.log10(distances / di0)

    if np.allclose(received_powers, expected, rtol=1e-10):
        results.record_pass("log_distance_path_loss (basic)")
    else:
        results.record_fail("log_distance_path_loss (basic)",
                           f"Max error: {np.max(np.abs(received_powers - expected))}")


def test_idw(results):
    """Test IDW interpolation."""
    if not IDW_AVAILABLE:
        print("\nSkipping IDW tests (module not available)")
        return

    print("\nTesting IDW interpolation...")

    # Test IDW weights
    distances = np.array([1.0, 2.0, 4.0])
    power = 2

    weights = idw_weights(distances, power)

    # Expected: 1/d^p normalized
    raw_weights = 1.0 / (distances ** power)
    expected_weights = raw_weights / np.sum(raw_weights)

    if np.allclose(weights, expected_weights, rtol=1e-10):
        results.record_pass("idw_weights")
    else:
        results.record_fail("idw_weights",
                           f"Expected {expected_weights}, got {weights}")

    # Test that weights sum to 1
    if np.isclose(np.sum(weights), 1.0, rtol=1e-10):
        results.record_pass("idw_weights sum to 1")
    else:
        results.record_fail("idw_weights sum to 1",
                           f"Sum is {np.sum(weights)}, expected 1.0")


def test_confidence(results):
    """Test confidence level calculation."""
    if not CONFIDENCE_AVAILABLE:
        print("\nSkipping confidence tests (module not available)")
        return

    print("\nTesting confidence level...")

    # Test confidence at sensor location (dp = 0)
    dp = 0.0
    dmax = 100.0
    alpha = 0.01

    confidence = calculate_confidence_level(dp, dmax, alpha)

    # Expected: (1 - 0) * exp(-0.01 * 0) = 1.0
    expected = 1.0

    if np.isclose(confidence, expected, rtol=1e-10):
        results.record_pass("confidence at sensor location (dp=0)")
    else:
        results.record_fail("confidence at sensor location (dp=0)",
                           f"Expected {expected}, got {confidence}")


def main():
    """Run all basic function tests."""
    print(f"\n{'#'*60}")
    print("# BASIC FUNCTION TESTING")
    print(f"{'#'*60}\n")
    print(f"Testing core mathematical functions")

    results = TestResults()

    # Run all test suites
    test_conversions(results)
    test_coordinates(results)
    test_path_loss(results)
    test_idw(results)
    test_confidence(results)

    # Print summary
    results.print_summary()

    return 0 if results.tests_failed == 0 else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
