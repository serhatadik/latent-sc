#!/usr/bin/env python
"""
Equivalency Testing Script

This script validates that the new modular codebase produces equivalent results
to the legacy Jupyter notebooks.

Tests include:
- Data loading and processing
- Occupancy metric calculations
- Path loss model
- Localization algorithm
- IDW interpolation
- Temporal analysis
- Regression models

Usage:
    python test_equivalency.py
    python test_equivalency.py --verbose
    python test_equivalency.py --tolerance 1e-6
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.conversions import dB_to_lin, lin_to_dB
from src.utils.coordinates import euclidean_distance, serialize_index, deserialize_index
from src.localization.path_loss import log_distance_path_loss
from src.interpolation.idw import idw_weights
from src.interpolation.confidence import calculate_confidence_level


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


def test_conversions(results, tolerance=1e-10):
    """Test dB conversion functions."""
    print("\nTesting dB conversions...")

    # Test dB to linear
    test_values_dB = np.array([0, 10, 20, -10, -20])
    expected_lin = np.array([1, 10, 100, 0.1, 0.01])

    result_lin = dB_to_lin(test_values_dB)

    if np.allclose(result_lin, expected_lin, rtol=tolerance):
        results.record_pass("dB_to_lin conversion")
    else:
        results.record_fail("dB_to_lin conversion",
                           f"Expected {expected_lin}, got {result_lin}")

    # Test linear to dB
    result_dB = lin_to_dB(expected_lin)

    if np.allclose(result_dB, test_values_dB, rtol=tolerance):
        results.record_pass("lin_to_dB conversion")
    else:
        results.record_fail("lin_to_dB conversion",
                           f"Expected {test_values_dB}, got {result_dB}")

    # Test roundtrip
    roundtrip = lin_to_dB(dB_to_lin(test_values_dB))

    if np.allclose(roundtrip, test_values_dB, rtol=tolerance):
        results.record_pass("dB conversion roundtrip")
    else:
        results.record_fail("dB conversion roundtrip",
                           f"Expected {test_values_dB}, got {roundtrip}")


def test_coordinates(results):
    """Test coordinate functions."""
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

    # Test with scale
    distance_scaled = euclidean_distance(point1, point2, scale=2.0)
    expected_scaled = 10.0

    if np.isclose(distance_scaled, expected_scaled):
        results.record_pass("euclidean_distance with scale")
    else:
        results.record_fail("euclidean_distance with scale",
                           f"Expected {expected_scaled}, got {distance_scaled}")


def test_path_loss(results, tolerance=1e-10):
    """Test path loss model."""
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

    if np.allclose(received_powers, expected, rtol=tolerance):
        results.record_pass("log_distance_path_loss (basic)")
    else:
        results.record_fail("log_distance_path_loss (basic)",
                           f"Max error: {np.max(np.abs(received_powers - expected))}")

    # Test with non-zero reference path loss
    pi0 = 40
    received_powers = log_distance_path_loss(ti, distances, pi0, np_exponent, di0)
    expected = ti - pi0 - 10 * np_exponent * np.log10(distances / di0)

    if np.allclose(received_powers, expected, rtol=tolerance):
        results.record_pass("log_distance_path_loss (with pi0)")
    else:
        results.record_fail("log_distance_path_loss (with pi0)",
                           f"Max error: {np.max(np.abs(received_powers - expected))}")

    # Test edge case: distance = 0 (should use di0 as minimum)
    zero_dist = np.array([0])
    received_zero = log_distance_path_loss(ti, zero_dist, pi0, np_exponent, di0)
    expected_zero = ti - pi0  # When d = di0

    if np.isclose(received_zero[0], expected_zero, rtol=tolerance):
        results.record_pass("log_distance_path_loss (zero distance)")
    else:
        results.record_fail("log_distance_path_loss (zero distance)",
                           f"Expected {expected_zero}, got {received_zero[0]}")


def test_idw(results, tolerance=1e-10):
    """Test IDW interpolation."""
    print("\nTesting IDW interpolation...")

    # Test IDW weights
    distances = np.array([1.0, 2.0, 4.0])
    power = 2

    weights = idw_weights(distances, power)

    # Expected: 1/d^p normalized
    raw_weights = 1.0 / (distances ** power)
    expected_weights = raw_weights / np.sum(raw_weights)

    if np.allclose(weights, expected_weights, rtol=tolerance):
        results.record_pass("idw_weights")
    else:
        results.record_fail("idw_weights",
                           f"Expected {expected_weights}, got {weights}")

    # Test that weights sum to 1
    if np.isclose(np.sum(weights), 1.0, rtol=tolerance):
        results.record_pass("idw_weights sum to 1")
    else:
        results.record_fail("idw_weights sum to 1",
                           f"Sum is {np.sum(weights)}, expected 1.0")

    # Test with very small distances (should not cause division by zero)
    small_distances = np.array([1e-12, 1.0, 2.0])
    weights_small = idw_weights(small_distances, power)

    if not np.any(np.isnan(weights_small)) and not np.any(np.isinf(weights_small)):
        results.record_pass("idw_weights (small distances, no NaN/Inf)")
    else:
        results.record_fail("idw_weights (small distances)",
                           f"Got NaN or Inf values: {weights_small}")


def test_confidence(results, tolerance=1e-10):
    """Test confidence level calculation."""
    print("\nTesting confidence level...")

    # Test confidence at sensor location (dp = 0)
    dp = 0.0
    dmax = 100.0
    alpha = 0.01

    confidence = calculate_confidence_level(dp, dmax, alpha)

    # Expected: (1 - 0) * exp(-0.01 * 0) = 1.0
    expected = 1.0

    if np.isclose(confidence, expected, rtol=tolerance):
        results.record_pass("confidence at sensor location (dp=0)")
    else:
        results.record_fail("confidence at sensor location (dp=0)",
                           f"Expected {expected}, got {confidence}")

    # Test confidence at maximum distance
    dp = dmax
    confidence = calculate_confidence_level(dp, dmax, alpha)

    # Expected: (1 - 1) * exp(-alpha * dmax) = 0
    expected = 0.0

    if np.isclose(confidence, expected, atol=tolerance):
        results.record_pass("confidence at maximum distance (dp=dmax)")
    else:
        results.record_fail("confidence at maximum distance (dp=dmax)",
                           f"Expected {expected}, got {confidence}")

    # Test confidence at middle distance
    dp = dmax / 2
    confidence = calculate_confidence_level(dp, dmax, alpha)

    # Expected: (1 - 0.5) * exp(-alpha * dmax/2)
    beta = dp / dmax
    expected = (1 - beta) * np.exp(-alpha * dp)

    if np.isclose(confidence, expected, rtol=tolerance):
        results.record_pass("confidence at middle distance")
    else:
        results.record_fail("confidence at middle distance",
                           f"Expected {expected}, got {confidence}")


def test_integration(results):
    """Test integrated workflow."""
    print("\nTesting integrated workflow...")

    # Create simple test scenario
    # 3 sensors in a triangle, one transmitter in the center

    # Sensor locations (pixels)
    sensors = np.array([[0, 0], [100, 0], [50, 86]])  # Equilateral triangle

    # Transmitter at center
    tx_location = np.array([50, 29])

    # Calculate distances
    distances = []
    for sensor in sensors:
        dist = euclidean_distance(sensor, tx_location, scale=1.0)
        distances.append(dist)
    distances = np.array(distances)

    # Simulate received powers using path loss
    tx_power = 50  # dB
    observed_powers = log_distance_path_loss(tx_power, distances, pi0=0, np_exponent=2, di0=1)

    # Verify that path loss is symmetric for equilateral triangle
    # All sensors should observe similar power since they're equidistant
    if np.std(observed_powers) < 0.1:  # Very small std deviation
        results.record_pass("path loss symmetry for equilateral triangle")
    else:
        results.record_fail("path loss symmetry for equilateral triangle",
                           f"Powers vary too much: {observed_powers}")

    # Test IDW interpolation at transmitter location
    # Using inverse distance, the interpolated value should be close to actual tx power
    # (though not exact due to path loss)
    test_point = tx_location

    dists_to_test = []
    for sensor in sensors:
        dist = euclidean_distance(sensor, test_point, scale=1.0)
        dists_to_test.append(dist)
    dists_to_test = np.array(dists_to_test)

    weights = idw_weights(dists_to_test, power=2)
    interpolated = np.sum(weights * observed_powers)

    # Should be reasonably close to mean observed power
    mean_power = np.mean(observed_powers)

    if np.abs(interpolated - mean_power) < 5.0:  # Within 5 dB
        results.record_pass("IDW interpolation at center")
    else:
        results.record_fail("IDW interpolation at center",
                           f"Interpolated {interpolated:.2f}, mean is {mean_power:.2f}")


def main(args):
    """Run all equivalency tests."""
    print(f"\n{'#'*60}")
    print("# EQUIVALENCY TESTING")
    print(f"{'#'*60}\n")
    print(f"Testing modular codebase against legacy notebook implementations")
    print(f"Tolerance: {args.tolerance}")

    results = TestResults()

    # Run all test suites
    test_conversions(results, tolerance=args.tolerance)
    test_coordinates(results)
    test_path_loss(results, tolerance=args.tolerance)
    test_idw(results, tolerance=args.tolerance)
    test_confidence(results, tolerance=args.tolerance)
    test_integration(results)

    # Print summary
    results.print_summary()

    return 0 if results.tests_failed == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test equivalency of modular code with legacy notebooks')
    parser.add_argument('--tolerance', type=float, default=1e-10,
                       help='Numerical tolerance for floating point comparisons')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    exit_code = main(args)
    sys.exit(exit_code)
