"""
Test bigO's unit testing utilities.

This demonstrates the fix for https://github.com/plasma-umass/bigO/issues/16
"""
import os
import bigO
from bigO import assert_bounds, disable_persistence, clear_performance_data, BigOError


def find_intersection_quadratic(a: list, b: list) -> list:
    """O(n^2) implementation using nested iteration."""
    result = []
    for value in a:
        if value in b:
            result.append(value)
    return result


def find_intersection_linear(a: list, b: list) -> list:
    """O(n) implementation using sets."""
    a = set(a)
    b = set(b)
    return list(a.intersection(b))


def test_assert_bounds_linear():
    """Test that a linear function passes O(n) bounds check."""
    data = list(range(10_000))
    inputs = [
        (data[:i], data[1:i+1])
        for i in range(1_000, 10_001, 1_000)
    ]

    def input_length(a, b):
        return len(a) + len(b)

    # This should pass
    assert_bounds(
        find_intersection_linear,
        input_length,
        inputs,
        time="O(n)",
    )
    print("Linear function correctly passes O(n) bounds check")


def test_assert_bounds_quadratic_fails():
    """Test that a quadratic function fails O(n) bounds check."""
    data = list(range(5_000))
    inputs = [
        (data[:i], data[1:i+1])
        for i in range(500, 5_001, 500)
    ]

    def input_length(a, b):
        return len(a) + len(b)

    # This should fail - quadratic function claimed to be O(n)
    try:
        assert_bounds(
            find_intersection_quadratic,
            input_length,
            inputs,
            time="O(n)",
        )
        print("ERROR: Quadratic function incorrectly passed O(n) bounds check")
    except BigOError as e:
        print(f"Quadratic function correctly failed O(n) bounds check: {e.message[:50]}...")


def test_multiple_functions_isolated():
    """Test that multiple function checks don't interfere with each other."""
    data = list(range(5_000))
    inputs = [
        (data[:i], data[1:i+1])
        for i in range(500, 5_001, 500)
    ]

    def input_length(a, b):
        return len(a) + len(b)

    # First check the linear function (should pass)
    assert_bounds(
        find_intersection_linear,
        input_length,
        inputs,
        time="O(n)",
    )
    print("First check (linear, O(n)): PASSED")

    # Now check the quadratic function (should fail)
    # This previously failed due to data mixing with the linear function
    try:
        assert_bounds(
            find_intersection_quadratic,
            input_length,
            inputs,
            time="O(n)",
        )
        print("Second check (quadratic, O(n)): INCORRECTLY PASSED - data isolation broken!")
    except BigOError:
        print("Second check (quadratic, O(n)): Correctly FAILED")


def test_no_json_file_written():
    """Test that disable_persistence() prevents JSON file creation."""
    disable_persistence()

    # Remove any existing file
    json_file = "bigO_data.json"
    if os.path.exists(json_file):
        os.remove(json_file)

    data = list(range(1_000))
    inputs = [(data[:i], data[1:i+1]) for i in range(100, 1_001, 100)]

    def input_length(a, b):
        return len(a) + len(b)

    assert_bounds(
        find_intersection_linear,
        input_length,
        inputs,
        time="O(n)",
    )

    # File should not exist
    if not os.path.exists(json_file):
        print("No JSON file written (as expected with disable_persistence)")
    else:
        print("ERROR: JSON file was written despite disable_persistence()")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing bigO unit testing utilities")
    print("=" * 60)
    print()

    print("Test 1: Linear function with O(n) bounds")
    print("-" * 40)
    test_assert_bounds_linear()
    print()

    print("Test 2: Quadratic function should fail O(n) bounds")
    print("-" * 40)
    test_assert_bounds_quadratic_fails()
    print()

    print("Test 3: Multiple functions should be isolated")
    print("-" * 40)
    test_multiple_functions_isolated()
    print()

    print("Test 4: No JSON file with disable_persistence()")
    print("-" * 40)
    test_no_json_file_written()
    print()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
