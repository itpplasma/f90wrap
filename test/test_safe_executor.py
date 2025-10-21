"""
Unit tests for safe_executor module.

Tests process-isolated execution with shared memory for Direct-C wrappers.
"""

import unittest
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from f90wrap.safe_executor import (
    SafeDirectCExecutor,
    WorkerCrashError,
    TimeoutError as SafeTimeoutError
)


# Mock Direct-C module for testing
class MockDirectCModule:
    """Simulates a Direct-C wrapped Fortran module."""

    __name__ = 'mock_fortran_lib'

    @staticmethod
    def simple_sum(arr):
        """Simple function that returns array sum."""
        return np.sum(arr)

    @staticmethod
    def modify_array(arr):
        """Modifies array in-place."""
        arr[:] = arr * 2.0
        return np.sum(arr)

    @staticmethod
    def with_kwargs(arr, multiplier=1.0, offset=0.0):
        """Function with keyword arguments."""
        return np.sum(arr) * multiplier + offset

    @staticmethod
    def slow_function(arr, duration=0.1):
        """Simulates slow computation."""
        time.sleep(duration)
        return np.sum(arr)

    @staticmethod
    def crashing_function(arr):
        """Simulates a crash (raises exception)."""
        raise RuntimeError("Simulated Fortran crash")

    @staticmethod
    def multiple_arrays(arr1, arr2):
        """Works with multiple arrays."""
        return np.sum(arr1) + np.sum(arr2)


class TestSafeExecutorBasics(unittest.TestCase):
    """Test basic functionality of SafeDirectCExecutor."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = MockDirectCModule()
        self.executor = SafeDirectCExecutor(self.module, timeout=5.0, max_workers=2)

    def test_simple_function_call(self):
        """Test basic function execution."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.sum(arr)

        result = self.executor.simple_sum(arr)

        self.assertAlmostEqual(result, expected)

    def test_array_not_modified_unless_intended(self):
        """Test that arrays aren't modified by accident."""
        arr = np.array([1.0, 2.0, 3.0])
        original = arr.copy()

        result = self.executor.simple_sum(arr)

        np.testing.assert_array_equal(arr, original)

    def test_array_modification_synced_back(self):
        """Test that in-place modifications are synced back."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        expected = arr * 2.0

        result = self.executor.modify_array(arr)

        np.testing.assert_array_almost_equal(arr, expected)

    def test_keyword_arguments(self):
        """Test functions with keyword arguments."""
        arr = np.array([1.0, 2.0, 3.0])

        result1 = self.executor.with_kwargs(arr, multiplier=2.0)
        self.assertAlmostEqual(result1, np.sum(arr) * 2.0)

        result2 = self.executor.with_kwargs(arr, multiplier=1.0, offset=10.0)
        self.assertAlmostEqual(result2, np.sum(arr) + 10.0)

    def test_multiple_arrays(self):
        """Test passing multiple arrays."""
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([3.0, 4.0, 5.0])

        result = self.executor.multiple_arrays(arr1, arr2)

        expected = np.sum(arr1) + np.sum(arr2)
        self.assertAlmostEqual(result, expected)


class TestSafeExecutorArrayTypes(unittest.TestCase):
    """Test different array types and sizes."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = MockDirectCModule()
        self.executor = SafeDirectCExecutor(self.module)

    def test_different_dtypes(self):
        """Test various numpy dtypes."""
        dtypes = [np.float32, np.float64, np.int32, np.int64]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                arr = np.array([1, 2, 3, 4], dtype=dtype)
                result = self.executor.simple_sum(arr)
                expected = np.sum(arr)
                self.assertAlmostEqual(result, expected)

    def test_multidimensional_arrays(self):
        """Test 2D and 3D arrays."""
        arr_2d = np.random.rand(10, 20)
        result_2d = self.executor.simple_sum(arr_2d)
        self.assertAlmostEqual(result_2d, np.sum(arr_2d))

        arr_3d = np.random.rand(5, 10, 15)
        result_3d = self.executor.simple_sum(arr_3d)
        self.assertAlmostEqual(result_3d, np.sum(arr_3d))

    def test_large_arrays(self):
        """Test with large arrays (tests shared memory efficiency)."""
        large_arr = np.random.rand(10000)
        result = self.executor.simple_sum(large_arr)
        expected = np.sum(large_arr)
        self.assertAlmostEqual(result, expected, places=5)

    def test_empty_array(self):
        """Test edge case: empty array."""
        arr = np.array([])
        result = self.executor.simple_sum(arr)
        self.assertEqual(result, 0.0)


class TestSafeExecutorErrorHandling(unittest.TestCase):
    """Test error handling and isolation."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = MockDirectCModule()
        self.executor = SafeDirectCExecutor(self.module, timeout=2.0)

    def test_timeout_raises_exception(self):
        """Test that timeout is properly detected."""
        arr = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(SafeTimeoutError) as cm:
            self.executor.slow_function(arr, duration=5.0)

        self.assertIn("timeout", str(cm.exception).lower())

    def test_crash_raises_worker_crash_error(self):
        """Test that crashes are caught and reported."""
        arr = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(WorkerCrashError) as cm:
            self.executor.crashing_function(arr)

        self.assertIn("crash", str(cm.exception).lower())

    def test_main_process_survives_crash(self):
        """Test that main process continues after worker crash."""
        arr = np.array([1.0, 2.0, 3.0])

        # First call crashes
        with self.assertRaises(WorkerCrashError):
            self.executor.crashing_function(arr)

        # Second call should still work (new worker spawned)
        result = self.executor.simple_sum(arr)
        self.assertAlmostEqual(result, np.sum(arr))


class TestSafeExecutorPerformance(unittest.TestCase):
    """Test performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = MockDirectCModule()
        self.executor = SafeDirectCExecutor(self.module)

    def test_overhead_is_reasonable(self):
        """Test that IPC overhead is reasonable for small arrays."""
        arr = np.array([1.0, 2.0, 3.0])

        # Warmup
        self.executor.simple_sum(arr)

        # Measure 10 calls
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = self.executor.simple_sum(arr)
            elapsed = (time.perf_counter() - start) * 1e6  # microseconds
            times.append(elapsed)

        avg_time = sum(times) / len(times)

        # With real importable modules: ~18 µs
        # With pickled functions (mock objects): ~1-2 ms
        # Use 5 ms as upper bound to catch regressions
        self.assertLess(avg_time, 5000.0,
                       f"Average overhead {avg_time:.1f} µs too high")

    def test_persistent_workers_reused(self):
        """Test that workers are reused (calls are consistently fast)."""
        arr = np.array([1.0, 2.0, 3.0])

        # Make several calls and verify consistent performance
        # (if worker was re-spawned each time, we'd see huge variance)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            self.executor.simple_sum(arr)
            elapsed = (time.perf_counter() - start) * 1e6
            times.append(elapsed)

        # Calculate coefficient of variation (std/mean)
        import statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times)
        cv = std_time / mean_time

        # If workers are reused, variance should be moderate
        # If workers are re-spawned every call, CV would be much higher (>3.0)
        # Note: Pickling adds variance, so we use a generous threshold
        self.assertLess(cv, 2.0,
                       f"High variance suggests workers not reused (CV={cv:.2f}, mean={mean_time:.0f}µs)")


class TestSafeExecutorConcurrency(unittest.TestCase):
    """Test concurrent execution with multiple workers."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = MockDirectCModule()
        self.executor = SafeDirectCExecutor(self.module, max_workers=2)

    def test_multiple_workers(self):
        """Test that multiple workers can be used."""
        arr = np.array([1.0, 2.0, 3.0])

        # Make multiple calls (should use different workers)
        results = []
        for i in range(5):
            result = self.executor.simple_sum(arr)
            results.append(result)

        # All results should be correct
        for result in results:
            self.assertAlmostEqual(result, np.sum(arr))


class TestSafeExecutorNonCallables(unittest.TestCase):
    """Test access to non-callable module attributes."""

    def setUp(self):
        """Set up test fixtures with attributes."""
        class ModuleWithAttrs:
            __name__ = 'test_module'
            constant = 42
            name = "test"

            @staticmethod
            def func(arr):
                return np.sum(arr)

        self.module = ModuleWithAttrs()
        self.executor = SafeDirectCExecutor(self.module)

    def test_access_non_callable_attributes(self):
        """Test that non-callable attributes pass through."""
        self.assertEqual(self.executor.constant, 42)
        self.assertEqual(self.executor.name, "test")


def suite():
    """Build test suite."""
    test_suite = unittest.TestSuite()

    test_suite.addTest(unittest.makeSuite(TestSafeExecutorBasics))
    test_suite.addTest(unittest.makeSuite(TestSafeExecutorArrayTypes))
    test_suite.addTest(unittest.makeSuite(TestSafeExecutorErrorHandling))
    test_suite.addTest(unittest.makeSuite(TestSafeExecutorPerformance))
    test_suite.addTest(unittest.makeSuite(TestSafeExecutorConcurrency))
    test_suite.addTest(unittest.makeSuite(TestSafeExecutorNonCallables))

    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    sys.exit(0 if result.wasSuccessful() else 1)
