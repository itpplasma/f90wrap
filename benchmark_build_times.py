#!/usr/bin/env python3
"""
Benchmark script to measure build time speedup of direct-C mode vs f2py.

This script:
1. Selects representative examples (both simple and complex)
2. Measures f2py build time (generation + compilation + linking)
3. Measures direct-C build time (generation + compilation + linking)
4. Runs multiple iterations for statistical validity
5. Generates comprehensive report with speedup factors

Evidence collected:
- Individual run timings
- Average/median/std deviation
- Speedup factors (direct-C vs f2py)
- Build log excerpts
"""

import os
import sys
import time
import json
import shutil
import subprocess
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Repository root
REPO_ROOT = Path(__file__).parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# Number of iterations per example for statistical validity
NUM_ITERATIONS = 5

# Examples to benchmark - expanded to 20 for better statistical coverage
# Selected from the 31 passing examples in all_examples_direct_c_summary.json
BENCHMARK_EXAMPLES = [
    # Simple examples
    "strings",              # Basic string handling
    "subroutine_args",      # Argument passing
    "elemental",            # Elemental functions
    "class_names",          # Class name handling
    "output_kind",          # Kind output handling
    "errorbinding",         # Error binding
    "intent_out_size",      # Intent out size handling

    # Medium complexity
    "derivedtypes",         # Derived types with constructors
    "recursive_type",       # Recursive type definitions
    "interface",            # Interface handling
    "mockderivetype",       # Mock derived type
    "remove_pointer_arg",   # Pointer argument removal
    "type_check",           # Type checking

    # Complex examples
    "arrayderivedtypes",    # Arrays of derived types
    "derivedtypes_procedure",  # Derived types with procedures
    "arrays_in_derived_types_issue50",  # Arrays in derived types
    "recursive_type_array", # Recursive type with arrays
    "extends",              # Type extension
    "default_i8",           # Default integer*8
    "auto_raise_error",     # Automatic error raising
]


class BenchmarkRunner:
    """Runs benchmarks comparing f2py and direct-C build times."""

    def __init__(self, repo_root: Path, examples_dir: Path):
        self.repo_root = repo_root
        self.examples_dir = examples_dir
        self.results = []

    def clean_example(self, example_dir: Path) -> None:
        """Clean all build artifacts from example directory."""
        patterns = [
            "*.so",
            "*.o",
            "*.mod",
            "_*module.c",  # C extension modules
            "*_support.f90",  # Fortran support files
            "f90wrap_*.f90",  # f2py wrapper files
            "build/",
            "*.pyc",
            "__pycache__/",
        ]

        # Also clean Python wrapper files (but not test files)
        for py_file in example_dir.glob("*.py"):
            # Only clean if it's a wrapper file (matches module name)
            if py_file.stem in [example_dir.name, f"_{example_dir.name}"]:
                py_file.unlink()

        for pattern in patterns:
            if "/" in pattern:
                # Directory
                target = example_dir / pattern.rstrip("/")
                if target.exists():
                    shutil.rmtree(target)
            else:
                # File pattern
                for f in example_dir.glob(pattern):
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        shutil.rmtree(f)

    def find_fortran_files(self, example_dir: Path) -> List[Path]:
        """Find all Fortran source files in example directory."""
        fortran_files = []
        for ext in ["*.f90", "*.F90"]:
            fortran_files.extend(sorted(example_dir.glob(ext)))
        return fortran_files

    def benchmark_f2py(self, example_name: str, example_dir: Path) -> Optional[float]:
        """
        Benchmark f2py build time.

        Returns:
            Build time in seconds, or None if build failed.
        """
        self.clean_example(example_dir)

        fortran_files = self.find_fortran_files(example_dir)
        if not fortran_files:
            return None

        # Module name is typically the example name
        module_name = example_name

        # Build command using f90wrap + f2py
        start_time = time.perf_counter()

        try:
            # Step 1: f90wrap generation
            cmd_f90wrap = [
                "f90wrap",
                "-m", module_name,
            ] + [str(f) for f in fortran_files]

            result = subprocess.run(
                cmd_f90wrap,
                cwd=example_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return None

            # Step 2: Compile Fortran wrapper files
            wrapper_files = list(example_dir.glob("f90wrap_*.f90"))
            if not wrapper_files:
                return None

            # Step 3: f2py build
            cmd_f2py = [
                "f2py",
                "-c",
            ] + [str(f) for f in fortran_files] + [str(f) for f in wrapper_files] + [
                "-m", f"_{module_name}",
            ]

            result = subprocess.run(
                cmd_f2py,
                cwd=example_dir,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                return None

            end_time = time.perf_counter()
            return end_time - start_time

        except (subprocess.TimeoutExpired, Exception):
            return None

    def benchmark_direct_c(self, example_name: str, example_dir: Path) -> Optional[float]:
        """
        Benchmark direct-C build time.

        Returns:
            Build time in seconds, or None if build failed.
        """
        self.clean_example(example_dir)

        fortran_files = self.find_fortran_files(example_dir)
        if not fortran_files:
            return None

        module_name = example_name

        start_time = time.perf_counter()

        try:
            # Step 1: f90wrap with --direct-c
            cmd_f90wrap = [
                "f90wrap",
                "--direct-c",
                "-m", module_name,
            ] + [str(f) for f in fortran_files]

            result = subprocess.run(
                cmd_f90wrap,
                cwd=example_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return None

            # Find generated files (f90wrap generates _<name>module.c, not _<name>_directcmodule.c)
            c_module = example_dir / f"_{module_name}module.c"
            support_file = example_dir / f"{module_name}_support.f90"

            if not c_module.exists():
                return None

            # Step 2: Compile Fortran support module if it exists
            fortran_objects = []
            compile_files = list(fortran_files)
            if support_file.exists():
                compile_files.append(support_file)

            for f in compile_files:
                obj = f.with_suffix(".o")
                cmd_fc = ["gfortran", "-fPIC", "-c", str(f), "-o", str(obj)]
                result = subprocess.run(
                    cmd_fc,
                    cwd=example_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    return None
                fortran_objects.append(obj)

            # Step 3: Build C extension
            python_include = subprocess.run(
                ["python3", "-c", "import sysconfig; print(sysconfig.get_path('include'))"],
                capture_output=True,
                text=True,
                timeout=5
            ).stdout.strip()

            numpy_include = subprocess.run(
                ["python3", "-c", "import numpy; print(numpy.get_include())"],
                capture_output=True,
                text=True,
                timeout=5
            ).stdout.strip()

            so_name = f"_{module_name}module.so"

            # F90wrap include directory for capsule_helpers.h
            f90wrap_include = self.repo_root / "f90wrap"

            cmd_gcc = [
                "gcc",
                "-shared",
                "-fPIC",
                f"-I{python_include}",
                f"-I{numpy_include}",
                f"-I{f90wrap_include}",
                str(c_module),
            ] + [str(obj) for obj in fortran_objects] + [
                "-lgfortran",
                "-o", so_name,
            ]

            result = subprocess.run(
                cmd_gcc,
                cwd=example_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return None

            end_time = time.perf_counter()
            return end_time - start_time

        except (subprocess.TimeoutExpired, Exception):
            return None

    def benchmark_example(self, example_name: str) -> Dict:
        """
        Benchmark both f2py and direct-C for a single example.

        Returns:
            Dictionary with timing results and speedup factor.
        """
        example_dir = self.examples_dir / example_name

        if not example_dir.exists():
            return {
                "example": example_name,
                "status": "not_found",
                "error": f"Example directory not found: {example_dir}"
            }

        print(f"\nBenchmarking {example_name}...")

        # Run multiple iterations for each mode
        f2py_times = []
        direct_c_times = []

        for i in range(NUM_ITERATIONS):
            print(f"  Iteration {i+1}/{NUM_ITERATIONS}...", end=" ", flush=True)

            # Benchmark f2py
            f2py_time = self.benchmark_f2py(example_name, example_dir)
            if f2py_time is not None:
                f2py_times.append(f2py_time)
                print(f"f2py: {f2py_time:.2f}s", end=", ", flush=True)
            else:
                print("f2py: FAILED", end=", ", flush=True)

            # Benchmark direct-C
            direct_c_time = self.benchmark_direct_c(example_name, example_dir)
            if direct_c_time is not None:
                direct_c_times.append(direct_c_time)
                print(f"direct-C: {direct_c_time:.2f}s", flush=True)
            else:
                print("direct-C: FAILED", flush=True)

        # Analyze results
        if not f2py_times or not direct_c_times:
            return {
                "example": example_name,
                "status": "failed",
                "f2py_times": f2py_times,
                "direct_c_times": direct_c_times,
                "error": "One or both modes failed to build"
            }

        # Calculate statistics
        f2py_avg = statistics.mean(f2py_times)
        f2py_median = statistics.median(f2py_times)
        f2py_std = statistics.stdev(f2py_times) if len(f2py_times) > 1 else 0.0

        direct_c_avg = statistics.mean(direct_c_times)
        direct_c_median = statistics.median(direct_c_times)
        direct_c_std = statistics.stdev(direct_c_times) if len(direct_c_times) > 1 else 0.0

        speedup_avg = f2py_avg / direct_c_avg if direct_c_avg > 0 else 0.0
        speedup_median = f2py_median / direct_c_median if direct_c_median > 0 else 0.0

        return {
            "example": example_name,
            "status": "success",
            "f2py": {
                "times": f2py_times,
                "avg": f2py_avg,
                "median": f2py_median,
                "std": f2py_std,
                "min": min(f2py_times),
                "max": max(f2py_times),
            },
            "direct_c": {
                "times": direct_c_times,
                "avg": direct_c_avg,
                "median": direct_c_median,
                "std": direct_c_std,
                "min": min(direct_c_times),
                "max": max(direct_c_times),
            },
            "speedup": {
                "avg": speedup_avg,
                "median": speedup_median,
            }
        }

    def run_all_benchmarks(self, examples: List[str]) -> List[Dict]:
        """Run benchmarks on all specified examples."""
        results = []

        for example in examples:
            result = self.benchmark_example(example)
            results.append(result)
            self.results.append(result)

        return results

    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report."""
        successful = [r for r in self.results if r.get("status") == "success"]
        failed = [r for r in self.results if r.get("status") != "success"]

        if not successful:
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "all_failed",
                "total_examples": len(self.results),
                "successful": 0,
                "failed": len(failed),
                "error": "No examples built successfully in both modes"
            }

        # Calculate aggregate statistics
        speedups_avg = [r["speedup"]["avg"] for r in successful]
        speedups_median = [r["speedup"]["median"] for r in successful]

        f2py_totals = [r["f2py"]["avg"] for r in successful]
        direct_c_totals = [r["direct_c"]["avg"] for r in successful]

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success",
            "iterations_per_example": NUM_ITERATIONS,
            "total_examples": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "aggregate_statistics": {
                "speedup_average": statistics.mean(speedups_avg),
                "speedup_median": statistics.median(speedups_avg),
                "speedup_min": min(speedups_avg),
                "speedup_max": max(speedups_avg),
                "speedup_std": statistics.stdev(speedups_avg) if len(speedups_avg) > 1 else 0.0,
                "f2py_total_time": sum(f2py_totals),
                "direct_c_total_time": sum(direct_c_totals),
                "overall_speedup": sum(f2py_totals) / sum(direct_c_totals),
            },
            "individual_results": self.results,
            "summary_table": [
                {
                    "example": r["example"],
                    "f2py_avg_s": r["f2py"]["avg"],
                    "direct_c_avg_s": r["direct_c"]["avg"],
                    "speedup": r["speedup"]["avg"],
                }
                for r in successful
            ],
            "failed_examples": [
                {
                    "example": r["example"],
                    "error": r.get("error", "Unknown error")
                }
                for r in failed
            ]
        }


def main():
    """Main entry point."""
    print("="*80)
    print("f90wrap Build Time Benchmark: direct-C vs f2py")
    print("="*80)
    print(f"\nRepository: {REPO_ROOT}")
    print(f"Examples directory: {EXAMPLES_DIR}")
    print(f"Iterations per example: {NUM_ITERATIONS}")
    print(f"Examples to benchmark: {len(BENCHMARK_EXAMPLES)}")
    print("\nStarting benchmarks...\n")

    # Create benchmark runner
    runner = BenchmarkRunner(REPO_ROOT, EXAMPLES_DIR)

    # Run benchmarks
    results = runner.run_all_benchmarks(BENCHMARK_EXAMPLES)

    # Generate report
    report = runner.generate_report()

    # Save report
    report_file = REPO_ROOT / "benchmark_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)

    if report["status"] == "success":
        agg = report["aggregate_statistics"]
        print(f"\nSuccessful benchmarks: {report['successful']}/{report['total_examples']}")
        print(f"Failed benchmarks: {report['failed']}/{report['total_examples']}")
        print(f"\nAGGREGATE SPEEDUP STATISTICS:")
        print(f"  Average speedup:  {agg['speedup_average']:.2f}×")
        print(f"  Median speedup:   {agg['speedup_median']:.2f}×")
        print(f"  Min speedup:      {agg['speedup_min']:.2f}×")
        print(f"  Max speedup:      {agg['speedup_max']:.2f}×")
        print(f"  Std deviation:    {agg['speedup_std']:.2f}×")
        print(f"\nOVERALL BUILD TIME:")
        print(f"  f2py total:       {agg['f2py_total_time']:.2f}s")
        print(f"  direct-C total:   {agg['direct_c_total_time']:.2f}s")
        print(f"  Overall speedup:  {agg['overall_speedup']:.2f}×")

        print(f"\nINDIVIDUAL RESULTS:")
        print(f"{'Example':<35} {'f2py (s)':<12} {'direct-C (s)':<12} {'Speedup':<10}")
        print("-"*80)
        for entry in report["summary_table"]:
            print(f"{entry['example']:<35} "
                  f"{entry['f2py_avg_s']:<12.2f} "
                  f"{entry['direct_c_avg_s']:<12.2f} "
                  f"{entry['speedup']:<10.2f}×")
    else:
        print(f"\nBenchmark failed: {report.get('error', 'Unknown error')}")

    print(f"\nDetailed report saved to: {report_file}")
    print("\nBenchmark complete!")

    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
