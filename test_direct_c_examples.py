#!/usr/bin/env python3
"""
Test representative examples with direct-C mode end-to-end.
Verifies: generate -> compile -> import -> run tests.
"""
import subprocess
import sys
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Select diverse examples covering different features
EXAMPLES = [
    "arrays",                          # Basic array handling
    "strings",                         # String handling
    "derivedtypes",                    # Derived types (multi-file)
    "subroutine_args",                 # Subroutine argument passing
    "arrayderivedtypes",              # Arrays of derived types (single-file)
    "recursive_type",                  # Recursive type definitions
    "kind_map_default",               # Kind mapping
    "auto_raise_error",               # Error handling
    "callback_print_function_issue93", # Callbacks
]


def analyze_fortran_dependencies(f90_files: List[Path]) -> List[Path]:
    """
    Analyze Fortran file dependencies and return compilation order.

    Uses simple heuristics:
    1. parameters.f90 always comes first (common dependency)
    2. Files with no 'use' statements come early
    3. Files that use other modules come later

    Returns files sorted in dependency order.
    """
    # Read all files and extract module definitions and dependencies
    file_info = {}

    for f90_file in f90_files:
        with open(f90_file, 'r') as f:
            content = f.read()

        # Find module definitions
        modules_defined = re.findall(r'^\s*module\s+(\w+)', content, re.MULTILINE | re.IGNORECASE)
        modules_defined = [m.lower() for m in modules_defined if m.lower() not in ['procedure']]

        # Find module dependencies (use statements)
        uses = re.findall(r'^\s*use\s+(\w+)', content, re.MULTILINE | re.IGNORECASE)
        uses = [u.lower() for u in uses if u.lower() not in ['iso_fortran_env', 'iso_c_binding']]

        file_info[f90_file] = {
            'defines': set(modules_defined),
            'uses': set(uses),
        }

    # Build dependency graph
    sorted_files = []
    remaining = set(f90_files)
    defined_modules = set()

    # Special case: parameters.f90 always first
    for f90_file in list(remaining):
        if f90_file.name.lower() == 'parameters.f90':
            sorted_files.append(f90_file)
            remaining.remove(f90_file)
            defined_modules.update(file_info[f90_file]['defines'])
            break

    # Iteratively add files whose dependencies are satisfied
    max_iterations = len(remaining) + 1
    iteration = 0

    while remaining and iteration < max_iterations:
        iteration += 1
        added_this_iteration = []

        for f90_file in list(remaining):
            info = file_info[f90_file]
            # Check if all dependencies are satisfied
            if info['uses'].issubset(defined_modules):
                sorted_files.append(f90_file)
                added_this_iteration.append(f90_file)
                defined_modules.update(info['defines'])

        for f90_file in added_this_iteration:
            remaining.remove(f90_file)

        if not added_this_iteration and remaining:
            # Can't resolve dependencies, add remaining in alphabetical order
            sorted_files.extend(sorted(remaining, key=lambda x: x.name))
            break

    return sorted_files


def clean_example(example_dir: Path):
    """Clean build artifacts from example directory."""
    patterns = ['*.so', '*.o', '*.mod', '*_support.f90', '_*.c', '*.fpp']
    for pattern in patterns:
        for file in example_dir.glob(pattern):
            file.unlink()


def get_python_includes() -> Tuple[str, str, str]:
    """Get Python, NumPy, and f90wrap include paths."""
    python_include = subprocess.check_output(
        [sys.executable, '-c', 'import sysconfig; print(sysconfig.get_path("include"))'],
        text=True
    ).strip()

    numpy_include = subprocess.check_output(
        [sys.executable, '-c', 'import numpy; print(numpy.get_include())'],
        text=True
    ).strip()

    f90wrap_include = subprocess.check_output(
        [sys.executable, '-c', 'import f90wrap, os; print(os.path.dirname(f90wrap.__file__))'],
        text=True
    ).strip()

    return python_include, numpy_include, f90wrap_include


def build_example_direct_c(example_name: str, repo_root: Path) -> Dict:
    """Build a single example with direct-C mode."""
    example_dir = repo_root / "examples" / example_name

    if not example_dir.exists():
        return {
            "example": example_name,
            "status": "skip",
            "reason": "Example directory not found"
        }

    # Clean first
    clean_example(example_dir)

    # Find Fortran source files (exclude generated support files)
    f90_files = sorted(example_dir.glob("*.f90"))
    f90_files = [f for f in f90_files if not f.name.endswith('_support.f90')]
    if not f90_files:
        return {
            "example": example_name,
            "status": "skip",
            "reason": "No .f90 files found"
        }

    # Analyze dependencies and sort files
    f90_files = analyze_fortran_dependencies(f90_files)

    try:
        # Preprocess all Fortran files
        fpp_files = []
        for f90_file in f90_files:
            fpp_file = f90_file.with_suffix('.fpp')
            result = subprocess.run(
                ['gfortran', '-E', '-x', 'f95-cpp-input', '-fPIC', str(f90_file), '-o', str(fpp_file)],
                cwd=str(example_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return {
                    "example": example_name,
                    "status": "preprocess_fail",
                    "file": f90_file.name,
                    "stderr": result.stderr[-500:]
                }
            fpp_files.append(fpp_file)

        # Determine module name
        module_name = f"{example_name}_directc"

        # Run f90wrap with --direct-c
        kind_map_file = example_dir / "kind_map"
        f90wrap_cmd = [
            'f90wrap',
            '--direct-c',
            '-m', module_name,
        ] + [str(f) for f in fpp_files]

        if kind_map_file.exists():
            f90wrap_cmd.extend(['-k', str(kind_map_file)])

        result = subprocess.run(
            f90wrap_cmd,
            cwd=str(example_dir),
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return {
                "example": example_name,
                "status": "f90wrap_fail",
                "returncode": result.returncode,
                "stderr": result.stderr[-1000:],
                "stdout": result.stdout[-1000:]
            }

        # Compile Fortran sources in dependency order
        for f90_file in f90_files:
            obj_file = f90_file.with_suffix('.o')
            result = subprocess.run(
                ['gfortran', '-c', '-fPIC', str(f90_file), '-o', str(obj_file)],
                cwd=str(example_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return {
                    "example": example_name,
                    "status": "fortran_compile_fail",
                    "file": f90_file.name,
                    "stderr": result.stderr[-500:]
                }

        # Compile support module if it exists
        support_f90 = example_dir / f"{module_name}_support.f90"
        if support_f90.exists():
            support_obj = support_f90.with_suffix('.o')
            result = subprocess.run(
                ['gfortran', '-c', '-fPIC', str(support_f90), '-o', str(support_obj)],
                cwd=str(example_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return {
                    "example": example_name,
                    "status": "support_compile_fail",
                    "stderr": result.stderr[-500:]
                }

        # Compile C wrapper
        c_wrapper = example_dir / f"_{module_name}module.c"
        if not c_wrapper.exists():
            return {
                "example": example_name,
                "status": "no_c_wrapper",
                "reason": "C wrapper not generated by f90wrap"
            }

        python_inc, numpy_inc, f90wrap_inc = get_python_includes()
        c_obj = c_wrapper.with_suffix('.o')

        result = subprocess.run(
            [
                'gcc', '-c', '-fPIC',
                f'-I{python_inc}',
                f'-I{numpy_inc}',
                f'-I{f90wrap_inc}',
                str(c_wrapper),
                '-o', str(c_obj)
            ],
            cwd=str(example_dir),
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return {
                "example": example_name,
                "status": "c_compile_fail",
                "stderr": result.stderr[-500:]
            }

        # Link extension module
        all_objects = sorted(example_dir.glob("*.o"))
        so_file = example_dir / f"_{module_name}.so"

        result = subprocess.run(
            ['gcc', '-shared', '-fPIC'] + [str(o) for o in all_objects] +
            ['-o', str(so_file), '-lgfortran', '-lm'],
            cwd=str(example_dir),
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return {
                "example": example_name,
                "status": "link_fail",
                "stderr": result.stderr[-500:]
            }

        # Test import
        result = subprocess.run(
            [
                sys.executable, '-c',
                f"import sys; sys.path.insert(0, '.'); import {module_name}; print('Import successful')"
            ],
            cwd=str(example_dir),
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return {
                "example": example_name,
                "status": "import_fail",
                "stderr": result.stderr[-500:],
                "stdout": result.stdout[-500:]
            }

        # Run tests if available
        test_file = example_dir / "tests.py"
        if test_file.exists():
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(example_dir),
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                return {
                    "example": example_name,
                    "status": "test_fail",
                    "returncode": result.returncode,
                    "stderr": result.stderr[-500:],
                    "stdout": result.stdout[-500:]
                }
            test_output = result.stdout[-200:]
        else:
            test_output = "No tests.py found (import successful)"

        return {
            "example": example_name,
            "status": "pass",
            "test_output": test_output,
            "files_compiled": [f.name for f in f90_files]
        }

    except subprocess.TimeoutExpired as e:
        return {
            "example": example_name,
            "status": "timeout",
            "reason": f"Command exceeded timeout: {e.cmd}"
        }
    except Exception as e:
        return {
            "example": example_name,
            "status": "error",
            "reason": str(e)
        }


def main():
    repo_root = Path("/home/ert/code/f90wrap")
    results = []

    print(f"Testing {len(EXAMPLES)} representative examples with direct-C mode\n")
    print("=" * 70)

    for i, example in enumerate(EXAMPLES, 1):
        print(f"\n[{i}/{len(EXAMPLES)}] Testing: {example}")
        print("-" * 70)

        result = build_example_direct_c(example, repo_root)
        results.append(result)

        status = result["status"]
        if status == "pass":
            print(f"✓ {example}: PASS")
            if "files_compiled" in result:
                print(f"  Files: {', '.join(result['files_compiled'])}")
        elif status in ["skip", "no_tests"]:
            print(f"- {example}: {status.upper()} ({result.get('reason', 'N/A')})")
        else:
            print(f"✗ {example}: {status.upper()}")
            if "file" in result:
                print(f"  File: {result['file']}")
            if "stderr" in result:
                print(f"  Error output:\n{result['stderr'][:300]}")
            if "stdout" in result and status in ["test_fail", "import_fail", "f90wrap_fail"]:
                print(f"  Output:\n{result['stdout'][:300]}")

    # Calculate statistics
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] not in ["pass", "skip", "no_tests"])
    skipped = sum(1 for r in results if r["status"] in ["skip", "no_tests"])
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0

    print("\n" + "=" * 70)
    print(f"\nRESULTS SUMMARY:")
    print(f"  Total:   {total}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Success: {success_rate:.1f}%")

    # Output JSON report
    report = {
        "total_examples": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "success_rate": success_rate,
        "results": results
    }

    report_file = repo_root / "direct_c_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report: {report_file}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
