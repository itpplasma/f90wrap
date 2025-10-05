#!/usr/bin/env python3
"""
Test all f90wrap examples with the --direct-c flag.
Documents baseline compatibility status for each example.
"""

import os
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import traceback

EXAMPLES_DIR = Path("/home/ert/code/f90wrap/examples")
RESULTS_DIR = Path("/home/ert/code/f90wrap/direct_c_test_results")
REPORT_FILE = RESULTS_DIR / "compatibility_report.md"
JSON_REPORT = RESULTS_DIR / "compatibility_results.json"

# Skip __pycache__ and other non-example directories
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}

def run_command(cmd, cwd=None, timeout=60):
    """Run a command and return result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "success": False
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False
        }

def find_fortran_files(example_dir):
    """Find Fortran source files in example directory."""
    f90_files = list(example_dir.glob("*.f90"))
    f_files = list(example_dir.glob("*.f"))
    fpp_files = list(example_dir.glob("*.fpp"))
    return f90_files + f_files + fpp_files

def test_example_direct_c(example_name, example_dir):
    """Test a single example with --direct-c flag."""
    result = {
        "name": example_name,
        "path": str(example_dir),
        "status": "SKIP",
        "f90wrap_output": "",
        "f90wrap_error": "",
        "test_output": "",
        "test_error": "",
        "error_category": None,
        "notes": []
    }

    # Create a temporary working directory for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy example files to temp directory
        try:
            for item in example_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, tmpdir / item.name)
                elif item.is_dir() and item.name not in SKIP_DIRS:
                    shutil.copytree(item, tmpdir / item.name)
        except Exception as e:
            result["status"] = "SKIP"
            result["notes"].append(f"Failed to copy example files: {e}")
            return result

        # Find Fortran source files
        fortran_files = find_fortran_files(tmpdir)
        if not fortran_files:
            result["status"] = "SKIP"
            result["notes"].append("No Fortran source files found")
            return result

        # Check for preprocessed files (.fpp) - use them if available
        fpp_files = list(tmpdir.glob("*.fpp"))
        if fpp_files:
            # .fpp files are already preprocessed, but f90wrap expects them
            source_files = fpp_files
            result["notes"].append(f"Using preprocessed files: {[f.name for f in fpp_files]}")
            # We also need the .f90 files for compilation
            f90_files = list(tmpdir.glob("*.f90"))
            if not f90_files:
                # If no .f90 files exist, the .fpp are the preprocessed versions
                # Strip the preprocessor markers to get clean .f90 files
                for fpp_file in fpp_files:
                    clean_f90 = tmpdir / f"{fpp_file.stem}.f90"
                    if not clean_f90.exists():
                        # Read fpp and strip preprocessor line markers
                        with open(fpp_file, 'r') as f:
                            lines = f.readlines()
                        clean_lines = [line for line in lines if not line.startswith('#')]
                        with open(clean_f90, 'w') as f:
                            f.writelines(clean_lines)
        else:
            source_files = fortran_files
            result["notes"].append(f"Using source files: {[f.name for f in source_files]}")

        # Check for kind_map file
        kind_map_file = tmpdir / "kind_map"
        kind_map_arg = f"-k {kind_map_file}" if kind_map_file.exists() else ""

        # Build f90wrap command with --direct-c flag
        source_file_args = " ".join(str(f) for f in source_files)
        f90wrap_cmd = f"f90wrap -m {example_name}_direct {source_file_args} {kind_map_arg} --direct-c -v"

        result["notes"].append(f"Command: {f90wrap_cmd}")

        # Run f90wrap with --direct-c
        wrap_result = run_command(f90wrap_cmd, cwd=tmpdir, timeout=30)
        result["f90wrap_output"] = wrap_result["stdout"]
        result["f90wrap_error"] = wrap_result["stderr"]

        if not wrap_result["success"]:
            result["status"] = "FAIL"
            result["error_category"] = categorize_error(wrap_result["stderr"])
            result["notes"].append(f"f90wrap failed with return code {wrap_result['returncode']}")
            return result

        # Check if C files were generated
        c_files = list(tmpdir.glob("*.c"))
        if not c_files:
            result["status"] = "FAIL"
            result["error_category"] = "no_c_files_generated"
            result["notes"].append("No C files generated by f90wrap --direct-c")
            return result

        result["notes"].append(f"Generated C files: {[f.name for f in c_files]}")

        # Try to compile the generated C code
        # First compile Fortran sources to object files
        compile_success = True
        # Get the .f90 files for compilation (not .fpp)
        f90_for_compile = list(tmpdir.glob("*.f90"))
        if not f90_for_compile:
            # Try .f files if no .f90 found
            f90_for_compile = list(tmpdir.glob("*.f"))

        for f_file in f90_for_compile:
            compile_cmd = f"gfortran -fPIC -c {f_file} -o {f_file.stem}.o"
            compile_result = run_command(compile_cmd, cwd=tmpdir)
            if not compile_result["success"]:
                compile_success = False
                result["notes"].append(f"Failed to compile {f_file.name}: {compile_result['stderr'][:500]}")
                break

        if not compile_success:
            result["status"] = "FAIL"
            result["error_category"] = "fortran_compilation_failed"
            return result

        # Compile C files
        for c_file in c_files:
            # Get Python include paths
            python_includes_cmd = "python3 -c 'import sysconfig; print(sysconfig.get_paths()[\"include\"])'"
            includes_result = run_command(python_includes_cmd)
            python_include = includes_result["stdout"].strip() if includes_result["success"] else ""

            numpy_includes_cmd = "python3 -c 'import numpy; print(numpy.get_include())'"
            numpy_result = run_command(numpy_includes_cmd)
            numpy_include = numpy_result["stdout"].strip() if numpy_result["success"] else ""

            # Add f90wrap include directory for capsule_helpers.h
            f90wrap_include = "/home/ert/code/f90wrap/f90wrap"

            include_flags = f" -I{f90wrap_include}"
            if python_include:
                include_flags += f" -I{python_include}"
            if numpy_include:
                include_flags += f" -I{numpy_include}"

            compile_c_cmd = f"gcc -fPIC -c {c_file} {include_flags} -o {c_file.stem}.o"
            compile_c_result = run_command(compile_c_cmd, cwd=tmpdir)
            if not compile_c_result["success"]:
                result["status"] = "FAIL"
                result["error_category"] = "c_compilation_failed"
                result["notes"].append(f"Failed to compile {c_file.name}: {compile_c_result['stderr'][:500]}")
                return result

        # Link into shared library
        obj_files = list(tmpdir.glob("*.o"))
        if obj_files:
            obj_file_args = " ".join(str(f) for f in obj_files)
            link_cmd = f"gcc -shared {obj_file_args} -lgfortran -o _{example_name}_direct.so"
            link_result = run_command(link_cmd, cwd=tmpdir)
            if not link_result["success"]:
                result["status"] = "FAIL"
                result["error_category"] = "linking_failed"
                result["notes"].append(f"Linking failed: {link_result['stderr'][:500]}")
                return result

        # Check if tests.py exists and try to run it
        test_file = tmpdir / "tests.py"
        if test_file.exists():
            # Modify tests.py to import the direct-c module
            try:
                test_content = test_file.read_text()
                # Simple replacement - this might need refinement for complex cases
                modified_content = test_content.replace(f"import {example_name}", f"import {example_name}_direct")
                modified_content = modified_content.replace(f"from {example_name}", f"from {example_name}_direct")
                test_file.write_text(modified_content)
                result["notes"].append("Modified tests.py to use direct-c module")
            except Exception as e:
                result["notes"].append(f"Failed to modify tests.py: {e}")

            # Run the test
            test_cmd = f"cd {tmpdir} && python3 tests.py"
            test_result = run_command(test_cmd, cwd=tmpdir, timeout=60)
            result["test_output"] = test_result["stdout"]
            result["test_error"] = test_result["stderr"]

            if test_result["success"]:
                result["status"] = "PASS"
                result["notes"].append("Tests passed successfully")
            else:
                result["status"] = "FAIL"
                result["error_category"] = "test_execution_failed"
                result["notes"].append(f"Test failed with return code {test_result['returncode']}")
        else:
            # No tests.py, but compilation succeeded
            result["status"] = "PASS"
            result["notes"].append("No tests.py found, but compilation succeeded")

    return result

def categorize_error(error_text):
    """Categorize error messages for analysis."""
    error_lower = error_text.lower()

    if "notimplementederror" in error_lower:
        if "abstract" in error_lower or "interface" in error_lower:
            return "abstract_interface_not_supported"
        elif "type-bound" in error_lower or "procedure" in error_lower:
            return "type_bound_procedures_not_supported"
        elif "optional" in error_lower:
            return "optional_args_not_supported"
        elif "pointer" in error_lower:
            return "pointer_not_supported"
        elif "allocatable" in error_lower:
            return "allocatable_not_supported"
        elif "callback" in error_lower or "function pointer" in error_lower:
            return "callback_not_supported"
        else:
            return "general_not_implemented"
    elif "attributeerror" in error_lower:
        return "attribute_error"
    elif "typeerror" in error_lower:
        return "type_error"
    elif "syntaxerror" in error_lower:
        return "syntax_error"
    elif "no such file" in error_lower or "not found" in error_lower:
        return "file_not_found"
    elif "undefined symbol" in error_lower:
        return "undefined_symbol"
    elif "segmentation fault" in error_lower:
        return "segmentation_fault"
    else:
        return "unknown_error"

def generate_report(results):
    """Generate markdown and JSON reports."""
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Count statistics
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")

    # Group failures by error category
    error_categories = {}
    for r in results:
        if r["status"] == "FAIL" and r["error_category"]:
            if r["error_category"] not in error_categories:
                error_categories[r["error_category"]] = []
            error_categories[r["error_category"]].append(r["name"])

    # Generate markdown report
    report_lines = [
        "# F90wrap Direct-C Compatibility Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\n## Summary",
        f"- Total Examples: {total}",
        f"- ✅ Passed: {passed} ({100*passed/total:.1f}%)",
        f"- ❌ Failed: {failed} ({100*failed/total:.1f}%)",
        f"- ⊘ Skipped: {skipped} ({100*skipped/total:.1f}%)",
        f"\n## Results by Example\n"
    ]

    # Sort results by status for better readability
    sorted_results = sorted(results, key=lambda x: (x["status"] != "PASS", x["status"] != "FAIL", x["name"]))

    for r in sorted_results:
        status_symbol = "✅" if r["status"] == "PASS" else "❌" if r["status"] == "FAIL" else "⊘"
        report_lines.append(f"### {status_symbol} {r['name']}")
        report_lines.append(f"- **Status**: {r['status']}")
        if r["error_category"]:
            report_lines.append(f"- **Error Category**: {r['error_category']}")
        if r["notes"]:
            report_lines.append("- **Notes**:")
            for note in r["notes"]:
                report_lines.append(f"  - {note}")
        if r["status"] == "FAIL" and r["f90wrap_error"]:
            report_lines.append("- **Error Output**:")
            report_lines.append("```")
            report_lines.append(r["f90wrap_error"][:1000])  # Limit error output
            if len(r["f90wrap_error"]) > 1000:
                report_lines.append("... (truncated)")
            report_lines.append("```")
        report_lines.append("")

    # Add error categorization section
    if error_categories:
        report_lines.append("\n## Error Categories\n")
        for category, examples in sorted(error_categories.items()):
            report_lines.append(f"### {category} ({len(examples)} examples)")
            for ex in sorted(examples):
                report_lines.append(f"- {ex}")
            report_lines.append("")

    # Write markdown report
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))

    # Write JSON report for programmatic access
    with open(JSON_REPORT, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "pass_rate": passed / total if total > 0 else 0
            },
            "error_categories": error_categories,
            "results": results
        }, f, indent=2)

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "error_categories": error_categories
    }

def main():
    """Main test runner."""
    print("=" * 70)
    print("F90wrap Direct-C Compatibility Testing")
    print("=" * 70)

    # Find all example directories
    example_dirs = []
    for item in EXAMPLES_DIR.iterdir():
        if item.is_dir() and item.name not in SKIP_DIRS:
            example_dirs.append(item)

    example_dirs.sort()
    print(f"\nFound {len(example_dirs)} examples to test\n")

    # Test each example
    results = []
    for i, example_dir in enumerate(example_dirs, 1):
        example_name = example_dir.name
        print(f"[{i}/{len(example_dirs)}] Testing {example_name}...", end=" ")
        sys.stdout.flush()

        try:
            result = test_example_direct_c(example_name, example_dir)
            results.append(result)

            status_symbol = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⊘"
            print(f"{status_symbol} {result['status']}")

            if result["status"] == "FAIL" and result["error_category"]:
                print(f"     Error category: {result['error_category']}")
        except Exception as e:
            print(f"❌ EXCEPTION")
            print(f"     {str(e)}")
            results.append({
                "name": example_name,
                "path": str(example_dir),
                "status": "FAIL",
                "error_category": "test_framework_error",
                "notes": [f"Exception during testing: {str(e)}", traceback.format_exc()],
                "f90wrap_output": "",
                "f90wrap_error": "",
                "test_output": "",
                "test_error": ""
            })

    # Generate report
    print("\n" + "=" * 70)
    print("Generating reports...")
    stats = generate_report(results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Examples: {stats['total']}")
    print(f"✅ Passed: {stats['passed']} ({100*stats['passed']/stats['total']:.1f}%)")
    print(f"❌ Failed: {stats['failed']} ({100*stats['failed']/stats['total']:.1f}%)")
    print(f"⊘ Skipped: {stats['skipped']} ({100*stats['skipped']/stats['total']:.1f}%)")

    if stats["error_categories"]:
        print("\nTop Error Categories:")
        for category, examples in sorted(stats["error_categories"].items(), key=lambda x: -len(x[1]))[:5]:
            print(f"  - {category}: {len(examples)} examples")

    print(f"\nDetailed report: {REPORT_FILE}")
    print(f"JSON results: {JSON_REPORT}")

    return 0 if stats["failed"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())