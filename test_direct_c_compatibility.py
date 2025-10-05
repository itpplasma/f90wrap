#!/usr/bin/env python3
"""
Test all f90wrap examples with the --direct-c flag.
Documents baseline compatibility status for each example.

V2: Improved compilation handling:
- Handles preprocessing for .F90 and files with #include
- Compiles all Fortran files together to handle module dependencies
- Better error reporting
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
    F90_files = list(example_dir.glob("*.F90"))  # Uppercase needs preprocessing
    f_files = list(example_dir.glob("*.f"))
    fpp_files = list(example_dir.glob("*.fpp"))
    return f90_files + F90_files + f_files + fpp_files

def needs_preprocessing(filepath):
    """Check if a file needs preprocessing."""
    # .F90 (uppercase) always needs preprocessing
    if str(filepath).endswith('.F90'):
        return True

    # Check for preprocessor directives in the file
    try:
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read(1000)  # Check first 1000 chars
            if '#include' in content or '#ifdef' in content or '#define' in content:
                return True
    except:
        pass

    return False

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

        # Determine which files to pass to f90wrap
        # .fpp files are already preprocessed and should be used preferentially
        fpp_files = list(tmpdir.glob("*.fpp"))
        if fpp_files:
            source_files_for_wrap = fpp_files
            result["notes"].append(f"Using preprocessed files for f90wrap: {[f.name for f in fpp_files]}")
        else:
            source_files_for_wrap = fortran_files
            result["notes"].append(f"Using source files for f90wrap: {[f.name for f in source_files_for_wrap]}")

        # Check for kind_map file
        kind_map_file = tmpdir / "kind_map"
        kind_map_arg = f"-k {kind_map_file}" if kind_map_file.exists() else ""

        # Build f90wrap command with --direct-c flag
        source_file_args = " ".join(str(f) for f in source_files_for_wrap)
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

        # Prepare files for compilation
        # We need to compile the original Fortran sources, not the .fpp files
        files_to_compile = []

        # If we used .fpp files for wrapping, we need to preprocess them for compilation
        if fpp_files:
            for fpp_file in fpp_files:
                # Use gfortran -E to preprocess .fpp to .f90
                f90_output = tmpdir / f"{fpp_file.stem}_processed.f90"
                preprocess_cmd = f"gfortran -E -cpp {fpp_file} -o {f90_output}"
                pp_result = run_command(preprocess_cmd, cwd=tmpdir)
                if pp_result["success"]:
                    files_to_compile.append(f90_output)
                    result["notes"].append(f"Preprocessed {fpp_file.name} -> {f90_output.name}")
                else:
                    result["notes"].append(f"Failed to preprocess {fpp_file.name}: {pp_result['stderr'][:200]}")
                    # Fall back to using the fpp file directly
                    files_to_compile.append(fpp_file)
        else:
            # Process each Fortran file
            for f_file in fortran_files:
                if needs_preprocessing(f_file):
                    # Preprocess files that need it
                    output_name = f"{f_file.stem}_processed.f90"
                    f90_output = tmpdir / output_name
                    preprocess_cmd = f"gfortran -E -cpp {f_file.name} -o {output_name}"
                    pp_result = run_command(preprocess_cmd, cwd=tmpdir)
                    if pp_result["success"]:
                        files_to_compile.append(f90_output)
                        result["notes"].append(f"Preprocessed {f_file.name} -> {output_name}")
                    else:
                        result["notes"].append(f"Failed to preprocess {f_file.name}: {pp_result['stderr'][:200]}")
                        files_to_compile.append(f_file)
                else:
                    files_to_compile.append(f_file)

        # Add the generated support module
        support_module = tmpdir / f"{example_name}_direct_support.f90"
        if support_module.exists():
            files_to_compile.append(support_module)
            result["notes"].append(f"Found support module: {support_module.name}")

        # Compile all Fortran files together (handles module dependencies)
        if files_to_compile:
            compile_files = " ".join(str(f.name) for f in files_to_compile)
            compile_cmd = f"gfortran -fPIC -c {compile_files}"
            result["notes"].append(f"Compiling Fortran: {compile_cmd}")
            compile_result = run_command(compile_cmd, cwd=tmpdir)

            if not compile_result["success"]:
                result["status"] = "FAIL"
                result["error_category"] = "fortran_compilation_failed"
                result["notes"].append(f"Fortran compilation failed: {compile_result['stderr'][:500]}")
                return result
            else:
                result["notes"].append("Fortran compilation successful")

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

            compile_c_cmd = f"gcc -fPIC -c {c_file.name} {include_flags} -o {c_file.stem}.o"
            result["notes"].append(f"Compiling C: {compile_c_cmd}")
            compile_c_result = run_command(compile_c_cmd, cwd=tmpdir)
            if not compile_c_result["success"]:
                result["status"] = "FAIL"
                result["error_category"] = "c_compilation_failed"
                result["notes"].append(f"Failed to compile {c_file.name}: {compile_c_result['stderr'][:500]}")
                return result

        # Link into shared library
        obj_files = list(tmpdir.glob("*.o"))
        if obj_files:
            obj_file_args = " ".join(str(f.name) for f in obj_files)
            link_cmd = f"gcc -shared {obj_file_args} -lgfortran -o _{example_name}_direct.so"
            result["notes"].append(f"Linking: {link_cmd}")
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

    # Generate Markdown report
    with open(REPORT_FILE, "w") as f:
        f.write(f"# F90wrap Direct-C Compatibility Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Total Examples:** {total}\n")
        f.write(f"- **✅ Passed:** {passed} ({passed*100/total:.1f}%)\n")
        f.write(f"- **❌ Failed:** {failed} ({failed*100/total:.1f}%)\n")
        f.write(f"- **⊘ Skipped:** {skipped} ({skipped*100/total:.1f}%)\n\n")

        if error_categories:
            f.write(f"## Error Categories\n\n")
            for category, examples in sorted(error_categories.items(), key=lambda x: -len(x[1])):
                f.write(f"### {category} ({len(examples)} examples)\n")
                for example in sorted(examples):
                    f.write(f"- {example}\n")
                f.write("\n")

        f.write(f"## Detailed Results\n\n")
        f.write("| Example | Status | Error Category | Notes |\n")
        f.write("|---------|--------|----------------|-------|\n")

        for r in sorted(results, key=lambda x: (x["status"] != "PASS", x["name"])):
            status_icon = "✅" if r["status"] == "PASS" else "❌" if r["status"] == "FAIL" else "⊘"
            notes = " ".join(r["notes"][:2]) if r["notes"] else ""
            if len(notes) > 100:
                notes = notes[:97] + "..."
            f.write(f"| {r['name']} | {status_icon} {r['status']} | {r['error_category'] or 'N/A'} | {notes} |\n")

    # Generate JSON report
    report_data = {
        "generated": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": f"{passed*100/total:.1f}%"
        },
        "error_categories": error_categories,
        "results": results
    }

    with open(JSON_REPORT, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Total Examples: {total}")
    print(f"✅ Passed: {passed} ({passed*100/total:.1f}%)")
    print(f"❌ Failed: {failed} ({failed*100/total:.1f}%)")
    print(f"⊘ Skipped: {skipped} ({skipped*100/total:.1f}%)")

    if error_categories:
        print(f"\nTop Error Categories:")
        for category, examples in sorted(error_categories.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"  - {category}: {len(examples)} examples")

    print(f"\nDetailed report: {REPORT_FILE}")
    print(f"JSON results: {JSON_REPORT}")

def main():
    """Main test runner."""
    print("="*70)
    print("F90wrap Direct-C Compatibility Testing")
    print("="*70)

    # Find all example directories
    examples = []
    for item in EXAMPLES_DIR.iterdir():
        if item.is_dir() and item.name not in SKIP_DIRS:
            examples.append((item.name, item))

    examples.sort()
    print(f"\nFound {len(examples)} examples to test\n")

    # Test each example
    results = []
    for i, (name, path) in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] Testing {name}...", end=" ")
        sys.stdout.flush()

        try:
            result = test_example_direct_c(name, path)
            status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⊘"
            print(f"{status_icon} {result['status']}")
            if result["status"] == "FAIL":
                print(f"     Error category: {result['error_category']}")
        except Exception as e:
            result = {
                "name": name,
                "path": str(path),
                "status": "FAIL",
                "error_category": "test_framework_error",
                "notes": [f"Test framework error: {str(e)}"],
                "f90wrap_output": "",
                "f90wrap_error": traceback.format_exc(),
                "test_output": "",
                "test_error": ""
            }
            print(f"❌ ERROR: {str(e)}")

        results.append(result)

    # Generate reports
    print(f"\n{'='*70}")
    print("Generating reports...")
    generate_report(results)

    # Return non-zero if any tests failed
    failed_count = sum(1 for r in results if r["status"] == "FAIL")
    return 1 if failed_count > 0 else 0

if __name__ == "__main__":
    sys.exit(main())