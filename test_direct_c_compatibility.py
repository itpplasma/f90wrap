#!/usr/bin/env python3
"""Direct-C compatibility sweep for f90wrap examples.

Adapts the reference harness from feature/direct-c-generation to work from any
repository checkout and to prefer helper-based Direct-C code paths.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = REPO_ROOT / "examples"
RESULTS_DIR = REPO_ROOT / "direct_c_test_results"
REPORT_FILE = RESULTS_DIR / "compatibility_report.md"
JSON_REPORT = RESULTS_DIR / "compatibility_results.json"
FORTRAN_DIAGNOSTICS = RESULTS_DIR / "fortran_failures.md"

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}
INTRINSIC_MODULES = {
    "iso_fortran_env",
    "iso_c_binding",
    "ieee_arithmetic",
    "ieee_exceptions",
    "ieee_features",
}

CommandResult = Dict[str, object]


def sanitize_module_name(name: str) -> str:
    """Return a valid Python identifier derived from *name*."""

    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def direct_module_name(example_name: str) -> str:
    """Return the sanitized direct-C module name for the example."""

    return f"{sanitize_module_name(example_name)}_direct"


def direct_extension_filename(example_name: str) -> str:
    """Return the shared library filename for the direct-C module."""

    return f"_{direct_module_name(example_name)}.so"


def run_command(command: str, cwd: Optional[Path] = None, timeout: int = 60) -> CommandResult:
    """Run a shell command and capture stdout/stderr."""

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "success": completed.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "success": False,
        }
    except Exception as exc:  # pragma: no cover - subprocess failure path
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "success": False,
        }


def find_fortran_files(example_dir: Path) -> List[Path]:
    """Return all Fortran sources in the example directory."""

    patterns = ("*.f90", "*.F90", "*.f", "*.fpp")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(example_dir.glob(pattern)))
    return files


def needs_preprocessing(path: Path) -> bool:
    """Return True if the file should be preprocessed before compilation."""

    if path.suffix.upper() == ".F90":
        return True

    try:
        snippet = path.read_text(errors="ignore")[:2000]
    except OSError:
        return False

    return any(token in snippet for token in ("#include", "#ifdef", "#define"))


def strip_fortran_comments(lines: Iterable[str]) -> List[str]:
    """Remove traditional Fortran comment lines and inline ! comments."""

    cleaned: List[str] = []
    for line in lines:
        body = line.split("!", 1)[0]
        if not body:
            cleaned.append("")
            continue
        stripped = body.lstrip()
        if stripped and stripped[0] in {"c", "C", "*"}:
            cleaned.append("")
        else:
            cleaned.append(body)
    return cleaned


def extract_module_info(path: Path) -> Tuple[Optional[str], List[str]]:
    """Return (module_defined, modules_used) for a Fortran source file."""

    try:
        lines = strip_fortran_comments(path.read_text(errors="ignore").splitlines())
        content = "\n".join(lines)
    except OSError:
        return None, []

    module_name: Optional[str] = None
    module_pattern = re.compile(r"^\s*module\s+(\w+)\s*$", re.IGNORECASE | re.MULTILINE)
    for match in module_pattern.finditer(content):
        candidate = match.group(1).lower()
        if candidate not in INTRINSIC_MODULES:
            module_name = candidate
            break

    use_pattern = re.compile(r"^\s*use\s*(?:,\s*\w+\s*)?\s*(?:::)?\s*(\w+)", re.IGNORECASE | re.MULTILINE)
    dependencies: List[str] = []
    for match in use_pattern.finditer(content):
        dep = match.group(1).lower()
        if dep not in INTRINSIC_MODULES and dep not in dependencies:
            dependencies.append(dep)

    return module_name, dependencies


def topo_sort_fortran(files: List[Path]) -> List[Path]:
    """Perform a dependency-aware ordering of Fortran compilation units."""

    info: Dict[Path, Tuple[Optional[str], List[str]]] = {
        path: extract_module_info(path) for path in files
    }
    module_to_file: Dict[str, Path] = {
        module: path for path, (module, _) in info.items() if module
    }

    graph: Dict[Path, Set[Path]] = {path: set() for path in files}
    indegree: Dict[Path, int] = {path: 0 for path in files}

    for path, (_, deps) in info.items():
        for dep in deps:
            provider = module_to_file.get(dep)
            if provider and provider != path and provider not in graph[path]:
                graph[path].add(provider)
                indegree[path] += 1

    queue = sorted([path for path, degree in indegree.items() if degree == 0], key=lambda p: p.name)
    ordered: List[Path] = []

    while queue:
        current = queue.pop(0)
        ordered.append(current)
        for other in files:
            if current in graph[other]:
                indegree[other] -= 1
                if indegree[other] == 0 and other not in queue:
                    queue.append(other)
        queue.sort(key=lambda p: p.name)

    if len(ordered) != len(files):
        no_deps = [f for f, (_, deps) in info.items() if not deps]
        with_deps = [f for f, (_, deps) in info.items() if deps]
        ordered = sorted(no_deps, key=lambda p: p.name) + sorted(with_deps, key=lambda p: p.name)

    return ordered


def preprocess_file(path: Path, cwd: Path) -> Path:
    """Preprocess a Fortran source file, returning the new path."""

    output = cwd / f"{path.stem}_pp.f90"
    command = f"gfortran -E -cpp {path.name} | grep -v '^#' > {output.name}"
    result = run_command(command, cwd=cwd)
    if result["success"]:
        return output
    return path


def python_include_flags() -> str:
    """Return include flags for Python, NumPy, and f90wrap headers."""

    python_include = run_command("python3 -c 'import sysconfig; print(sysconfig.get_path(\"include\"))'")
    numpy_include = run_command("python3 -c 'import numpy; print(numpy.get_include())'")
    flags = []
    if python_include["success"]:
        include = python_include["stdout"].strip()
        if include:
            flags.append(f"-I{include}")
    if numpy_include["success"]:
        include = numpy_include["stdout"].strip()
        if include:
            flags.append(f"-I{include}")
    flags.append(f"-I{(REPO_ROOT / 'f90wrap').resolve()}")
    return " ".join(flags)


def categorize_error(message: str) -> str:
    """Map error output to a coarse failure category."""

    text = message.lower()
    for needle, category in (
        ("notimplementederror", "not_implemented"),
        ("attributeerror", "attribute_error"),
        ("typeerror", "type_error"),
        ("syntaxerror", "syntax_error"),
        ("undefined symbol", "undefined_symbol"),
        ("no such file", "file_not_found"),
        ("not found", "file_not_found"),
        ("segmentation fault", "segmentation_fault"),
    ):
        if needle in text:
            return category
    return "unknown_error"


def copy_example(source: Path, destination: Path) -> None:
    """Copy an example directory into the working directory."""

    for entry in source.iterdir():
        target = destination / entry.name
        if entry.is_dir():
            if entry.name in SKIP_DIRS:
                continue
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)


def compile_fortran_sources(files: List[Path], cwd: Path, notes: List[str]) -> bool:
    """Compile Fortran sources in dependency order."""

    if not files:
        return True

    ordered = topo_sort_fortran(files)
    notes.append(f"Compilation order: {[path.name for path in ordered]}")
    command = "gfortran -fPIC -c " + " ".join(path.name for path in ordered)
    result = run_command(command, cwd=cwd)
    if not result["success"]:
        snippet = result["stderr"][:500]
        notes.append(f"Fortran compilation failed: {snippet}")
        deps = []
        for path in files:
            mod, used = extract_module_info(path)
            if mod or used:
                description = []
                if mod:
                    description.append(f"defines {mod}")
                if used:
                    description.append(f"uses {used}")
                deps.append(f"{path.name}: {', '.join(description)}")
        if deps:
            notes.append("Module dependency hints: " + "; ".join(deps))
        return False

    notes.append("Fortran compilation succeeded")
    return True


def compile_c_sources(c_files: List[Path], cwd: Path, notes: List[str]) -> bool:
    """Compile generated C wrappers."""

    if not c_files:
        notes.append("No C sources generated")
        return False

    include_flags = python_include_flags()
    for c_file in c_files:
        command = f"gcc -fPIC -c {c_file.name} {include_flags} -o {c_file.stem}.o"
        result = run_command(command, cwd=cwd)
        if not result["success"]:
            notes.append(f"Failed to compile {c_file.name}: {result['stderr'][:500]}")
            return False
    notes.append("C compilation succeeded")
    return True


def link_shared_library(objects: List[Path], output_name: str, cwd: Path, notes: List[str]) -> bool:
    """Link object files into a Python extension."""

    if not objects:
        notes.append("No object files for linking")
        return False

    command = "gcc -shared " + " ".join(obj.name for obj in objects) + f" -lgfortran -o {output_name}"
    result = run_command(command, cwd=cwd)
    if not result["success"]:
        notes.append(f"Linking failed: {result['stderr'][:500]}")
        return False

    notes.append("Linking succeeded")
    return True


def alias_extension_modules(c_files: List[Path], extension_name: str, cwd: Path, notes: List[str]) -> None:
    """Duplicate the linked extension so each generated C module has a matching filename."""

    if not c_files:
        return

    source = cwd / extension_name
    if not source.exists():
        notes.append(f"Linked artifact {extension_name} missing; cannot alias")
        return

    suffix_result = run_command("python3-config --extension-suffix", cwd=cwd)
    if not suffix_result["success"]:
        notes.append("Failed to determine extension suffix for aliasing")
        return

    ext_suffix = suffix_result["stdout"].strip()
    if not ext_suffix:
        notes.append("Empty extension suffix; skipping alias creation")
        return

    for c_file in c_files:
        target = cwd / f"{c_file.stem}{ext_suffix}"
        if target.name == extension_name:
            continue
        try:
            shutil.copy2(source, target)
            notes.append(f"Alias created: {target.name}")
        except OSError as exc:
            notes.append(f"Failed to alias {target.name}: {exc}")



def modify_tests_py(
        example_name: str,
        direct_mod: str,
        tests_file: Path,
        notes: List[str],
    ) -> None:
        """Rewrite imports in tests.py to target the direct extension."""

        try:
            content = tests_file.read_text()
        except OSError as exc:
            notes.append(f"Could not read tests.py: {exc}")
            return

        extension_mod = direct_mod
        skip_modules = {"numpy", "unittest", "sys"}
        dotted_helpers_defined = False
        helper_preamble: List[str] = []
        new_lines: List[str] = []

        def append_callback(prefix: str, alias_name: str) -> None:
            new_lines.append(f"{prefix}if hasattr({alias_name}, '_cback'):")
            new_lines.append(f"{prefix}    class _F90WrapCallbackProxy:")
            new_lines.append(f"{prefix}        def __init__(self, module):")
            new_lines.append(f"{prefix}            self._module = module")
            new_lines.append(f"{prefix}            self._backend = module._cback")
            new_lines.append(f"{prefix}        def __getattr__(self, name):")
            new_lines.append(f"{prefix}            if hasattr(self._module, name):")
            new_lines.append(f"{prefix}                return getattr(self._module, name)")
            new_lines.append(f"{prefix}            return getattr(self._backend, name)")
            new_lines.append(f"{prefix}        def __setattr__(self, name, value):")
            new_lines.append(f"{prefix}            if name in ('_module', '_backend'):")
            new_lines.append(f"{prefix}                object.__setattr__(self, name, value)")
            new_lines.append(f"{prefix}                return")
            new_lines.append(f"{prefix}            setattr(self._module, name, value)")
            new_lines.append(f"{prefix}            try:")
            new_lines.append(f"{prefix}                setattr(self._backend, name, value)")
            new_lines.append(f"{prefix}            except AttributeError:")
            new_lines.append(f"{prefix}                pass")
            new_lines.append(f"{prefix}        def __dir__(self):")
            new_lines.append(f"{prefix}            return sorted(set(dir(self._module)) | set(dir(self._backend)))")
            new_lines.append(f"{prefix}    {alias_name}._CBF = _F90WrapCallbackProxy({alias_name})")
            new_lines.append(f"{prefix}elif not hasattr({alias_name}, '_CBF'):")
            new_lines.append(f"{prefix}    {alias_name}._CBF = {alias_name}")

        def ensure_dotted_helpers() -> None:
            nonlocal dotted_helpers_defined
            if dotted_helpers_defined:
                return
            dotted_helpers_defined = True
            helper_preamble.append("import sys as _f90wrap_sys")
            helper_preamble.append("import types as _f90wrap_types")
            helper_preamble.append("")
            helper_preamble.append("def _f90wrap_bind_dotted(name: str, module) -> None:")
            helper_preamble.append("    parts = name.split('.')")
            helper_preamble.append("    parent = None")
            helper_preamble.append("    prefix = []")
            helper_preamble.append("    for index, part in enumerate(parts):")
            helper_preamble.append("        prefix.append(part)")
            helper_preamble.append("        qual = '.'.join(prefix)")
            helper_preamble.append("        if index == len(parts) - 1:")
            helper_preamble.append("            if parent is None:")
            helper_preamble.append("                globals()[part] = module")
            helper_preamble.append("            else:")
            helper_preamble.append("                setattr(parent, part, module)")
            helper_preamble.append("            _f90wrap_sys.modules[qual] = module")
            helper_preamble.append("        else:")
            helper_preamble.append("            pkg = _f90wrap_sys.modules.get(qual)")
            helper_preamble.append("            if pkg is None:")
            helper_preamble.append("                pkg = _f90wrap_types.ModuleType(part)")
            helper_preamble.append("                _f90wrap_sys.modules[qual] = pkg")
            helper_preamble.append("                if parent is None:")
            helper_preamble.append("                    globals()[part] = pkg")
            helper_preamble.append("                else:")
            helper_preamble.append("                    setattr(parent, part, pkg)")
            helper_preamble.append("            parent = pkg")
            helper_preamble.append("")

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("from __future__"):
                new_lines.append(line)
                continue

            prefix = line[: len(line) - len(line.lstrip())]

            if stripped.startswith("import "):
                parts = stripped.split()
                if len(parts) >= 2:
                    original = parts[1]
                    module = original.split(",")[0]
                    if module not in skip_modules:
                        alias = module
                        if " as " in stripped:
                            alias = stripped.split(" as ", 1)[1].split()[0]

                        if "." in module:
                            ensure_dotted_helpers()
                            package, leaf = module.rsplit(".", 1)
                            direct_alias = f"_{sanitize_module_name(module)}_direct"
                            target_var = f"{direct_alias}_{sanitize_module_name(leaf)}"
                            new_lines.append(f"{prefix}import {extension_mod} as {direct_alias}")
                            new_lines.append(
                                f"{prefix}{target_var} = getattr({direct_alias}, '{leaf}', {direct_alias})"
                            )
                            append_callback(prefix, target_var)
                            new_lines.append(f"{prefix}_f90wrap_bind_dotted('{module}', {target_var})")
                            if alias != module:
                                new_lines.append(f"{prefix}{alias} = {target_var}")
                            continue

                        sanitized_alias = alias
                        new_lines.append(f"{prefix}import {extension_mod} as {sanitized_alias}")
                        append_callback(prefix, sanitized_alias)
                        continue

            if stripped.startswith("from "):
                parts = stripped.split()
                if len(parts) >= 2:
                    module = parts[1]
                    if module not in skip_modules:
                        new_lines.append(line.replace(f"from {module}", f"from {extension_mod}"))
                        continue

            new_lines.append(line)

        if helper_preamble:
            insert_at = 0
            in_docstring = False
            doc_delim = ""
            while insert_at < len(new_lines):
                stripped = new_lines[insert_at].strip()
                if in_docstring:
                    insert_at += 1
                    if stripped.endswith(doc_delim):
                        in_docstring = False
                    continue
                if not stripped or stripped.startswith("#"):
                    insert_at += 1
                    continue
                if stripped.startswith(('"""', "'''")):
                    doc_delim = stripped[:3]
                    in_docstring = not stripped.endswith(doc_delim) or len(stripped) == 3
                    insert_at += 1
                    continue
                if stripped.startswith("from __future__"):
                    insert_at += 1
                    continue
                break
            new_lines = new_lines[:insert_at] + helper_preamble + new_lines[insert_at:]

        rewritten = "\n".join(new_lines)
        if content.endswith("\n"):
            rewritten += "\n"

        if rewritten != content:
            tests_file.write_text(rewritten)
            notes.append("tests.py rewritten for direct module")

def gather_objects(cwd: Path, sources: Iterable[Path]) -> List[Path]:
    """Collect object files that correspond to the provided sources."""

    objects: List[Path] = []
    for source in sources:
        candidate = cwd / f"{source.stem}.o"
        if candidate.exists():
            objects.append(candidate)
    return objects


def test_example(example_dir: Path) -> Dict[str, object]:
    """Run direct-C workflow for a single example directory."""

    example_name = example_dir.name
    outcome: Dict[str, object] = {
        "name": example_name,
        "path": str(example_dir),
        "status": "SKIP",
        "error_category": None,
        "notes": [],
        "f90wrap_output": "",
        "f90wrap_error": "",
        "test_output": "",
        "test_error": "",
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        workdir = Path(temp_dir)
        try:
            copy_example(example_dir, workdir)
        except Exception as exc:
            outcome["notes"].append(f"Failed to copy files: {exc}")
            return outcome

        fortran_files = find_fortran_files(workdir)
        if not fortran_files:
            outcome["notes"].append("No Fortran sources; skipping")
            return outcome

        fpp_files = list(workdir.glob("*.fpp"))
        if fpp_files:
            wrap_sources = fpp_files
        else:
            wrap_sources = fortran_files
        outcome["notes"].append(f"f90wrap inputs: {[path.name for path in wrap_sources]}")
        wrapper_input_stems = {
            path.stem.lower()
            for path in wrap_sources
            if path.stem.lower().startswith("f90wrap_")
        }
        callback_names: List[str] = []
        callback_pattern = re.compile(r"intent\(callback[^)]*\)\s+([A-Za-z0-9_]+)", re.IGNORECASE)
        for source in wrap_sources:
            try:
                text = source.read_text(errors="ignore")
            except OSError:
                continue
            for match in callback_pattern.finditer(text):
                callback_names.append(match.group(1))
        callback_names = sorted({name for name in callback_names if name})
        if callback_names:
            outcome["notes"].append(f"Callbacks: {callback_names}")

        kind_map = None
        for candidate in ("kind_map", "kind.map"):
            path = workdir / candidate
            if path.exists():
                kind_map = path
                break
        direct_mod = direct_module_name(example_name)
        cmd_parts: List[str] = [
            sys.executable,
            "-m",
            "f90wrap.scripts.main",
            "--direct-c",
            "-m",
            direct_mod,
        ]
        if callback_names:
            cmd_parts.append("--callback")
            cmd_parts.extend(callback_names)
        if kind_map is not None:
            cmd_parts.extend(["-k", kind_map.name])
        cmd_parts.append("--")
        cmd_parts.extend(path.name for path in wrap_sources)

        wrap_cmd = " ".join(shlex.quote(part) for part in cmd_parts)
        wrap_result = run_command(wrap_cmd, cwd=workdir, timeout=60)
        outcome["f90wrap_output"] = wrap_result["stdout"]
        outcome["f90wrap_error"] = wrap_result["stderr"]
        if not wrap_result["success"]:
            outcome["status"] = "FAIL"
            outcome["error_category"] = categorize_error(wrap_result["stderr"])
            outcome["notes"].append(f"f90wrap failed (rc={wrap_result['returncode']})")
            return outcome

        c_files = sorted(workdir.glob("*.c"))
        if not c_files:
            outcome["status"] = "FAIL"
            outcome["error_category"] = "no_c_output"
            outcome["notes"].append("No Direct-C source generated")
            return outcome
        outcome["notes"].append(f"Generated C: {[path.name for path in c_files]}")

        compilation_units: List[Path] = []
        if fpp_files:
            for source in fpp_files:
                compilation_units.append(preprocess_file(source, workdir))
        else:
            for source in fortran_files:
                compilation_units.append(preprocess_file(source, workdir) if needs_preprocessing(source) else source)

        support_module = workdir / f"{direct_mod}_support.f90"
        if not support_module.exists():
            legacy_support = workdir / f"{example_name}_direct_support.f90"
            if legacy_support.exists():
                support_module = legacy_support
        if support_module.exists():
            compilation_units.append(support_module)

        wrapper_sources = []
        for source in sorted(workdir.glob("f90wrap_*.f90")):
            if source.stem.lower() in wrapper_input_stems:
                outcome["notes"].append(f"Skipping duplicate wrapper {source.name}")
                continue
            wrapper_sources.append(source)

        if wrapper_sources:
            compilation_units.extend(wrapper_sources)
            outcome["notes"].append(f"Wrapper sources: {[path.name for path in wrapper_sources]}")

        # Deduplicate while preserving order
        unique_units: List[Path] = []
        seen_units: Set[Path] = set()
        for unit in compilation_units:
            if unit not in seen_units:
                unique_units.append(unit)
                seen_units.add(unit)

        if not compile_fortran_sources(unique_units, workdir, outcome["notes"]):
            outcome["status"] = "FAIL"
            outcome["error_category"] = "fortran_compilation_failed"
            return outcome

        if not compile_c_sources(c_files, workdir, outcome["notes"]):
            outcome["status"] = "FAIL"
            outcome["error_category"] = "c_compilation_failed"
            return outcome

        objects: List[Path] = []
        objects.extend(gather_objects(workdir, unique_units))
        objects.extend(gather_objects(workdir, c_files))
        objects = list(dict.fromkeys(objects))  # preserve order, drop dups

        extension_name = direct_extension_filename(example_name)
        if not link_shared_library(objects, extension_name, workdir, outcome["notes"]):
            outcome["status"] = "FAIL"
            outcome["error_category"] = "linking_failed"
            return outcome

        alias_extension_modules(c_files, extension_name, workdir, outcome["notes"])

        tests_py = workdir / "tests.py"
        if tests_py.exists():
            modify_tests_py(example_name, direct_mod, tests_py, outcome["notes"])
            test_result = run_command("python3 tests.py", cwd=workdir, timeout=60)
            outcome["test_output"] = test_result["stdout"]
            outcome["test_error"] = test_result["stderr"]
            if test_result["success"]:
                outcome["status"] = "PASS"
                outcome["notes"].append("tests.py passed")
            else:
                outcome["status"] = "FAIL"
                outcome["error_category"] = categorize_error(test_result["stderr"])
                outcome["notes"].append(f"tests.py failed (rc={test_result['returncode']})")
        else:
            outcome["status"] = "PASS"
            outcome["notes"].append("Compilation succeeded; no tests.py present")

    return outcome


def _summarise_blob(blob: str, limit: int = 320) -> str:
    """Return a trimmed single-line summary for stderr/stdout snippets."""

    if not blob:
        return ""
    text = " ".join(blob.strip().splitlines())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text

def write_fortran_diagnostics(results: List[Dict[str, object]]) -> None:
    """Emit a markdown summary of Fortran compilation failures."""

    failures = [
        record for record in results if record.get("error_category") == "fortran_compilation_failed"
    ]

    if not failures:
        if FORTRAN_DIAGNOSTICS.exists():
            FORTRAN_DIAGNOSTICS.unlink()
        return

    lines: List[str] = [
        "# Fortran Compilation Diagnostics",
        "",
        "Automatically captured compiler output for parity planning.",
        "",
    ]

    for record in failures:
        lines.append(f"## {record['name']}")
        lines.append("")
        notes = record.get("notes", [])
        snippet = next((note for note in notes if note.startswith("Fortran compilation failed")), "")
        if snippet:
            lines.append("### Compiler Output")
            lines.append("")
            lines.append("````text")
            lines.append(snippet.replace("Fortran compilation failed: ", ""))
            lines.append("````")
            lines.append("")
        deps = next((note for note in notes if note.startswith("Module dependency hints")), "")
        if deps:
            lines.append("### Module Dependencies")
            lines.append("")
            lines.append(deps.replace("Module dependency hints: ", ""))
            lines.append("")
    FORTRAN_DIAGNOSTICS.write_text("\n".join(lines) + "\n")


def generate_report(results: List[Dict[str, object]]) -> None:
    """Persist markdown and JSON summaries in RESULTS_DIR."""

    RESULTS_DIR.mkdir(exist_ok=True)
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = total - passed - failed

    pass_rate = (passed / total * 100.0) if total else 0.0
    summary_lines = [
        "# F90wrap Direct-C Compatibility Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Total Examples:** {total}",
        f"- **✅ Passed:** {passed} ({pass_rate:.1f}%)",
        f"- **❌ Failed:** {failed} ({(failed / total * 100.0) if total else 0.0:.1f}%)",
        f"- **⊘ Skipped:** {skipped} ({(skipped / total * 100.0) if total else 0.0:.1f}%)",
        "",
    ]

    failure_groups: Dict[str, List[str]] = {}
    for record in results:
        if record["status"] == "FAIL" and record["error_category"]:
            failure_groups.setdefault(record["error_category"], []).append(record["name"])

    if failure_groups:
        summary_lines.append("## Error Categories")
        summary_lines.append("")
        for category, names in sorted(failure_groups.items(), key=lambda item: -len(item[1])):
            summary_lines.append(f"### {category} ({len(names)} examples)")
            for name in sorted(names):
                summary_lines.append(f"- {name}")
            summary_lines.append("")

    summary_lines.append("## Detailed Results")
    summary_lines.append("")
    summary_lines.append("| Example | Status | Category | Note |")
    summary_lines.append("|---------|--------|----------|------|")

    for record in sorted(results, key=lambda r: (r["status"] != "PASS", r["name"])):
        icon = "✅" if record["status"] == "PASS" else ("❌" if record["status"] == "FAIL" else "⊘")
        category = record["error_category"] or "N/A"
        note = " ".join(record["notes"][:2]) if record["notes"] else ""
        if len(note) > 100:
            note = note[:97] + "..."
        summary_lines.append(f"| {record['name']} | {icon} {record['status']} | {category} | {note} |")

    REPORT_FILE.write_text("\n".join(summary_lines) + "\n")

    top_failures: Dict[str, List[Dict[str, str]]] = {}
    for category, names in failure_groups.items():
        bucket: List[Dict[str, str]] = []
        for record in results:
            if record["name"] in names:
                snippet_err = record.get("test_error", "") or record.get("f90wrap_error", "")
                snippet_out = record.get("test_output", "") or record.get("f90wrap_output", "")
                snippet_err = _summarise_blob(snippet_err)
                snippet_out = _summarise_blob(snippet_out)
                bucket.append({
                    "name": record["name"],
                    "stderr": snippet_err,
                    "stdout": snippet_out,
                })
        top_failures[category] = bucket

    data = {
        "generated": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": f"{pass_rate:.1f}%",
        },
        "results": results,
        "error_categories": failure_groups,
        "failure_summaries": top_failures,
    }
    JSON_REPORT.write_text(json.dumps(data, indent=2) + "\n")
    write_fortran_diagnostics(results)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Examples: {total}")
    print(f"✅ Passed: {passed} ({pass_rate:.1f}%)")
    print(f"❌ Failed: {failed} ({(failed / total * 100.0) if total else 0.0:.1f}%)")
    print(f"⊘ Skipped: {skipped} ({(skipped / total * 100.0) if total else 0.0:.1f}%)")
    if failure_groups:
        print("\nTop error categories:")
        for category, names in sorted(failure_groups.items(), key=lambda item: -len(item[1]))[:5]:
            print(f"  - {category}: {len(names)}")
    print(f"\nReport: {REPORT_FILE}")
    print(f"JSON:   {JSON_REPORT}")


def discover_examples() -> List[Path]:
    """Return sorted list of example directories."""

    if not EXAMPLES_DIR.exists():
        print(f"No examples directory found at {EXAMPLES_DIR}", file=sys.stderr)
        return []
    entries = [path for path in EXAMPLES_DIR.iterdir() if path.is_dir() and path.name not in SKIP_DIRS]
    return sorted(entries, key=lambda p: p.name)


def main() -> int:
    print("=" * 70)
    print("F90wrap Direct-C Compatibility Testing")
    print("=" * 70)

    examples = discover_examples()
    print(f"Found {len(examples)} examples\n")
    results: List[Dict[str, object]] = []
    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] {example.name}...", end=" ")
        sys.stdout.flush()
        try:
            record = test_example(example)
            icon = "✅" if record["status"] == "PASS" else ("❌" if record["status"] == "FAIL" else "⊘")
            print(f"{icon} {record['status']}")
            if record["status"] == "FAIL" and record["error_category"]:
                print(f"    category: {record['error_category']}")
        except Exception as exc:  # pragma: no cover - framework error
            record = {
                "name": example.name,
                "path": str(example),
                "status": "FAIL",
                "error_category": "framework_error",
                "notes": [f"Framework exception: {exc}"],
                "f90wrap_output": "",
                "f90wrap_error": traceback.format_exc(),
                "test_output": "",
                "test_error": "",
            }
            print(f"❌ ERROR: {exc}")
        results.append(record)

    print("\nGenerating reports...")
    generate_report(results)

    failures = sum(1 for record in results if record["status"] == "FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
