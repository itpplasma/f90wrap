# Direct-C Clean Execution Plan (Detailed)

## Goals
- Direct-C backend matches f2py behaviour for every supported example.
- Generated file names and runtime contract identical to classic f90wrap.
- All direct-C specific logic isolated under `f90wrap/directc.py` and associated helpers.

## Branch Strategy
- Active branch: `feature/direct-c-clean` (branched from `origin/master`).
- Reference branch: `feature/direct-c-generation` (read-only; cherry-pick ideas such as NumPy dtype handling and unit tests).

## Implementation Steps

### 1. Routine Classification (`f90wrap/directc.py`)
- Build on existing analyser (`analyse_interop`).
- Extend `_argument_is_iso_c` to recognise explicit-shape arrays, `value` arguments, and simple characters with explicit length.
- Add helpers:
  ```python
  def is_iso_c_procedure(proc: ft.Procedure) -> bool:
      ...  # reuse `_procedure_requires_helper`
  def needs_helper(proc): return not is_iso_c_procedure(proc)
  ```
- Provide public API:
  ```python
  def classify(tree, kind_map): -> Dict[ProcedureKey, InteropInfo]
  ```
  and a utility `bind_c_symbol(prefix, key)` that standardises shim names.

### 2. Fortran Generator Updates (`f90wrap/f90wrapgen.py`)
- Accept the classification map (`direct_c_interop`).
- In `visit_Procedure`:
  - Compute shim name via `directc.bind_c_symbol`.
  - If `requires_helper` is `False`:
    * Do **not** emit the legacy `f90wrap_*` subroutine.
    * Emit a `bind(C, name='<shim>')` subroutine that:
      ```fortran
      use iso_c_binding, only: c_int, c_double, c_ptr, ...
      implicit none
      ! direct argument list mirrors ISO C types
      call <original> (...)
      end subroutine
      ```
  - If `requires_helper` is `True`:
    * Emit the existing helper unchanged.
    * After the helper, emit the `bind(C)` shim that simply calls the helper, using the same handle buffer layout.
- Ensure shim emission reuses existing utilities (`write_uses_lines`, `write_arg_decl_lines`) where possible.
- Keep files named `f90wrap_<module>.f90`—no `_direct.f90`.

### 3. C Generator (`f90wrap/directc_cgen.py` new module)
- Introduce a new generator module to retain modularity.
- Inputs: parse tree + classification map.
- For each exported symbol:
  ```c
  static PyObject* wrapper(PyObject *self, PyObject *args) {
      /* marshal Python args */
      /* call shim: bind(C) symbol */
      /* translate results back */
  }
  ```
- Use NumPy conversions already developed in `feature/direct-c-generation` (copy the relevant `numpy_capi.py` improvements into a new helper module if needed).
- Update `setup`/meson to build `_module_direct.c` from the new generator output.

### 4. CLI Integration (`f90wrap/scripts/main.py`)
- Detect `--direct-c` flag, run classification (`directc.classify`) and pass result to Fortran / C generators.
- Skip f2py invocation; instead, write generated C code and optional build stubs.

### 5. Tests & Evidence
- Bring over the direct-C unit tests from the reference branch (adapt to new shim paths).
- After each major change run:
  - `pytest` (unit tests).
  - `python test_direct_c_compatibility.py` to capture example pass/fail stats.
- Update `direct_c_test_results/compatibility_report.md` and `.json` only when a run is green.

### 6. Cleanup & Documentation
- Once all examples pass, remove unused capsule helpers from the tree.
- Update CLI help / CHANGELOG only.

## Copy Guidance from `feature/direct-c-generation`
- NumPy conversion routines (`PyArray_FROM_OTF` usage) located in `f90wrap/numpy_capi.py` (lines ~150–230).
- Improved name mangling tests in `test/test_cwrapgen.py`.
- Compatibility script `test_direct_c_compatibility.py` (for reuse, not committed).

## Rollback Strategy
- Every logical change committed separately on `feature/direct-c-clean`; revert specific commits if regressions occur.
- If work stalls, discard branch; `origin/master` untouched.

## Definition of Done
- `python test_direct_c_compatibility.py`: all examples that pass under f2py also pass under direct-C.
- `pytest`: green.
- No generated artefacts in `examples/`.
- Documentation limited to CLI help and a short CHANGELOG entry.
