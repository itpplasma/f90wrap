# Direct-C Clean Execution Plan

## Goals
- Direct-C backend matches the classic f2py-backed workflow for every supported example.
- Generated filenames and Python module structure are **identical** to the standard flow (no `_direct.py`, no `_direct.f90`).
- All direct-C specific logic lives in new helper modules (`f90wrap/directc.py`, `f90wrap/directc_cgen.py`); existing core files stay untouched whenever possible.

## Branch Strategy
- Active branch: `feature/direct-c-clean` (from `origin/master`).
- Reference branch: `feature/direct-c-generation` (read-only for reusable snippets).

## Implementation Steps

### 1. Classification (done)
- Use `f90wrap/directc.py` to mark each procedure with `requires_helper`. Helpers are always emitted; classification informs the C generator about marshalling requirements.

### 2. Fortran helpers (unchanged)
- Keep `F90WrapperGenerator` exactly as it is today. The canonical `f90wrap_<module>.f90` files remain the single Fortran surface the C layer will call.

### 3. Direct-C C generator (`f90wrap/directc_cgen.py`)
- Implement a new generator that emits `_module.c`:
  ```c
  static PyObject* wrap_foo(PyObject *self, PyObject *args) {
      /* Parse Python inputs (NumPy arrays, ints, strings) */
      /* Call existing helper: f90wrap_module__foo(...) */
      /* Convert results back to Python objects */
  }
  ```
- Reuse NumPy conversion utilities (copy from `feature/direct-c-generation`’s `numpy_capi.py`).
- Export the same symbols the Python wrappers already expect (`f90wrap_module__foo`, `f90wrap_type__bar`, ...).

### 4. CLI updates (`f90wrap/scripts/main.py`)
- When `--direct-c` is specified:
  - Generate Fortran helpers and Python wrappers as usual.
  - Invoke `directc_cgen.py` to write the C extension instead of f2py.
  - Emit a brief message telling the user to compile `_module.c` with their toolchain.
- Normal mode (`--direct-c` absent) remains unchanged.

### 5. Tests & Evidence
- Port the NumPy coercion tests and name-mangling checks from the old branch.
- After each major change run `pytest` **and** `python test_direct_c_compatibility.py`.
- Update `direct_c_test_results/compatibility_report.md` and `.json` only once the suite is green.

### 6. Cleanup & Docs
- Remove the capsule helpers and unused artifacts after the new C generator is proven.
- Update CLI help / CHANGELOG briefly; no other docs.

## Reusable Pieces from `feature/direct-c-generation`
- NumPy coercion code (`PyArray_FROM_OTF`, dtype enforcement) in `f90wrap/numpy_capi.py`.
- Direct-C unit tests (`test/test_cwrapgen.py`).
- Compatibility script (`test_direct_c_compatibility.py`).

## Rollback Strategy
- Commit each logical step separately on `feature/direct-c-clean`; use `git revert` if regressions appear.
- Drop the branch if the redesign stalls; `origin/master` remains untouched.

## Definition of Done
- `python test_direct_c_compatibility.py`: all examples that pass under f2py also pass under direct-C.
- `pytest`: green.
- No generated artifacts committed under `examples/`.
- Documentation impact limited to CLI help and a short CHANGELOG entry.
