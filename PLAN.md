# Direct-C Clean Execution Plan

## Goals
- Direct-C backend matches f2py behaviour for every supported example.
- Generated file names and runtime contract identical to classic f90wrap (helpers remain in `f90wrap_<module>.f90`).
- New logic isolated in fresh modules (`f90wrap/directc.py`, `f90wrap/directc_cgen.py`) so the existing code stays intact.

## Branch Strategy
- Active branch: `feature/direct-c-clean` (branched from `origin/master`).
- Reference branch: `feature/direct-c-generation` (read-only; reuse NumPy conversion snippets and tests).

## Implementation Steps

### 1. Interop Classification (done)
- Use `f90wrap/directc.py` to tag each procedure with `requires_helper`. All routines continue to emit helpers; classification is used only by the C generator to decide how much marshalling is needed.

### 2. Fortran helpers (unchanged)
- Continue running `F90WrapperGenerator` exactly as today. Helpers remain the single Fortran surface we call from C; no additional `bind(C)` shims are emitted.

### 3. Direct-C C Generator (`f90wrap/directc_cgen.py`)
- Implement a new generator that traverses the parse tree and writes `_module.c` files that:
  ```c
  static PyObject* wrap_foo(PyObject *self, PyObject *args) {
      /* marshal Python → C (NumPy, ints, strings) */
      /* call the existing helper: f90wrap_module__foo(...) */
      /* translate results back to Python */
  }
  ```
- Reuse NumPy conversion utilities from the old branch (move code into a small helper module under `f90wrap/` so the generator stays clean).
- Export the same symbol names the Python wrappers expect (`f90wrap_module__foo`, etc.).

### 4. CLI updates (`f90wrap/scripts/main.py`)
- When `--direct-c` is passed:
  - Run the helper/classification steps as usual.
  - Emit the C extension via `directc_cgen.py` instead of invoking f2py.
  - Leave the Python wrapper emission untouched.
- Add minimal build notes/logging so the user knows to compile `_module.c` (mirrors how f2py reports its output).

### 5. Tests & Evidence
- Port the NumPy coercion tests and name-mangling checks from `feature/direct-c-generation`.
- After each major change run:
  - `pytest` (unit tests).
  - `python test_direct_c_compatibility.py` to capture example pass/fail counts.
- Update `direct_c_test_results/compatibility_report.md` and `.json` once the suite is green.

### 6. Cleanup & Documentation
- Remove capsule helpers and other dead code once the new C generator is confirmed working.
- Documentation limited to CLI help and a short CHANGELOG entry.

## Copy Guidance from `feature/direct-c-generation`
- NumPy conversions: `f90wrap/numpy_capi.py` (lines ~150–230).
- Direct-C unit tests: `test/test_cwrapgen.py`.
- Compatibility script: `test_direct_c_compatibility.py` (adapt paths, keep out of git).

## Rollback Strategy
- Commit each logical step separately on `feature/direct-c-clean`; revert individual commits if regressions appear.
- If the approach stalls, delete the branch; `origin/master` remains untouched.

## Definition of Done
- `python test_direct_c_compatibility.py`: all examples that pass under f2py also pass under direct-C.
- `pytest`: green.
- No generated artefacts tracked under `examples/`.
- Documentation impact limited to CLI help and a brief CHANGELOG entry.
