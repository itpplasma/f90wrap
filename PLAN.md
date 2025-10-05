# Direct-C Clean Execution Plan

## Goals
- Deliver a direct-C backend that matches f2py behaviour for every supported example.
- Keep generated file names and runtime expectations identical to classic f90wrap.
- Minimise moving parts: reuse existing Fortran helpers, add only the shims required to expose an ISO C ABI, and keep the C extension lean.

## Repo Setup
1. Work on `feature/direct-c-clean` (already branched from `origin/master`).
2. Use `feature/direct-c-generation` purely as reference for useful snippets (NumPy helpers, tests).

## Implementation Steps

### 1. Generator Updates
- Extend `f90wrap/f90wrapgen.py` so every emitted `f90wrap_<module>.f90` also contains:
  - The existing helper routines (unchanged).
  - A minimal `bind(C)` wrapper per entry point that forwards to the helper. Scalars/explicit arrays go through C types; anything else is passed as the handle buffer.
- Add a tiny helper module if we need shared shims (e.g. handle-to-`c_ptr` conversions), but keep everything inside the same generated `.f90` file.

### 2. C Emission
- Replace the capsule experiment in `f90wrap/cwrapgen.py` with a thin generator that:
  - Converts Python objects to C arguments (NumPy arrays via `PyArray_FROM_OTF`, integers via `PyLong_AsLong`, etc.).
  - Calls the new `bind(C)` wrappers.
  - Returns results using the existing runtime helpers (handle buffers, NumPy views).
- Delete unused capsule utilities once the new path is in place.

### 3. Python Wrappers
- Keep the canonical Python outputs from `pywrapgen` verbatim. No behavioural changes should be visible to downstream code.

### 4. Tests & Evidence
- Port the improved NumPy conversion tests and name-mangling assertions from the old branch.
- Run `python test_direct_c_compatibility.py` after each logical change and record pass/fail counts.
- Update `direct_c_test_results/compatibility_report.md` and `.json` only when the run is green.

## Rollback Strategy
- Every change lands as an isolated commit on `feature/direct-c-clean`; revert the offending commit if a regression appears.
- If the overall approach stalls, delete the branch and start a new one—the `main` branch remains untouched.

## Definition of Done
- `python test_direct_c_compatibility.py` reports all applicable examples ✅.
- `pytest` (existing unit tests) passes.
- Only intentional source changes appear in `git status`; no generated artefacts under `examples/`.
- Documentation limited to CLI help and a brief CHANGELOG entry.
