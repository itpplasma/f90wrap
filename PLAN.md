# Direct-C Clean Execution Plan

## Goals
- 100% example parity with the classic f2py-backed workflow.
- Zero behavioural changes for downstream Python code (same module names, same runtime contract).
- Minimal new code: reuse existing Fortran helpers wherever ISO C falls short; emit shims only when necessary.

## Branch Strategy
- Work exclusively on `feature/direct-c-clean` (branched from `origin/master`).
- Treat `feature/direct-c-generation` as a read-only reference for reusable snippets (NumPy coercion, tests).

## Implementation Steps

### 1. Classify interfaces for ISO C interop
- Extend the visitor in `transform.py`/`f90wrapgen.py` to flag each routine as either "interop-friendly" or "needs helper" based on:
  - dummy argument type (scalar vs. array, derived vs. intrinsic).
  - attributes (optional, pointer, allocatable, assumed-shape, character length, callbacks).
- Expose a helper predicate (e.g. `needs_c_helper(node)`) so downstream stages can query the classification.

### 2. Fortran emission (`f90wrap_<module>.f90`)
- Always create the canonical wrapper file (same filename as today).
- For each routine:
  - If it’s ISO C friendly: emit a `bind(C)` subroutine/function that directly invokes the original Fortran implementation.
  - If it requires marshalling: emit the existing `f90wrap_*` helper **and** a `bind(C)` shim that forwards to that helper.
- Shared utilities (handle buffers, `c_ptr` conversions) live in the same file to keep the build unchanged.

### 3. C emission (`f90wrap/cwrapgen.py`)
- Regenerate the extension so every exported symbol calls the appropriate `bind(C)` routine (direct call or helper shim based on the classification).
- Use the improved NumPy conversions (`PyArray_FROM_OTF` + dtype enforcement) to coerce arrays.
- Delete the capsule-specific code paths once the new shims are in place.

### 4. Python wrappers (`pywrapgen.py`)
- Keep the generated Python modules unchanged; they continue to import the compiled extension and rely on `FortranDerivedType`, `FortranDerivedTypeArray`, etc.
- No Python-side knowledge of which helper is used—everything funnels through the C extension.

### 5. Tests & Evidence
- Port the numeric dtype/unit tests from the experimental branch; add new cases for the interop classifier.
- After each logical change, run `python test_direct_c_compatibility.py` and capture the pass/fail counts.
- Update `direct_c_test_results/compatibility_report.md` and `.json` only when the run is green.
- Keep `pytest` passing throughout.

## Rollback Strategy
- Every logical change is its own commit on `feature/direct-c-clean`; revert specific commits if regressions appear.
- If the approach stalls, drop the branch—`origin/master` stays intact.

## Definition of Done
- Compatibility suite reports ✅ for every example that passes under f2py.
- CI (pytest + direct-C job) green.
- No generated artefacts checked into `examples/`.
- Documentation impact limited to CLI help and a short CHANGELOG entry.
