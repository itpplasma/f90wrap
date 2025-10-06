# Direct-C Compatibility Plan (October 2025 Update)

## Mission
Deliver a production-quality `--direct-c` backend that mirrors the helper-based Python API, achieves ≥95 % pass rate across `examples/`, and integrates cleanly with the existing f90wrap workflow.

## Current Baseline (7 Oct 2025, midday sweep)
- Branch: `feature/direct-c-clean`
- Harness: `python3 test_direct_c_compatibility.py`
- Latest sweep: **26 / 50 PASS (52 %)**, 1 skip (`example2`).
- Newly green suites cover derived-type scalar access (`arrayderivedtypes`, `keyword_renaming_issue160`) and auto-raise handling (`auto_raise_error`). Character outputs now surface as Python `bytes`, matching helper semantics.
- Regression suite artifacts stored in `direct_c_test_results/` (untracked).

## Key Improvements Landed
1. **Module helper coverage** — `_module.c` generation now exports `get_/set_/array__*` wrappers plus derived-type accessors.
2. **Derived-type constructor/destructor hooks** — Direct-C wrappers synthesize handles for helper-based allocators and accept handle lists for destructors.
3. **Auto-raise parity** — Optional/error arguments are suppressed from Python signatures, internal ierr/errmsg buffers are allocated automatically, and non-zero ierr now reliably triggers `RuntimeError`.
4. **Derived-type scalar plumbing** — Type-member getters/setters accept handles, marshal values (including characters) and return Python-friendly bytes/ints, eliminating property-type mismatches.
5. **NumPy handle fallback** — `_library.c` and peers expose `_array__*` data identical to helper mode, and Python wrappers fall back to `f90wrap.runtime.direct_c_array` when the helper returns metadata.
6. **Harness aliasing** — Direct-C build step now copies the shared object to every generated module stem (e.g. `_library*.so`), eliminating import mismatches.

## Failure Analysis
| Category | Count | Representative examples | Root cause snapshot |
| --- | --- | --- | --- |
| `c_compilation_failed` | 9 | `optional_string`, `return_array`, `output_kind` | Optional character buffers still emit missing length locals; some generated modules include duplicate helper declarations. |
| `fortran_compilation_failed` | 4 | `fortran_oo`, `kind_map_default` | Upstream sources rely on helper-emitted pointer scaffolding; Direct-C still misses the equivalent support code. |
| `unknown_error` | 5 | `derivedtypes`, `intent_out_size`, `subroutine_args` | Python wrappers expect helper-packaged namespaces (`ExampleDerivedTypes`) or mishandle optional defaults, causing runtime import errors. |
| `linking_failed` | 1 | `docstring` | Both transformed and pre-generated `f90wrap_*` sources are compiled, yielding multiply-defined wrapper symbols. |
| `undefined_symbol` | 1 | `callback_print_function_issue93` | Direct-C output lacks the callback trampoline (`pyfunc_print_`) that helper mode injects. |
| `no_c_output` | 1 | `cylinder` | Direct-C generator still skips ISO-C-only procedures (Phase A2).

## Path Forward

### Phase B – Stabilise Runtime Surface (target ≥70 %)
1. **Character/optional argument parity**
   - Extend `_write_helper_call` and `_write_arg_preparation` so intent(`out`) character arguments allocate scratch buffers and marshal back to Python (`auto_raise_error`, `docstring`).
   - Mirror helper logic for optional arguments: skip parsing optional outputs, honour Python `None`, and wire hidden length arguments.

2. **Derived-type scalar access**
   - Parent-handle propagation for derived-type arrays is in place; finish plumbing setters/getters for scalar members invoked through Python properties (`keyword_renaming_issue160`).
   - Rehydrate derived-type returns via `lookup_class` to eliminate bare handle tuples (`derivedtypes`).

3. **Abort/runtime shims**
   - Bundle the lightweight `f90wrap_abort` C helper into every generated `_module.c` and audit remaining undefined symbols (e.g. `_abort`, `_traceback`).

4. **Harness resilience**
   - Persist f90wrap stderr into the JSON report (`f90wrap_error`) and surface top failure classes in the markdown summary.

### Phase C – Helper Parity for Derived Arrays (target ≥85 %)
1. **Type array helpers**
   - Generate `*_array__field`, `*_array_getitem__field`, `*_array_setitem__field`, and `*_array_len__field` for derived-type members. This feeds examples such as `arrays_in_derived_types_issue50` and `recursive_type`. (Partially landed; complete coverage + parent handle propagation still required.)
2. **Module-level allocatables**
   - Ensure module arrays containing derived types rebuild Python objects via `f90wrap.runtime.lookup_class`, mirroring helper mode caches.
3. **Error reporting alignment**
   - Propagate Fortran exceptions (via `f90wrap_abort`) so failing tests emit informative errors rather than silent mismatches.

### Phase D – ISO-C Coverage & Build Integration (target ≥95 %)
1. **Emit non-helper wrappers**
   - Update Direct-C generator to include ISO-C compatible routines (Phase A2 of original roadmap) by calling either the helper shim or `F90WRAP_F_SYMBOL` directly.
2. **Meson/ninja build hooks**
   - Ensure editable installs always keep `_build/cp3xx` in sync (documented, re-run `ninja` when needed) to avoid stale `_f90wrap_editable_loader` rebuild issues.
3. **Regression sweep**
   - Run the harness after each milestone and append pass-rate deltas to `direct_c_test_results/compatibility_report.md`.

## Immediate Next Actions (Week 41)
1. **Optional character array bookkeeping** — Emit length locals for every optional/output character array and plumb them through helper calls (fixes `optional_string`, `output_kind`, `return_array`).
2. **Deduplicate hybrid wrapper builds** — Detect pre-generated `f90wrap_*` sources when `.fpp` shims are present and skip regenerating duplicates to avoid link collisions (`docstring`).
3. **Callback trampolines** — Generate tiny C shims (and Fortran declarations) for helper-defined callbacks so direct-C modules export `pyfunc_print_` and similar symbols (`callback_print_function_issue93`).
4. **Package-level module exports** — Adjust the harness or generator to expose the expected `ExampleDerivedTypes` namespace, covering suites that import helper-style aggregate modules (`derivedtypes`).

Tracking: rerun `python3 test_direct_c_compatibility.py` after each fix, update this plan with new pass rates, and stash harness logs for audit.
