# Direct-C Compatibility Plan (October 2025 Update)

## Mission
Deliver a production-quality `--direct-c` backend that mirrors the helper-based Python API, achieves ≥95 % pass rate across `examples/`, and integrates cleanly with the existing f90wrap workflow.

## Current Baseline (7 Oct 2025, 20:54 UTC sweep)
- Branch: `feature/direct-c-clean`
- Harness: `python3 test_direct_c_compatibility.py`
- Latest sweep (07 Oct 2025 20:54 UTC): **43 / 50 PASS (86 %)**, 1 skip (`example2`).
- Scalar intent(in/out) arguments now reuse NumPy buffers and copy results back, unblocking `fixed_1D_derived_type_array_argument`, `arrays`, and `return_array`. Type-bound alias registration bridges now attach missing `_CBF`-era helpers in generated Python, and the direct-C module exports the alias wrappers, so `derivedtypes_procedure` completes without segfaults. Range-bound dimension metadata now collapses to explicit extents, clearing the `issue261_array_shapes` C compilation failure.

## Key Improvements Landed
1. **Module helper coverage** — `_module.c` generation now exports `get_/set_/array__*` wrappers plus derived-type accessors.
2. **Derived-type constructor/destructor hooks** — Direct-C wrappers synthesize handles for helper-based allocators and accept handle lists for destructors.
3. **Auto-raise parity** — Optional/error arguments are suppressed from Python signatures, internal ierr/errmsg buffers are allocated automatically, and non-zero ierr now reliably triggers `RuntimeError`.
4. **Derived-type scalar plumbing** — Type-member getters/setters accept handles, marshal values (including characters) and return Python-friendly bytes/ints, eliminating property-type mismatches.
5. **NumPy handle fallback** — `_library.c` and peers expose `_array__*` data identical to helper mode, and Python wrappers fall back to `f90wrap.runtime.direct_c_array` when the helper returns metadata.
6. **Harness aliasing** — Direct-C build step now copies the shared object to every generated module stem (e.g. `_library*.so`), eliminating import mismatches.
7. **Callback trampolines** — Direct-C C generation emits `_pyfunc_*` trampolines plus module-level placeholders so helper-style callbacks load without undefined symbols (`callback_print_function_issue93`).
8. **Harness duplicate filtering & proxy shims** — The compatibility sweep skips pre-generated `f90wrap_*.f90` sources, feeds detected `intent(callback)` shims back into `f90wrap`, and rewrites legacy tests to stand up helper-style proxy objects (e.g. `_CBF`) so callback registration continues to work without undefined symbols.
9. **Intent(out) array auto-allocation** — Direct-C wrappers now declare NumPy buffers for pure outputs, surface hidden dimension tokens case-insensitively, and recover helper parity for `arrays`, `arrays_fixed`, `output_kind`, and related suites.
10. **Pass-by-reference scalars** — Scalar arguments without explicit kind or helper metadata now map to concrete C/NumPy dtypes (including `double precision`), restoring the `passbyreference` suite.
11. **Namespace bridge scaffolding** — The direct-C generator now emits helper declarations for every module/type member, giving us working direct-C `derivedtypes` coverage.
12. **Alias wrapper exports** — `_module.c` generation now materializes binding-alias wrappers, matching the helper-era `_CBF` entry points and keeping direct-C modules in sync with Python alias installers.
13. **Range extent lowering** — Explicit lower:upper bounds (e.g. `1:n`) translate into concrete lengths when auto-allocating NumPy buffers, restoring Direct-C parity for fixed-range outputs (`issue261_array_shapes`).
14. **Harness failure snapshots** — Compatibility JSON now summarizes stderr/stdout per failing category, speeding post-run triage and aligning with the diagnostics action items.
15. **Import rewrite coverage** — `tests.py` rewriting now sanitizes module names and handles dotted imports via fallback binding helpers, keeping helper-era package layouts working for direct-C (`mod_arg_clash`, `arrays`).
16. **Character argument parity** — Direct-C wrappers now accept both `bytes` and `str` inputs for `character(*)` arguments and surface module strings as `bytes`, restoring the `strings` suite.

## Failure Analysis
| Category | Count | Representative examples | Root cause snapshot |
| --- | --- | --- | --- |
| `fortran_compilation_failed` | 4 | `fortran_oo`, `kind_map_default`, `type_check`, `issue258_derived_type_attributes` | Transformed Fortran wrappers assume helper-generated pointer scaffolding and ISO_C prototypes that Direct-C does not yet emit. |
| `undefined_symbol` | 1 | `derived-type-aliases` | Direct-C still misses helper-style exports for secondary module bindings, leaving `_othertype_mod__plus_b` unresolved. |
| `no_c_output` | 1 | `cylinder` | Procedures that require ISO_C bindings are still filtered out at generation time (Phase D). |

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
1. **Fortran OO parity plan** — Leverage the new `direct_c_test_results/fortran_failures.md` summary to map compiler diagnostics to missing direct-C features, then outline an execution order that keeps ≥85 % pass rate while tackling ISO_C gaps.
2. **Undefined symbol triage** — Investigate the remaining `derived-type-aliases` undefined symbol path and co-plan the Direct-C ISO_C coverage needed to clear `cylinder` without regressing helper compatibility.
3. **ISO_C build coverage** — Prototype the direct-C path for ISO_C-visible routines so `cylinder` can emit C wrappers instead of being skipped, validating the approach on one small example before rolling out.

### Session Checklist — 07 Oct 2025 20:54 UTC
- Extract and categorize the Fortran compiler diagnostics for the four failing suites to inform the OO parity remediation plan.
- Diff the generated `_mytype_mod.c` artifacts between helper and direct-C to identify the missing `_othertype_mod__plus_b` export.
- Draft the ISO_C emission prototype scope (target procedure list, helper reuse) ahead of implementation.

### Session Summary — 07 Oct 2025 20:54 UTC
- Alias wrapper export path landed, the import rewrite scaffolding handles sanitized and dotted modules, and the direct-C sweep now reports **43 / 50 PASS (86 %)** with 1 skip.
- Range-lowered dimension handling unblocked `issue261_array_shapes`, diagnostics capture stderr/stdout slices per failure category, and dotted import binding keeps `arrays`/`mod_arg_clash` green.
- Direct-C character handling now accepts bytes/str inputs and returns bytes for module scalars, closing the `strings` parity gap while preserving the ≥86 % pass plateau.

Tracking: rerun `python3 test_direct_c_compatibility.py` after each fix, update this plan with new pass rates, and stash harness logs for audit.
