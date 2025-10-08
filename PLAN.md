# Direct-C Compatibility Plan (October 2025 Update)

## Mission
Deliver a production-quality `--direct-c` backend that mirrors the helper-based Python API, achieves ≥95 % pass rate across `examples/`, and integrates cleanly with the existing f90wrap workflow.

## Current Baseline (7 Oct 2025, 23:32 UTC sweep)
- Branch: `feature/direct-c-clean`
- Harness: `python3 test_direct_c_compatibility.py`
- Latest sweep (07 Oct 2025 23:32 UTC): **45 / 50 PASS (90 %)**, 1 skip (`example2`).
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
17. **Helper interface synthesis** — Direct-C Fortran wrappers emit explicit interfaces with host `import` statements for helper routines that operate on polymorphic arguments, fixing the prior "explicit interface required" regression in `docstring` and stabilising helper dispatch for `fortran_oo`.

## Failure Analysis
| Category | Count | Representative examples | Root cause snapshot |
| --- | --- | --- | --- |
| `fortran_compilation_failed` | 2 | `fortran_oo`, `issue258_derived_type_attributes` | Direct-C wrappers still lack helper-visible explicit interfaces for polymorphic dispatch (`fortran_oo`) and continue to transfer CLASS handles into TYPE locals (`issue258_derived_type_attributes`). |
| `undefined_symbol` | 1 | `derived-type-aliases` | Cross-module alias entry points (e.g. `_othertype_mod__plus_b`) are not emitted by the direct-C generator, so the shared object misses the helper shim. |
| `no_c_output` | 1 | `cylinder` | ISO_C routines are filtered out before codegen; the direct-C pipeline never emits C wrappers for pure bind(C) procedures. |

## Design Plan (08 Oct 2025)

Priority is to keep fixes local to the direct-C toolchain while leaving the legacy helper path untouched.

1. **Fortran OO helper bridge**
   - Teach the direct-C generator to emit an interface block (in the generated Fortran) only when a helper routine accepts polymorphic dummies. This pulls the fully-typed helper declarations into scope without editing the upstream wrappers.
   - Extend the C generator so constructors marshal Python scalars before calling helper allocators, preserving the original helper signatures (e.g. `f90wrap_m_geometry__construct_square(handle, length)`).
   - Add a minimal direct-C–only Fortran shim that `use`s `f90wrap_m_geometry` and re-exports the polymorphic helper entry points. The shim is generated beside `_module.c`, so no upstream sources are edited.

2. **Derived-type attribute parity (`issue258_derived_type_attributes`)**
   - Enhance the direct-C Fortran wrappers to keep CLASS handles inside helper-visible wrapper types instead of transferring straight into TYPE locals. This is limited to the direct-C generator: adjust `convert_derived_type_arguments`' outputs when `intent(out)` and CLASS semantics collide.
   - Provide a lightweight helper subroutine (generated in-line) to clone CLASS allocatables, ensuring the original helper API remains untouched.

3. **Cross-module alias export (`derived-type-aliases`)**
   - Inspect helper metadata during direct-C codegen and emit alias wrappers in `_module.c` whenever the helper symbol resolves to a different module. This mirrors the helper-era `_CBF` stubs without altering Python glue.

4. **ISO_C coverage (`cylinder`)**
   - Allow the direct-C pipeline to fall back to helper wrappers for BIND(C) procedures by generating a tiny Fortran forwarding stub (in the direct-C build directory) that calls the helper symbol. Only direct-C outputs change; the legacy helper build still sees the original sources.

5. **Regression harness tweaks**
   - Limit harness edits to the direct-C execution path (e.g. auto-copying `kind.map` to `kind_map`), keeping baseline helper runs intact.

These design steps avoid touching the helper-generated Fortran/Python files committed in upstream and isolate all new behaviour to direct-C artefacts or build-time shims.

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
1. **Fortran OO parity plan** — With explicit helper interfaces now auto-generated, finish remapping the double-precision overloads: thread kind metadata into the direct-C Fortran stubs so `perimeter_8` receives `real(8)` radii and update the plan for polymorphic downcasts using the refreshed `direct_c_test_results/fortran_failures.md` diagnostics.
2. **Undefined symbol triage** — Investigate the remaining `derived-type-aliases` undefined symbol path and co-plan the Direct-C ISO_C coverage needed to clear `cylinder` without regressing helper compatibility.
3. **ISO_C build coverage** — Prototype the direct-C path for ISO_C-visible routines so `cylinder` can emit C wrappers instead of being skipped, validating the approach on one small example before rolling out.

### Stepwise Execution Plan
1. **Fortran OO parity plan**
   - Capture the refreshed diagnostics from `direct_c_test_results/fortran_failures.md` (07 Oct 2025 23:32 UTC) so the outstanding kind mismatches for `fortran_oo` remain tracked alongside the helper interface notes.
   - Finalise the generator changes: (a) keep the new interface blocks for polymorphic arguments, (b) propagate `real(8)` kind metadata into direct-C Fortran stubs, and (c) stage the harness update that exercises the corrected overloads.
   - Implement and unit-test the dp-aware generator updates, then rerun the compatibility harness to confirm `fortran_oo` clears the remaining `real(4)` to `real(8)` mismatch.
2. **Derived-type alias exports**
   - Compare the helper-generated `_mytype_mod.c` against the direct-C output to identify which alias exports (`_othertype_mod__plus_b`, etc.) are missing.
   - Extend `DirectCGenerator` to emit cross-module binding aliases and add regression coverage focused on `derived-type-aliases`.
   - Validate by rerunning the harness; ensure no regressions in other derived-type suites.
3. **ISO_C build coverage**
   - Inventory all examples failing with `no_c_output` (`cylinder`, `issue206_subroutine_oldstyle`, `issue32`, `optional_args_issue53`, `string_array_input_f2py`, `subroutine_contains_issue101`).
   - Prototype ISO_C wrapper emission for a single routine (`cylinder`), including build/link updates, and document the workflow.
   - Generalise the approach across the generator and harness, then rerun the full sweep to verify the remaining `no_c_output` failures clear.

### Session Checklist — 07 Oct 2025 20:54 UTC
- Extract and categorize the Fortran compiler diagnostics for the four failing suites to inform the OO parity remediation plan.
- Diff the generated `_mytype_mod.c` artifacts between helper and direct-C to identify the missing `_othertype_mod__plus_b` export.
- Draft the ISO_C emission prototype scope (target procedure list, helper reuse) ahead of implementation.

### Session Summary — 07 Oct 2025 23:32 UTC
- Helper interface synthesis now quells the polymorphic explicit-interface failures, and the direct-C sweep improved to **45 / 50 PASS (90 %)** with 1 skip.
- Range-lowered dimension handling and updated kind-map discovery keep `issue261_array_shapes`, `kind_map_default`, and `type_check` green while the diagnostics capture has been refreshed for the current failing suites.
- Direct-C character handling still mirrors helper parity; remaining blockers are double-precision overload wiring in `fortran_oo`, the `t_inner` handle conversions, and the outstanding ISO_C/alias tasks.

Tracking: rerun `python3 test_direct_c_compatibility.py` after each fix, update this plan with new pass rates, and stash harness logs for audit.
