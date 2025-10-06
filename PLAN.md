# Direct-C Compatibility Plan (October 2025 Update)

## Mission
Deliver a production-quality `--direct-c` backend that mirrors the helper-based Python API, achieves ≥95 % pass rate across `examples/`, and integrates cleanly with the existing f90wrap workflow.

## Current Baseline (6 Oct 2025, evening sweep)
- Branch: `feature/direct-c-clean`
- Harness: `python3 test_direct_c_compatibility.py`
- Latest sweep: **28 / 50 PASS (56 %)**, 1 skip (`example2`).
- Newly green suites include array handling families (`arrays`, `arrays_fixed`) and retain recent wins around optional error handling (`auto_raise_error`). Regression artifacts live under `direct_c_test_results/` for reproducibility.

## Key Improvements Landed
1. **Module helper coverage** — `_module.c` generation now exports `get_/set_/array__*` wrappers plus derived-type accessors.
2. **Derived-type constructor/destructor hooks** — Direct-C wrappers synthesize handles for helper-based allocators and accept handle lists for destructors.
3. **Auto-raise parity** — Optional/error arguments are suppressed from Python signatures, internal ierr/errmsg buffers are allocated automatically, and non-zero ierr now reliably triggers `RuntimeError`.
4. **Derived-type scalar plumbing** — Type-member getters/setters accept handles, marshal values (including characters) and return Python-friendly bytes/ints, eliminating property-type mismatches.
5. **NumPy handle fallback** — `_library.c` and peers expose `_array__*` data identical to helper mode, and Python wrappers fall back to `f90wrap.runtime.direct_c_array` when the helper returns metadata.
6. **Harness aliasing** — Direct-C build step now copies the shared object to every generated module stem (e.g. `_library*.so`), eliminating import mismatches.
7. **Callback trampolines** — Direct-C C generation emits `_pyfunc_*` trampolines plus module-level placeholders so helper-style callbacks load without undefined symbols (`callback_print_function_issue93`).
8. **Harness duplicate filtering** — The compatibility sweep skips pre-generated `f90wrap_*.f90` sources and pipes detected `intent(callback)` shims back into `f90wrap`, preventing link clashes (`docstring`) and ensuring trampoline emission stays wired.

## Failure Analysis
| Category | Count | Representative examples | Root cause snapshot |
| --- | --- | --- | --- |
| `c_compilation_failed` | 9 | `optional_string`, `return_array`, `output_kind` | Optional character buffers still emit missing length locals; some generated modules include duplicate helper declarations. |
| `fortran_compilation_failed` | 4 | `fortran_oo`, `kind_map_default` | Upstream sources rely on helper-emitted pointer scaffolding; Direct-C still misses the equivalent support code. |
| `unknown_error` | 3 | `intent_out_size`, `strings`, `subroutine_args` | Python wrappers still expect helper-packaged namespaces (`ExampleDerivedTypes`) or fail when optional defaults return helper-only wrappers. |
| `syntax_error` | 2 | `derived-type-aliases`, `mod_arg_clash` | Harness rewrites `tests.py` imports but still needs to respect nested aliasing patterns in legacy modules. |
| `type_error` | 1 | `strings` | Direct-C character buffers are returned as `bytes`, breaking tests that assume helper-mode Unicode conversions. |
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
2. **Helper-style namespace bridge** — Provide a proxy or generation-time hook so modules expose helper-era aggregate namespaces (`ExampleDerivedTypes`, `_CBF`), unblocking `derivedtypes` and callback examples without manual harness shims.
3. **Direct-C array metadata parity** — Capture shape/kind metadata for derived arrays so `_array__*` wrappers can rebuild NumPy views without helper assistance (`recursive_type`, `return_array`).
4. **Harness diagnostics** — Extend the JSON report with summarized stderr/stdout excerpts for the top failure classes to accelerate root-cause triage.

Tracking: rerun `python3 test_direct_c_compatibility.py` after each fix, update this plan with new pass rates, and stash harness logs for audit.
