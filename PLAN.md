# Direct-C Compatibility Plan (October 2025 Update)

## Mission
Deliver a production-quality `--direct-c` backend that mirrors the helper-based Python API, achieves ≥95 % pass rate across `examples/`, and integrates cleanly with the existing f90wrap workflow.

## Current Baseline (6 Oct 2025, late PM sweep)
- Branch: `feature/direct-c-clean`
- Harness: `python3 test_direct_c_compatibility.py`
- Latest sweep: **26 / 50 PASS (52 %)**, 1 skip (`example2`).
- Green scenarios now include module helper access (`arrays`, `class_names`, `arrayderivedtypes`, `recursive_type_array`) and scalar/module character access (`default_i8`, `strings` getters).
- Regression suite artifacts stored in `direct_c_test_results/` (untracked).

## Key Improvements Landed
1. **Module helper coverage** — `_module.c` generation now exports `get_/set_/array__*` wrappers plus derived-type accessors.
2. **Derived-type constructor/destructor hooks** — Direct-C wrappers synthesize handles for helper-based allocators and accept handle lists for destructors.
3. **NumPy handle fallback** — `_library.c` and peers expose `_array__*` data identical to helper mode, and Python wrappers fall back to `f90wrap.runtime.direct_c_array` when the helper returns metadata.
4. **Harness aliasing** — Direct-C build step now copies the shared object to every generated module stem (e.g. `_library*.so`), eliminating import mismatches.

## Failure Analysis
| Category | Count | Representative examples | Root cause snapshot |
| --- | --- | --- | --- |
| `c_compilation_failed` | 7 | `docstring`, `mod_arg_clash`, `return_array` | Remaining helpers still expect hidden string buffers or parent handles for nested arguments. |
| `fortran_compilation_failed` | 4 | `fortran_oo`, `kind_map_default` | Upstream sources rely on helper-emitted pointer scaffolding; Direct-C still misses the equivalent support code. |
| `type_error` | 3 | `auto_raise_error`, `keyword_renaming_issue160`, `remove_pointer_arg` | Optional/output arguments are still exposed to Python, causing signature drift and runtime exceptions. |
| `unknown_error` | 5 | `derivedtypes`, `intent_out_size`, `subroutine_args` | Semantic mismatches surfaced at runtime (handle reconstruction / optional defaults) once compilation succeeds. |
| `undefined_symbol` | 1 | `callback_print_function_issue93` | Missing runtime shims (e.g. `f90wrap_abort`) from directly generated C. |
| `no_c_output` | 1 | `cylinder` | Direct-C generator still skips ISO-C-only procedures (Phase A2). |

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
1. **Optional/intent(out) plumbing** — Teach Direct-C wrappers to omit optional outputs from Python signatures, allocate temporary buffers, and raise `RuntimeError` when ierr/errmsg pairs signal failure.
2. **Helper parity for module characters** — Generalise the module-helper character support to procedure wrappers and ensure tests like `auto_raise_error` pass end-to-end.
3. **Document rebuild workflow** — Add a README section covering `pip install -e . --no-build-isolation` + `ninja` to rebuild the editable wheel, preventing future `fortranobject.h` build errors.

Tracking: rerun `python3 test_direct_c_compatibility.py` after each fix, update this plan with new pass rates, and stash harness logs for audit.
