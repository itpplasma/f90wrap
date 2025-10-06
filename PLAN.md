# Direct-C Compatibility Plan (October 2025 Update)

## Mission
Deliver a production-quality `--direct-c` backend that mirrors the helper-based Python API, achieves ≥95 % pass rate across `examples/`, and integrates cleanly with the existing f90wrap workflow.

## Current Baseline (6 Oct 2025, PM sweep)
- Branch: `feature/direct-c-clean`
- Harness: `python3 test_direct_c_compatibility.py`
- Latest sweep: **24 / 50 PASS (48 %)**, 1 skip (`example2`).
- Green scenarios now include module helper access (`arrays`, `class_names`, `arrayderivedtypes`) and scalar module state (`default_i8`).
- Regression suite artifacts stored in `direct_c_test_results/` (untracked).

## Key Improvements Landed
1. **Module helper coverage** — `_module.c` generation now exports `get_/set_/array__*` wrappers plus derived-type accessors.
2. **Derived-type constructor/destructor hooks** — Direct-C wrappers synthesize handles for helper-based allocators and accept handle lists for destructors.
3. **NumPy handle fallback** — `_library.c` and peers expose `_array__*` data identical to helper mode, and Python wrappers fall back to `f90wrap.runtime.direct_c_array` when the helper returns metadata.
4. **Harness aliasing** — Direct-C build step now copies the shared object to every generated module stem (e.g. `_library*.so`), eliminating import mismatches.

## Failure Analysis
| Category | Count | Representative examples | Root cause snapshot |
| --- | --- | --- | --- |
| `c_compilation_failed` | 13 | `auto_raise_error`, `docstring`, `mod_arg_clash` | Helper signatures still expect extra hidden arguments (e.g. `character(len=*)` buffers) and we dont yet coerce strings/optional args for minimal wrappers. |
| `fortran_compilation_failed` | 4 | `fortran_oo`, `kind_map_default` | Upstream sources rely on f2py transform assumptions (pointer arguments inserted by helper path); the Direct-C pass must generate equivalent support code. |
| `attribute_error` / `type_error` | 5 | `keyword_renaming_issue160`, `recursive_type` | Generated Python wrappers call `module.set_foo(...)` that our Direct-C code doesnt (yet) expose, or we still return raw tuples instead of proper derived-type handles. |
| `unknown_error` | 3 | `derivedtypes`, `intent_out_size` | Semantic mismatches surfaced at runtime (handle reconstruction / optional argument defaults). |
| `no_c_output` | 1 | `cylinder` | Direct-C generator still skips ISO-C-only procedures (Phase A2). |

## Path Forward

### Phase B – Stabilise Runtime Surface (target ≥70 %)
1. **Character/optional argument parity**
   - Extend `_write_module_helper_wrapper` and `_write_helper_call` to allocate temporary `char*` buffers (with length arguments) for module setters (`auto_raise_error`, `docstring`).
   - Mirror helper logic for optional arguments: track hidden `f90wrap_*` length arguments and respect defaulted keywords.

2. **Derived-type scalar access**
   - Replace ad-hoc constructor/destructor wrappers with helper calls that accept parent handles (implemented for module scalars, still pending for derived-type members). Ensure `_module_*__set__foo` receives both parent and child handles.
   - Add Python-side convenience (`Module.array` setter/getter) to pass handles explicitly when tests expect raw types.

3. **Abort/runtime shims**
   - Bundle the lightweight `f90wrap_abort` C helper we emit directly into every module (already added) and audit remaining missing symbols (e.g. `f90wrap_abort_` duplicates).

4. **Harness resilience**
   - Capture f90wrap failure logs in the JSON report (`f90wrap_error`) for quicker triage.

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
1. **Complete derived-type array setters** — Finish propagating parent handles through getters/setters to unblock `recursive_type` and similar recursive fixtures (currently 48 %).
2. **String helper support** — Fix `auto_raise_error` by parsing `(char*, int)` pairs and forwarding hidden length arguments when calling character helpers.
3. **Document rebuild workflow** — Add a README section covering `pip install -e . --no-build-isolation` + `ninja` to rebuild the editable wheel, preventing future `fortranobject.h` build errors.

Tracking: rerun `python3 test_direct_c_compatibility.py` after each fix, update this plan with new pass rates, and stash harness logs for audit.
