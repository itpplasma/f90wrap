# Direct-C Coverage Plan

Goals:
- Ensure every example under `examples/` can be built and executed in Direct-C mode.
- Keep legacy f2py builds as the default; Direct-C must be opt-in.
- Provide coverage reports (which examples succeed/fall back) after each major step.

## 1. Audit Current Analyzer & Generator Limitations
- [x] Enumerate which `examples/*` fail today when forced through the Direct-C pipeline.
- [ ] For each failure, classify the blocker (non-ISO C binding arguments, callbacks, assumed-shape arrays, etc.).
- [x] Produce an initial coverage report mapping example → (passes via Direct-C | requires fallback).

## 2. Extend Interop Analyzer
- [ ] Teach `directc.analyse_interop` to detect additional safe patterns (e.g., value-result scalars, interoperable derived types with `bind(c)` components).
- [ ] Flag unsupported constructs with explicit reasons and expose them in the coverage report.
- [ ] Re-run coverage sweep; note improvements.

## 3. Generator Enhancements
- [ ] Support array shape handling for assumed-shape/explicit-shape arguments that already have interoperable interfaces.
- [ ] Generate BIND(C) wrappers for derived types that contain only interoperable fields.
- [ ] Ensure namespace helper covers nested modules/types to avoid symbol clashes.
- [ ] Update coverage report.

## 4. Runtime & CLI Support
- [ ] Add CLI/Meson switches (e.g., `--direct-c`) without changing existing defaults.
- [ ] Ensure runtime helpers lazily import Direct-C modules so legacy builds are unaffected.
- [ ] Provide documentation on how to opt in per example.
- [ ] Update coverage report, confirming CLI flag triggers new path.

## 5. Example Harness
- [ ] Extend `examples/Makefile` with a `test_directc` target that rebuilds each example in Direct-C mode and falls back gracefully.
- [ ] Capture pass/fail status for every example in a machine-readable format (`direct_c_results.json`).
- [ ] Update coverage report after running the harness.

## 6. Automated Testing
- [ ] Integrate the Direct-C example sweep into CI (optional matrix entry) while keeping it opt-in.
- [ ] Ensure pytest suite runs both legacy and Direct-C smoke tests.
- [ ] Update coverage report with CI results.

## 7. Documentation & Cleanup
- [ ] Document limitations and fallback behaviour in README/CHANGELOG.
- [ ] Remove any temporary scripts or debug output used during the rollout.
- [ ] Produce final coverage report with 100% example support.

## Reporting
After each numbered section, append a summary to `direct_c_coverage.md`:
- Examples tested
- Count passing via Direct-C, falling back, failing
- Notable blockers/new support added
