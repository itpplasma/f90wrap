# Direct C Generation Roadmap

## Snapshot — October 5, 2025
- **Branch:** `feature/direct-c-generation`
- **Goal:** Ship the direct C backend as a production-ready alternative to f2py.
- **Current Health:** Unit tests green (58/58 pass). Generator now emits complete
  wrapper functions. Ready to test with real examples.

## Completed Work
- Core machinery in `cwrapgen.py` for type maps, name mangling, templates, and
  wrapper orchestration.
- Python CLI flag `--direct-c` with basic integration into `f90wrap` driver.
- Unit coverage for scalar argument mapping (`test/test_cwrapgen.py`).
- Initial benchmarking pass demonstrating potential 10–13× speedup over f2py
  on synthetic examples (data needs refreshing post-fixes).
- **Fixed critical AST attribute bug** (module.procedures → module.routines).
  All 58 unit tests now pass.

## Blocking Issues
1. **Generated C may not compile for real examples (needs verification).**
   - ~~`PyArg_ParseTuple` signatures mismatched to actual arguments.~~ **FIXED**
   - ~~Generator skipped wrapper functions entirely.~~ **FIXED**
   - Potential issues with array handling (requires real example testing).
   - Optional argument handling not yet implemented.
   - Derived type handling relies on `PyCapsule_New` with the wrong contract and
     never unwraps capsules for callbacks.
   - No replacement for the Fortran helper shims that f2py used to emit; direct
     mode must author equivalent routines or reuse `f90wrap_*` outputs.
2. **Example test suite stays red.** Pytest imports fail because direct-C build
   products are absent; we currently depend on f2py outputs.
3. **Documentation/UX gap.** README, CLI help, and CHANGELOG still describe
   f2py-only workflow.
4. **CI coverage.** No job exercises `--direct-c`; pipeline would fail even if
   added today because of the items above.

## Execution Plan
1. **Stabilise Code Generation (High Priority)**
   - Implement wrapper emission for callbacks, optional arguments, and derived
     type constructors/destructors.
   - Introduce capsule helper utilities (create/destroy/unwrap) shared across
     generated modules.
   - Ensure function wrappers always terminate with valid closing braces and
     call `function_wrapper_end`.
   - Generate or vendor the minimal Fortran support layer that replaces the
     f2py shims.
   - Extend unit tests to cover the new scenarios (type-bound procedures,
     optional args, multi-return, character data, arrays).

2. **Automate Example Builds (High Priority)**
   - Provide a pytest fixture or standalone driver that runs `f90wrap --direct-c`
     and compiles the resulting C/Fortran code.
   - Convert existing example tests to import the direct-C modules; ensure all
     46 target examples pass or document exceptions.
   - Capture real build logs/timings to validate the speedup claim.

3. **Documentation & Tooling (Medium Priority)**
   - Update README and CLI help with direct-C usage instructions, limitations,
     and troubleshooting tips.
   - Add CHANGELOG entry summarising the feature.
   - Produce a migration note comparing f2py vs direct-C flows.

4. **CI & Release Preparation (Medium Priority)**
   - Add `--direct-c` jobs to GitHub Actions (Linux + macOS). Gate on example
     build success.
   - Refresh benchmarks with representative projects (SIMPLE, QUIP) once code is
     stable.
   - Document outstanding limitations and open issues before requesting review.

## Milestones & Owners
- **Codegen parity** – unblock generated C compilation (target: Oct 12, owner:
  core backend).
- **Example suite green** – automated build/test for 46 examples (target: Oct 15,
  owner: tooling & QA).
- **Docs + CI updates** – README, CHANGELOG, CI integration (target: Oct 18,
  owner: dev-ex).
- **Ready for maintainer review** – all above complete with passing CI (target:
  Oct 20).

## Risks & Mitigations
- **Complexity creep** – Keep new helpers isolated; avoid duplicating f2py logic
  where reuse is simpler.
- **Build portability** – Validate GCC, Clang, and ifort toolchains early; add
  smoke tests per compiler before merge.
- **Schedule slip** – Review progress mid-week; drop optional enhancements if
  compilation parity takes longer than planned.

## Immediate Actions
- Focus on fixing the generator defects outlined under *Blocking Issues*.
- Re-run unit suite (`pytest test/test_cwrapgen.py`) after each fix; add new
  tests alongside code.
- Once direct examples compile, capture evidence (commands + timings) for the PR
  description and PLAN revision.
