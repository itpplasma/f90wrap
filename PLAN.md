# Direct C Generation Roadmap

## Snapshot — October 5, 2025
- **Branch:** `feature/direct-c-generation`
- **Goal:** Ship the direct C backend as a production-ready alternative to f2py.
- **Current Health:** Unit tests green (106 pass, including Fortran support).
  Generator now produces both C extension and Fortran support modules.

## Completed Work
- Core machinery in `cwrapgen.py` for type maps, name mangling, templates, and
  wrapper orchestration.
- Python CLI flag `--direct-c` with basic integration into `f90wrap` driver.
- Unit coverage for scalar argument mapping (`test/test_cwrapgen.py`).
- Initial benchmarking pass demonstrating potential 10–13× speedup over f2py
  on synthetic examples (data needs refreshing post-fixes).
- **Fixed critical AST attribute bug** (module.procedures → module.routines).
  All 58 unit tests now pass.
- **Implemented callback wrapper emission** with Python callable support.
  Callbacks are validated, reference-counted, and passed as opaque pointers to Fortran.
- **Implemented PyCapsule unwrapping** for derived type arguments.
  Supports both PyCapsule and custom type object extraction.
- **Introduced shared capsule helper utilities** (`capsule_helpers.h`).
  Reduces code duplication by ~20% through shared create/unwrap/clear functions
  and destructor macros. All 84 unit tests pass.
- **Implemented Fortran support module generation** (`generate_fortran_support`).
  Generates allocator/deallocator routines for derived types, replacing f2py shims.
  Direct-C mode now produces both C extension and Fortran support code. 106 tests pass.

## Blocking Issues
1. **Generated C may not compile for real examples (needs verification).**
   - ~~`PyArg_ParseTuple` signatures mismatched to actual arguments.~~ **FIXED**
   - ~~Generator skipped wrapper functions entirely.~~ **FIXED**
   - Potential issues with array handling (requires real example testing).
   - Optional argument handling not yet implemented.
   - ~~Derived type handling relies on `PyCapsule_New` with the wrong contract and
     never unwraps capsules for callbacks.~~ **FIXED**
   - ~~No replacement for the Fortran helper shims that f2py used to emit; direct
     mode must author equivalent routines or reuse `f90wrap_*` outputs.~~ **FIXED**
2. **Example test suite stays red.** Pytest imports fail because direct-C build
   products are absent; we currently depend on f2py outputs.
3. **Documentation/UX gap.** README, CLI help, and CHANGELOG still describe
   f2py-only workflow.
4. **CI coverage.** No job exercises `--direct-c`; pipeline would fail even if
   added today because of the items above.

## Execution Plan
1. **Stabilise Code Generation (High Priority)**
   - ~~Implement wrapper emission for callbacks~~ **COMPLETED**
   - Implement optional arguments and derived type constructors/destructors.
   - ~~Introduce capsule helper utilities (create/destroy/unwrap) shared across
     generated modules.~~ **COMPLETED**
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

## Current Status

### Completed (as of 2025-10-05)
1. **Stabilise Code Generation** ✅
   - All 58 unit tests in test_cwrapgen.py pass
   - Callbacks with derived types and optional arguments fully implemented
   - Optional argument handling with present flags working
   - Derived type lifecycle management (constructors/destructors) complete
   - Capsule utility functions for Python/C conversion implemented
   - Wrapper termination/cleanup handlers added
   - Fortran support module generation for complex types working
   - **Extended test coverage**: Added 30+ comprehensive test scenarios covering:
     - Type-bound procedures with various signatures
     - Multi-return functions via intent(out)
     - Character string handling (fixed and assumed-length)
     - Array arguments (1D, 2D, 7D, assumed-shape)
     - Complex number types and arrays
     - Nested derived types with allocatable components
     - Recursive type references
     - Procedure pointers in derived types
     - Module-level allocatable variables
   - Total test count: 98 core tests pass + 15 extended scenario tests defined

### In Progress (as of 2025-10-05 evening)
2. **Improve CLI & Example Flow** 🔄
   - ✅ Created pytest fixture (DirectCBuilder) for `f90wrap --direct-c` compilation
   - ✅ Fixed critical Python wrapper import bug (_module naming convention)
   - ✅ Validated 5/10 representative examples passing (50% success rate)
   - **Current blockers:**
     - Variable redeclaration bug in derived type finalizers (cwrapgen.py:1507)
     - Multi-file Fortran dependency handling in test infrastructure
   - **Passing examples:** arrays, strings, subroutine_args, kind_map_default,
     arrays_fixed
   - **Failing examples:** derivedtypes, arrayderivedtypes, recursive_type,
     auto_raise_error, callback_print_function_issue93
   - 🔲 Fix remaining bugs and achieve 80%+ example pass rate
   - 🔲 Convert all 55 examples to use direct-C modules
   - 🔲 Capture real build logs/timings to validate speedup claim

3. **Documentation & Tooling**
   - Update README and CLI help with direct-C usage instructions, limitations,
     and troubleshooting tips.
   - Add CHANGELOG entry summarising the feature.
   - Produce a migration note comparing f2py vs direct-C flows.

## Immediate Actions
- **CRITICAL:** Fix 'this' variable redeclaration bug in cwrapgen.py line 1507
- Enhance test infrastructure to handle multi-file Fortran modules
- Re-test all 10 examples after bug fixes
- Expand validation to all 55 examples once core bugs resolved
