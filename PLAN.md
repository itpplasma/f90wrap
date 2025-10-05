# Direct-C Generation: Minimization Plan

## Guiding Principles

1. **100% Example Compatibility** - All examples that work with f2py MUST work with --direct-c
2. **Minimal Changes** - Add new code, don't modify existing behavior
3. **Minimal Documentation** - Only CLI help and CHANGELOG, no README/guides
4. **Zero Regressions** - Existing f2py mode completely unchanged
5. **Evidence-Based** - Every claim backed by test results

## Current Status (After Restoration)

All accidentally deleted files from master have been restored. The branch now properly adds direct-C functionality without breaking existing tests.

**Files changed:** 38 (down from 161 after cleanup)
**Net change:** +3,838 insertions, -1,557 deletions
**Core f90wrap changes:** 7 files (1 modified, 6 new)

## What Changed vs Master

### Core f90wrap Package (7 files)
1. `f90wrap/scripts/main.py` - **MODIFIED** (+72, -17 lines)
   - Added `--direct-c` CLI flag
   - Added direct-C generation branch in main workflow
2. `f90wrap/cwrapgen.py` - **NEW** (1,902 lines)
   - C wrapper code generator
3. `f90wrap/cerror.py` - **NEW** (380 lines)
   - Error handling utilities
4. `f90wrap/numpy_capi.py` - **NEW** (381 lines)
   - NumPy C API helpers
5. `f90wrap/capsule_helpers.py` - **NEW** (46 lines)
   - PyCapsule utilities (Python)
6. `f90wrap/capsule_helpers.h` - **NEW** (171 lines)
   - PyCapsule utilities (C header)
7. `f90wrap/meson.build` - **MODIFIED** (+4 lines)
   - Added new module files

### Test Infrastructure (4 files)
1. `test/test_cwrapgen.py` - **NEW** (873 lines)
   - Unit tests for direct-C wrapper generator

Existing tests restored and preserved:
- `test/test_parser.py` (preserved)
- `test/test_transform.py` (preserved)
- `test/.gitignore`, `test/samples/*` (preserved)

### Documentation (1 file)
- `PLAN.md` - This file

### Example Files (26 files)
- All existing `examples/*/tests.py` restored
- Generated direct-C outputs in examples/ (should not be committed)

## Minimization Strategy

### Priority 1: Remove Generated Files (IMMEDIATE)
Generated files should NOT be in the branch:

**Find and remove:**
```bash
# Direct-C generated outputs in examples/
examples/*/_*module.c
examples/*/*_directc.py
examples/*/*_directc_support.f90
examples/*/*.o
examples/*/*.mod
examples/*/*.so
examples/*/f90wrap.log
```

These are build artifacts, not source code.

### Priority 2: Simplify Core Implementation (OPTIONAL)
Consider consolidating helper modules:

**Current:** 4 separate utility files
- `cerror.py` (380 lines)
- `numpy_capi.py` (381 lines)
- `capsule_helpers.py` (46 lines)
- `capsule_helpers.h` (171 lines)

**Option:** Merge into `cwrapgen.py` or create single `cwrapgen_utils.py`
- Would reduce file count: 7 → 4 files
- Trade-off: Larger individual files vs more files

### Priority 3: Minimize main.py Changes (REVIEW)
Current: +72, -17 lines in `main.py`

**Review:**
- Are all 72 added lines necessary?
- Can any logic be moved to `cwrapgen.py`?
- Is the branch clean (`if args.direct_c:` vs messy interleaving)?

### Priority 4: Test Coverage (ASSESS)
Current: `test/test_cwrapgen.py` (873 lines)

**Options:**
1. Keep as-is (comprehensive coverage)
2. Reduce to smoke tests only (~100 lines)
3. Split into multiple focused test files

**Recommendation:** Keep as-is. Good test coverage is worth the lines.

## Production Readiness Checklist

### Must Have (Blocking Merge)
- [x] Core functionality working
- [x] Existing tests preserved
- [ ] **ALL existing examples must work with --direct-c flag**
  - Every example in `examples/*/tests.py` must pass with direct-C mode
  - Target: 100% compatibility (all examples that work with f2py must work with direct-C)
  - If any example cannot work with direct-C, document why and mark as "not applicable"
  - Zero regressions: No example that works with f2py should fail with direct-C
- [ ] Remove generated files from examples/
- [ ] Verify no accidental deletions remain
- [ ] **Minimal documentation updates ONLY:**
  - Add `--direct-c` flag to CLI help text (already in main.py)
  - Add 3-line CHANGELOG entry: "Added --direct-c flag for direct C generation"
  - **NO README changes** (users can use --help)
  - **NO guides, tutorials, or extensive documentation**
- [ ] CI job added for direct-C mode (runs ALL passing examples)

### Post-100% Enhancements (Optional - NOT for initial merge)
- [ ] Consolidate utility modules
- [ ] Optimize main.py changes
- [ ] Documentation: User guides, tutorials (only if users request)
- [ ] Performance benchmarks in docs/
- [ ] Advanced features (callbacks, abstract types, etc.)

## Remaining Tasks

### Path to 100% Example Compatibility

**Phase 1: Baseline Assessment (Day 1)**
1. **Run all examples with direct-C and document status**
   ```bash
   # Create test script that runs all examples
   for dir in examples/*/; do
     echo "Testing $dir"
     cd "$dir"
     # Extract f90wrap command from existing build
     # Run with --direct-c flag
     # Run tests.py
     # Document: Pass/Fail/Skip
   done
   ```

   **Categorize results:**
   - ✅ PASS: Works perfectly with direct-C
   - ❌ FAIL: Fails with direct-C (needs fixing)
   - ⊘ SKIP: Not applicable (no Fortran code, special cases)

**Phase 2: Fix All Failures (Days 2-4)**
For each failing example:
1. Identify root cause (code generation bug, missing feature, etc.)
2. Fix the bug in cwrapgen.py or related files
3. Add test case to test_cwrapgen.py to prevent regression
4. Re-run example, verify it passes
5. Move to next failure

**Common failure categories to expect:**
- Missing type support (complex numbers, character arrays, etc.)
- Array handling edge cases (assumed-shape, allocatable, etc.)
- Derived type complications (nested types, type-bound procedures)
- Module structure issues (multiple modules, dependencies)
- Name mangling edge cases

**Acceptance criteria for Phase 2:**
- 100% of examples that work with f2py also work with direct-C
- Every fix has corresponding test case
- Zero regressions (all previously passing examples still pass)

**Phase 3: Systematic Debugging Workflow**
When an example fails with direct-C:

1. **Compare outputs:**
   ```bash
   # Run with f2py (working)
   f90wrap -m example example.f90
   python tests.py > f2py_output.txt

   # Run with direct-C (failing)
   f90wrap --direct-c -m example example.f90
   python tests.py > directc_output.txt 2>&1

   # Compare
   diff f2py_output.txt directc_output.txt
   ```

2. **Identify failure stage:**
   - Code generation (syntax errors in .c file)
   - Compilation (gcc errors)
   - Linking (undefined symbols)
   - Runtime (Python import errors, segfaults)
   - Test execution (wrong results)

3. **Fix root cause:**
   - **Code generation bug:** Fix in cwrapgen.py, add unit test
   - **Missing feature:** Implement in cwrapgen.py
   - **Name mangling:** Fix symbol generation
   - **Type conversion:** Fix in numpy_capi.py or cerror.py

4. **Verify fix:**
   ```bash
   # Test the specific example
   pytest test/test_cwrapgen.py -k new_test

   # Re-run the failing example
   cd examples/failing_example
   f90wrap --direct-c -m example example.f90
   python tests.py  # Should pass

   # Run ALL examples to ensure no regression
   ./test_all_examples_direct_c.sh
   ```

5. **Document and move on:**
   - Update compatibility matrix
   - Commit fix with reference to example
   - Proceed to next failure

**Phase 4: Clean and Document**
   ```bash
   git rm examples/*/*_directc* examples/*/*.o examples/*/*.mod examples/*/*.so examples/*/f90wrap.log 2>/dev/null
   git commit -m "Remove generated build artifacts from examples"
   ```

3. **Verify diff is clean**
   ```bash
   git diff master --stat
   # Should show only source files, no .o/.mod/.so
   ```

4. **Update documentation**
   - Add `--direct-c` section to README.md
   - Create CHANGELOG.md entry
   - Document known limitations in `docs/direct-c-compatibility.md`

5. **Add CI integration**
   - Create `.github/workflows/direct-c.yml`
   - Test on Linux (minimum)
   - Run subset of passing examples in CI
   - Optional: macOS

6. **Final review**
   - Verify f2py mode unchanged (run existing tests without --direct-c)
   - Run existing test suite (pytest test/)
   - Verify ≥50% example compatibility documented

### Final Target
- **~10-15 files changed** (vs current 38)
- **Core code only** (no generated artifacts)
- **Clean diff** (additive, not destructive)
- **Tests pass** (both f2py and direct-C)

## Risk Assessment

### Low Risk ✅
- Feature properly isolated behind `--direct-c` flag
- Existing f2py workflow completely preserved
- All existing tests restored and passing

### Medium Risk ⚠️
- +2,900 lines added to f90wrap/ package
  - Mitigation: Well-tested, isolated in separate modules
- Generated files in examples/ (if not removed)
  - Mitigation: Remove before merge
- Meson.build changes
  - Mitigation: Only adds new files to build, doesn't modify existing

### No Risk ❌
- Breaking existing f2py functionality (verified)
- Test regressions (all tests restored)

## Success Criteria

**For merge:**
- ✅ All existing files from master restored
- ⏳ **100% of applicable examples work with --direct-c** (zero regressions)
- ⏳ Compatibility matrix documented in docs/direct-c-compatibility.md (internal only)
  - List of all passing examples
  - List of skipped examples (with justification)
  - ZERO failing examples (all bugs fixed)
- ⏳ No generated artifacts in branch
- ⏳ **Minimal documentation:**
  - CLI help text includes --direct-c flag (already done in main.py)
  - 3-line CHANGELOG entry
  - **NO README changes, NO user guides**
- ⏳ CI passing (runs ALL passing examples in both modes)
- ⏳ Maintainer approval

**Measurement:**
```bash
# Should show ~10-15 files changed
git diff master --stat | tail -1

# Should show zero deletions (all additive)
git diff master --numstat | awk '$2 > 0 {print $3}'
# (Should be empty or only justified deletions in main.py)
```

## Next Steps

### Week 1: Achieve 100% Example Compatibility
**Day 1: Assessment**
1. Create automated test script for all examples
2. Run all examples with --direct-c flag
3. Document baseline: X passing, Y failing, Z skipped
4. Categorize failures by root cause
5. Create GitHub issues for each failure category

**Days 2-4: Fix All Failures**
6. Fix Category 1 failures (e.g., array handling)
7. Fix Category 2 failures (e.g., derived types)
8. Fix Category 3 failures (e.g., character handling)
9. Continue until ZERO failures remain
10. Add regression tests for each fix

**Day 5: Validation**
11. Re-run all examples, confirm 100% pass rate
12. Document results in docs/direct-c-compatibility.md (internal reference only)
13. Clean generated files from branch
14. Update PLAN.md with final statistics

### Week 2: Minimal Documentation and CI
15. **Skip README updates** (--help is sufficient)
16. Add 3-line CHANGELOG.md entry (just mention the flag exists)
17. Add CI job that runs ALL passing examples
18. Final review and testing
19. Create pull request

### Timeline Summary
- **Merge target:** End of Week 2
- **Prerequisite:** 100% example compatibility (zero regressions)
- **Acceptable exceptions:** Examples marked as "skip" with clear justification
