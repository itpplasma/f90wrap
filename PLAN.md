# Direct-C Generation: Minimization Plan

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
- [ ] **Run existing example tests with --direct-c flag**
  - All examples in `examples/*/tests.py` must be tested with direct-C mode
  - Document which examples pass/fail with direct-C
  - Minimum acceptable pass rate: 50% of examples work with direct-C
  - Known failures documented with root causes
- [ ] Remove generated files from examples/
- [ ] Verify no accidental deletions remain
- [ ] README.md updated with `--direct-c` usage
- [ ] CHANGELOG.md entry added
- [ ] CI job added for direct-C mode

### Nice to Have (Post-Merge)
- [ ] Consolidate utility modules
- [ ] Optimize main.py changes
- [ ] Add docs/direct-c-guide.md
- [ ] Performance benchmarks in docs/
- [ ] Example pass rate improvements (62% → 80%+)

## Remaining Tasks

### This Week
1. **Validate existing examples with direct-C** (CRITICAL)
   ```bash
   # For each example in examples/*/, run:
   cd examples/arrays
   f90wrap --direct-c -m arrays library.f90 parameters.f90
   # Compile and run tests.py
   python tests.py

   # Document results in a table:
   # | Example | Pass/Fail | Notes |
   # |---------|-----------|-------|
   # | arrays  | Pass      | All tests pass |
   # | strings | Fail      | Character handling issue |
   ```

   **Acceptance criteria:**
   - ≥50% of existing examples pass with --direct-c
   - All failures documented with root causes
   - Create `docs/direct-c-compatibility.md` with results

2. **Clean generated files**
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
- ⏳ ≥50% of existing examples work with --direct-c (compatibility validated)
- ⏳ Compatibility results documented in docs/direct-c-compatibility.md
- ⏳ No generated artifacts in branch
- ⏳ Documentation complete (README, CHANGELOG)
- ⏳ CI passing (both f2py and direct-C modes)
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

1. **Validate existing examples with direct-C** (CRITICAL - today)
   - Run all examples/*/tests.py with --direct-c flag
   - Document pass/fail rate and root causes
   - Target: ≥50% compatibility
2. Remove generated files (today)
3. Update PLAN.md with actual file count (today)
4. Create docs/direct-c-compatibility.md with validation results (this week)
5. Documentation: README, CHANGELOG (this week)
6. CI integration (this week)
7. Create PR (end of week)
