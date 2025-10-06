# Direct-C Autonomous Execution Plan

## Mission
Achieve 100% Direct-C example compatibility through systematic implementation, testing, and debugging.

## Current State
- Branch: `feature/direct-c-clean`
- Phase 1 COMPLETE: 30 Direct-C tests passing (27 unit + 3 e2e)
- Infrastructure: directc.py, directc_cgen.py, numpy_utils.py all functional
- CLI: --direct-c flag wired and working

## Execution Strategy

All phases execute autonomously. Each step verifies success with concrete evidence before proceeding.

---

## PHASE 2: First Working Example

### Step 2.1: Select Simplest Example ✅

**Simplest examples identified (analysis complete):**
1. `issue32/test.f90` - 5 lines, 1 subroutine, scalar args only
2. `elemental/elemental_module.f90` - 13 lines, 1 elemental function

**SELECTED: `issue32/test.f90`**
- Single subroutine `foo(a,b)`
- `real(kind=8), intent(in) :: a`
- `integer :: b` (no intent = inout by default)
- No arrays, no derived types, no callbacks
- Simplest possible test case

### Step 2.2: Generate Direct-C for issue32

**Task:** Generate wrappers and fix any bugs in directc_cgen.py

**Commands:**
```bash
cd examples/issue32
f90wrap --direct-c test.f90 2>&1 | tee directc_gen.log
```

**Expected files:**
- `f90wrap_test.f90` - Fortran helper
- `mod.py` - Python wrapper
- `_test.c` - Direct-C extension

**Success criteria:**
- Exit code 0
- All 3 files exist
- No errors in directc_gen.log
- `_test.c` contains valid C syntax

**If fails:** Debug and fix directc_cgen.py, commit fix, retry

### Step 2.3: Compile Direct-C Extension

**Task:** Compile the C extension and link with Fortran helper

**Create:** `examples/issue32/build_directc.sh`
```bash
#!/bin/bash
set -e

# Compile Fortran sources
gfortran -c -fPIC test.f90 -o test.o
gfortran -c -fPIC f90wrap_test.f90 -o f90wrap_test.o

# Compile C extension
gcc -shared -fPIC _test.c test.o f90wrap_test.o \
    -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
    -I$(python3 -c "import numpy; print(numpy.get_include())") \
    -lgfortran \
    -o _test$(python3-config --extension-suffix)

echo "Build successful"
ls -lh _test*.so
```

**Commands:**
```bash
chmod +x build_directc.sh
./build_directc.sh 2>&1 | tee build.log
```

**Success criteria:**
- Exit code 0
- `_test.*.so` file exists
- No compilation errors
- No undefined symbol errors

**If fails:**
- Check C code generation (extern declarations, function signatures)
- Check Fortran name mangling (lowercase vs uppercase, trailing underscore)
- Fix directc_cgen.py, rebuild, retry

### Step 2.4: Functional Test

**Task:** Import and call the extension

**Create:** `examples/issue32/test_directc_func.py`
```python
#!/usr/bin/env python3
import sys
import test  # The Direct-C extension

# Test calling foo
try:
    test.foo(3.14, 42)
    print("SUCCESS: Direct-C call completed")
    sys.exit(0)
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
```

**Commands:**
```bash
python3 test_directc_func.py
```

**Success criteria:**
- Exit code 0
- Prints "SUCCESS"
- No Python errors
- Fortran print statement executes

**If fails:**
- Check argument parsing in C wrapper
- Check helper call signature
- Check return value handling
- Fix directc_cgen.py, rebuild, retry

---

## PHASE 3: Systematic Validation

### Step 3.1: Port Compatibility Script

**Task:** Create automated test harness for all examples

**Source:** `git show feature/direct-c-generation:test_direct_c_compatibility.py`

**Adapt for helpers-only:**
- Remove BIND(C) logic
- Always use helper wrappers
- Add build step for each example
- Track pass/fail/skip per example

**Create:** `test_direct_c_compatibility.py` in project root

**Key functions:**
```python
def test_example(example_dir):
    """
    1. Run f90wrap --direct-c
    2. Compile Fortran + C extension
    3. Import and test (if test.py exists)
    4. Return: PASS/FAIL/SKIP + error details
    """

def main():
    """
    - Scan examples/ directory
    - Test each with Direct-C
    - Generate compatibility_report.md
    - Generate compatibility_results.json
    - Print summary table
    """
```

**Commands:**
```bash
python3 test_direct_c_compatibility.py 2>&1 | tee compat_test.log
```

**Success criteria:**
- Script runs to completion
- `compatibility_report.md` created
- `compatibility_results.json` created
- At least issue32 shows PASS

### Step 3.2: Fix Failing Examples

**Strategy:** Iterate through failures, fix root causes

**For each FAIL:**

1. **Reproduce:**
   ```bash
   cd examples/<failing>
   f90wrap --direct-c *.f90
   # Check error
   ```

2. **Diagnose category:**
   - **Generation error** → Bug in directc_cgen.py (e.g. type mapping, array handling)
   - **Compilation error** → Signature mismatch, name mangling, missing declarations
   - **Runtime error** → Array strides, memory management, type conversion
   - **Wrong output** → NumPy conversion bug, Fortran call convention issue

3. **Fix:**
   - Update directc_cgen.py or numpy_utils.py
   - Add test case to test/test_directc.py if new functionality
   - Commit fix with clear message

4. **Verify:**
   - Re-run compatibility script
   - Confirm example now PASS
   - Ensure no regressions

5. **Repeat** until 100% of examples that work with f2py also work with Direct-C

**Common issues to fix:**
- Arrays with assumed-shape dimensions
- Character string length parameters
- Intent(out) argument handling
- Function return values
- Array-of-derived-types
- Optional arguments
- Assumed-size arrays (dimension(*))

**Target:** All f2py-compatible examples also pass Direct-C

---

## PHASE 4: Finalization

### Step 4.1: Cleanup

**Tasks:**
- Remove any test artifacts from examples/
- Clean up build scripts (keep only if documented)
- Verify no generated files committed

**Commands:**
```bash
git status examples/
# If any generated files:
git clean -fdx examples/
```

### Step 4.2: Final Validation

**Run full test suite:**
```bash
pytest test/ -v
python3 test_direct_c_compatibility.py
```

**Verify:**
- All pytest tests pass (including 30 Direct-C tests)
- Compatibility script shows 100% (or documented known limitations)
- No regressions in normal mode (without --direct-c)

### Step 4.3: Documentation

**Update CHANGELOG.md:**
```markdown
## [Unreleased]

### Added
- Direct-C code generation via `--direct-c` flag
- Generates C extensions calling f90wrap helpers (no f2py dependency)
- 100% compatibility with f2py for supported examples

### Implementation
- `f90wrap/directc.py`: Procedure classification
- `f90wrap/directc_cgen.py`: C code generator
- `f90wrap/numpy_utils.py`: Type mapping utilities
- 30 new tests (27 unit + 3 e2e)

### Compatibility
- X/Y examples pass (Y% compatibility)
- Known limitations: [list any]
```

**Update README (optional, per CLAUDE.md minimal docs):**
Brief usage example if requested

### Step 4.4: Create Pull Request

**Commands:**
```bash
git push origin feature/direct-c-clean

gh pr create \
  --title "Add Direct-C code generation (helpers-only)" \
  --body "$(cat <<'EOF'
## Summary
Implements Direct-C code generation as an alternative to f2py, eliminating the f2py dependency while maintaining 100% API compatibility.

## Implementation
- **Helpers-only approach**: All C wrappers call existing `f90wrap_<module>__<proc>` Fortran helpers
- **No BIND(C)**: Avoids ISO C interoperability constraints
- **Drop-in replacement**: Generated Python modules have identical API to f2py versions

## Compatibility Results
- X/Y examples passing (Y%)
- [Link to compatibility_report.md]

## Testing
- 30 new Direct-C tests (all passing)
- No regressions in existing tests
- Systematic validation across example suite

## Usage
```bash
# Generate Direct-C wrapper instead of f2py
f90wrap --direct-c mymodule.f90

# Compile (user's toolchain)
gfortran -c -fPIC f90wrap_*.f90
gcc -shared -fPIC _*.c *.o -lgfortran -o _mymodule.so
```

## Closes
Related to #[issue] (if any)
EOF
)" \
  --base master \
  --head feature/direct-c-clean
```

---

## Success Criteria (Definition of Done)

- [x] Phase 1: Foundation tests (30 tests passing)
- [ ] Phase 2: First working example (issue32 compiles and runs)
- [ ] Phase 3: Systematic validation (compatibility script complete, X% passing)
- [ ] Phase 4: Finalization (PR created, CI green)

## Execution Notes

- **Autonomous mode**: Execute all steps without user intervention
- **Evidence required**: Every success claim must have concrete proof (test output, build logs, file existence)
- **Commit frequently**: Each significant fix is its own commit
- **No shortcuts**: No stubs, no placeholders, complete implementation only

## Current Blocker

**NONE** - All information gathered, ready for autonomous execution.

## Next Action

Execute Phase 2.1-2.4 sequentially, then Phase 3.1-3.2, then Phase 4.1-4.4.
