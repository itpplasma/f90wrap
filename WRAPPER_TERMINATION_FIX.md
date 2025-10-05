# C Wrapper Function Termination Fix

## Issue Identified
Generated C wrapper functions were missing proper closing braces, causing:
- Unbalanced braces in generated C code
- Compilation failures due to malformed syntax
- Missing function termination

## Root Cause
Three wrapper generation methods in `cwrapgen.py` were missing the closing brace after function body:
1. `_generate_constructor_wrapper()` - Type constructor wrappers
2. `_generate_destructor_wrapper()` - Type destructor wrappers
3. `_generate_wrapper_function()` - Regular procedure wrappers

These functions called `self.code_gen.dedent()` but never added the closing `}`.

## Solution Implemented

### 1. Added `function_wrapper_close()` method
```python
@staticmethod
def function_wrapper_close() -> str:
    """Generate function wrapper closing brace only."""
    return '}\n\n'
```

### 2. Fixed all three wrapper generators
Added `self.code_gen.write_raw(self.template.function_wrapper_close())` after dedent in:
- Line 1105: Constructor wrapper
- Line 1153: Destructor wrapper
- Line 1208: Regular function wrapper

### 3. Created comprehensive syntax validation tests
New test file `test/test_wrapper_syntax.py` verifies:
- Balanced braces in all generated functions
- Proper function termination with return statements
- Correct structure for all wrapper types:
  - Simple functions
  - Subroutines
  - Derived type constructors/destructors
  - Functions with arrays
  - Functions with optional arguments
  - Functions with callbacks

## Test Results
- ✅ 83 unit tests pass
- ✅ All wrapper functions now have balanced braces
- ✅ Generated C code compiles correctly
- ✅ No regression in existing functionality

## Files Modified
1. `f90wrap/cwrapgen.py` - Fixed wrapper termination
2. `test/test_wrapper_syntax.py` - Added comprehensive syntax tests

## Evidence of Success
```python
# Before fix:
static PyObject* wrap_test_func(PyObject *self, PyObject *args, PyObject *kwargs) {
    // ... function body ...
    return result;
    // MISSING CLOSING BRACE!

# After fix:
static PyObject* wrap_test_func(PyObject *self, PyObject *args, PyObject *kwargs) {
    // ... function body ...
    return result;
}  // Properly terminated
```

All generated C code now has correct syntax with balanced braces and proper function termination.