# Direct-C Mode Bug Fix Plan

**Date:** 2025-10-05
**Branch:** feature/direct-c-generation
**Current Success Rate:** 56.0% (28/50 examples)
**Target Success Rate:** 78%+ (39/50 examples)

## Priority 1: Support File Module Dependencies (9 failures → HIGH IMPACT)

### Bug Description
Support files missing imports for kind parameters used in getter/setter type declarations.

**Example Error:**
```fortran
Error: Symbol 'idp' at (1) has no IMPLICIT type
```

**Root Cause:**
When generating getter/setter routines, the support file uses type declarations like `real(idp)` but doesn't import the kind parameter `idp` from the parameters module.

**Current Code (cwrapgen.py:1765-1769):**
```python
# Use statements for all modules with types
for module in modules_with_types:
    fortran_lines.append(f"    use {module.name}")

fortran_lines.append("    use iso_c_binding")
```

**Fix Strategy:**
Add transitive dependencies from each module to the support file imports.

**Implementation:**
1. For each module with types, also import modules it depends on
2. Option A: Use module.uses attribute to get dependencies
3. Option B: Simpler - collect ALL modules from AST and import them (safe and simple)

**Affected Examples:** errorbinding, issue254_getter, issue258_derived_type_attributes, issue41_abstract_classes, keyword_renaming_issue160, long_subroutine_name, mod_arg_clash, optional_derived_arrays, type_bn

**Expected Impact:** +18% success rate (9/50)

---

## Priority 2: Variable Name Collision (2 failures → HIGH IMPACT)

### Bug Description
Variable `self` declared twice in same C function scope.

**Example Error:**
```c
error: 'self' redeclared as different kind of symbol
  532 |     void* self;
```

**Root Cause:**
C wrapper function uses `self` as PyObject* parameter name, then declares local `void* self` variable.

**Current Pattern:**
```c
static PyObject* wrap_asum(PyObject *self, PyObject *args, PyObject *kwargs) {
    void* self;  // ← ERROR: redeclaration
    ...
}
```

**Fix Strategy:**
Rename local Fortran object pointer variable to avoid collision.

**Implementation:**
Search for pattern in C code generation where we declare `void* self` and rename to `void* fortran_self` or `void* f90_obj`.

**Location:** cwrapgen.py - function wrapper generation

**Affected Examples:** derivedtypes_procedure, docstring (likely)

**Expected Impact:** +4% success rate (2/50)

---

## Priority 3: Module Name Sanitization (1 failure → MEDIUM IMPACT)

### Bug Description
Module names containing hyphens create invalid C identifiers.

**Example Error:**
```c
error: expected '=', ',', ';', 'asm' or '__attribute__' before '-' token
  595 | static struct PyModuleDef _derived-type-aliases_directc_module = {
```

**Root Cause:**
Hyphens are valid in directory/file names but invalid in C identifiers.

**Fix Strategy:**
Sanitize module names by replacing hyphens with underscores when generating C code.

**Implementation:**
Add sanitization function: `module_name.replace('-', '_')` in C code generation paths.

**Location:** cwrapgen.py - multiple locations where module name is used in C identifiers

**Affected Examples:** derived-type-aliases

**Expected Impact:** +2% success rate (1/50)

---

## Priority 4: Callback Symbol Linking (1 failure → MEDIUM IMPACT)

### Bug Description
Callback wrapper symbols not found during Python import.

**Example Error:**
```
undefined symbol: pyfunc_print_
```

**Root Cause:**
Callback wrappers not being emitted in direct-C mode, or not properly linked.

**Fix Strategy:**
Ensure callback wrapper generation works in direct-C mode similar to f2py mode.

**Investigation Needed:**
1. Check if callback wrappers are being generated
2. Verify they're included in the build
3. Ensure proper symbol naming/mangling

**Location:** cwrapgen.py - callback handling code

**Affected Examples:** callback_print_function_issue93

**Expected Impact:** +2% success rate (1/50)

---

## Priority 5: Additional C Compile Failures (2 failures → MEDIUM IMPACT)

### Bug Description
Two examples fail C compilation with unclear errors.

**Affected Examples:** issue235_allocatable_classes, optional_string

**Fix Strategy:**
Investigate detailed error messages for each case.

**Expected Impact:** +4% success rate (2/50)

---

## Priority 6: F90wrap Transformation Failure (1 failure → LOW IMPACT)

### Bug Description
Internal exception during code generation transformation.

**Affected Examples:** fixed_1D_derived_type_array_argument

**Fix Strategy:**
Debug the transformation pipeline for this specific edge case.

**Expected Impact:** +2% success rate (1/50)

---

## Out of Scope: Fortran Source Code Issues (2 failures)

**Affected Examples:** cylinder, fortran_oo

These are pre-existing bugs in the example Fortran source code (type mismatches), not direct-C backend issues. Can be addressed separately or examples excluded from validation.

---

## Implementation Order

1. **Fix support file dependencies** (Priority 1)
   - File: `f90wrap/cwrapgen.py`
   - Function: `generate_fortran_support()`
   - Change: Import all modules from AST, not just modules with types

2. **Fix variable name collision** (Priority 2)
   - File: `f90wrap/cwrapgen.py`
   - Search for: `void* self` declarations in function wrappers
   - Change: Rename to `void* fortran_self`

3. **Test and verify** 78% success rate achieved

4. **Continue with medium-priority fixes** if time permits

---

## Success Criteria

After implementing Priority 1 and 2 fixes:
- Success rate ≥ 78% (39/50 examples passing)
- All support_compile_fail examples resolved (9 → 0)
- All variable redeclaration errors resolved (2 → 0)
- Evidence: Re-run test_all_examples_direct_c.py and capture results
