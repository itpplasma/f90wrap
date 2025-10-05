# Getter/Setter Implementation Evidence

## Summary

Implemented automatic generation of Fortran getter/setter routines for derived type elements in direct-C mode. The generated routines allow Python code to access and modify scalar elements of Fortran derived types through the C wrapper layer.

## Changes Made

### File: /home/ert/code/f90wrap/f90wrap/cwrapgen.py

Added three new functions:
1. `_is_scalar_element(element)` - Helper to identify scalar (non-array) elements
2. `_is_derived_type_element(element)` - Helper to identify nested derived type elements
3. Enhanced `generate_fortran_support()` - Now generates getter/setter routines for each scalar non-derived-type element

### Implementation Details

For each scalar element in a derived type, the generator now creates:
- **Getter routine**: `f90wrap_{typename}__get__{elementname}(self_ptr, value)`
  - Takes a c_ptr to the instance
  - Returns the element value via intent(out) parameter
  - Uses c_f_pointer to access Fortran data

- **Setter routine**: `f90wrap_{typename}__set__{elementname}(self_ptr, value)`
  - Takes a c_ptr to the instance and new value
  - Sets the element value via intent(in) parameter
  - Uses c_f_pointer to access Fortran data

## Testing Evidence

### Manual Test (derivedtypes example)

Created instance and tested getter/setters for all three element types:

```python
import _derivedtypes_directc

DT = _derivedtypes_directc.different_types
obj = DT()

# Test logical
obj.alpha = True
print(f"alpha = {obj.alpha}")  # Output: alpha = True

# Test integer
obj.beta = 42
print(f"beta = {obj.beta}")  # Output: beta = 42

# Test real
obj.delta = 3.14159
print(f"delta = {obj.delta}")  # Output: delta = 3.14159
```

**Result**: All getter/setter operations work correctly ✓

### Generated Code Example

From `derivedtypes_directc_support.f90`:

```fortran
subroutine f90wrap_different_types__get__alpha(self_ptr, value) &
    bind(C, name='__datatypes_MOD_f90wrap_different_types__get__alpha_')
    type(c_ptr), value :: self_ptr
    logical, intent(out) :: value
    type(different_types), pointer :: self

    call c_f_pointer(self_ptr, self)
    value = self%alpha
end subroutine f90wrap_different_types__get__alpha

subroutine f90wrap_different_types__set__alpha(self_ptr, value) &
    bind(C, name='__datatypes_MOD_f90wrap_different_types__set__alpha_')
    type(c_ptr), value :: self_ptr
    logical, intent(in) :: value
    type(different_types), pointer :: self

    call c_f_pointer(self_ptr, self)
    self%alpha = value
end subroutine f90wrap_different_types__set__alpha
```

### Symbol Verification

Confirmed generated symbols exist in compiled library:

```
$ nm derivedtypes_directc_support.o | grep __get__
000000000000fda0 T __datatypes_MOD_f90wrap_different_types__get__alpha_
000000000000fd59 T __datatypes_MOD_f90wrap_different_types__get__beta_
000000000000fd0d T __datatypes_MOD_f90wrap_different_types__get__delta_

$ nm derivedtypes_directc_support.o | grep __set__
000000000000fc7f T __datatypes_MOD_f90wrap_different_types__set__alpha_
000000000000fc38 T __datatypes_MOD_f90wrap_different_types__set__beta_
000000000000fbec T __datatypes_MOD_f90wrap_different_types__set__delta_
```

## Limitations

Current implementation handles:
- ✓ Scalar primitive types (logical, integer, real, complex)
- ✗ Array elements (requires separate array access infrastructure)
- ✗ Nested derived type elements (requires type-safe pointer wrapping)

## Files Modified

- `/home/ert/code/f90wrap/f90wrap/cwrapgen.py` - Added getter/setter generation logic

## Next Steps

This implementation resolves the missing getter/setter symbol errors. However, the full test suite still requires:
1. Kind map files (.f2py_f2cmap) for examples with kind parameters
2. Python wrapper compatibility fixes for direct-C calling conventions
3. Additional work on array and nested type element access

## Conclusion

The core getter/setter functionality is implemented and verified working. The generated Fortran routines correctly provide C-compatible access to derived type elements.
