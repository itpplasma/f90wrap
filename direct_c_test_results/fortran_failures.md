# Fortran Compilation Diagnostics

Automatically captured compiler output for parity planning.

## fortran_oo

### Compiler Output

````text
f90wrap_fortran_oo_direct.f90:185:55:

  185 |     call f90wrap_m_geometry__obj_name__binding__circle(obj_ptr%p%obj)
      |                                                       1
Error: Explicit interface required for polymorphic argument at (1)
f90wrap_fortran_oo_direct.f90:352:45:

  352 |     call f90wrap_m_geometry__circle_obj_name(obj_ptr%p%obj)
      |                                             1
Error: Explicit interface required for polymorphic argument at (1)
f90wrap_main-oo_pp.f90:4
````

### Module Dependencies

main-oo_pp.f90: defines m_geometry, uses ['m_base_poly']; f90wrap_main-oo_pp.f90: uses ['m_geometry']; base_poly_pp.f90: defines m_base_poly; f90wrap_base_poly_pp.f90: uses ['m_base_poly']; f90wrap_fortran_oo_direct.f90: uses ['m_geometry']

## issue258_derived_type_attributes

### Compiler Output

````text
f90wrap_dta_cc.f90:152:45:

  152 |     call t_inner_print(inner=inner_ptr%p%obj)
      |                                             1
Error: Type mismatch in argument ‘inner’ at (1); passed CLASS(t_inner) to CLASS(t_inner)
f90wrap_dta_ct.f90:43:23:

   43 |         class(t_inner), allocatable :: obj
      |                       1
Error: Type name ‘t_inner’ at (1) is ambiguous
f90wrap_dta_ct.f90:52:24:

   52 |     ret_inner_ptr%p%obj = t_inner(value=value)
      |                        1
Err
````

### Module Dependencies

dta_ct_pp.f90: defines dta_ct; dta_cc_pp.f90: defines dta_cc; dta_tt_pp.f90: defines dta_tt; dta_tc_pp.f90: defines dta_tc; f90wrap_dta_cc.f90: uses ['dta_tc', 'dta_cc']; f90wrap_dta_ct.f90: uses ['dta_tc', 'dta_ct']; f90wrap_dta_tc.f90: uses ['dta_tc']; f90wrap_dta_tt.f90: uses ['dta_tc']

## kind_map_default

### Compiler Output

````text
f90wrap_main.f90:27:18:

   27 |     ret_out_int = test_real8(in_real=in_real)
      |                  1
Error: Type mismatch in argument ‘in_real’ at (1); passed REAL(4) to REAL(8)

````

### Module Dependencies

main_pp.f90: defines m_test; f90wrap_main.f90: uses ['m_test']

## type_check

### Compiler Output

````text
f90wrap_main.f90:275:17:

  275 |     ret_output = in_scalar(input=input)
      |                 1
Error: There is no specific function for the generic ‘in_scalar’ at (1)
f90wrap_main.f90:322:17:

  322 |     ret_output = in_scalar(input=input)
      |                 1
Error: There is no specific function for the generic ‘in_scalar’ at (1)

````

### Module Dependencies

main_pp.f90: defines m_type_test; f90wrap_main.f90: uses ['m_type_test']

