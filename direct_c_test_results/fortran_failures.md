# Fortran Compilation Diagnostics

Automatically captured compiler output for parity planning.

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

