# F90wrap Direct-C Compatibility Report

Generated: 2025-10-05T15:48:34.113032

## Summary
- Total Examples: 50
- ✅ Passed: 8 (16.0%)
- ❌ Failed: 40 (80.0%)
- ⊘ Skipped: 2 (4.0%)

## Results by Example

### ✅ elemental
- **Status**: PASS
- **Notes**:
  - Using preprocessed files: ['elemental_module.fpp']
  - Command: f90wrap -m elemental_direct /tmp/tmpqvyko3p3/elemental_module.fpp -k /tmp/tmpqvyko3p3/kind_map --direct-c -v
  - Generated C files: ['_elemental_directmodule.c']
  - No tests.py found, but compilation succeeded

### ✅ issue105_function_definition_with_empty_lines
- **Status**: PASS
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m issue105_function_definition_with_empty_lines_direct /tmp/tmpl1n6ci3_/main.fpp  --direct-c -v
  - Generated C files: ['_issue105_function_definition_with_empty_lines_directmodule.c']
  - No tests.py found, but compilation succeeded

### ✅ issue206_subroutine_oldstyle
- **Status**: PASS
- **Notes**:
  - Using source files: ['subroutine_oldstyle.f']
  - Command: f90wrap -m issue206_subroutine_oldstyle_direct /tmp/tmpmabrt_by/subroutine_oldstyle.f  --direct-c -v
  - Generated C files: ['_issue206_subroutine_oldstyle_directmodule.c']
  - No tests.py found, but compilation succeeded

### ✅ issue32
- **Status**: PASS
- **Notes**:
  - Using preprocessed files: ['test.fpp']
  - Command: f90wrap -m issue32_direct /tmp/tmp7jono7jl/test.fpp -k /tmp/tmp7jono7jl/kind_map --direct-c -v
  - Generated C files: ['_issue32_directmodule.c']
  - No tests.py found, but compilation succeeded

### ✅ optional_args_issue53
- **Status**: PASS
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m optional_args_issue53_direct /tmp/tmpclva1bx0/main.fpp  --direct-c -v
  - Generated C files: ['_optional_args_issue53_directmodule.c']
  - No tests.py found, but compilation succeeded

### ✅ output_kind
- **Status**: PASS
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m output_kind_direct /tmp/tmpmnkyln2b/main.fpp  --direct-c -v
  - Generated C files: ['_output_kind_directmodule.c']
  - No tests.py found, but compilation succeeded

### ✅ string_array_input_f2py
- **Status**: PASS
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m string_array_input_f2py_direct /tmp/tmp8odku62f/main.fpp  --direct-c -v
  - Generated C files: ['_string_array_input_f2py_directmodule.c']
  - No tests.py found, but compilation succeeded

### ✅ subroutine_contains_issue101
- **Status**: PASS
- **Notes**:
  - Using preprocessed files: ['test.fpp']
  - Command: f90wrap -m subroutine_contains_issue101_direct /tmp/tmpaihn7vm0/test.fpp  --direct-c -v
  - Generated C files: ['_subroutine_contains_issue101_directmodule.c']
  - No tests.py found, but compilation succeeded

### ❌ arrayderivedtypes
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['test.fpp']
  - Command: f90wrap -m arrayderivedtypes_direct /tmp/tmp1p311zmm/test.fpp -k /tmp/tmp1p311zmm/kind_map --direct-c -v
  - Generated C files: ['_arrayderivedtypes_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmp1p311zmm/arrayderivedtypes_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmp1p311zmm/arrayderivedtypes_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmp1p311zmm/arrayderivedtypes_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmp1p311zmm/arrayderivedtypes_direct_support.o:(.bss+0x8): first defined here
/usr/bin/ld: /tmp/tmp1p311zmm

### ❌ arrays
- **Status**: FAIL
- **Error Category**: test_execution_failed
- **Notes**:
  - Using preprocessed files: ['library.fpp', 'parameters.fpp']
  - Command: f90wrap -m arrays_direct /tmp/tmpoixont_k/library.fpp /tmp/tmpoixont_k/parameters.fpp -k /tmp/tmpoixont_k/kind_map --direct-c -v
  - Generated C files: ['_arrays_directmodule.c', '_ExampleArraymodule.c']
  - Modified tests.py to use direct-c module
  - Test failed with return code 1

### ❌ arrays_fixed
- **Status**: FAIL
- **Error Category**: test_execution_failed
- **Notes**:
  - Using source files: ['parameters.f', 'library.f']
  - Command: f90wrap -m arrays_fixed_direct /tmp/tmp4srh5fvf/parameters.f /tmp/tmp4srh5fvf/library.f -k /tmp/tmp4srh5fvf/kind_map --direct-c -v
  - Generated C files: ['_arrays_fixed_directmodule.c']
  - Modified tests.py to use direct-c module
  - Test failed with return code 1

### ❌ arrays_in_derived_types_issue50
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['test.fpp']
  - Command: f90wrap -m arrays_in_derived_types_issue50_direct /tmp/tmptpxz_xng/test.fpp -k /tmp/tmptpxz_xng/kind_map --direct-c -v
  - Generated C files: ['_arrays_in_derived_types_issue50_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmptpxz_xng/arrays_in_derived_types_issue50_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmptpxz_xng/arrays_in_derived_types_issue50_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmptpxz_xng/arrays_in_derived_types_issue50_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmptpxz_xng/arrays_in_derived_types_issue50_direct_support.o:(.bs

### ❌ auto_raise_error
- **Status**: FAIL
- **Error Category**: test_execution_failed
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m auto_raise_error_direct /tmp/tmpa6u0j_aw/main.fpp  --direct-c -v
  - Generated C files: ['_auto_raise_error_directmodule.c']
  - Modified tests.py to use direct-c module
  - Test failed with return code 1

### ❌ callback_print_function_issue93
- **Status**: FAIL
- **Error Category**: test_execution_failed
- **Notes**:
  - Using preprocessed files: ['cback.fpp', 'caller.fpp']
  - Command: f90wrap -m callback_print_function_issue93_direct /tmp/tmphj6ej7ch/cback.fpp /tmp/tmphj6ej7ch/caller.fpp -k /tmp/tmphj6ej7ch/kind_map --direct-c -v
  - Generated C files: ['_callback_print_function_issue93_directmodule.c']
  - Modified tests.py to use direct-c module
  - Test failed with return code 1

### ❌ class_names
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['test.fpp']
  - Command: f90wrap -m class_names_direct /tmp/tmpqsh_svfg/test.fpp -k /tmp/tmpqsh_svfg/kind_map --direct-c -v
  - Generated C files: ['_class_names_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmpqsh_svfg/class_names_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmpqsh_svfg/class_names_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmpqsh_svfg/class_names_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmpqsh_svfg/class_names_direct_support.o:(.bss+0x8): first defined here
/usr/bin/ld: /tmp/tmpqsh_svfg/class_names_support.o:(

### ❌ cylinder
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['cyldnad.fpp', 'DNAD.fpp']
  - Command: f90wrap -m cylinder_direct /tmp/tmpoecdsf7o/cyldnad.fpp /tmp/tmpoecdsf7o/DNAD.fpp -k /tmp/tmpoecdsf7o/kind_map --direct-c -v
  - Generated C files: ['_cylinder_directmodule.c']
  - Failed to compile cylinder_direct_support.f90: /tmp/tmpoecdsf7o/cylinder_direct_support.f90:6:9:

    6 |     use mcyldnad
      |         1
Fatal Error: Cannot open module file ‘mcyldnad.mod’ for reading at (1): No such file or directory
compilation terminated.


### ❌ default_i8
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['test.fpp']
  - Command: f90wrap -m default_i8_direct /tmp/tmp85y8ejkg/test.fpp -k /tmp/tmp85y8ejkg/kind_map --direct-c -v
  - Generated C files: ['_default_i8_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmp85y8ejkg/default_i8_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmp85y8ejkg/default_i8_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmp85y8ejkg/default_i8_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmp85y8ejkg/default_i8_direct_support.o:(.bss+0x8): first defined here
/usr/bin/ld: /tmp/tmp85y8ejkg/default_i8_support.o:(.data

### ❌ derived-type-aliases
- **Status**: FAIL
- **Error Category**: c_compilation_failed
- **Notes**:
  - Using preprocessed files: ['othertype_mod.fpp', 'mytype_mod.fpp']
  - Command: f90wrap -m derived-type-aliases_direct /tmp/tmp3bo4d2s1/othertype_mod.fpp /tmp/tmp3bo4d2s1/mytype_mod.fpp -k /tmp/tmp3bo4d2s1/kind_map --direct-c -v
  - Generated C files: ['_derived-type-aliases_directmodule.c']
  - Failed to compile _derived-type-aliases_directmodule.c: /tmp/tmp3bo4d2s1/_derived-type-aliases_directmodule.c:522:13: error: redefinition of ‘wrap_plus_b__doc__’
  522 | static char wrap_plus_b__doc__[] = "Wrapper for plus_b";
      |             ^~~~~~~~~~~~~~~~~~
/tmp/tmp3bo4d2s1/_derived-type-aliases_directmodule.c:462:13: note: previous definition of ‘wrap_plus_b__doc__’ with type ‘char[19]’
  462 | static char wrap_plus_b__doc__[] = "Wrapper for plus_b";
      |             ^~~~~~~~~~~~~~~~~~
/tmp/tmp3bo4d2s1/_derived-type-aliases_directmodule.c

### ❌ derivedtypes
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['datatypes.fpp', 'library.fpp', 'parameters.fpp']
  - Command: f90wrap -m derivedtypes_direct /tmp/tmpb4bsqwyz/datatypes.fpp /tmp/tmpb4bsqwyz/library.fpp /tmp/tmpb4bsqwyz/parameters.fpp -k /tmp/tmpb4bsqwyz/kind_map --direct-c -v
  - Generated C files: ['_derivedtypes_directmodule.c']
  - Failed to compile derivedtypes_direct_support.f90: /tmp/tmpb4bsqwyz/derivedtypes_direct_support.f90:6:9:

    6 |     use datatypes_allocatable
      |         1
Fatal Error: Cannot open module file ‘datatypes_allocatable.mod’ for reading at (1): No such file or directory
compilation terminated.


### ❌ derivedtypes_procedure
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['library.fpp']
  - Command: f90wrap -m derivedtypes_procedure_direct /tmp/tmpvkqy3h_u/library.fpp  --direct-c -v
  - Generated C files: ['_derivedtypes_procedure_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmpvkqy3h_u/derivedtypes_procedure_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmpvkqy3h_u/derivedtypes_procedure_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmpvkqy3h_u/derivedtypes_procedure_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmpvkqy3h_u/derivedtypes_procedure_direct_support.o:(.bss+0x8): first defined here
/usr/bin/

### ❌ docstring
- **Status**: FAIL
- **Error Category**: c_compilation_failed
- **Notes**:
  - Using preprocessed files: ['main.fpp', 'f90wrap_main.fpp']
  - Command: f90wrap -m docstring_direct /tmp/tmp6lk1k0p2/main.fpp /tmp/tmp6lk1k0p2/f90wrap_main.fpp  --direct-c -v
  - Generated C files: ['_docstring_directmodule.c']
  - Failed to compile _docstring_directmodule.c: /tmp/tmp6lk1k0p2/_docstring_directmodule.c:297:240: warning: unknown escape sequence: ‘\p’
  297 | static char wrap_doc_inside__doc__[] = "=========================================================================== >  \brief Doc inside  \param[in,out] circle      t_circle to initialize  \param[in]     radius      radius of the circle <";
      |                                                                                                                                                         

### ❌ errorbinding
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['datatypes.fpp', 'parameters.fpp']
  - Command: f90wrap -m errorbinding_direct /tmp/tmpwu_tlvnl/datatypes.fpp /tmp/tmpwu_tlvnl/parameters.fpp -k /tmp/tmpwu_tlvnl/kind_map --direct-c -v
  - Generated C files: ['_errorbinding_directmodule.c']
  - Failed to compile errorbinding_direct_support.f90: /tmp/tmpwu_tlvnl/errorbinding_direct_support.f90:6:9:

    6 |     use datatypes
      |         1
Fatal Error: Cannot open module file ‘datatypes.mod’ for reading at (1): No such file or directory
compilation terminated.


### ❌ extends
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['testextends.fpp']
  - Command: f90wrap -m extends_direct /tmp/tmprt5g7o4t/testextends.fpp  --direct-c -v
  - Generated C files: ['_extends_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmprt5g7o4t/extends_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmprt5g7o4t/extends_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmprt5g7o4t/extends_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmprt5g7o4t/extends_direct_support.o:(.bss+0x8): first defined here
/usr/bin/ld: /tmp/tmprt5g7o4t/extends_support.o:(.data.rel+0x0): mult

### ❌ fixed_1D_derived_type_array_argument
- **Status**: FAIL
- **Error Category**: attribute_error
- **Notes**:
  - Using preprocessed files: ['functions.fpp']
  - Command: f90wrap -m fixed_1D_derived_type_array_argument_direct /tmp/tmp0j5u_d5m/functions.fpp  --direct-c -v
  - f90wrap failed with return code 1
- **Error Output**:
```
Traceback (most recent call last):
  File "/home/ert/code/.venv/lib/python3.13/site-packages/f90wrap/scripts/main.py", line 357, in main
    tree = tf.transform_to_generic_wrapper(tree,
                                           types,
    ...<11 lines>...
                                           remove_optional_arguments,
                                           force_public=force_public)
  File "/home/ert/code/.venv/lib/python3.13/site-packages/f90wrap/transform.py", line 1399, in transform_to_generic_wrapper
    tree = add_missing_constructors(tree)
  File "/home/ert/code/.venv/lib/python3.13/site-packages/f90wrap/transform.py", line 865, in add_missing_constructors
    if 'abstract' in node.attributes:
                     ^^^^^^^^^^^^^^^
AttributeError: 'Type' object has no attribute 'attributes'
f90wrap: AttributeError("'Type' object has no attribute 'attributes'")
         for help use --help
Traceback (most recent call last):
  File "/home/ert/code/.venv/bin/f90wrap", line 
... (truncated)
```

### ❌ fortran_oo
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['main-oo.fpp', 'f90wrap_main-oo.fpp', 'base_poly.fpp', 'f90wrap_base_poly.fpp']
  - Command: f90wrap -m fortran_oo_direct /tmp/tmpw4_i6hab/main-oo.fpp /tmp/tmpw4_i6hab/f90wrap_main-oo.fpp /tmp/tmpw4_i6hab/base_poly.fpp /tmp/tmpw4_i6hab/f90wrap_base_poly.fpp  --direct-c -v
  - Generated C files: ['_fortran_oo_directmodule.c']
  - Failed to compile fortran_oo_direct_support.f90: /tmp/tmpw4_i6hab/fortran_oo_direct_support.f90:117:37:

  117 |         type(rectangle), pointer :: self
      |                                     1~~~
Error: ‘self’ at (1) is of the ABSTRACT type ‘rectangle’
/tmp/tmpw4_i6hab/fortran_oo_direct_support.f90:108:37:

  108 |         type(rectangle), pointer :: self
      |                                     1~~~
Error: ‘self’ at (1) is of the ABSTRACT type ‘rectangle’
/tmp/tmpw4_i6hab/fortran_oo_direct_support.f90:99:37:

   99 |         type(re

### ❌ intent_out_size
- **Status**: FAIL
- **Error Category**: test_execution_failed
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m intent_out_size_direct /tmp/tmp0jrymjvx/main.fpp  --direct-c -v
  - Generated C files: ['_intent_out_size_directmodule.c']
  - Modified tests.py to use direct-c module
  - Test failed with return code 1

### ❌ interface
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['example.fpp']
  - Command: f90wrap -m interface_direct /tmp/tmpzcxyu5co/example.fpp  --direct-c -v
  - Generated C files: ['_interface_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmpzcxyu5co/interface_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmpzcxyu5co/interface_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmpzcxyu5co/interface_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmpzcxyu5co/interface_direct_support.o:(.bss+0x8): first defined here
/usr/bin/ld: /tmp/tmpzcxyu5co/interface_support.o:(.data.rel+

### ❌ issue227_allocatable
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['alloc_output.fpp']
  - Command: f90wrap -m issue227_allocatable_direct /tmp/tmp2u6586km/alloc_output.fpp  --direct-c -v
  - Generated C files: ['_issue227_allocatable_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmp2u6586km/issue227_allocatable_directc_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmp2u6586km/issue227_allocatable_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmp2u6586km/issue227_allocatable_directc_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmp2u6586km/issue227_allocatable_direct_support.o:(.bss+0x8): first defined here
/

### ❌ issue235_allocatable_classes
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['myclass_factory.fpp', 'mytype.fpp', 'myclass.fpp']
  - Command: f90wrap -m issue235_allocatable_classes_direct /tmp/tmpcmq07byx/myclass_factory.fpp /tmp/tmpcmq07byx/mytype.fpp /tmp/tmpcmq07byx/myclass.fpp  --direct-c -v
  - Generated C files: ['_issue235_allocatable_classes_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmpcmq07byx/issue235_allocatable_classes_directc_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmpcmq07byx/issue235_allocatable_classes_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmpcmq07byx/issue235_allocatable_classes_directc_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmpcmq07byx/issue235_allocatable_classes_direct_support.o:

### ❌ issue254_getter
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['KIMDispersion_Horton.fpp', 'KIMDispersionEquation.fpp']
  - Command: f90wrap -m issue254_getter_direct /tmp/tmp9ol777sc/KIMDispersion_Horton.fpp /tmp/tmp9ol777sc/KIMDispersionEquation.fpp  --direct-c -v
  - Generated C files: ['_issue254_getter_directmodule.c']
  - Failed to compile issue254_getter_direct_support.f90: /tmp/tmp9ol777sc/issue254_getter_direct_support.f90:78:49:

   78 |         type(kimdispersionequation), pointer :: fptr
      |                                                 1~~~
Error: ‘fptr’ at (1) is of the ABSTRACT type ‘kimdispersionequation’
/tmp/tmp9ol777sc/issue254_getter_direct_support.f90:70:49:

   70 |         type(kimdispersionequation), pointer :: fptr
      |                                                 1~~~
Error: ‘fptr’ at (1) is of the ABSTRACT type ‘kimdispersionequation

### ❌ issue258_derived_type_attributes
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['dta_ct.fpp', 'dta_cc.fpp', 'dta_tt.fpp', 'dta_tc.fpp']
  - Command: f90wrap -m issue258_derived_type_attributes_direct /tmp/tmpo9mn32lq/dta_ct.fpp /tmp/tmpo9mn32lq/dta_cc.fpp /tmp/tmpo9mn32lq/dta_tt.fpp /tmp/tmpo9mn32lq/dta_tc.fpp  --direct-c -v
  - Generated C files: ['_issue258_derived_type_attributes_directmodule.c']
  - Failed to compile issue258_derived_type_attributes_direct_support.f90: /tmp/tmpo9mn32lq/issue258_derived_type_attributes_direct_support.f90:17:22:

   17 |         type(t_inner), pointer :: fptr
      |                      1
Error: Type name ‘t_inner’ at (1) is ambiguous
/tmp/tmpo9mn32lq/issue258_derived_type_attributes_direct_support.f90:19:18:

   19 |         allocate(fptr)
      |                  1~~~
Error: Allocate-object at (1) is neither a data pointer nor an allocatable variable
/tmp/tmpo9mn32lq/issue258_derived_type_attributes_direct_support.f90:25:22:


### ❌ issue261_array_shapes
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['array_shapes.fpp']
  - Command: f90wrap -m issue261_array_shapes_direct /tmp/tmpmks1hj8m/array_shapes.fpp  --direct-c -v
  - Generated C files: ['_issue261_array_shapes_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmpmks1hj8m/issue261_array_shapes_directc_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmpmks1hj8m/issue261_array_shapes_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmpmks1hj8m/issue261_array_shapes_directc_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmpmks1hj8m/issue261_array_shapes_direct_support.o:(.bss+0x8): first defined he

### ❌ issue41_abstract_classes
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['myclass_factory.fpp', 'main.fpp', 'myclass_impl.fpp', 'myclass_base.fpp', 'myclass_impl2.fpp']
  - Command: f90wrap -m issue41_abstract_classes_direct /tmp/tmp7mkne9oz/myclass_factory.fpp /tmp/tmp7mkne9oz/main.fpp /tmp/tmp7mkne9oz/myclass_impl.fpp /tmp/tmp7mkne9oz/myclass_base.fpp /tmp/tmp7mkne9oz/myclass_impl2.fpp  --direct-c -v
  - Generated C files: ['_issue41_abstract_classes_directmodule.c']
  - Failed to compile issue41_abstract_classes_direct_support.f90: /tmp/tmp7mkne9oz/issue41_abstract_classes_direct_support.f90:25:37:

   25 |         type(myclass_t), pointer :: fptr
      |                                     1~~~
Error: ‘fptr’ at (1) is of the ABSTRACT type ‘myclass_t’
/tmp/tmp7mkne9oz/issue41_abstract_classes_direct_support.f90:17:37:

   17 |         type(myclass_t), pointer :: fptr
      |                                     1~~~
Error: ‘fptr’ at (1) is of the ABSTRACT type ‘myclass_t’


### ❌ keyword_renaming_issue160
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['rename.fpp']
  - Command: f90wrap -m keyword_renaming_issue160_direct /tmp/tmpqhdn57mu/rename.fpp -k /tmp/tmpqhdn57mu/kind_map --direct-c -v
  - Generated C files: ['_keyword_renaming_issue160_directmodule.c']
  - Failed to compile keyword_renaming_issue160_direct_support.f90: /tmp/tmpqhdn57mu/keyword_renaming_issue160_direct_support.f90:6:9:

    6 |     use global_
      |         1
Fatal Error: Cannot open module file ‘global_.mod’ for reading at (1): No such file or directory
compilation terminated.


### ❌ kind_map_default
- **Status**: FAIL
- **Error Category**: test_execution_failed
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m kind_map_default_direct /tmp/tmpm3tprc44/main.fpp  --direct-c -v
  - Generated C files: ['_kind_map_default_directmodule.c']
  - Modified tests.py to use direct-c module
  - Test failed with return code 1

### ❌ long_subroutine_name
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m long_subroutine_name_direct /tmp/tmp3gg9527n/main.fpp  --direct-c -v
  - Generated C files: ['_long_subroutine_name_directmodule.c']
  - Failed to compile long_subroutine_name_direct_support.f90: /tmp/tmp3gg9527n/long_subroutine_name_direct_support.f90:49:80:

   49 |     subroutine f90wrap_m_long_subroutine_name_type__get__m_long_subroutine_name_type_integer(self_ptr, value) bind(C, name='__m_long_subroutine_name_MOD_f90wrap_m_long_subroutine_name_type__get__m_long_subroutine_name_type_integer')
      |                                                                                1
Error: Name at (1) is too long
/tmp/tmp3gg9527n/long_subroutine_name_direct_support.f90:50:38:

   50 |  

### ❌ mockderivetype
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['leveltwomod.fpp', 'define.fpp', 'fwrap.fpp']
  - Command: f90wrap -m mockderivetype_direct /tmp/tmpn2h1f6jc/leveltwomod.fpp /tmp/tmpn2h1f6jc/define.fpp /tmp/tmpn2h1f6jc/fwrap.fpp -k /tmp/tmpn2h1f6jc/kind_map --direct-c -v
  - Generated C files: ['_mockderivetype_directmodule.c']
  - Failed to compile mockderivetype_direct_support.f90: /tmp/tmpn2h1f6jc/mockderivetype_direct_support.f90:6:9:

    6 |     use leveltwomod
      |         1
Fatal Error: Cannot open module file ‘leveltwomod.mod’ for reading at (1): No such file or directory
compilation terminated.


### ❌ mod_arg_clash
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['test.fpp']
  - Command: f90wrap -m mod_arg_clash_direct /tmp/tmpn1ni6xkm/test.fpp  --direct-c -v
  - Generated C files: ['_mod_arg_clash_directmodule.c']
  - Failed to compile mod_arg_clash_direct_support.f90: /tmp/tmpn1ni6xkm/mod_arg_clash_direct_support.f90:67:65:

   67 |     subroutine f90wrap_unit_cell__get__species_symbol(self_ptr, value) bind(C, name='__cell_MOD_f90wrap_unit_cell__get__species_symbol')
      |                                                                 1~~~~
Error: Character dummy argument ‘value’ at (1) must be of constant length of one or assumed length, unless it has assumed shape or assumed rank, as procedure ‘f90wrap_unit_cell__get__species_symbol’ has the BIND(C) attr

### ❌ optional_derived_arrays
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['test.fpp']
  - Command: f90wrap -m optional_derived_arrays_direct /tmp/tmp_wznlz6z/test.fpp  --direct-c -v
  - Generated C files: ['_optional_derived_arrays_directmodule.c']
  - Failed to compile optional_derived_arrays_direct_support.f90: /tmp/tmp_wznlz6z/optional_derived_arrays_direct_support.f90:85:60:

   85 |     subroutine f90wrap_keyword__get__description(self_ptr, value) bind(C, name='__io_MOD_f90wrap_keyword__get__description')
      |                                                            1~~~~
Error: Character dummy argument ‘value’ at (1) must be of constant length of one or assumed length, unless it has assumed shape or assumed rank, as procedure ‘f90wrap_keyword__get__description’ has the BIND(C) attribute
/tmp/t

### ❌ optional_string
- **Status**: FAIL
- **Error Category**: c_compilation_failed
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m optional_string_direct /tmp/tmpre36s1nm/main.fpp  --direct-c -v
  - Generated C files: ['_optional_string_directmodule.c']
  - Failed to compile _optional_string_directmodule.c: /tmp/tmpre36s1nm/_optional_string_directmodule.c: In function ‘wrap_string_out_optional_array’:
/tmp/tmpre36s1nm/_optional_string_directmodule.c:363:69: error: ‘output_data’ undeclared (first use in this function)
  363 |     __m_string_test_MOD_string_out_optional_array((output_present ? output_data : NULL));
      |                                                                     ^~~~~~~~~~~
/tmp/tmpre36s1nm/_optional_string_directmodule.c:363:69: note: each undeclared identifier is reporte

### ❌ recursive_type
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['tree.fpp']
  - Command: f90wrap -m recursive_type_direct /tmp/tmpfd4jtzjz/tree.fpp -k /tmp/tmpfd4jtzjz/kind_map --direct-c -v
  - Generated C files: ['_recursive_type_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmpfd4jtzjz/recursive_type_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmpfd4jtzjz/recursive_type_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmpfd4jtzjz/recursive_type_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmpfd4jtzjz/recursive_type_direct_support.o:(.bss+0x8): first defined here
/usr/bin/ld: /tmp/tmpfd4jtzjz/recursive_t

### ❌ recursive_type_array
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['test.fpp']
  - Command: f90wrap -m recursive_type_array_direct /tmp/tmpkuhwoo39/test.fpp -k /tmp/tmpkuhwoo39/kind_map --direct-c -v
  - Generated C files: ['_recursive_type_array_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmpkuhwoo39/recursive_type_array_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmpkuhwoo39/recursive_type_array_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmpkuhwoo39/recursive_type_array_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmpkuhwoo39/recursive_type_array_direct_support.o:(.bss+0x8): first defined here
/usr/bin/ld: /tmp

### ❌ remove_pointer_arg
- **Status**: FAIL
- **Error Category**: test_execution_failed
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m remove_pointer_arg_direct /tmp/tmpbo8wwlm9/main.fpp  --direct-c -v
  - Generated C files: ['_remove_pointer_arg_directmodule.c']
  - Modified tests.py to use direct-c module
  - Test failed with return code 1

### ❌ return_array
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m return_array_direct /tmp/tmpcddno35a/main.fpp  --direct-c -v
  - Generated C files: ['_return_array_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmpcddno35a/return_array_directc_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmpcddno35a/return_array_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmpcddno35a/return_array_directc_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmpcddno35a/return_array_direct_support.o:(.bss+0x8): first defined here
/usr/bin/ld: /tmp/tmpcddno35a/ret

### ❌ strings
- **Status**: FAIL
- **Error Category**: test_execution_failed
- **Notes**:
  - Using preprocessed files: ['string_io.fpp']
  - Command: f90wrap -m strings_direct /tmp/tmpp30jckzv/string_io.fpp -k /tmp/tmpp30jckzv/kind_map --direct-c -v
  - Generated C files: ['_strings_directmodule.c']
  - Modified tests.py to use direct-c module
  - Test failed with return code 1

### ❌ subroutine_args
- **Status**: FAIL
- **Error Category**: test_execution_failed
- **Notes**:
  - Using preprocessed files: ['subroutine_mod.fpp']
  - Command: f90wrap -m subroutine_args_direct /tmp/tmp29duj3ec/subroutine_mod.fpp -k /tmp/tmp29duj3ec/kind_map --direct-c -v
  - Generated C files: ['_subroutine_args_directmodule.c']
  - Modified tests.py to use direct-c module
  - Test failed with return code 1

### ❌ type_bn
- **Status**: FAIL
- **Error Category**: fortran_compilation_failed
- **Notes**:
  - Using preprocessed files: ['type_bn.fpp']
  - Command: f90wrap -m type_bn_direct /tmp/tmpstqh_m25/type_bn.fpp  --direct-c -v
  - Generated C files: ['_type_bn_directmodule.c']
  - Failed to compile type_bn_direct_support.f90: /tmp/tmpstqh_m25/type_bn_direct_support.f90:55:15:

   55 |         value = self%type_bn
      |               1
Error: Syntax error in VALUE statement at (1)
/tmp/tmpstqh_m25/type_bn_direct_support.f90:64:21:

   64 |         self%type_bn = value
      |                     1
Error: ‘type_bn’ at (1) is not a member of the ‘type_face’ structure; did you mean ‘type’?


### ❌ type_check
- **Status**: FAIL
- **Error Category**: linking_failed
- **Notes**:
  - Using preprocessed files: ['main.fpp']
  - Command: f90wrap -m type_check_direct /tmp/tmpq2xrkw9b/main.fpp  --direct-c -v
  - Generated C files: ['_type_check_directmodule.c']
  - Linking failed: /usr/bin/ld: /tmp/tmpq2xrkw9b/type_check_support.o:(.bss+0x0): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_funptr'; /tmp/tmpq2xrkw9b/type_check_direct_support.o:(.bss+0x0): first defined here
/usr/bin/ld: /tmp/tmpq2xrkw9b/type_check_support.o:(.bss+0x8): multiple definition of `__f90wrap_support_MOD___def_init___iso_c_binding_C_ptr'; /tmp/tmpq2xrkw9b/type_check_direct_support.o:(.bss+0x8): first defined here
/usr/bin/ld: /tmp/tmpq2xrkw9b/type_check_support.o:(.data

### ⊘ example2
- **Status**: SKIP
- **Notes**:
  - No Fortran source files found

### ⊘ passbyreference
- **Status**: SKIP
- **Notes**:
  - No Fortran source files found


## Error Categories

### attribute_error (1 examples)
- fixed_1D_derived_type_array_argument

### c_compilation_failed (3 examples)
- derived-type-aliases
- docstring
- optional_string

### fortran_compilation_failed (13 examples)
- cylinder
- derivedtypes
- errorbinding
- fortran_oo
- issue254_getter
- issue258_derived_type_attributes
- issue41_abstract_classes
- keyword_renaming_issue160
- long_subroutine_name
- mockderivetype
- mod_arg_clash
- optional_derived_arrays
- type_bn

### linking_failed (14 examples)
- arrayderivedtypes
- arrays_in_derived_types_issue50
- class_names
- default_i8
- derivedtypes_procedure
- extends
- interface
- issue227_allocatable
- issue235_allocatable_classes
- issue261_array_shapes
- recursive_type
- recursive_type_array
- return_array
- type_check

### test_execution_failed (9 examples)
- arrays
- arrays_fixed
- auto_raise_error
- callback_print_function_issue93
- intent_out_size
- kind_map_default
- remove_pointer_arg
- strings
- subroutine_args
