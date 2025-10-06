# F90wrap Direct-C Compatibility Report

**Generated:** 2025-10-06 22:58:54

## Summary

- **Total Examples:** 50
- **✅ Passed:** 28 (56.0%)
- **❌ Failed:** 21 (42.0%)
- **⊘ Skipped:** 1 (2.0%)

## Error Categories

### c_compilation_failed (9 examples)
- derivedtypes_procedure
- issue235_allocatable_classes
- issue261_array_shapes
- issue41_abstract_classes
- optional_string
- output_kind
- passbyreference
- recursive_type
- return_array

### fortran_compilation_failed (4 examples)
- fortran_oo
- issue258_derived_type_attributes
- kind_map_default
- type_check

### attribute_error (3 examples)
- callback_print_function_issue93
- derivedtypes
- fixed_1D_derived_type_array_argument

### syntax_error (2 examples)
- derived-type-aliases
- mod_arg_clash

### no_c_output (1 examples)
- cylinder

### unknown_error (1 examples)
- intent_out_size

### type_error (1 examples)
- strings

## Detailed Results

| Example | Status | Category | Note |
|---------|--------|----------|------|
| arrayderivedtypes | ✅ PASS | N/A | f90wrap inputs: ['test.fpp'] Generated C: ['_module_calcul.c'] |
| arrays | ✅ PASS | N/A | f90wrap inputs: ['library.fpp', 'parameters.fpp'] Generated C: ['_library.c'] |
| arrays_fixed | ✅ PASS | N/A | f90wrap inputs: ['library.f', 'parameters.f'] Generated C: ['_library.c'] |
| arrays_in_derived_types_issue50 | ✅ PASS | N/A | f90wrap inputs: ['test.fpp'] Generated C: ['_module_test.c'] |
| auto_raise_error | ✅ PASS | N/A | f90wrap inputs: ['main.fpp'] Generated C: ['_m_error.c'] |
| class_names | ✅ PASS | N/A | f90wrap inputs: ['test.fpp'] Generated C: ['_module_snake_mod.c'] |
| default_i8 | ✅ PASS | N/A | f90wrap inputs: ['test.fpp'] Generated C: ['_my_module.c'] |
| docstring | ✅ PASS | N/A | f90wrap inputs: ['main.fpp', 'f90wrap_main.fpp'] Generated C: ['_m_circle.c'] |
| elemental | ✅ PASS | N/A | f90wrap inputs: ['elemental_module.fpp'] Generated C: ['_elemental_module.c'] |
| errorbinding | ✅ PASS | N/A | f90wrap inputs: ['datatypes.fpp', 'parameters.fpp'] Generated C: ['_datatypes.c'] |
| extends | ✅ PASS | N/A | f90wrap inputs: ['testextends.fpp'] Generated C: ['_testextends_mod.c'] |
| interface | ✅ PASS | N/A | f90wrap inputs: ['example.fpp'] Generated C: ['_class_example.c'] |
| issue105_function_definition_with_empty_lines | ✅ PASS | N/A | f90wrap inputs: ['main.fpp'] Generated C: ['_itestit.c'] |
| issue206_subroutine_oldstyle | ✅ PASS | N/A | f90wrap inputs: ['subroutine_oldstyle.f'] Generated C: ['_issue206_subroutine_oldstyle_direct.c'] |
| issue227_allocatable | ✅ PASS | N/A | f90wrap inputs: ['alloc_output.fpp'] Generated C: ['_alloc_output.c'] |
| issue254_getter | ✅ PASS | N/A | f90wrap inputs: ['KIMDispersion_Horton.fpp', 'KIMDispersionEquation.fpp'] Generated C: ['_kimdisp... |
| issue32 | ✅ PASS | N/A | f90wrap inputs: ['test.fpp'] Generated C: ['_issue32_direct.c'] |
| keyword_renaming_issue160 | ✅ PASS | N/A | f90wrap inputs: ['rename.fpp'] Generated C: ['_global_.c'] |
| long_subroutine_name | ✅ PASS | N/A | f90wrap inputs: ['main.fpp'] Generated C: ['_m_long_subroutine_name.c'] |
| mockderivetype | ✅ PASS | N/A | f90wrap inputs: ['leveltwomod.fpp', 'define.fpp', 'fwrap.fpp'] Generated C: ['_leveltwomod.c'] |
| optional_args_issue53 | ✅ PASS | N/A | f90wrap inputs: ['main.fpp'] Generated C: ['_optional_args_issue53_direct.c'] |
| optional_derived_arrays | ✅ PASS | N/A | f90wrap inputs: ['test.fpp'] Generated C: ['_io.c'] |
| recursive_type_array | ✅ PASS | N/A | f90wrap inputs: ['test.fpp'] Generated C: ['_mod_recursive_type_array.c'] |
| remove_pointer_arg | ✅ PASS | N/A | f90wrap inputs: ['main.fpp'] Generated C: ['_m_test.c'] |
| string_array_input_f2py | ✅ PASS | N/A | f90wrap inputs: ['main.fpp'] Generated C: ['_string_array_input_f2py_direct.c'] |
| subroutine_args | ✅ PASS | N/A | f90wrap inputs: ['subroutine_mod.fpp'] Generated C: ['_subroutine_mod.c'] |
| subroutine_contains_issue101 | ✅ PASS | N/A | f90wrap inputs: ['test.fpp'] Generated C: ['_subroutine_contains_issue101_direct.c'] |
| type_bn | ✅ PASS | N/A | f90wrap inputs: ['type_bn.fpp'] Generated C: ['_module_structure.c'] |
| callback_print_function_issue93 | ❌ FAIL | attribute_error | f90wrap inputs: ['cback.fpp', 'caller.fpp'] Callbacks: ['pyfunc_print', 'pyfunc_return'] |
| cylinder | ❌ FAIL | no_c_output | f90wrap inputs: ['.fpp'] No Direct-C source generated |
| derived-type-aliases | ❌ FAIL | syntax_error | f90wrap inputs: ['mytype_mod.f90', 'othertype_mod.f90'] Generated C: ['_mytype_mod.c'] |
| derivedtypes | ❌ FAIL | attribute_error | f90wrap inputs: ['datatypes.fpp', 'library.fpp', 'parameters.fpp'] Generated C: ['_datatypes_allo... |
| derivedtypes_procedure | ❌ FAIL | c_compilation_failed | f90wrap inputs: ['library.fpp'] Generated C: ['_test.c'] |
| example2 | ⊘ SKIP | N/A | No Fortran sources; skipping |
| fixed_1D_derived_type_array_argument | ❌ FAIL | attribute_error | f90wrap inputs: ['functions.fpp'] f90wrap failed (rc=2) |
| fortran_oo | ❌ FAIL | fortran_compilation_failed | f90wrap inputs: ['main-oo.fpp', 'f90wrap_main-oo.fpp', 'base_poly.fpp', 'f90wrap_base_poly.fpp'] ... |
| intent_out_size | ❌ FAIL | unknown_error | f90wrap inputs: ['main.fpp'] Generated C: ['_m_intent_out.c'] |
| issue235_allocatable_classes | ❌ FAIL | c_compilation_failed | f90wrap inputs: ['myclass_factory.fpp', 'mytype.fpp', 'myclass.fpp'] Generated C: ['_myclass_fact... |
| issue258_derived_type_attributes | ❌ FAIL | fortran_compilation_failed | f90wrap inputs: ['dta_ct.fpp', 'dta_cc.fpp', 'dta_tt.fpp', 'dta_tc.fpp'] Generated C: ['_dta_ct.c'] |
| issue261_array_shapes | ❌ FAIL | c_compilation_failed | f90wrap inputs: ['array_shapes.fpp'] Generated C: ['_array_shapes.c'] |
| issue41_abstract_classes | ❌ FAIL | c_compilation_failed | f90wrap inputs: ['myclass_factory.fpp', 'main.fpp', 'myclass_impl.fpp', 'myclass_base.fpp', 'mycl... |
| kind_map_default | ❌ FAIL | fortran_compilation_failed | f90wrap inputs: ['main.fpp'] Generated C: ['_m_test.c'] |
| mod_arg_clash | ❌ FAIL | syntax_error | f90wrap inputs: ['test.fpp'] Generated C: ['_cell.c'] |
| optional_string | ❌ FAIL | c_compilation_failed | f90wrap inputs: ['main.f90'] Generated C: ['_m_string_test.c'] |
| output_kind | ❌ FAIL | c_compilation_failed | f90wrap inputs: ['main.fpp'] Generated C: ['_m_out_test.c'] |
| passbyreference | ❌ FAIL | c_compilation_failed | f90wrap inputs: ['mycode.F90'] Generated C: ['_mymodule.c'] |
| recursive_type | ❌ FAIL | c_compilation_failed | f90wrap inputs: ['tree.fpp'] Generated C: ['_tree.c'] |
| return_array | ❌ FAIL | c_compilation_failed | f90wrap inputs: ['main.fpp'] Generated C: ['_m_test.c'] |
| strings | ❌ FAIL | type_error | f90wrap inputs: ['string_io.fpp'] Generated C: ['_string_io.c'] |
| type_check | ❌ FAIL | fortran_compilation_failed | f90wrap inputs: ['main.fpp'] Generated C: ['_m_type_test.c'] |
