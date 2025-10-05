# F90wrap Direct-C Compatibility Report

**Generated:** 2025-10-05 21:13:17

## Summary

- **Total Examples:** 50
- **✅ Passed:** 29 (58.0%)
- **❌ Failed:** 20 (40.0%)
- **⊘ Skipped:** 1 (2.0%)

## Error Categories

### test_execution_failed (13 examples)
- arrayderivedtypes
- arrays_in_derived_types_issue50
- auto_raise_error
- callback_print_function_issue93
- class_names
- cylinder
- default_i8
- derivedtypes
- derivedtypes_procedure
- mod_arg_clash
- recursive_type
- recursive_type_array
- subroutine_args

### fortran_compilation_failed (5 examples)
- arrays
- fortran_oo
- keyword_renaming_issue160
- mockderivetype
- strings

### unknown_error (1 examples)
- derived-type-aliases

### attribute_error (1 examples)
- fixed_1D_derived_type_array_argument

## Detailed Results

| Example | Status | Error Category | Notes |
|---------|--------|----------------|-------|
| arrays_fixed | ✅ PASS | N/A | Using source files for f90wrap: ['parameters.f', 'library.f'] Command: f90wrap -m arrays_fixed_di... |
| docstring | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp', 'f90wrap_main.fpp'] Command: f90wrap -m docstr... |
| elemental | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['elemental_module.fpp'] Command: f90wrap -m elemental_dire... |
| errorbinding | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['datatypes.fpp', 'parameters.fpp'] Command: f90wrap -m err... |
| extends | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['testextends.fpp'] Command: f90wrap -m extends_direct /tmp... |
| intent_out_size | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m intent_out_size_direct /tm... |
| interface | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['example.fpp'] Command: f90wrap -m interface_direct /tmp/t... |
| issue105_function_definition_with_empty_lines | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m issue105_function_definiti... |
| issue206_subroutine_oldstyle | ✅ PASS | N/A | Using source files for f90wrap: ['subroutine_oldstyle.f'] Command: f90wrap -m issue206_subroutine... |
| issue227_allocatable | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['alloc_output.fpp'] Command: f90wrap -m issue227_allocatab... |
| issue235_allocatable_classes | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['myclass_factory.fpp', 'mytype.fpp', 'myclass.fpp'] Comman... |
| issue254_getter | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['KIMDispersion_Horton.fpp', 'KIMDispersionEquation.fpp'] C... |
| issue258_derived_type_attributes | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['dta_ct.fpp', 'dta_cc.fpp', 'dta_tt.fpp', 'dta_tc.fpp'] Co... |
| issue261_array_shapes | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['array_shapes.fpp'] Command: f90wrap -m issue261_array_sha... |
| issue32 | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m issue32_direct /tmp/tmp9vi... |
| issue41_abstract_classes | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['myclass_factory.fpp', 'main.fpp', 'myclass_impl.fpp', 'my... |
| kind_map_default | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m kind_map_default_direct /t... |
| long_subroutine_name | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m long_subroutine_name_direc... |
| optional_args_issue53 | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m optional_args_issue53_dire... |
| optional_derived_arrays | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m optional_derived_arrays_di... |
| optional_string | ✅ PASS | N/A | Using source files for f90wrap: ['main.f90'] Command: f90wrap -m optional_string_direct /tmp/tmp1... |
| output_kind | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m output_kind_direct /tmp/tm... |
| passbyreference | ✅ PASS | N/A | Using source files for f90wrap: ['mycode.F90'] Command: f90wrap -m passbyreference_direct /tmp/tm... |
| remove_pointer_arg | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m remove_pointer_arg_direct ... |
| return_array | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m return_array_direct /tmp/t... |
| string_array_input_f2py | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m string_array_input_f2py_di... |
| subroutine_contains_issue101 | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m subroutine_contains_issue1... |
| type_bn | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['type_bn.fpp'] Command: f90wrap -m type_bn_direct /tmp/tmp... |
| type_check | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m type_check_direct /tmp/tmp... |
| arrayderivedtypes | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m arrayderivedtypes_direct /... |
| arrays | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['library.fpp', 'parameters.fpp'] Command: f90wrap -m array... |
| arrays_in_derived_types_issue50 | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m arrays_in_derived_types_is... |
| auto_raise_error | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m auto_raise_error_direct /t... |
| callback_print_function_issue93 | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['cback.fpp', 'caller.fpp'] Command: f90wrap -m callback_pr... |
| class_names | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m class_names_direct /tmp/tm... |
| cylinder | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['.fpp'] Command: f90wrap -m cylinder_direct /tmp/tmpp5wmbw... |
| default_i8 | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m default_i8_direct /tmp/tmp... |
| derived-type-aliases | ❌ FAIL | unknown_error | Using source files for f90wrap: ['mytype_mod.f90', 'dta_direct_support.f90', 'dtypes_support.f90'... |
| derivedtypes | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['datatypes.fpp', 'library.fpp', 'parameters.fpp'] Command:... |
| derivedtypes_procedure | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['library.fpp'] Command: f90wrap -m derivedtypes_procedure_... |
| example2 | ⊘ SKIP | N/A | No Fortran source files found |
| fixed_1D_derived_type_array_argument | ❌ FAIL | attribute_error | Using preprocessed files for f90wrap: ['functions.fpp'] Command: f90wrap -m fixed_1D_derived_type... |
| fortran_oo | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['main-oo.fpp', 'f90wrap_main-oo.fpp', 'base_poly.fpp', 'f9... |
| keyword_renaming_issue160 | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['rename.fpp'] Command: f90wrap -m keyword_renaming_issue16... |
| mockderivetype | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['leveltwomod.fpp', 'define.fpp', 'fwrap.fpp'] Command: f90... |
| mod_arg_clash | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m mod_arg_clash_direct /tmp/... |
| recursive_type | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['tree.fpp'] Command: f90wrap -m recursive_type_direct /tmp... |
| recursive_type_array | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m recursive_type_array_direc... |
| strings | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['string_io.fpp'] Command: f90wrap -m strings_direct /tmp/t... |
| subroutine_args | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['subroutine_mod.fpp'] Command: f90wrap -m subroutine_args_... |
