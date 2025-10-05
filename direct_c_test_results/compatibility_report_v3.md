# F90wrap Direct-C Compatibility Report

**Generated:** 2025-10-05 16:39:14

## Summary

- **Total Examples:** 50
- **✅ Passed:** 21 (42.0%)
- **❌ Failed:** 28 (56.0%)
- **⊘ Skipped:** 1 (2.0%)

## Error Categories

### test_execution_failed (13 examples)
- arrays_fixed
- arrays_in_derived_types_issue50
- auto_raise_error
- callback_print_function_issue93
- default_i8
- derivedtypes_procedure
- intent_out_size
- kind_map_default
- recursive_type
- recursive_type_array
- remove_pointer_arg
- strings
- subroutine_args

### fortran_compilation_failed (9 examples)
- cylinder
- derived-type-aliases
- fortran_oo
- issue258_derived_type_attributes
- keyword_renaming_issue160
- long_subroutine_name
- mod_arg_clash
- optional_derived_arrays
- type_bn

### linking_failed (3 examples)
- arrayderivedtypes
- arrays
- derivedtypes

### c_compilation_failed (2 examples)
- docstring
- optional_string

### attribute_error (1 examples)
- fixed_1D_derived_type_array_argument

## Detailed Results

| Example | Status | Error Category | Notes |
|---------|--------|----------------|-------|
| class_names | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m class_names_direct /tmp/tm... |
| elemental | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['elemental_module.fpp'] Command: f90wrap -m elemental_dire... |
| errorbinding | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['datatypes.fpp', 'parameters.fpp'] Command: f90wrap -m err... |
| extends | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['testextends.fpp'] Command: f90wrap -m extends_direct /tmp... |
| interface | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['example.fpp'] Command: f90wrap -m interface_direct /tmp/t... |
| issue105_function_definition_with_empty_lines | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m issue105_function_definiti... |
| issue206_subroutine_oldstyle | ✅ PASS | N/A | Using source files for f90wrap: ['subroutine_oldstyle.f'] Command: f90wrap -m issue206_subroutine... |
| issue227_allocatable | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['alloc_output.fpp'] Command: f90wrap -m issue227_allocatab... |
| issue235_allocatable_classes | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['myclass_factory.fpp', 'mytype.fpp', 'myclass.fpp'] Comman... |
| issue254_getter | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['KIMDispersion_Horton.fpp', 'KIMDispersionEquation.fpp'] C... |
| issue261_array_shapes | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['array_shapes.fpp'] Command: f90wrap -m issue261_array_sha... |
| issue32 | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m issue32_direct /tmp/tmpto3... |
| issue41_abstract_classes | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['myclass_factory.fpp', 'main.fpp', 'myclass_impl.fpp', 'my... |
| mockderivetype | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['leveltwomod.fpp', 'define.fpp', 'fwrap.fpp'] Command: f90... |
| optional_args_issue53 | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m optional_args_issue53_dire... |
| output_kind | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m output_kind_direct /tmp/tm... |
| passbyreference | ✅ PASS | N/A | Using source files for f90wrap: ['mycode.F90'] Command: f90wrap -m passbyreference_direct /tmp/tm... |
| return_array | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m return_array_direct /tmp/t... |
| string_array_input_f2py | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m string_array_input_f2py_di... |
| subroutine_contains_issue101 | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m subroutine_contains_issue1... |
| type_check | ✅ PASS | N/A | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m type_check_direct /tmp/tmp... |
| arrayderivedtypes | ❌ FAIL | linking_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m arrayderivedtypes_direct /... |
| arrays | ❌ FAIL | linking_failed | Using preprocessed files for f90wrap: ['library.fpp', 'parameters.fpp'] Command: f90wrap -m array... |
| arrays_fixed | ❌ FAIL | test_execution_failed | Using source files for f90wrap: ['parameters.f', 'library.f'] Command: f90wrap -m arrays_fixed_di... |
| arrays_in_derived_types_issue50 | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m arrays_in_derived_types_is... |
| auto_raise_error | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m auto_raise_error_direct /t... |
| callback_print_function_issue93 | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['cback.fpp', 'caller.fpp'] Command: f90wrap -m callback_pr... |
| cylinder | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['cyldnad.fpp', 'DNAD.fpp'] Command: f90wrap -m cylinder_di... |
| default_i8 | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m default_i8_direct /tmp/tmp... |
| derived-type-aliases | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['othertype_mod.fpp', 'mytype_mod.fpp'] Command: f90wrap -m... |
| derivedtypes | ❌ FAIL | linking_failed | Using preprocessed files for f90wrap: ['datatypes.fpp', 'library.fpp', 'parameters.fpp'] Command:... |
| derivedtypes_procedure | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['library.fpp'] Command: f90wrap -m derivedtypes_procedure_... |
| docstring | ❌ FAIL | c_compilation_failed | Using preprocessed files for f90wrap: ['main.fpp', 'f90wrap_main.fpp'] Command: f90wrap -m docstr... |
| example2 | ⊘ SKIP | N/A | No Fortran source files found |
| fixed_1D_derived_type_array_argument | ❌ FAIL | attribute_error | Using preprocessed files for f90wrap: ['functions.fpp'] Command: f90wrap -m fixed_1D_derived_type... |
| fortran_oo | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['main-oo.fpp', 'f90wrap_main-oo.fpp', 'base_poly.fpp', 'f9... |
| intent_out_size | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m intent_out_size_direct /tm... |
| issue258_derived_type_attributes | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['dta_ct.fpp', 'dta_cc.fpp', 'dta_tt.fpp', 'dta_tc.fpp'] Co... |
| keyword_renaming_issue160 | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['rename.fpp'] Command: f90wrap -m keyword_renaming_issue16... |
| kind_map_default | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m kind_map_default_direct /t... |
| long_subroutine_name | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m long_subroutine_name_direc... |
| mod_arg_clash | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m mod_arg_clash_direct /tmp/... |
| optional_derived_arrays | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m optional_derived_arrays_di... |
| optional_string | ❌ FAIL | c_compilation_failed | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m optional_string_direct /tm... |
| recursive_type | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['tree.fpp'] Command: f90wrap -m recursive_type_direct /tmp... |
| recursive_type_array | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['test.fpp'] Command: f90wrap -m recursive_type_array_direc... |
| remove_pointer_arg | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['main.fpp'] Command: f90wrap -m remove_pointer_arg_direct ... |
| strings | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['string_io.fpp'] Command: f90wrap -m strings_direct /tmp/t... |
| subroutine_args | ❌ FAIL | test_execution_failed | Using preprocessed files for f90wrap: ['subroutine_mod.fpp'] Command: f90wrap -m subroutine_args_... |
| type_bn | ❌ FAIL | fortran_compilation_failed | Using preprocessed files for f90wrap: ['type_bn.fpp'] Command: f90wrap -m type_bn_direct /tmp/tmp... |
