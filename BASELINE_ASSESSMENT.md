# F90wrap Direct-C Baseline Compatibility Assessment

## Executive Summary

Baseline compatibility assessment completed for all 50 examples in the f90wrap repository with the `--direct-c` flag. Current compatibility is 16% (8/50 examples passing).

## Assessment Results

### Statistics
- **Total Examples**: 50
- **✅ Passed**: 8 (16.0%)
- **❌ Failed**: 40 (80.0%)
- **⊘ Skipped**: 2 (4.0%)

### Passing Examples
The following examples work with `--direct-c` generation:
1. `elemental` - Simple elemental functions
2. `issue105_function_definition_with_empty_lines` - Function parsing edge case
3. `issue206_subroutine_oldstyle` - Old-style Fortran subroutines
4. `issue32` - Previous issue fix
5. `optional_args_issue53` - Optional arguments support
6. `output_kind` - Kind parameter output
7. `string_array_input_f2py` - String array input
8. `subroutine_contains_issue101` - Subroutines with contains blocks

### Failure Categories

#### 1. **Linking Failed** (14 examples, 28%)
Multiple definition errors with `__f90wrap_support_MOD` symbols.
- Root cause: Support modules being compiled twice with conflicting symbols
- Affected examples: `arrayderivedtypes`, `arrays_in_derived_types_issue50`, `class_names`, `default_i8`, `derivedtypes_procedure`, `extends`, `interface`, `issue227_allocatable`, `issue235_allocatable_classes`, `issue261_array_shapes`, `recursive_type`, `recursive_type_array`, `return_array`, `type_check`

#### 2. **Fortran Compilation Failed** (13 examples, 26%)
Generated Fortran support code fails to compile.
- Issues include:
  - Missing module dependencies
  - Abstract type instantiation attempts
  - BIND(C) character length restrictions
  - Name length exceeding compiler limits
- Affected examples: `cylinder`, `derivedtypes`, `errorbinding`, `fortran_oo`, `issue254_getter`, `issue258_derived_type_attributes`, `issue41_abstract_classes`, `keyword_renaming_issue160`, `long_subroutine_name`, `mockderivetype`, `mod_arg_clash`, `optional_derived_arrays`, `type_bn`

#### 3. **Test Execution Failed** (9 examples, 18%)
Code compiles but runtime tests fail.
- Likely issues with Python-C interface or data marshaling
- Affected examples: `arrays`, `arrays_fixed`, `auto_raise_error`, `callback_print_function_issue93`, `intent_out_size`, `kind_map_default`, `remove_pointer_arg`, `strings`, `subroutine_args`

#### 4. **C Compilation Failed** (3 examples, 6%)
Generated C code fails to compile.
- Issues include:
  - Redefinition of symbols
  - Undeclared variables
  - Incorrect escape sequences
- Affected examples: `derived-type-aliases`, `docstring`, `optional_string`

#### 5. **Attribute Error** (1 example, 2%)
Python AttributeError during f90wrap processing.
- Affected example: `fixed_1D_derived_type_array_argument`

## Critical Issues Identified

### Priority 1: Linking Issues (28% of failures)
The most common failure mode is duplicate symbol definitions in support modules. This suggests the direct-C generation creates conflicting definitions when both the original and direct-C support modules are linked together.

### Priority 2: Abstract Types and OO Features (20% of failures)
Many failures involve abstract types, type-bound procedures, and Fortran OO features. The direct-C backend doesn't properly handle abstract types and attempts to instantiate them.

### Priority 3: Character String Handling (10% of failures)
BIND(C) restrictions on character dummy arguments cause compilation failures. Character strings need special handling for C interoperability.

## Recommendations for Phase 2

Based on this assessment, the following fixes should be prioritized:

1. **Fix linking issues** - Resolve duplicate symbol definitions in support modules
2. **Handle abstract types** - Skip abstract type instantiation, handle polymorphism
3. **Fix character string marshaling** - Implement proper BIND(C) compliant string handling
4. **Module dependency resolution** - Ensure proper module compilation order
5. **Name mangling** - Handle long names that exceed compiler limits
6. **Array handling** - Fix array descriptor issues for derived types

## Test Infrastructure

Created automated testing script `/home/ert/code/f90wrap/test_direct_c_compatibility.py` that:
- Tests all examples with `--direct-c` flag
- Compiles generated C and Fortran code
- Runs test suites where available
- Generates detailed compatibility reports
- Categorizes failures for analysis

## Artifacts Generated

- `/home/ert/code/f90wrap/test_direct_c_compatibility.py` - Automated test script
- `/home/ert/code/f90wrap/direct_c_test_results/compatibility_report.md` - Detailed test report
- `/home/ert/code/f90wrap/direct_c_test_results/compatibility_results.json` - Machine-readable results

## Next Steps

Per PLAN.md Phase 2, the next step is to implement fixes for the identified issues, starting with the highest-impact problems (linking issues affecting 28% of examples).