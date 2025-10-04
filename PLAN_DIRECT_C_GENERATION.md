# f90wrap Direct C Generation Mode - Implementation Plan

## Executive Summary

Replace the f90wrap → f2py pipeline with direct C/Python API code generation, eliminating the 13x performance bottleneck in f2py while maintaining full functionality.

**Current Performance:** 0.6s (f90wrap) + 7.7s (f2py) = 8.3s total
**Target Performance:** <1s total (13x speedup)

## Architecture Overview

### Current Pipeline (Indirect)
```
Fortran → f90wrap → Fortran wrappers → f2py → C module → Compile
          (0.6s)                        (7.7s)
```

### New Pipeline (Direct)
```
Fortran → f90wrap → C module → Compile
          (<1s)
```

## Design Principles

1. **Minimal Code, Maximum Performance**
   - Direct template-based C generation
   - No intermediate Fortran wrappers
   - No inefficient rule-application system

2. **Modern Best Practices**
   - Clean separation of concerns
   - Type-safe code generation
   - Comprehensive error handling
   - Extensive test coverage

3. **Compatibility**
   - Maintain backward compatibility with f90wrap Python API
   - Support all existing f90wrap features
   - NumPy C API for array handling
   - Python 3.9+ support

4. **Zero Tolerance for Incomplete Work**
   - No stubs or placeholders
   - Full implementation required for each phase
   - Mandatory validation after each phase

## Phase 1: C Code Generator Infrastructure

### 1.1 Core C Code Generator (2-3 days)

**File:** `f90wrap/cwrapgen.py`

**Responsibilities:**
- Template-based C code generation
- Type conversion tables (Fortran ↔ C ↔ NumPy)
- Module initialization
- Error handling framework

**Implementation:**
```python
class CWrapperGenerator:
    """
    Direct C/Python API code generator.
    Replaces f2py with efficient template-based generation.
    """

    def __init__(self, ast, module_name, config):
        self.ast = ast
        self.module_name = module_name
        self.config = config
        self.type_map = FortranCTypeMap()
        self.code_gen = CCodeGenerator()

    def generate(self):
        """Main entry point - generates complete C module"""
        self._generate_includes()
        self._generate_type_definitions()
        self._generate_fortran_prototypes()
        self._generate_wrapper_functions()
        self._generate_method_table()
        self._generate_module_init()
        return str(self.code_gen)
```

**Key Components:**

1. **Type Mapping System**
   ```python
   class FortranCTypeMap:
       # Fortran → C type conversions
       # Fortran → NumPy dtype conversions
       # Intent handling (in/out/inout)
       # Array dimension handling
   ```

2. **Template System**
   ```python
   class CCodeTemplate:
       # Function wrapper template
       # Getter/setter templates
       # Array conversion templates
       # Error handling templates
   ```

3. **Fortran Name Mangling**
   ```python
   class FortranNameMangler:
       # Handle f77/f90/gfortran conventions
       # Underscore rules
       # Case handling
   ```

**Tests Required:**
- ✅ Type conversion correctness (all Fortran types)
- ✅ Name mangling (all compiler conventions)
- ✅ Template rendering
- ✅ Code validity (compiles without warnings)

**Validation Criteria:**
- No placeholder functions
- All type conversions implemented
- 100% code coverage for type mapping
- Compiles with `-Wall -Werror`

### 1.2 NumPy C API Integration (1-2 days)

**File:** `f90wrap/numpy_capi.py`

**Responsibilities:**
- NumPy array creation/conversion
- Memory management
- Reference counting
- Array descriptor handling

**Implementation:**
```python
class NumpyArrayHandler:
    """Handle NumPy C API array operations"""

    @staticmethod
    def generate_array_from_fortran(arg, code_gen):
        """Generate C code to create NumPy array from Fortran pointer"""
        # Handle dimensions, strides, data pointer
        # Fortran column-major → NumPy row-major
        # Memory ownership

    @staticmethod
    def generate_fortran_from_array(arg, code_gen):
        """Generate C code to extract Fortran data from NumPy array"""
        # Type checking
        # Dimension checking
        # Contiguity handling
```

**Tests Required:**
- ✅ 1D/2D/3D array conversions
- ✅ All NumPy dtypes
- ✅ Memory leak detection
- ✅ Reference counting correctness
- ✅ Column/row major handling

**Validation Criteria:**
- Valgrind clean (no leaks)
- All array operations functional
- Performance within 5% of f2py

### 1.3 Error Handling & Exceptions (1 day)

**File:** `f90wrap/cerror.py`

**Responsibilities:**
- Python exception raising from C
- Fortran error propagation
- f90wrap_abort support

**Implementation:**
```python
class CErrorHandler:
    """Generate error handling code"""

    @staticmethod
    def generate_exception_check(code_gen):
        """Generate PyErr_Occurred() checks"""

    @staticmethod
    def generate_abort_handler(code_gen):
        """Generate f90wrap_abort mechanism (setjmp/longjmp)"""
```

**Tests Required:**
- ✅ Exception propagation
- ✅ Fortran abort handling
- ✅ Resource cleanup on error
- ✅ No memory leaks on exception paths

**Phase 1 Deliverables:**
- ✅ Complete `cwrapgen.py` with all type conversions
- ✅ Complete `numpy_capi.py` with array handling
- ✅ Complete `cerror.py` with error handling
- ✅ 95%+ test coverage
- ✅ No stubs, no TODOs, no placeholders
- ✅ Full documentation

**Phase 1 Validation:**
```bash
# Run full test suite
pytest f90wrap/tests/test_cwrapgen.py -v --cov=f90wrap.cwrapgen --cov-report=term-missing
# Check for stubs
grep -r "NotImplementedError\|TODO\|FIXME\|pass  #" f90wrap/cwrapgen.py && exit 1
# Compile test
gcc -Wall -Werror -shared test_output.c -o test.so
```

## Phase 2: Function & Subroutine Wrappers

### 2.1 Scalar Arguments (2 days)

**Implementation:**
- Generate wrappers for subroutines/functions with scalar args
- Handle all Fortran scalar types (integer, real, complex, logical, character)
- Intent handling (in/out/inout)

**Template Example:**
```c
static PyObject* wrap_{{name}}(PyObject *self, PyObject *args) {
    // Argument declarations
    {{arg_declarations}}

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "{{format_string}}", {{arg_pointers}})) {
        return NULL;
    }

    // Type conversions Python → Fortran
    {{py_to_fortran_conversions}}

    // Call Fortran function
    {{fortran_call}}

    // Type conversions Fortran → Python (for intent(out))
    {{fortran_to_py_conversions}}

    // Build return value
    return {{build_return}};
}
```

**Tests Required:**
- ✅ All scalar types (int, real, complex, logical, character)
- ✅ All intent combinations
- ✅ Optional arguments
- ✅ Return values vs out arguments

### 2.2 Array Arguments (3 days)

**Implementation:**
- 1D/2D/3D/nD array handling
- Assumed-shape, explicit-shape, allocatable
- Fortran column-major ↔ NumPy row-major
- Contiguity requirements

**Tests Required:**
- ✅ Fixed-size arrays
- ✅ Assumed-shape arrays
- ✅ Allocatable arrays
- ✅ Array slicing
- ✅ Non-contiguous arrays
- ✅ All combinations of dimensions (1D to 7D)

### 2.3 Character/String Handling (1 day)

**Implementation:**
- Fixed-length strings
- Assumed-length strings
- String arrays

**Tests Required:**
- ✅ String input/output
- ✅ String length handling
- ✅ Unicode support
- ✅ String arrays

**Phase 2 Deliverables:**
- ✅ Complete scalar wrapper generation
- ✅ Complete array wrapper generation
- ✅ Complete string handling
- ✅ 500+ test cases covering all combinations
- ✅ Performance benchmarks (must match or exceed f2py)

**Phase 2 Validation:**
```bash
# Run exhaustive test suite
pytest f90wrap/tests/test_wrappers.py -v --tb=short
# Performance check
python benchmark.py  # Must be <1s for SIMPLE codebase
# Check for incomplete implementations
grep -r "raise NotImplementedError\|TODO\|stub" f90wrap/ && exit 1
```

## Phase 3: Derived Type Support

### 3.1 Type Definition Wrappers (2 days)

**Implementation:**
- Generate Python class for each Fortran derived type
- Constructor/destructor
- Getter/setter for components
- Opaque pointer handling (f90wrap's existing mechanism)

**Template:**
```c
// Type definition
typedef struct {
    PyObject_HEAD
    {{fortran_type}}_ptr_type fortran_ptr;
} Py{{TypeName}};

// Getter/setter
static PyObject* {{type}}_get_{{field}}(Py{{TypeName}} *self, void *closure) {
    {{field_type}} value;
    f90wrap_{{type}}__get__{{field}}(self->fortran_ptr, &value);
    return {{c_to_py_conversion}}(value);
}
```

**Tests Required:**
- ✅ Simple derived types
- ✅ Nested derived types
- ✅ Derived types with allocatable components
- ✅ Type-bound procedures
- ✅ Inheritance (extends)

### 3.2 Type-Bound Procedures (2 days)

**Implementation:**
- Method binding to Python class
- Self argument handling
- Overloading support

**Tests Required:**
- ✅ Simple methods
- ✅ Methods with array arguments
- ✅ Overloaded methods
- ✅ Generic interfaces

### 3.3 Array of Derived Types (2 days)

**Implementation:**
- Special handling for type arrays
- Super-type generation (f90wrap's existing approach)

**Tests Required:**
- ✅ 1D arrays of types
- ✅ Fixed-size type arrays
- ✅ Type array item access

**Phase 3 Deliverables:**
- ✅ Complete derived type support
- ✅ All f90wrap type features working
- ✅ 300+ derived type tests
- ✅ Performance validation

## Phase 4: Advanced Features

### 4.1 Interfaces & Overloading (2 days)

**Implementation:**
- Generate Python function with dispatch logic
- Type signature checking
- Argument count/type-based dispatch

**Tests Required:**
- ✅ Generic interfaces
- ✅ Multiple implementations
- ✅ Type-specific dispatch

### 4.2 Callbacks (2 days)

**Implementation:**
- Python function → Fortran callback
- Trampoline function generation
- GIL handling

**Tests Required:**
- ✅ Simple callbacks
- ✅ Callbacks with array arguments
- ✅ Exception handling in callbacks

### 4.3 Optional Arguments (1 day)

**Implementation:**
- Python None handling
- Fortran PRESENT() intrinsic
- Default values

**Tests Required:**
- ✅ Optional scalars
- ✅ Optional arrays
- ✅ Optional derived types

**Phase 4 Deliverables:**
- ✅ All advanced features working
- ✅ 200+ advanced feature tests
- ✅ Full f90wrap feature parity

## Phase 5: Integration & Optimization

### 5.1 CLI Integration (1 day)

**File:** `f90wrap/scripts/main.py` (modifications)

**Implementation:**
```python
# Add --direct-c flag
if args.direct_c:
    from f90wrap.cwrapgen import CWrapperGenerator
    generator = CWrapperGenerator(tree, args.mod_name, config)
    c_code = generator.generate()
    with open(f"{args.mod_name}module.c", 'w') as f:
        f.write(c_code)
```

**Tests Required:**
- ✅ CLI flag parsing
- ✅ File generation
- ✅ Error reporting

### 5.2 Build System Integration (1 day)

**Implementation:**
- CMake integration examples
- Setup.py template
- Meson build examples

**Tests Required:**
- ✅ CMake build
- ✅ setuptools build
- ✅ Meson build

### 5.3 Performance Optimization (2 days)

**Focus Areas:**
- Minimize PyArg_ParseTuple calls
- Inline small conversions
- Optimize array strides calculation
- Reduce memory allocations

**Tests Required:**
- ✅ Benchmark vs f2py (must be faster)
- ✅ Memory profiling (no regressions)
- ✅ Scalability tests (large codebases)

### 5.4 Documentation (2 days)

**Deliverables:**
- User guide for --direct-c mode
- API documentation
- Migration guide from f2py mode
- Performance comparison docs

## Phase 6: Validation & Release

### 6.1 Comprehensive Testing (3 days)

**Test Matrix:**
- ✅ All f90wrap examples (50+ examples)
- ✅ SIMPLE codebase (9,176 lines)
- ✅ Comparison with f2py output (functional equivalence)
- ✅ Cross-platform (Linux, macOS, Windows)
- ✅ Multiple compilers (gfortran, ifort, ifx)
- ✅ Python 3.9, 3.10, 3.11, 3.12, 3.13

### 6.2 Real-World Validation (2 days)

**Test Codebases:**
- SIMPLE (this codebase)
- QUIP
- Example from f90wrap repo

**Success Criteria:**
- All tests pass
- Performance ≥ 10x faster than f2py
- Binary compatibility with f2py version
- No memory leaks
- No segfaults

### 6.3 Code Quality Assurance (1 day)

**Checks:**
```bash
# No incomplete implementations
! grep -r "NotImplementedError\|TODO\|FIXME\|stub\|placeholder" f90wrap/

# Code coverage
pytest --cov=f90wrap --cov-report=html --cov-fail-under=90

# Type checking
mypy f90wrap/cwrapgen.py --strict

# Linting
ruff check f90wrap/
black --check f90wrap/

# Memory leaks
valgrind --leak-check=full python test_simple.py

# Security scan
bandit -r f90wrap/
```

## Test Coverage Requirements

### Unit Tests (500+ tests)

**Type Conversions (100 tests):**
- All Fortran types → C types
- All C types → Python types
- Intent combinations
- Edge cases (NaN, Inf, overflow)

**Array Handling (150 tests):**
- All dimensions (1D-7D)
- All dtypes
- Memory layouts
- Slicing, strides
- Non-contiguous

**Derived Types (100 tests):**
- Simple types
- Nested types
- Type arrays
- Type-bound procedures
- Inheritance

**Functions/Subroutines (150 tests):**
- All argument combinations
- Return values
- Optional arguments
- Interfaces

### Integration Tests (100+ tests)

**Examples (50 tests):**
- All f90wrap examples must pass
- Output comparison with f2py mode

**Real Codebases (10 tests):**
- SIMPLE integration
- Build + import + run
- Performance validation

**Cross-Platform (40 tests):**
- Linux x86_64
- macOS ARM64
- Windows MSVC

### Performance Tests (20 tests)

**Benchmarks:**
- Build time < 1s for SIMPLE
- Import time < 100ms
- Function call overhead < f2py
- Array conversion overhead < f2py

## Implementation Schedule

| Phase | Duration | Deliverable |
|-------|----------|------------|
| Phase 1: Infrastructure | 4-6 days | Core C generator + NumPy + Errors |
| Phase 2: Functions | 6 days | Complete function/subroutine support |
| Phase 3: Derived Types | 6 days | Full derived type support |
| Phase 4: Advanced | 5 days | Interfaces, callbacks, optional args |
| Phase 5: Integration | 4 days | CLI, build, optimization, docs |
| Phase 6: Validation | 6 days | Testing, validation, QA |
| **Total** | **31-33 days** | Production-ready direct C mode |

## Success Metrics

1. **Performance:** ≥10x faster than f2py (target: <1s for SIMPLE)
2. **Coverage:** ≥90% code coverage
3. **Compatibility:** 100% of f90wrap features working
4. **Quality:** Zero stubs, placeholders, or TODOs
5. **Reliability:** Zero crashes, zero memory leaks
6. **Portability:** Works on Linux, macOS, Windows

## Risk Mitigation

### Risk 1: NumPy C API Complexity
**Mitigation:** Start with simple arrays, iterate to complex cases

### Risk 2: Fortran Compiler Variations
**Mitigation:** Extensive name mangling tests, compiler matrix

### Risk 3: Performance Regression
**Mitigation:** Continuous benchmarking, profiling at each phase

### Risk 4: Incomplete Implementation
**Mitigation:** Mandatory validation after each phase, no shortcuts allowed

## Stub/Shortcut Detection

After each phase, run:

```bash
#!/bin/bash
# detect_incomplete.sh

echo "Checking for incomplete implementations..."

# Check for stubs
if grep -r "NotImplementedError\|raise NotImplementedError" f90wrap/cw*.py; then
    echo "ERROR: Found NotImplementedError stubs"
    exit 1
fi

# Check for TODOs
if grep -r "TODO\|FIXME\|XXX\|HACK" f90wrap/cw*.py; then
    echo "ERROR: Found TODO markers"
    exit 1
fi

# Check for placeholder functions
if grep -r "def.*:$\s*pass\s*$" f90wrap/cw*.py; then
    echo "ERROR: Found placeholder functions"
    exit 1
fi

# Check for disabled tests
if grep -r "@pytest.skip\|@unittest.skip" f90wrap/tests/; then
    echo "WARNING: Found skipped tests"
fi

echo "✓ No incomplete implementations detected"
```

## Conclusion

This plan provides a complete roadmap to implement direct C generation in f90wrap, eliminating the f2py bottleneck while maintaining full functionality. Each phase has clear deliverables, comprehensive tests, and strict validation criteria to ensure a production-ready implementation with no shortcuts or incomplete work.

The result will be a modern, high-performance Fortran-Python interface generator that is 13x faster than the current pipeline.
