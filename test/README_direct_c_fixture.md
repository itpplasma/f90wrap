# Direct-C Build Fixture for f90wrap Testing

## Overview

This directory contains a pytest fixture and utilities for testing the `f90wrap --direct-c` functionality, which generates C extension modules directly without using f2py.

## Files

- `direct_c_fixture.py`: Standalone module with `DirectCBuilder` class and fixtures
- `test_direct_c_build.py`: Complete test suite with integrated fixtures
- `example_direct_c_usage.py`: Example script demonstrating usage

## Key Components

### DirectCBuilder Class

Handles the complete workflow for building Python extensions from Fortran code using f90wrap's direct-C generation:

1. Preprocessing Fortran files with gfortran
2. Running `f90wrap --direct-c` to generate C wrapper and Python interface
3. Compiling Fortran source to object files
4. Compiling generated C wrapper with Python/NumPy headers
5. Linking everything into a Python extension module
6. Fixing import statements in generated Python wrapper (workaround for naming issue)

### Pytest Fixtures

- `direct_c_builder`: Provides a configured DirectCBuilder instance with temporary workspace
- `simple_fortran_module`: Creates a test Fortran module
- `build_and_test_module`: Complete workflow helper function

## Usage

```python
from test_direct_c_build import DirectCBuilder, build_and_test_module

# Create builder
builder = DirectCBuilder(work_dir, verbose=True)

# Build module
results = build_and_test_module(
    builder,
    [fortran_file],
    module_name="my_module"
)

if results['success']:
    module = results['module']
    # Use the module...
```

## Known Issues and Workarounds

1. **Module Naming**: The generated Python wrapper uses `import _module_name` but the C extension exports `module_name`. The fixture automatically fixes this by modifying the import statement.

2. **No Fortran Support Module**: In direct-C mode, no separate Fortran support module is generated (unlike f2py mode).

3. **Type Recognition**: Some iso_c_binding types like `real(c_double)` in derived type definitions may not be recognized. Use standard Fortran types (`real*8`) as a workaround.

4. **Empty Modules**: Some simple modules may generate empty C wrappers. This appears to be a limitation of the current direct-C implementation.

## Test Coverage

The test suite validates:
- Simple module compilation and import
- Complex types and arrays
- Multiple interdependent modules
- C code generation validity
- Error handling and reporting

## Requirements

- Python 3.6+
- NumPy
- gfortran or compatible Fortran compiler
- gcc or compatible C compiler
- f90wrap with direct-C support

## Running Tests

```bash
# Run all direct-C build tests
pytest test/test_direct_c_build.py -v

# Run specific test
pytest test/test_direct_c_build.py::TestDirectCBuild::test_simple_module_compilation -xvs

# Run example script
python test/example_direct_c_usage.py
```