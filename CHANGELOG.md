# Changelog

## [Unreleased]

### Added
- **Direct-C mode**: Alternative to f2py for generating Python extension modules. Use `--direct-c` flag to generate C code that directly calls f90wrap Fortran helpers via the Python C API, eliminating the f2py dependency.

### Implementation
- `f90wrap/directc.py`: ISO C interoperability analysis and procedure classification
- `f90wrap/directc_cgen/`: C code generator package for Python C API wrappers
- `f90wrap/numpy_utils.py`: NumPy C API type mapping utilities
- `f90wrap/runtime.py`: Runtime support for Direct-C array handling
- CLI: `--direct-c` flag generates `_module.c` files alongside standard Fortran wrappers

### Direct-C Mode Details
- Generates standalone C extension modules using Python C API
- All procedures call existing `f90wrap_<module>__<proc>` Fortran helpers
- Generated C files must be compiled manually (see examples/arrays/Makefile)
- Normal f2py workflow unchanged when `--direct-c` not specified
- See README.md for complete usage instructions
