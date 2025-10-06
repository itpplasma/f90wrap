#!/bin/bash
set -euo pipefail

# Compile Fortran sources
gfortran -c -fPIC test.f90 -o test.o
gfortran -c -fPIC f90wrap_test.f90 -o f90wrap_test.o

# Compile C extension
gcc -shared -fPIC _test.c test.o f90wrap_test.o \
    -I"$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")" \
    -I"$(python3 -c "import numpy; print(numpy.get_include())")" \
    -lgfortran \
    -o "_test$(python3-config --extension-suffix)"

echo "Build successful"
ls -lh _test*.so
