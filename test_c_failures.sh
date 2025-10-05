#!/bin/bash
set -e

PYTHON_INC="-I/usr/include/python3.13"
NUMPY_INC="-I/home/ert/code/.venv/lib/python3.13/site-packages/numpy/_core/include"

echo "=== Testing derived-type-aliases ==="
cd /home/ert/code/f90wrap/examples/derived-type-aliases
if [ -f _dta_directmodule.c ]; then
    gcc -c -I. $PYTHON_INC $NUMPY_INC _dta_directmodule.c 2>&1 | head -50
else
    echo "C file not found, running f90wrap first"
    f90wrap -m dta_direct --direct-c mytype_mod.f90 othertype_mod.f90 2>&1 | tail -20
    gcc -c -I. $PYTHON_INC $NUMPY_INC _dta_directmodule.c 2>&1 | head -50
fi

echo ""
echo "=== Testing docstring ==="
cd /home/ert/code/f90wrap/examples/docstring
if [ -f _docstring_directmodule.c ]; then
    gcc -c -I. $PYTHON_INC $NUMPY_INC _docstring_directmodule.c 2>&1 | head -50
else
    echo "C file not found, running f90wrap first"
    f90wrap -m docstring_direct --direct-c main.f90 f90wrap_main.f90 2>&1 | tail -20
    gcc -c -I. $PYTHON_INC $NUMPY_INC _docstring_directmodule.c 2>&1 | head -50
fi

echo ""
echo "=== Testing issue258_derived_type_attributes ==="
cd /home/ert/code/f90wrap/examples/issue258_derived_type_attributes
if [ -f _issue258_derived_type_attributes_directmodule.c ]; then
    gcc -c -I. $PYTHON_INC $NUMPY_INC _issue258_derived_type_attributes_directmodule.c 2>&1 | head -50
else
    echo "C file not found, running f90wrap first"
    f90wrap -m issue258_derived_type_attributes_direct --direct-c dta_ct.f90 dta_cc.f90 dta_tt.f90 dta_tc.f90 2>&1 | tail -20
    gcc -c -I. $PYTHON_INC $NUMPY_INC _issue258_derived_type_attributes_directmodule.c 2>&1 | head -50
fi
