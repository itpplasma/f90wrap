from __future__ import print_function, absolute_import, division
import _ExampleDerivedTypes_pkg
import f90wrap.runtime
import logging
import numpy
import warnings
import ExampleDerivedTypes_pkg.library
import ExampleDerivedTypes_pkg.datatypes
import ExampleDerivedTypes_pkg.datatypes_allocatable
import ExampleDerivedTypes_pkg.parameters
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_ExampleDerivedTypes_pkg = _SafeDirectCExecutor(_ExampleDerivedTypes_pkg, \
    module_import_name='_ExampleDerivedTypes_pkg')

