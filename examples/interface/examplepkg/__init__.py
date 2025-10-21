from __future__ import print_function, absolute_import, division
import _examplepkg
import f90wrap.runtime
import logging
import numpy
import warnings
import examplepkg.class_example
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_examplepkg = _SafeDirectCExecutor(_examplepkg, \
    module_import_name='_examplepkg')

