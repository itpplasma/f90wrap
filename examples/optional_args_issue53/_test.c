#include <Python.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define F90WRAP_F_SYMBOL(name) name##_

void f90wrap_abort_(char *message, int len_message)
{
    /* Acquire GIL since we're calling Python C-API from Fortran */
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    if (message == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "f90wrap_abort called");
        PyGILState_Release(gstate);
        return;
    }
    while (len_message > 0 && message[len_message - 1] == ' ') {
        --len_message;
    }
    if (len_message <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "f90wrap_abort called");
        PyGILState_Release(gstate);
        return;
    }
    PyObject* unicode = PyUnicode_FromStringAndSize(message, len_message);
    if (unicode == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "f90wrap_abort called");
        PyGILState_Release(gstate);
        return;
    }
    PyErr_SetObject(PyExc_RuntimeError, unicode);
    Py_DECREF(unicode);
    PyGILState_Release(gstate);
}

void f90wrap_abort__(char *message, int len_message)
{
    f90wrap_abort_(message, len_message);
}

/* External f90wrap helper functions */
extern void F90WRAP_F_SYMBOL(f90wrap_wrap)(int* opt, int* def_);

static PyObject* wrap__test_wrap(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_opt = Py_None;
    int opt_val = 0;
    PyArrayObject* opt_scalar_arr = NULL;
    int opt_scalar_copyback = 0;
    int opt_scalar_is_array = 0;
    PyObject* py_def_ = NULL;
    int def__val = 0;
    PyArrayObject* def__scalar_arr = NULL;
    int def__scalar_copyback = 0;
    int def__scalar_is_array = 0;
    static char *kwlist[] = {"opt", "def_", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist, &py_opt, &py_def_)) {
        return NULL;
    }
    
    int* opt = &opt_val;
    if (py_opt == Py_None) {
        opt_val = 0;
    } else {
        if (PyArray_Check(py_opt)) {
            opt_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
                py_opt, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
            if (opt_scalar_arr == NULL) {
                return NULL;
            }
            if (PyArray_SIZE(opt_scalar_arr) != 1) {
                PyErr_SetString(PyExc_ValueError, "Argument opt must have exactly one element");
                Py_DECREF(opt_scalar_arr);
                return NULL;
            }
            opt_scalar_is_array = 1;
            opt = (int*)PyArray_DATA(opt_scalar_arr);
            opt_val = opt[0];
            if (PyArray_DATA(opt_scalar_arr) != PyArray_DATA((PyArrayObject*)py_opt) || PyArray_TYPE(opt_scalar_arr) != \
                PyArray_TYPE((PyArrayObject*)py_opt)) {
                opt_scalar_copyback = 1;
            }
        } else if (PyNumber_Check(py_opt)) {
            opt_val = (int)PyLong_AsLong(py_opt);
            if (PyErr_Occurred()) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument opt must be a scalar number or NumPy array");
            return NULL;
        }
    }
    int* def_ = &def__val;
    if (PyArray_Check(py_def_)) {
        def__scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_def_, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (def__scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(def__scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument def_ must have exactly one element");
            Py_DECREF(def__scalar_arr);
            return NULL;
        }
        def__scalar_is_array = 1;
        def_ = (int*)PyArray_DATA(def__scalar_arr);
        def__val = def_[0];
        if (PyArray_DATA(def__scalar_arr) != PyArray_DATA((PyArrayObject*)py_def_) || PyArray_TYPE(def__scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_def_)) {
            def__scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_def_)) {
        def__val = (int)PyLong_AsLong(py_def_);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument def_ must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_wrap)(opt, def_);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (opt_scalar_is_array) {
        if (opt_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_opt, opt_scalar_arr) < 0) {
                Py_DECREF(opt_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(opt_scalar_arr);
    }
    if (def__scalar_is_array) {
        if (def__scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_def_, def__scalar_arr) < 0) {
                Py_DECREF(def__scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(def__scalar_arr);
    }
    Py_RETURN_NONE;
}

/* Method table for _test module */
static PyMethodDef _test_methods[] = {
    {"f90wrap_wrap", (PyCFunction)wrap__test_wrap, METH_VARARGS | METH_KEYWORDS, "Wrapper for wrap"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _testmodule = {
    PyModuleDef_HEAD_INIT,
    "test",
    "Direct-C wrapper for _test module",
    -1,
    _test_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__test(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_testmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
