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
extern void F90WRAP_F_SYMBOL(f90wrap_foo)(double* a, int* b);

static PyObject* wrap__test_foo(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_a = NULL;
    double a_val = 0;
    PyArrayObject* a_scalar_arr = NULL;
    int a_scalar_copyback = 0;
    int a_scalar_is_array = 0;
    PyObject* py_b = NULL;
    int b_val = 0;
    PyArrayObject* b_scalar_arr = NULL;
    int b_scalar_copyback = 0;
    int b_scalar_is_array = 0;
    static char *kwlist[] = {"a", "b", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_a, &py_b)) {
        return NULL;
    }
    
    double* a = &a_val;
    if (PyArray_Check(py_a)) {
        a_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_a, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (a_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(a_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument a must have exactly one element");
            Py_DECREF(a_scalar_arr);
            return NULL;
        }
        a_scalar_is_array = 1;
        a = (double*)PyArray_DATA(a_scalar_arr);
        a_val = a[0];
        if (PyArray_DATA(a_scalar_arr) != PyArray_DATA((PyArrayObject*)py_a) || PyArray_TYPE(a_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_a)) {
            a_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_a)) {
        a_val = (double)PyFloat_AsDouble(py_a);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument a must be a scalar number or NumPy array");
        return NULL;
    }
    int* b = &b_val;
    if (PyArray_Check(py_b)) {
        b_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_b, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (b_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(b_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument b must have exactly one element");
            Py_DECREF(b_scalar_arr);
            return NULL;
        }
        b_scalar_is_array = 1;
        b = (int*)PyArray_DATA(b_scalar_arr);
        b_val = b[0];
        if (PyArray_DATA(b_scalar_arr) != PyArray_DATA((PyArrayObject*)py_b) || PyArray_TYPE(b_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_b)) {
            b_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_b)) {
        b_val = (int)PyLong_AsLong(py_b);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument b must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_foo)(a, b);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (a_scalar_is_array) {
        if (a_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_a, a_scalar_arr) < 0) {
                Py_DECREF(a_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(a_scalar_arr);
    }
    if (b_scalar_is_array) {
        if (b_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_b, b_scalar_arr) < 0) {
                Py_DECREF(b_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(b_scalar_arr);
    }
    Py_RETURN_NONE;
}

/* Method table for _test module */
static PyMethodDef _test_methods[] = {
    {"f90wrap_foo", (PyCFunction)wrap__test_foo, METH_VARARGS | METH_KEYWORDS, "Wrapper for foo"},
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
