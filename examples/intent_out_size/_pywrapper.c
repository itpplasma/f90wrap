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
extern void F90WRAP_F_SYMBOL(f90wrap_m_intent_out__interpolation)(int* f90wrap_n0, int* f90wrap_n1, int* f90wrap_n2, \
    int* f90wrap_n3, int* f90wrap_n4, int* f90wrap_n5, int* n1, int* n2, float* a1, float* a2, float* output);

static PyObject* wrap_m_intent_out_interpolation(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    int f90wrap_n2_val = 0;
    int f90wrap_n3_val = 0;
    int f90wrap_n4_val = 0;
    int f90wrap_n5_val = 0;
    PyObject* py_n1 = NULL;
    int n1_val = 0;
    PyArrayObject* n1_scalar_arr = NULL;
    int n1_scalar_copyback = 0;
    int n1_scalar_is_array = 0;
    PyObject* py_n2 = NULL;
    int n2_val = 0;
    PyArrayObject* n2_scalar_arr = NULL;
    int n2_scalar_copyback = 0;
    int n2_scalar_is_array = 0;
    PyObject* py_a1 = NULL;
    PyObject* py_a2 = NULL;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"n1", "n2", "a1", "a2", "output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO", kwlist, &py_n1, &py_n2, &py_a1, &py_a2, &py_output)) {
        return NULL;
    }
    
    int* n1 = &n1_val;
    if (PyArray_Check(py_n1)) {
        n1_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n1, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n1_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n1_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n1 must have exactly one element");
            Py_DECREF(n1_scalar_arr);
            return NULL;
        }
        n1_scalar_is_array = 1;
        n1 = (int*)PyArray_DATA(n1_scalar_arr);
        n1_val = n1[0];
        if (PyArray_DATA(n1_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n1) || PyArray_TYPE(n1_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n1)) {
            n1_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n1)) {
        n1_val = (int)PyLong_AsLong(py_n1);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n1 must be a scalar number or NumPy array");
        return NULL;
    }
    int* n2 = &n2_val;
    if (PyArray_Check(py_n2)) {
        n2_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n2, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n2_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n2_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n2 must have exactly one element");
            Py_DECREF(n2_scalar_arr);
            return NULL;
        }
        n2_scalar_is_array = 1;
        n2 = (int*)PyArray_DATA(n2_scalar_arr);
        n2_val = n2[0];
        if (PyArray_DATA(n2_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n2) || PyArray_TYPE(n2_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n2)) {
            n2_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n2)) {
        n2_val = (int)PyLong_AsLong(py_n2);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n2 must be a scalar number or NumPy array");
        return NULL;
    }
    PyArrayObject* a1_arr = NULL;
    float* a1 = NULL;
    /* Extract a1 array data */
    if (!PyArray_Check(py_a1)) {
        PyErr_SetString(PyExc_TypeError, "Argument a1 must be a NumPy array");
        return NULL;
    }
    a1_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_a1, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (a1_arr == NULL) {
        return NULL;
    }
    a1 = (float*)PyArray_DATA(a1_arr);
    int n0_a1 = (int)PyArray_DIM(a1_arr, 0);
    int n1_a1 = (int)PyArray_DIM(a1_arr, 1);
    f90wrap_n0_val = n0_a1;
    f90wrap_n1_val = n1_a1;
    
    PyArrayObject* a2_arr = NULL;
    float* a2 = NULL;
    /* Extract a2 array data */
    if (!PyArray_Check(py_a2)) {
        PyErr_SetString(PyExc_TypeError, "Argument a2 must be a NumPy array");
        return NULL;
    }
    a2_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_a2, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (a2_arr == NULL) {
        return NULL;
    }
    a2 = (float*)PyArray_DATA(a2_arr);
    int n0_a2 = (int)PyArray_DIM(a2_arr, 0);
    int n1_a2 = (int)PyArray_DIM(a2_arr, 1);
    f90wrap_n2_val = n0_a2;
    f90wrap_n3_val = n1_a2;
    
    PyArrayObject* output_arr = NULL;
    PyObject* py_output_arr = NULL;
    int output_needs_copyback = 0;
    float* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (float*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    int n1_output = (int)PyArray_DIM(output_arr, 1);
    f90wrap_n4_val = n0_output;
    f90wrap_n5_val = n1_output;
    Py_INCREF(py_output);
    py_output_arr = py_output;
    if (PyArray_DATA(output_arr) != PyArray_DATA((PyArrayObject*)py_output) || PyArray_TYPE(output_arr) != \
        PyArray_TYPE((PyArrayObject*)py_output)) {
        output_needs_copyback = 1;
    }
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_intent_out__interpolation)(&f90wrap_n0_val, &f90wrap_n1_val, &f90wrap_n2_val, \
        &f90wrap_n3_val, &f90wrap_n4_val, &f90wrap_n5_val, n1, n2, a1, a2, output);
    if (PyErr_Occurred()) {
        Py_XDECREF(a1_arr);
        Py_XDECREF(a2_arr);
        Py_XDECREF(py_output_arr);
        return NULL;
    }
    
    if (n1_scalar_is_array) {
        if (n1_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n1, n1_scalar_arr) < 0) {
                Py_DECREF(n1_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n1_scalar_arr);
    }
    if (n2_scalar_is_array) {
        if (n2_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n2, n2_scalar_arr) < 0) {
                Py_DECREF(n2_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n2_scalar_arr);
    }
    Py_DECREF(a1_arr);
    Py_DECREF(a2_arr);
    if (output_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_output, output_arr) < 0) {
            Py_DECREF(output_arr);
            Py_DECREF(py_output_arr);
            return NULL;
        }
    }
    Py_DECREF(output_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_output_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_output_arr != NULL) return py_output_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_output_arr != NULL) Py_DECREF(py_output_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_output_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_output_arr);
    }
    return result_tuple;
}

/* Method table for _pywrapper module */
static PyMethodDef _pywrapper_methods[] = {
    {"f90wrap_m_intent_out__interpolation", (PyCFunction)wrap_m_intent_out_interpolation, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for interpolation"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _pywrappermodule = {
    PyModuleDef_HEAD_INIT,
    "pywrapper",
    "Direct-C wrapper for _pywrapper module",
    -1,
    _pywrapper_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__pywrapper(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_pywrappermodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
