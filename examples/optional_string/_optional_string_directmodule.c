/* C Extension module for _optional_string_direct */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <complex.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Shared capsule helper functions */
/* Note: The capsule_helpers.h file should be in the same directory as this generated code
   or you can adjust the include path as needed */
#include "capsule_helpers.h"

/* Fortran subroutine prototypes */


/* Derived type definitions */

/* Fortran subroutine prototypes */

/* Python wrapper functions */


/* Wrapper for wrap_string_out_optional_array */
static char wrap_string_out_optional_array__doc__[] = "Wrapper for string_out_optional_array";

static PyObject* wrap_string_out_optional_array(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_output = NULL;
    int output_present = 0;
    char** output_data = NULL;

    if (!PyArg_ParseTuple(args, "|O", &py_output)) {
        return NULL;
    }

    if (py_output != NULL && py_output != Py_None) {
        output_present = 1;
        /* Extract Fortran array from NumPy py_output */
        if (!PyArray_Check(py_output)) {
            PyErr_SetString(PyExc_TypeError, "Expected NumPy array for output");
            return NULL;
        }

        if (PyArray_NDIM((PyArrayObject*)py_output) != 1) {
            PyErr_Format(PyExc_ValueError, "Array output must have 1 dimensions, got %d",
                         PyArray_NDIM((PyArrayObject*)py_output));
            return NULL;
        }

        if (PyArray_TYPE((PyArrayObject*)py_output) != NPY_STRING) {
            PyErr_SetString(PyExc_TypeError, "Array output has wrong dtype");
            return NULL;
        }

        PyArrayObject *output_data_array = (PyArrayObject*)py_output;
        if (!PyArray_IS_F_CONTIGUOUS(output_data_array)) {
            output_data_array = (PyArrayObject*)PyArray_FromArray(
                output_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
            if (output_data_array == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
                return NULL;
            }
        }

        output_data = (char**)PyArray_DATA(output_data_array);

    }

    /* Call Fortran subroutine */
    extern void __m_string_test_MOD_string_out_optional_array(void*);
    __m_string_test_MOD_string_out_optional_array((output_present ? output_data : NULL));

    Py_RETURN_NONE;
}



/* Wrapper for wrap_string_to_string */
static char wrap_string_to_string__doc__[] = "Wrapper for string_to_string";

static PyObject* wrap_string_to_string(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_input = NULL;
    char* input;
    char* output;

    if (!PyArg_ParseTuple(args, "s", &py_input)) {
        return NULL;
    }

    input = (char*)PyUnicode_AsUTF8(py_input);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument input");
        return NULL;
    }
    output = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void __m_string_test_MOD_string_to_string(char**, char**);
    __m_string_test_MOD_string_to_string(&input, &output);

    /* Build return tuple for output arguments */
    return PyUnicode_FromString(output);
}



/* Wrapper for wrap_string_in */
static char wrap_string_in__doc__[] = "Wrapper for string_in";

static PyObject* wrap_string_in(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_input = NULL;
    char* input;

    if (!PyArg_ParseTuple(args, "s", &py_input)) {
        return NULL;
    }

    input = (char*)PyUnicode_AsUTF8(py_input);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument input");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_string_test_MOD_string_in(char**);
    __m_string_test_MOD_string_in(&input);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_string_in_array */
static char wrap_string_in_array__doc__[] = "Wrapper for string_in_array";

static PyObject* wrap_string_in_array(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_input = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_input)) {
        return NULL;
    }

    /* Extract Fortran array from NumPy py_input */
    if (!PyArray_Check(py_input)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for input");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_input) != 1) {
        PyErr_Format(PyExc_ValueError, "Array input must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_input));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_input) != NPY_STRING) {
        PyErr_SetString(PyExc_TypeError, "Array input has wrong dtype");
        return NULL;
    }

    PyArrayObject *input_data_array = (PyArrayObject*)py_input;
    if (!PyArray_IS_F_CONTIGUOUS(input_data_array)) {
        input_data_array = (PyArrayObject*)PyArray_FromArray(
            input_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (input_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    char** input_data = (char**)PyArray_DATA(input_data_array);


    /* Call Fortran subroutine */
    extern void __m_string_test_MOD_string_in_array(void*);
    __m_string_test_MOD_string_in_array(input_data);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_string_out */
static char wrap_string_out__doc__[] = "Wrapper for string_out";

static PyObject* wrap_string_out(PyObject *self, PyObject *args, PyObject *kwargs) {

    char* output;

    output = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void __m_string_test_MOD_string_out(char**);
    __m_string_test_MOD_string_out(&output);

    /* Build return tuple for output arguments */
    return PyUnicode_FromString(output);
}



/* Wrapper for wrap_string_to_string_array */
static char wrap_string_to_string_array__doc__[] = "Wrapper for string_to_string_array";

static PyObject* wrap_string_to_string_array(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_input = NULL;
    PyObject *py_output = NULL;

    if (!PyArg_ParseTuple(args, "OO", &py_input, &py_output)) {
        return NULL;
    }

    /* Extract Fortran array from NumPy py_input */
    if (!PyArray_Check(py_input)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for input");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_input) != 1) {
        PyErr_Format(PyExc_ValueError, "Array input must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_input));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_input) != NPY_STRING) {
        PyErr_SetString(PyExc_TypeError, "Array input has wrong dtype");
        return NULL;
    }

    PyArrayObject *input_data_array = (PyArrayObject*)py_input;
    if (!PyArray_IS_F_CONTIGUOUS(input_data_array)) {
        input_data_array = (PyArrayObject*)PyArray_FromArray(
            input_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (input_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    char** input_data = (char**)PyArray_DATA(input_data_array);

    /* Extract Fortran array from NumPy py_output */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for output");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_output) != 1) {
        PyErr_Format(PyExc_ValueError, "Array output must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_output));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_output) != NPY_STRING) {
        PyErr_SetString(PyExc_TypeError, "Array output has wrong dtype");
        return NULL;
    }

    PyArrayObject *output_data_array = (PyArrayObject*)py_output;
    if (!PyArray_IS_F_CONTIGUOUS(output_data_array)) {
        output_data_array = (PyArrayObject*)PyArray_FromArray(
            output_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (output_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    char** output_data = (char**)PyArray_DATA(output_data_array);


    /* Call Fortran subroutine */
    extern void __m_string_test_MOD_string_to_string_array(void*, void*);
    __m_string_test_MOD_string_to_string_array(input_data, output_data);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_string_in_array_hardcoded_size */
static char wrap_string_in_array_hardcoded_size__doc__[] = "Wrapper for string_in_array_hardcoded_size";

static PyObject* wrap_string_in_array_hardcoded_size(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_input = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_input)) {
        return NULL;
    }

    /* Extract Fortran array from NumPy py_input */
    if (!PyArray_Check(py_input)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for input");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_input) != 1) {
        PyErr_Format(PyExc_ValueError, "Array input must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_input));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_input) != NPY_STRING) {
        PyErr_SetString(PyExc_TypeError, "Array input has wrong dtype");
        return NULL;
    }

    PyArrayObject *input_data_array = (PyArrayObject*)py_input;
    if (!PyArray_IS_F_CONTIGUOUS(input_data_array)) {
        input_data_array = (PyArrayObject*)PyArray_FromArray(
            input_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (input_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    char** input_data = (char**)PyArray_DATA(input_data_array);


    /* Call Fortran subroutine */
    extern void __m_string_test_MOD_string_in_array_hardcoded_size(void*);
    __m_string_test_MOD_string_in_array_hardcoded_size(input_data);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_string_out_optional */
static char wrap_string_out_optional__doc__[] = "Wrapper for string_out_optional";

static PyObject* wrap_string_out_optional(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_output = NULL;
    char* output;
    int output_present = 0;

    if (!PyArg_ParseTuple(args, "|s", &py_output)) {
        return NULL;
    }

    if (py_output != NULL) {
        output_present = 1;
        output = (char*)PyUnicode_AsUTF8(py_output);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Failed to convert argument output");
            return NULL;
        }
    }

    /* Call Fortran subroutine */
    extern void __m_string_test_MOD_string_out_optional(char**);
    __m_string_test_MOD_string_out_optional((output_present ? &output : NULL));

    /* Build return tuple for output arguments */
    return PyUnicode_FromString(output);
}



/* Method table */
static PyMethodDef _optional_string_direct_methods[] = {
    {"wrap_string_out_optional_array", (PyCFunction)wrap_string_out_optional_array, METH_VARARGS, wrap_string_out_optional_array__doc__},
    {"wrap_string_to_string", (PyCFunction)wrap_string_to_string, METH_VARARGS, wrap_string_to_string__doc__},
    {"wrap_string_in", (PyCFunction)wrap_string_in, METH_VARARGS, wrap_string_in__doc__},
    {"wrap_string_in_array", (PyCFunction)wrap_string_in_array, METH_VARARGS, wrap_string_in_array__doc__},
    {"wrap_string_out", (PyCFunction)wrap_string_out, METH_VARARGS, wrap_string_out__doc__},
    {"wrap_string_to_string_array", (PyCFunction)wrap_string_to_string_array, METH_VARARGS, wrap_string_to_string_array__doc__},
    {"wrap_string_in_array_hardcoded_size", (PyCFunction)wrap_string_in_array_hardcoded_size, METH_VARARGS, wrap_string_in_array_hardcoded_size__doc__},
    {"wrap_string_out_optional", (PyCFunction)wrap_string_out_optional, METH_VARARGS, wrap_string_out_optional__doc__},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Module definition */
static struct PyModuleDef _optional_string_direct_module = {
    PyModuleDef_HEAD_INIT,
    "_optional_string_direct",
    "Fortran module _optional_string_direct wrapped with f90wrap",
    -1,
    _optional_string_direct_methods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit__optional_string_direct(void) {
    PyObject *module;

    /* Import NumPy C API */
    import_array();

    /* Create module */
    module = PyModule_Create(&_optional_string_direct_module);
    if (module == NULL) {
        return NULL;
    }

    return module;
}
