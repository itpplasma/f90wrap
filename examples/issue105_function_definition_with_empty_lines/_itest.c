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
extern void F90WRAP_F_SYMBOL(f90wrap_itestit__testit1)(int* f90wrap_n0, float* x);
extern void F90WRAP_F_SYMBOL(f90wrap_itestit__testit2)(int* f90wrap_n0, float* x);

static PyObject* wrap_itestit_testit1(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* x_arr = NULL;
    PyObject* py_x_arr = NULL;
    int x_needs_copyback = 0;
    float* x = NULL;
    /* Extract x array data */
    if (!PyArray_Check(py_x)) {
        PyErr_SetString(PyExc_TypeError, "Argument x must be a NumPy array");
        return NULL;
    }
    x_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_x, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (x_arr == NULL) {
        return NULL;
    }
    x = (float*)PyArray_DATA(x_arr);
    int n0_x = (int)PyArray_DIM(x_arr, 0);
    f90wrap_n0_val = n0_x;
    Py_INCREF(py_x);
    py_x_arr = py_x;
    if (PyArray_DATA(x_arr) != PyArray_DATA((PyArrayObject*)py_x) || PyArray_TYPE(x_arr) != \
        PyArray_TYPE((PyArrayObject*)py_x)) {
        x_needs_copyback = 1;
    }
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_itestit__testit1)(&f90wrap_n0_val, x);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_x_arr);
        return NULL;
    }
    
    if (x_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_x, x_arr) < 0) {
            Py_DECREF(x_arr);
            Py_DECREF(py_x_arr);
            return NULL;
        }
    }
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_x_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_x_arr != NULL) return py_x_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_x_arr != NULL) Py_DECREF(py_x_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_x_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_x_arr);
    }
    return result_tuple;
}

static PyObject* wrap_itestit_testit2(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* x_arr = NULL;
    PyObject* py_x_arr = NULL;
    int x_needs_copyback = 0;
    float* x = NULL;
    /* Extract x array data */
    if (!PyArray_Check(py_x)) {
        PyErr_SetString(PyExc_TypeError, "Argument x must be a NumPy array");
        return NULL;
    }
    x_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_x, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (x_arr == NULL) {
        return NULL;
    }
    x = (float*)PyArray_DATA(x_arr);
    int n0_x = (int)PyArray_DIM(x_arr, 0);
    f90wrap_n0_val = n0_x;
    Py_INCREF(py_x);
    py_x_arr = py_x;
    if (PyArray_DATA(x_arr) != PyArray_DATA((PyArrayObject*)py_x) || PyArray_TYPE(x_arr) != \
        PyArray_TYPE((PyArrayObject*)py_x)) {
        x_needs_copyback = 1;
    }
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_itestit__testit2)(&f90wrap_n0_val, x);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_x_arr);
        return NULL;
    }
    
    if (x_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_x, x_arr) < 0) {
            Py_DECREF(x_arr);
            Py_DECREF(py_x_arr);
            return NULL;
        }
    }
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_x_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_x_arr != NULL) return py_x_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_x_arr != NULL) Py_DECREF(py_x_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_x_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_x_arr);
    }
    return result_tuple;
}

/* Method table for _itest module */
static PyMethodDef _itest_methods[] = {
    {"f90wrap_itestit__testit1", (PyCFunction)wrap_itestit_testit1, METH_VARARGS | METH_KEYWORDS, "Wrapper for testit1"},
    {"f90wrap_itestit__testit2", (PyCFunction)wrap_itestit_testit2, METH_VARARGS | METH_KEYWORDS, "Wrapper for testit2"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _itestmodule = {
    PyModuleDef_HEAD_INIT,
    "itest",
    "Direct-C wrapper for _itest module",
    -1,
    _itest_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__itest(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_itestmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
