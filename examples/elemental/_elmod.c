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
extern void F90WRAP_F_SYMBOL(f90wrap_elemental_module__sinc)(double* x, double* ret_sinc);

static PyObject* wrap_elemental_module_sinc(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_x = NULL;
    double x_val = 0;
    PyArrayObject* x_scalar_arr = NULL;
    int x_scalar_copyback = 0;
    int x_scalar_is_array = 0;
    double ret_sinc_val = 0;
    static char *kwlist[] = {"x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_x)) {
        return NULL;
    }
    
    double* x = &x_val;
    if (PyArray_Check(py_x)) {
        x_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_x, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (x_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(x_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument x must have exactly one element");
            Py_DECREF(x_scalar_arr);
            return NULL;
        }
        x_scalar_is_array = 1;
        x = (double*)PyArray_DATA(x_scalar_arr);
        x_val = x[0];
        if (PyArray_DATA(x_scalar_arr) != PyArray_DATA((PyArrayObject*)py_x) || PyArray_TYPE(x_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_x)) {
            x_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_x)) {
        x_val = (double)PyFloat_AsDouble(py_x);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument x must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_elemental_module__sinc)(x, &ret_sinc_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (x_scalar_is_array) {
        if (x_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_x, x_scalar_arr) < 0) {
                Py_DECREF(x_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(x_scalar_arr);
    }
    PyObject* py_ret_sinc_obj = Py_BuildValue("d", ret_sinc_val);
    if (py_ret_sinc_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_sinc_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_sinc_obj != NULL) return py_ret_sinc_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_sinc_obj != NULL) Py_DECREF(py_ret_sinc_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_sinc_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_sinc_obj);
    }
    return result_tuple;
}

/* Method table for _elmod module */
static PyMethodDef _elmod_methods[] = {
    {"f90wrap_elemental_module__sinc", (PyCFunction)wrap_elemental_module_sinc, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        sinc"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _elmodmodule = {
    PyModuleDef_HEAD_INIT,
    "elmod",
    "Direct-C wrapper for _elmod module",
    -1,
    _elmod_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__elmod(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_elmodmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
