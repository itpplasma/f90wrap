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
extern void F90WRAP_F_SYMBOL(f90wrap_library__do_array_stuff)(int* f90wrap_n0, int* f90wrap_n1, int* f90wrap_n2, int* \
    f90wrap_n3, int* n, double* x, double* y, double* br, double* co);
extern void F90WRAP_F_SYMBOL(f90wrap_library__only_manipulate)(int* f90wrap_n0, int* n, double* array);
extern void F90WRAP_F_SYMBOL(f90wrap_library__return_array)(int* f90wrap_n0, int* f90wrap_n1, int* m, int* n, int* \
    output);
extern void F90WRAP_F_SYMBOL(f90wrap_parameters__get__idp)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_parameters__get__isp)(int* value);

static PyObject* wrap_library_do_array_stuff(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    int f90wrap_n2_val = 0;
    int f90wrap_n3_val = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    PyObject* py_x = NULL;
    PyObject* py_y = NULL;
    PyObject* py_br = NULL;
    PyObject* py_co = NULL;
    static char *kwlist[] = {"n", "x", "y", "br", "co", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO", kwlist, &py_n, &py_x, &py_y, &py_br, &py_co)) {
        return NULL;
    }
    
    int* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n must have exactly one element");
            Py_DECREF(n_scalar_arr);
            return NULL;
        }
        n_scalar_is_array = 1;
        n = (int*)PyArray_DATA(n_scalar_arr);
        n_val = n[0];
        if (PyArray_DATA(n_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n) || PyArray_TYPE(n_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n)) {
            n_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n)) {
        n_val = (int)PyLong_AsLong(py_n);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n must be a scalar number or NumPy array");
        return NULL;
    }
    PyArrayObject* x_arr = NULL;
    double* x = NULL;
    /* Extract x array data */
    if (!PyArray_Check(py_x)) {
        PyErr_SetString(PyExc_TypeError, "Argument x must be a NumPy array");
        return NULL;
    }
    x_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_x, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (x_arr == NULL) {
        return NULL;
    }
    x = (double*)PyArray_DATA(x_arr);
    int n0_x = (int)PyArray_DIM(x_arr, 0);
    f90wrap_n0_val = n0_x;
    
    PyArrayObject* y_arr = NULL;
    double* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (double*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    f90wrap_n1_val = n0_y;
    
    PyArrayObject* br_arr = NULL;
    PyObject* py_br_arr = NULL;
    int br_needs_copyback = 0;
    double* br = NULL;
    /* Extract br array data */
    if (!PyArray_Check(py_br)) {
        PyErr_SetString(PyExc_TypeError, "Argument br must be a NumPy array");
        return NULL;
    }
    br_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_br, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (br_arr == NULL) {
        return NULL;
    }
    br = (double*)PyArray_DATA(br_arr);
    int n0_br = (int)PyArray_DIM(br_arr, 0);
    f90wrap_n2_val = n0_br;
    Py_INCREF(py_br);
    py_br_arr = py_br;
    if (PyArray_DATA(br_arr) != PyArray_DATA((PyArrayObject*)py_br) || PyArray_TYPE(br_arr) != \
        PyArray_TYPE((PyArrayObject*)py_br)) {
        br_needs_copyback = 1;
    }
    
    PyArrayObject* co_arr = NULL;
    PyObject* py_co_arr = NULL;
    int co_needs_copyback = 0;
    double* co = NULL;
    /* Extract co array data */
    if (!PyArray_Check(py_co)) {
        PyErr_SetString(PyExc_TypeError, "Argument co must be a NumPy array");
        return NULL;
    }
    co_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_co, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (co_arr == NULL) {
        return NULL;
    }
    co = (double*)PyArray_DATA(co_arr);
    int n0_co = (int)PyArray_DIM(co_arr, 0);
    int n1_co = (int)PyArray_DIM(co_arr, 1);
    f90wrap_n3_val = n1_co;
    Py_INCREF(py_co);
    py_co_arr = py_co;
    if (PyArray_DATA(co_arr) != PyArray_DATA((PyArrayObject*)py_co) || PyArray_TYPE(co_arr) != \
        PyArray_TYPE((PyArrayObject*)py_co)) {
        co_needs_copyback = 1;
    }
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__do_array_stuff)(&f90wrap_n0_val, &f90wrap_n1_val, &f90wrap_n2_val, &f90wrap_n3_val, n, \
        x, y, br, co);
    if (PyErr_Occurred()) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(py_br_arr);
        Py_XDECREF(py_co_arr);
        return NULL;
    }
    
    if (n_scalar_is_array) {
        if (n_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n, n_scalar_arr) < 0) {
                Py_DECREF(n_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n_scalar_arr);
    }
    Py_DECREF(x_arr);
    Py_DECREF(y_arr);
    if (br_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_br, br_arr) < 0) {
            Py_DECREF(br_arr);
            Py_DECREF(py_br_arr);
            return NULL;
        }
    }
    Py_DECREF(br_arr);
    if (co_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_co, co_arr) < 0) {
            Py_DECREF(co_arr);
            Py_DECREF(py_co_arr);
            return NULL;
        }
    }
    Py_DECREF(co_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_br_arr != NULL) result_count++;
    if (py_co_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_br_arr != NULL) return py_br_arr;
        if (py_co_arr != NULL) return py_co_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_br_arr != NULL) Py_DECREF(py_br_arr);
        if (py_co_arr != NULL) Py_DECREF(py_co_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_br_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_br_arr);
    }
    if (py_co_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_co_arr);
    }
    return result_tuple;
}

static PyObject* wrap_library_only_manipulate(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    PyObject* py_array = NULL;
    static char *kwlist[] = {"n", "array", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_n, &py_array)) {
        return NULL;
    }
    
    int* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n must have exactly one element");
            Py_DECREF(n_scalar_arr);
            return NULL;
        }
        n_scalar_is_array = 1;
        n = (int*)PyArray_DATA(n_scalar_arr);
        n_val = n[0];
        if (PyArray_DATA(n_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n) || PyArray_TYPE(n_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n)) {
            n_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n)) {
        n_val = (int)PyLong_AsLong(py_n);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n must be a scalar number or NumPy array");
        return NULL;
    }
    PyArrayObject* array_arr = NULL;
    PyObject* py_array_arr = NULL;
    int array_needs_copyback = 0;
    double* array = NULL;
    /* Extract array array data */
    if (!PyArray_Check(py_array)) {
        PyErr_SetString(PyExc_TypeError, "Argument array must be a NumPy array");
        return NULL;
    }
    array_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_array, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (array_arr == NULL) {
        return NULL;
    }
    array = (double*)PyArray_DATA(array_arr);
    int n0_array = (int)PyArray_DIM(array_arr, 0);
    int n1_array = (int)PyArray_DIM(array_arr, 1);
    f90wrap_n0_val = n1_array;
    Py_INCREF(py_array);
    py_array_arr = py_array;
    if (PyArray_DATA(array_arr) != PyArray_DATA((PyArrayObject*)py_array) || PyArray_TYPE(array_arr) != \
        PyArray_TYPE((PyArrayObject*)py_array)) {
        array_needs_copyback = 1;
    }
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__only_manipulate)(&f90wrap_n0_val, n, array);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_array_arr);
        return NULL;
    }
    
    if (n_scalar_is_array) {
        if (n_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n, n_scalar_arr) < 0) {
                Py_DECREF(n_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n_scalar_arr);
    }
    if (array_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_array, array_arr) < 0) {
            Py_DECREF(array_arr);
            Py_DECREF(py_array_arr);
            return NULL;
        }
    }
    Py_DECREF(array_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_array_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_array_arr != NULL) return py_array_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_array_arr != NULL) Py_DECREF(py_array_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_array_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_array_arr);
    }
    return result_tuple;
}

static PyObject* wrap_library_return_array(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_m = NULL;
    int m_val = 0;
    PyArrayObject* m_scalar_arr = NULL;
    int m_scalar_copyback = 0;
    int m_scalar_is_array = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"m", "n", "output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_m, &py_n, &py_output)) {
        return NULL;
    }
    
    int* m = &m_val;
    if (PyArray_Check(py_m)) {
        m_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_m, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (m_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(m_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument m must have exactly one element");
            Py_DECREF(m_scalar_arr);
            return NULL;
        }
        m_scalar_is_array = 1;
        m = (int*)PyArray_DATA(m_scalar_arr);
        m_val = m[0];
        if (PyArray_DATA(m_scalar_arr) != PyArray_DATA((PyArrayObject*)py_m) || PyArray_TYPE(m_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_m)) {
            m_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_m)) {
        m_val = (int)PyLong_AsLong(py_m);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument m must be a scalar number or NumPy array");
        return NULL;
    }
    int* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n must have exactly one element");
            Py_DECREF(n_scalar_arr);
            return NULL;
        }
        n_scalar_is_array = 1;
        n = (int*)PyArray_DATA(n_scalar_arr);
        n_val = n[0];
        if (PyArray_DATA(n_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n) || PyArray_TYPE(n_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n)) {
            n_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n)) {
        n_val = (int)PyLong_AsLong(py_n);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n must be a scalar number or NumPy array");
        return NULL;
    }
    PyArrayObject* output_arr = NULL;
    PyObject* py_output_arr = NULL;
    int output_needs_copyback = 0;
    int* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (int*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    int n1_output = (int)PyArray_DIM(output_arr, 1);
    f90wrap_n0_val = n0_output;
    f90wrap_n1_val = n1_output;
    Py_INCREF(py_output);
    py_output_arr = py_output;
    if (PyArray_DATA(output_arr) != PyArray_DATA((PyArrayObject*)py_output) || PyArray_TYPE(output_arr) != \
        PyArray_TYPE((PyArrayObject*)py_output)) {
        output_needs_copyback = 1;
    }
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__return_array)(&f90wrap_n0_val, &f90wrap_n1_val, m, n, output);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_output_arr);
        return NULL;
    }
    
    if (m_scalar_is_array) {
        if (m_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_m, m_scalar_arr) < 0) {
                Py_DECREF(m_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(m_scalar_arr);
    }
    if (n_scalar_is_array) {
        if (n_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n, n_scalar_arr) < 0) {
                Py_DECREF(n_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n_scalar_arr);
    }
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

static PyObject* wrap_parameters_helper_get_idp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_parameters__get__idp)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_parameters_helper_get_isp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_parameters__get__isp)(&value);
    return Py_BuildValue("i", value);
}

/* Method table for _ExampleArray_pkg module */
static PyMethodDef _ExampleArray_pkg_methods[] = {
    {"f90wrap_library__do_array_stuff", (PyCFunction)wrap_library_do_array_stuff, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        do_array_stuff"},
    {"f90wrap_library__only_manipulate", (PyCFunction)wrap_library_only_manipulate, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for only_manipulate"},
    {"f90wrap_library__return_array", (PyCFunction)wrap_library_return_array, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        return_array"},
    {"f90wrap_parameters__get__idp", (PyCFunction)wrap_parameters_helper_get_idp, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for idp"},
    {"f90wrap_parameters__get__isp", (PyCFunction)wrap_parameters_helper_get_isp, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for isp"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _ExampleArray_pkgmodule = {
    PyModuleDef_HEAD_INIT,
    "ExampleArray_pkg",
    "Direct-C wrapper for _ExampleArray_pkg module",
    -1,
    _ExampleArray_pkg_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__ExampleArray_pkg(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_ExampleArray_pkgmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
