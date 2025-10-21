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
extern void F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_int1)(short* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_int2)(short* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_int4)(int* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_int8)(long long* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_real4)(float* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_real8)(double* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_array_int4)(int* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_array_int8)(long long* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_array_real4)(float* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_array_real8)(double* ret_output);

static PyObject* wrap_m_out_test_out_scalar_int1(PyObject* self, PyObject* args, PyObject* kwargs)
{
    short ret_output_val = 0;
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_int1)(&ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_out_test_out_scalar_int2(PyObject* self, PyObject* args, PyObject* kwargs)
{
    short ret_output_val = 0;
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_int2)(&ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_out_test_out_scalar_int4(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int ret_output_val = 0;
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_int4)(&ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_out_test_out_scalar_int8(PyObject* self, PyObject* args, PyObject* kwargs)
{
    long long ret_output_val = 0;
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_int8)(&ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_out_test_out_scalar_real4(PyObject* self, PyObject* args, PyObject* kwargs)
{
    float ret_output_val = 0;
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_real4)(&ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_ret_output_obj = Py_BuildValue("d", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_out_test_out_scalar_real8(PyObject* self, PyObject* args, PyObject* kwargs)
{
    double ret_output_val = 0;
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_scalar_real8)(&ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_ret_output_obj = Py_BuildValue("d", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_out_test_out_array_int4(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyArrayObject* ret_output_arr = NULL;
    PyObject* py_ret_output_arr = NULL;
    int* ret_output = NULL;
    npy_intp ret_output_dim_0 = (npy_intp)(1);
    if (ret_output_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_output must be positive");
        return NULL;
    }
    npy_intp ret_output_dims[1] = {ret_output_dim_0};
    py_ret_output_arr = PyArray_SimpleNew(1, ret_output_dims, NPY_INT32);
    if (py_ret_output_arr == NULL) {
        return NULL;
    }
    ret_output_arr = (PyArrayObject*)py_ret_output_arr;
    ret_output = (int*)PyArray_DATA(ret_output_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_array_int4)(ret_output);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_ret_output_arr);
        return NULL;
    }
    
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_arr != NULL) return py_ret_output_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_arr != NULL) Py_DECREF(py_ret_output_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_out_test_out_array_int8(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyArrayObject* ret_output_arr = NULL;
    PyObject* py_ret_output_arr = NULL;
    long long* ret_output = NULL;
    npy_intp ret_output_dim_0 = (npy_intp)(1);
    if (ret_output_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_output must be positive");
        return NULL;
    }
    npy_intp ret_output_dims[1] = {ret_output_dim_0};
    py_ret_output_arr = PyArray_SimpleNew(1, ret_output_dims, NPY_INT64);
    if (py_ret_output_arr == NULL) {
        return NULL;
    }
    ret_output_arr = (PyArrayObject*)py_ret_output_arr;
    ret_output = (long long*)PyArray_DATA(ret_output_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_array_int8)(ret_output);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_ret_output_arr);
        return NULL;
    }
    
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_arr != NULL) return py_ret_output_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_arr != NULL) Py_DECREF(py_ret_output_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_out_test_out_array_real4(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyArrayObject* ret_output_arr = NULL;
    PyObject* py_ret_output_arr = NULL;
    float* ret_output = NULL;
    npy_intp ret_output_dim_0 = (npy_intp)(1);
    if (ret_output_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_output must be positive");
        return NULL;
    }
    npy_intp ret_output_dims[1] = {ret_output_dim_0};
    py_ret_output_arr = PyArray_SimpleNew(1, ret_output_dims, NPY_FLOAT32);
    if (py_ret_output_arr == NULL) {
        return NULL;
    }
    ret_output_arr = (PyArrayObject*)py_ret_output_arr;
    ret_output = (float*)PyArray_DATA(ret_output_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_array_real4)(ret_output);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_ret_output_arr);
        return NULL;
    }
    
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_arr != NULL) return py_ret_output_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_arr != NULL) Py_DECREF(py_ret_output_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_out_test_out_array_real8(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyArrayObject* ret_output_arr = NULL;
    PyObject* py_ret_output_arr = NULL;
    double* ret_output = NULL;
    npy_intp ret_output_dim_0 = (npy_intp)(1);
    if (ret_output_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_output must be positive");
        return NULL;
    }
    npy_intp ret_output_dims[1] = {ret_output_dim_0};
    py_ret_output_arr = PyArray_SimpleNew(1, ret_output_dims, NPY_FLOAT64);
    if (py_ret_output_arr == NULL) {
        return NULL;
    }
    ret_output_arr = (PyArrayObject*)py_ret_output_arr;
    ret_output = (double*)PyArray_DATA(ret_output_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_out_test__out_array_real8)(ret_output);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_ret_output_arr);
        return NULL;
    }
    
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_arr != NULL) return py_ret_output_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_arr != NULL) Py_DECREF(py_ret_output_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_arr);
    }
    return result_tuple;
}

/* Method table for _pywrapper module */
static PyMethodDef _pywrapper_methods[] = {
    {"f90wrap_m_out_test__out_scalar_int1", (PyCFunction)wrap_m_out_test_out_scalar_int1, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for out_scalar_int1"},
    {"f90wrap_m_out_test__out_scalar_int2", (PyCFunction)wrap_m_out_test_out_scalar_int2, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for out_scalar_int2"},
    {"f90wrap_m_out_test__out_scalar_int4", (PyCFunction)wrap_m_out_test_out_scalar_int4, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for out_scalar_int4"},
    {"f90wrap_m_out_test__out_scalar_int8", (PyCFunction)wrap_m_out_test_out_scalar_int8, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for out_scalar_int8"},
    {"f90wrap_m_out_test__out_scalar_real4", (PyCFunction)wrap_m_out_test_out_scalar_real4, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for out_scalar_real4"},
    {"f90wrap_m_out_test__out_scalar_real8", (PyCFunction)wrap_m_out_test_out_scalar_real8, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for out_scalar_real8"},
    {"f90wrap_m_out_test__out_array_int4", (PyCFunction)wrap_m_out_test_out_array_int4, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for out_array_int4"},
    {"f90wrap_m_out_test__out_array_int8", (PyCFunction)wrap_m_out_test_out_array_int8, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for out_array_int8"},
    {"f90wrap_m_out_test__out_array_real4", (PyCFunction)wrap_m_out_test_out_array_real4, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for out_array_real4"},
    {"f90wrap_m_out_test__out_array_real8", (PyCFunction)wrap_m_out_test_out_array_real8, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for out_array_real8"},
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
