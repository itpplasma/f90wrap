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
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__one_array_dynamic)(int* f90wrap_n0, int* f90wrap_n1, float* x, float* \
    ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__one_array_fixed)(float* x, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__one_array_fixed_range)(float* x, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__one_array_explicit)(int* f90wrap_n0, int* f90wrap_n1, float* x, int* \
    n, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__one_array_explicit_range)(int* f90wrap_n0, int* f90wrap_n1, float* x, \
    int* n, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_dynamic)(int* f90wrap_n0, int* f90wrap_n1, int* \
    f90wrap_n2, float* y, float* x, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_fixed)(float* y, float* x, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_mixed)(int* f90wrap_n0, int* f90wrap_n1, float* y, float* \
    x, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_2d_dynamic)(int* f90wrap_n0, int* f90wrap_n1, int* \
    f90wrap_n2, int* f90wrap_n3, float* y, float* x, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_2d_fixed)(float* y, float* x, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_2d_fixed_whitespace)(float* y, float* x, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_2d_mixed)(int* f90wrap_n0, int* f90wrap_n1, float* y, \
    float* x, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__get_container)(int* f90wrap_n0, float* x, int* ret_c);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__array_container_dynamic)(int* f90wrap_n0, int* f90wrap_n1, int* c, \
    float* y, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__array_container_fixed)(int* f90wrap_n0, int* c, float* y, float* \
    ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__array_container_dynamic_2d)(int* f90wrap_n0, int* f90wrap_n1, int* \
    f90wrap_n2, int* n, int* c, float* y, float* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__container_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__container_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__container__get__n_data)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__container__set__n_data)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_array_shapes__container__array__data)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);

static PyObject* wrap_array_shapes_one_array_dynamic(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* x_arr = NULL;
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
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(f90wrap_n1_val);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_res_dim_0;
    npy_intp ret_res_dims[1] = {ret_res_dim_0};
    py_ret_res_arr = PyArray_SimpleNew(1, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__one_array_dynamic)(&f90wrap_n0_val, &f90wrap_n1_val, x, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_one_array_fixed(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_x = NULL;
    static char *kwlist[] = {"x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* x_arr = NULL;
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
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(3);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    npy_intp ret_res_dims[1] = {ret_res_dim_0};
    py_ret_res_arr = PyArray_SimpleNew(1, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__one_array_fixed)(x, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_one_array_fixed_range(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_x = NULL;
    static char *kwlist[] = {"x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* x_arr = NULL;
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
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(((3) - (1) + 1));
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    npy_intp ret_res_dims[1] = {ret_res_dim_0};
    py_ret_res_arr = PyArray_SimpleNew(1, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__one_array_fixed_range)(x, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_one_array_explicit(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_x = NULL;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"x", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_x, &py_n)) {
        return NULL;
    }
    
    PyArrayObject* x_arr = NULL;
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
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(f90wrap_n1_val);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_res_dim_0;
    npy_intp ret_res_dims[1] = {ret_res_dim_0};
    py_ret_res_arr = PyArray_SimpleNew(1, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__one_array_explicit)(&f90wrap_n0_val, &f90wrap_n1_val, x, n, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
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
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_one_array_explicit_range(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_x = NULL;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"x", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_x, &py_n)) {
        return NULL;
    }
    
    PyArrayObject* x_arr = NULL;
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
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(f90wrap_n1_val);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_res_dim_0;
    npy_intp ret_res_dims[1] = {ret_res_dim_0};
    py_ret_res_arr = PyArray_SimpleNew(1, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__one_array_explicit_range)(&f90wrap_n0_val, &f90wrap_n1_val, x, n, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
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
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_two_arrays_dynamic(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    int f90wrap_n2_val = 0;
    PyObject* py_y = NULL;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"y", "x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_y, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* y_arr = NULL;
    float* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (float*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    f90wrap_n0_val = n0_y;
    
    PyArrayObject* x_arr = NULL;
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
    f90wrap_n1_val = n0_x;
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(f90wrap_n2_val);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n2_val = (int)ret_res_dim_0;
    npy_intp ret_res_dims[1] = {ret_res_dim_0};
    py_ret_res_arr = PyArray_SimpleNew(1, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_dynamic)(&f90wrap_n0_val, &f90wrap_n1_val, &f90wrap_n2_val, y, x, \
        ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(y_arr);
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_two_arrays_fixed(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_y = NULL;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"y", "x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_y, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* y_arr = NULL;
    float* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (float*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    
    PyArrayObject* x_arr = NULL;
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
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(3);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    npy_intp ret_res_dims[1] = {ret_res_dim_0};
    py_ret_res_arr = PyArray_SimpleNew(1, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_fixed)(y, x, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(y_arr);
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_two_arrays_mixed(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_y = NULL;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"y", "x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_y, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* y_arr = NULL;
    float* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (float*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    
    PyArrayObject* x_arr = NULL;
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
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(f90wrap_n1_val);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_res_dim_0;
    npy_intp ret_res_dims[1] = {ret_res_dim_0};
    py_ret_res_arr = PyArray_SimpleNew(1, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_mixed)(&f90wrap_n0_val, &f90wrap_n1_val, y, x, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(y_arr);
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_two_arrays_2d_dynamic(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    int f90wrap_n2_val = 0;
    int f90wrap_n3_val = 0;
    PyObject* py_y = NULL;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"y", "x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_y, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* y_arr = NULL;
    float* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (float*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    f90wrap_n0_val = n0_y;
    
    PyArrayObject* x_arr = NULL;
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
    f90wrap_n1_val = n0_x;
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(f90wrap_n2_val);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n2_val = (int)ret_res_dim_0;
    npy_intp ret_res_dim_1 = (npy_intp)(f90wrap_n3_val);
    if (ret_res_dim_1 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n3_val = (int)ret_res_dim_1;
    npy_intp ret_res_dims[2] = {ret_res_dim_0, ret_res_dim_1};
    py_ret_res_arr = PyArray_SimpleNew(2, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_2d_dynamic)(&f90wrap_n0_val, &f90wrap_n1_val, &f90wrap_n2_val, \
        &f90wrap_n3_val, y, x, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(y_arr);
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_two_arrays_2d_fixed(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_y = NULL;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"y", "x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_y, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* y_arr = NULL;
    float* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (float*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    
    PyArrayObject* x_arr = NULL;
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
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(3);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    npy_intp ret_res_dim_1 = (npy_intp)(2);
    if (ret_res_dim_1 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    npy_intp ret_res_dims[2] = {ret_res_dim_0, ret_res_dim_1};
    py_ret_res_arr = PyArray_SimpleNew(2, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_2d_fixed)(y, x, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(y_arr);
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_two_arrays_2d_fixed_whitespace(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_y = NULL;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"y", "x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_y, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* y_arr = NULL;
    float* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (float*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    
    PyArrayObject* x_arr = NULL;
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
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(3);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    npy_intp ret_res_dim_1 = (npy_intp)(2);
    if (ret_res_dim_1 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    npy_intp ret_res_dims[2] = {ret_res_dim_0, ret_res_dim_1};
    py_ret_res_arr = PyArray_SimpleNew(2, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_2d_fixed_whitespace)(y, x, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(y_arr);
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_two_arrays_2d_mixed(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_y = NULL;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"y", "x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_y, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* y_arr = NULL;
    float* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (float*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    
    PyArrayObject* x_arr = NULL;
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
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(f90wrap_n1_val);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_res_dim_0;
    npy_intp ret_res_dim_1 = (npy_intp)(2);
    if (ret_res_dim_1 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    npy_intp ret_res_dims[2] = {ret_res_dim_0, ret_res_dim_1};
    py_ret_res_arr = PyArray_SimpleNew(2, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__two_arrays_2d_mixed)(&f90wrap_n0_val, &f90wrap_n1_val, y, x, ret_res);
    if (PyErr_Occurred()) {
        Py_XDECREF(y_arr);
        Py_XDECREF(x_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_get_container(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_x = NULL;
    static char *kwlist[] = {"x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_x)) {
        return NULL;
    }
    
    PyArrayObject* x_arr = NULL;
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
    
    int ret_c[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__get_container)(&f90wrap_n0_val, x, ret_c);
    if (PyErr_Occurred()) {
        Py_XDECREF(x_arr);
        return NULL;
    }
    
    Py_DECREF(x_arr);
    PyObject* py_ret_c_obj = PyList_New(4);
    if (py_ret_c_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_c[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_c_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_c_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_c_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_c_obj != NULL) return py_ret_c_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_c_obj != NULL) Py_DECREF(py_ret_c_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_c_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_c_obj);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_array_container_dynamic(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_c = NULL;
    PyObject* py_y = NULL;
    static char *kwlist[] = {"c", "y", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_c, &py_y)) {
        return NULL;
    }
    
    PyObject* c_handle_obj = NULL;
    PyObject* c_sequence = NULL;
    Py_ssize_t c_handle_len = 0;
    if (PyObject_HasAttrString(py_c, "_handle")) {
        c_handle_obj = PyObject_GetAttrString(py_c, "_handle");
        if (c_handle_obj == NULL) {
            return NULL;
        }
        c_sequence = PySequence_Fast(c_handle_obj, "Failed to access handle sequence");
        if (c_sequence == NULL) {
            Py_DECREF(c_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_c)) {
        c_sequence = PySequence_Fast(py_c, "Argument c must be a handle sequence");
        if (c_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument c must be a Fortran derived-type instance");
        return NULL;
    }
    c_handle_len = PySequence_Fast_GET_SIZE(c_sequence);
    if (c_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument c has an invalid handle length");
        Py_DECREF(c_sequence);
        if (c_handle_obj) Py_DECREF(c_handle_obj);
        return NULL;
    }
    int* c = (int*)malloc(sizeof(int) * c_handle_len);
    if (c == NULL) {
        PyErr_NoMemory();
        Py_DECREF(c_sequence);
        if (c_handle_obj) Py_DECREF(c_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < c_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(c_sequence, i);
        if (item == NULL) {
            free(c);
            Py_DECREF(c_sequence);
            if (c_handle_obj) Py_DECREF(c_handle_obj);
            return NULL;
        }
        c[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(c);
            Py_DECREF(c_sequence);
            if (c_handle_obj) Py_DECREF(c_handle_obj);
            return NULL;
        }
    }
    (void)c_handle_len;  /* suppress unused warnings when unchanged */
    
    PyArrayObject* y_arr = NULL;
    float* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (float*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    f90wrap_n0_val = n0_y;
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(f90wrap_n1_val);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_res_dim_0;
    npy_intp ret_res_dims[1] = {ret_res_dim_0};
    py_ret_res_arr = PyArray_SimpleNew(1, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__array_container_dynamic)(&f90wrap_n0_val, &f90wrap_n1_val, c, y, ret_res);
    if (PyErr_Occurred()) {
        if (c_sequence) Py_DECREF(c_sequence);
        if (c_handle_obj) Py_DECREF(c_handle_obj);
        free(c);
        Py_XDECREF(y_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(y_arr);
    if (c_sequence) {
        Py_DECREF(c_sequence);
    }
    if (c_handle_obj) {
        Py_DECREF(c_handle_obj);
    }
    free(c);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_array_container_fixed(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_c = NULL;
    PyObject* py_y = NULL;
    static char *kwlist[] = {"c", "y", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_c, &py_y)) {
        return NULL;
    }
    
    PyObject* c_handle_obj = NULL;
    PyObject* c_sequence = NULL;
    Py_ssize_t c_handle_len = 0;
    if (PyObject_HasAttrString(py_c, "_handle")) {
        c_handle_obj = PyObject_GetAttrString(py_c, "_handle");
        if (c_handle_obj == NULL) {
            return NULL;
        }
        c_sequence = PySequence_Fast(c_handle_obj, "Failed to access handle sequence");
        if (c_sequence == NULL) {
            Py_DECREF(c_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_c)) {
        c_sequence = PySequence_Fast(py_c, "Argument c must be a handle sequence");
        if (c_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument c must be a Fortran derived-type instance");
        return NULL;
    }
    c_handle_len = PySequence_Fast_GET_SIZE(c_sequence);
    if (c_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument c has an invalid handle length");
        Py_DECREF(c_sequence);
        if (c_handle_obj) Py_DECREF(c_handle_obj);
        return NULL;
    }
    int* c = (int*)malloc(sizeof(int) * c_handle_len);
    if (c == NULL) {
        PyErr_NoMemory();
        Py_DECREF(c_sequence);
        if (c_handle_obj) Py_DECREF(c_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < c_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(c_sequence, i);
        if (item == NULL) {
            free(c);
            Py_DECREF(c_sequence);
            if (c_handle_obj) Py_DECREF(c_handle_obj);
            return NULL;
        }
        c[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(c);
            Py_DECREF(c_sequence);
            if (c_handle_obj) Py_DECREF(c_handle_obj);
            return NULL;
        }
    }
    (void)c_handle_len;  /* suppress unused warnings when unchanged */
    
    PyArrayObject* y_arr = NULL;
    float* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (float*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(f90wrap_n0_val);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n0_val = (int)ret_res_dim_0;
    npy_intp ret_res_dims[1] = {ret_res_dim_0};
    py_ret_res_arr = PyArray_SimpleNew(1, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__array_container_fixed)(&f90wrap_n0_val, c, y, ret_res);
    if (PyErr_Occurred()) {
        if (c_sequence) Py_DECREF(c_sequence);
        if (c_handle_obj) Py_DECREF(c_handle_obj);
        free(c);
        Py_XDECREF(y_arr);
        Py_XDECREF(py_ret_res_arr);
        return NULL;
    }
    
    Py_DECREF(y_arr);
    if (c_sequence) {
        Py_DECREF(c_sequence);
    }
    if (c_handle_obj) {
        Py_DECREF(c_handle_obj);
    }
    free(c);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_array_container_dynamic_2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    int f90wrap_n2_val = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    PyObject* py_c = NULL;
    PyObject* py_y = NULL;
    static char *kwlist[] = {"n", "c", "y", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_n, &py_c, &py_y)) {
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
    PyObject* c_handle_obj = NULL;
    PyObject* c_sequence = NULL;
    Py_ssize_t c_handle_len = 0;
    if (PyObject_HasAttrString(py_c, "_handle")) {
        c_handle_obj = PyObject_GetAttrString(py_c, "_handle");
        if (c_handle_obj == NULL) {
            return NULL;
        }
        c_sequence = PySequence_Fast(c_handle_obj, "Failed to access handle sequence");
        if (c_sequence == NULL) {
            Py_DECREF(c_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_c)) {
        c_sequence = PySequence_Fast(py_c, "Argument c must be a handle sequence");
        if (c_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument c must be a Fortran derived-type instance");
        return NULL;
    }
    c_handle_len = PySequence_Fast_GET_SIZE(c_sequence);
    if (c_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument c has an invalid handle length");
        Py_DECREF(c_sequence);
        if (c_handle_obj) Py_DECREF(c_handle_obj);
        return NULL;
    }
    int* c = (int*)malloc(sizeof(int) * c_handle_len);
    if (c == NULL) {
        PyErr_NoMemory();
        Py_DECREF(c_sequence);
        if (c_handle_obj) Py_DECREF(c_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < c_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(c_sequence, i);
        if (item == NULL) {
            free(c);
            Py_DECREF(c_sequence);
            if (c_handle_obj) Py_DECREF(c_handle_obj);
            return NULL;
        }
        c[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(c);
            Py_DECREF(c_sequence);
            if (c_handle_obj) Py_DECREF(c_handle_obj);
            return NULL;
        }
    }
    (void)c_handle_len;  /* suppress unused warnings when unchanged */
    
    PyArrayObject* y_arr = NULL;
    float* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (float*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    f90wrap_n0_val = n0_y;
    
    PyArrayObject* ret_res_arr = NULL;
    PyObject* py_ret_res_arr = NULL;
    float* ret_res = NULL;
    npy_intp ret_res_dim_0 = (npy_intp)(f90wrap_n1_val);
    if (ret_res_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_res_dim_0;
    npy_intp ret_res_dim_1 = (npy_intp)(f90wrap_n2_val);
    if (ret_res_dim_1 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_res must be positive");
        return NULL;
    }
    f90wrap_n2_val = (int)ret_res_dim_1;
    npy_intp ret_res_dims[2] = {ret_res_dim_0, ret_res_dim_1};
    py_ret_res_arr = PyArray_SimpleNew(2, ret_res_dims, NPY_FLOAT32);
    if (py_ret_res_arr == NULL) {
        return NULL;
    }
    ret_res_arr = (PyArrayObject*)py_ret_res_arr;
    ret_res = (float*)PyArray_DATA(ret_res_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__array_container_dynamic_2d)(&f90wrap_n0_val, &f90wrap_n1_val, &f90wrap_n2_val, n, \
        c, y, ret_res);
    if (PyErr_Occurred()) {
        if (c_sequence) Py_DECREF(c_sequence);
        if (c_handle_obj) Py_DECREF(c_handle_obj);
        free(c);
        Py_XDECREF(y_arr);
        Py_XDECREF(py_ret_res_arr);
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
    Py_DECREF(y_arr);
    if (c_sequence) {
        Py_DECREF(c_sequence);
    }
    if (c_handle_obj) {
        Py_DECREF(c_handle_obj);
    }
    free(c);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_arr != NULL) return py_ret_res_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_arr != NULL) Py_DECREF(py_ret_res_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_arr);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_container_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__container_initialise)(this);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_this_obj = PyList_New(4);
    if (py_this_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)this[i]);
        if (item == NULL) {
            Py_DECREF(py_this_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_this_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_this_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_this_obj != NULL) return py_this_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_this_obj != NULL) Py_DECREF(py_this_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_this_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_this_obj);
    }
    return result_tuple;
}

static PyObject* wrap_array_shapes_container_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    static char *kwlist[] = {"this", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_this)) {
        return NULL;
    }
    
    PyObject* this_handle_obj = NULL;
    PyObject* this_sequence = NULL;
    Py_ssize_t this_handle_len = 0;
    if (PyObject_HasAttrString(py_this, "_handle")) {
        this_handle_obj = PyObject_GetAttrString(py_this, "_handle");
        if (this_handle_obj == NULL) {
            return NULL;
        }
        this_sequence = PySequence_Fast(this_handle_obj, "Failed to access handle sequence");
        if (this_sequence == NULL) {
            Py_DECREF(this_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_this)) {
        this_sequence = PySequence_Fast(py_this, "Argument this must be a handle sequence");
        if (this_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument this must be a Fortran derived-type instance");
        return NULL;
    }
    this_handle_len = PySequence_Fast_GET_SIZE(this_sequence);
    if (this_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument this has an invalid handle length");
        Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        return NULL;
    }
    int* this = (int*)malloc(sizeof(int) * this_handle_len);
    if (this == NULL) {
        PyErr_NoMemory();
        Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < this_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(this_sequence, i);
        if (item == NULL) {
            free(this);
            Py_DECREF(this_sequence);
            if (this_handle_obj) Py_DECREF(this_handle_obj);
            return NULL;
        }
        this[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(this);
            Py_DECREF(this_sequence);
            if (this_handle_obj) Py_DECREF(this_handle_obj);
            return NULL;
        }
    }
    (void)this_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__container_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_array_shapes__container_helper_get_n_data(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_handle)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    int value;
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__container__get__n_data)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_array_shapes__container_helper_set_n_data(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "n_data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_handle, &value)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__container__set__n_data)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_array_shapes__container_helper_array_data(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* dummy_handle = Py_None;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &dummy_handle)) {
        return NULL;
    }
    
    int dummy_this[4] = {0, 0, 0, 0};
    if (dummy_handle != Py_None) {
        PyObject* handle_sequence = PySequence_Fast(dummy_handle, "Handle must be a sequence");
        if (handle_sequence == NULL) {
            return NULL;
        }
        Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
        if (handle_len != 4) {
            Py_DECREF(handle_sequence);
            PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
            return NULL;
        }
        for (int i = 0; i < 4; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
            if (item == NULL) {
                Py_DECREF(handle_sequence);
                return NULL;
            }
            dummy_this[i] = (int)PyLong_AsLong(item);
            if (PyErr_Occurred()) {
                Py_DECREF(handle_sequence);
                return NULL;
            }
        }
        Py_DECREF(handle_sequence);
    }
    int nd = 0;
    int dtype = 0;
    int dshape[10] = {0};
    long long handle = 0;
    F90WRAP_F_SYMBOL(f90wrap_array_shapes__container__array__data)(dummy_this, &nd, &dtype, dshape, &handle);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (nd < 0 || nd > 10) {
        PyErr_SetString(PyExc_ValueError, "Invalid dimensionality");
        return NULL;
    }
    PyObject* shape_tuple = PyTuple_New(nd);
    if (shape_tuple == NULL) {
        return NULL;
    }
    for (int i = 0; i < nd; ++i) {
        PyObject* dim = PyLong_FromLong((long)dshape[i]);
        if (dim == NULL) {
            Py_DECREF(shape_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(shape_tuple, i, dim);
    }
    PyObject* result = PyTuple_New(4);
    if (result == NULL) {
        Py_DECREF(shape_tuple);
        return NULL;
    }
    PyObject* nd_obj = PyLong_FromLong((long)nd);
    if (nd_obj == NULL) {
        Py_DECREF(shape_tuple);
        Py_DECREF(result);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, nd_obj);
    PyObject* dtype_obj = PyLong_FromLong((long)dtype);
    if (dtype_obj == NULL) {
        Py_DECREF(shape_tuple);
        Py_DECREF(result);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 1, dtype_obj);
    PyTuple_SET_ITEM(result, 2, shape_tuple);
    shape_tuple = NULL;
    PyObject* handle_obj = PyLong_FromLongLong(handle);
    if (handle_obj == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 3, handle_obj);
    return result;
}

/* Method table for _array_shapes module */
static PyMethodDef _array_shapes_methods[] = {
    {"f90wrap_array_shapes__one_array_dynamic", (PyCFunction)wrap_array_shapes_one_array_dynamic, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for one_array_dynamic"},
    {"f90wrap_array_shapes__one_array_fixed", (PyCFunction)wrap_array_shapes_one_array_fixed, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for one_array_fixed"},
    {"f90wrap_array_shapes__one_array_fixed_range", (PyCFunction)wrap_array_shapes_one_array_fixed_range, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for one_array_fixed_range"},
    {"f90wrap_array_shapes__one_array_explicit", (PyCFunction)wrap_array_shapes_one_array_explicit, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for one_array_explicit"},
    {"f90wrap_array_shapes__one_array_explicit_range", (PyCFunction)wrap_array_shapes_one_array_explicit_range, METH_VARARGS \
        | METH_KEYWORDS, "Wrapper for one_array_explicit_range"},
    {"f90wrap_array_shapes__two_arrays_dynamic", (PyCFunction)wrap_array_shapes_two_arrays_dynamic, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for two_arrays_dynamic"},
    {"f90wrap_array_shapes__two_arrays_fixed", (PyCFunction)wrap_array_shapes_two_arrays_fixed, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for two_arrays_fixed"},
    {"f90wrap_array_shapes__two_arrays_mixed", (PyCFunction)wrap_array_shapes_two_arrays_mixed, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for two_arrays_mixed"},
    {"f90wrap_array_shapes__two_arrays_2d_dynamic", (PyCFunction)wrap_array_shapes_two_arrays_2d_dynamic, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for two_arrays_2d_dynamic"},
    {"f90wrap_array_shapes__two_arrays_2d_fixed", (PyCFunction)wrap_array_shapes_two_arrays_2d_fixed, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for two_arrays_2d_fixed"},
    {"f90wrap_array_shapes__two_arrays_2d_fixed_whitespace", (PyCFunction)wrap_array_shapes_two_arrays_2d_fixed_whitespace, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for two_arrays_2d_fixed_whitespace"},
    {"f90wrap_array_shapes__two_arrays_2d_mixed", (PyCFunction)wrap_array_shapes_two_arrays_2d_mixed, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for two_arrays_2d_mixed"},
    {"f90wrap_array_shapes__get_container", (PyCFunction)wrap_array_shapes_get_container, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for get_container"},
    {"f90wrap_array_shapes__array_container_dynamic", (PyCFunction)wrap_array_shapes_array_container_dynamic, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for array_container_dynamic"},
    {"f90wrap_array_shapes__array_container_fixed", (PyCFunction)wrap_array_shapes_array_container_fixed, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for array_container_fixed"},
    {"f90wrap_array_shapes__array_container_dynamic_2d", (PyCFunction)wrap_array_shapes_array_container_dynamic_2d, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for array_container_dynamic_2d"},
    {"f90wrap_array_shapes__container_initialise", (PyCFunction)wrap_array_shapes_container_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for container"},
    {"f90wrap_array_shapes__container_finalise", (PyCFunction)wrap_array_shapes_container_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for container"},
    {"f90wrap_array_shapes__container__get__n_data", (PyCFunction)wrap_array_shapes__container_helper_get_n_data, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for n_data"},
    {"f90wrap_array_shapes__container__set__n_data", (PyCFunction)wrap_array_shapes__container_helper_set_n_data, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for n_data"},
    {"f90wrap_array_shapes__container__array__data", (PyCFunction)wrap_array_shapes__container_helper_array_data, \
        METH_VARARGS | METH_KEYWORDS, "Array helper for data"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _array_shapesmodule = {
    PyModuleDef_HEAD_INIT,
    "array_shapes",
    "Direct-C wrapper for _array_shapes module",
    -1,
    _array_shapes_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__array_shapes(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_array_shapesmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
