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
extern void F90WRAP_F_SYMBOL(f90wrap_string_io__func_generate_string)(int* n, char* ret_stringout, int \
    ret_stringout_len);
extern void F90WRAP_F_SYMBOL(f90wrap_string_io__func_return_string)(char* ret_stringout, int ret_stringout_len);
extern void F90WRAP_F_SYMBOL(f90wrap_string_io__generate_string)(int* n, char* stringout, int stringout_len);
extern void F90WRAP_F_SYMBOL(f90wrap_string_io__return_string)(char* stringout, int stringout_len);
extern void F90WRAP_F_SYMBOL(f90wrap_string_io__set_global_string)(int* n, char* newstring, int newstring_len);
extern void F90WRAP_F_SYMBOL(f90wrap_string_io__inout_string)(int* n, char* stringinout, int stringinout_len);
extern void F90WRAP_F_SYMBOL(f90wrap_string_io__get__global_string)(char* value, int value_len);
extern void F90WRAP_F_SYMBOL(f90wrap_string_io__set__global_string)(char* value, int value_len);

static PyObject* wrap_string_io_func_generate_string(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_n)) {
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
    int ret_stringout_len = 1024;
    if (ret_stringout_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character length for ret_stringout must be positive");
        return NULL;
    }
    char* ret_stringout = (char*)malloc((size_t)ret_stringout_len + 1);
    if (ret_stringout == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(ret_stringout, ' ', ret_stringout_len);
    ret_stringout[ret_stringout_len] = '\0';
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_string_io__func_generate_string)(n, ret_stringout, ret_stringout_len);
    if (PyErr_Occurred()) {
        free(ret_stringout);
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
    int ret_stringout_trim = ret_stringout_len;
    while (ret_stringout_trim > 0 && ret_stringout[ret_stringout_trim - 1] == ' ') {
        --ret_stringout_trim;
    }
    PyObject* py_ret_stringout_obj = PyBytes_FromStringAndSize(ret_stringout, ret_stringout_trim);
    free(ret_stringout);
    if (py_ret_stringout_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_stringout_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_stringout_obj != NULL) return py_ret_stringout_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_stringout_obj != NULL) Py_DECREF(py_ret_stringout_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_stringout_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_stringout_obj);
    }
    return result_tuple;
}

static PyObject* wrap_string_io_func_return_string(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int ret_stringout_len = 51;
    if (ret_stringout_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character length for ret_stringout must be positive");
        return NULL;
    }
    char* ret_stringout = (char*)malloc((size_t)ret_stringout_len + 1);
    if (ret_stringout == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(ret_stringout, ' ', ret_stringout_len);
    ret_stringout[ret_stringout_len] = '\0';
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_string_io__func_return_string)(ret_stringout, ret_stringout_len);
    if (PyErr_Occurred()) {
        free(ret_stringout);
        return NULL;
    }
    
    int ret_stringout_trim = ret_stringout_len;
    while (ret_stringout_trim > 0 && ret_stringout[ret_stringout_trim - 1] == ' ') {
        --ret_stringout_trim;
    }
    PyObject* py_ret_stringout_obj = PyBytes_FromStringAndSize(ret_stringout, ret_stringout_trim);
    free(ret_stringout);
    if (py_ret_stringout_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_stringout_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_stringout_obj != NULL) return py_ret_stringout_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_stringout_obj != NULL) Py_DECREF(py_ret_stringout_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_stringout_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_stringout_obj);
    }
    return result_tuple;
}

static PyObject* wrap_string_io_generate_string(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_n)) {
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
    int stringout_len = 1024;
    if (stringout_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character length for stringout must be positive");
        return NULL;
    }
    char* stringout = (char*)malloc((size_t)stringout_len + 1);
    if (stringout == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(stringout, ' ', stringout_len);
    stringout[stringout_len] = '\0';
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_string_io__generate_string)(n, stringout, stringout_len);
    if (PyErr_Occurred()) {
        free(stringout);
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
    int stringout_trim = stringout_len;
    while (stringout_trim > 0 && stringout[stringout_trim - 1] == ' ') {
        --stringout_trim;
    }
    PyObject* py_stringout_obj = PyBytes_FromStringAndSize(stringout, stringout_trim);
    free(stringout);
    if (py_stringout_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_stringout_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_stringout_obj != NULL) return py_stringout_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_stringout_obj != NULL) Py_DECREF(py_stringout_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_stringout_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_stringout_obj);
    }
    return result_tuple;
}

static PyObject* wrap_string_io_return_string(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int stringout_len = 51;
    if (stringout_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character length for stringout must be positive");
        return NULL;
    }
    char* stringout = (char*)malloc((size_t)stringout_len + 1);
    if (stringout == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(stringout, ' ', stringout_len);
    stringout[stringout_len] = '\0';
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_string_io__return_string)(stringout, stringout_len);
    if (PyErr_Occurred()) {
        free(stringout);
        return NULL;
    }
    
    int stringout_trim = stringout_len;
    while (stringout_trim > 0 && stringout[stringout_trim - 1] == ' ') {
        --stringout_trim;
    }
    PyObject* py_stringout_obj = PyBytes_FromStringAndSize(stringout, stringout_trim);
    free(stringout);
    if (py_stringout_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_stringout_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_stringout_obj != NULL) return py_stringout_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_stringout_obj != NULL) Py_DECREF(py_stringout_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_stringout_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_stringout_obj);
    }
    return result_tuple;
}

static PyObject* wrap_string_io_set_global_string(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    PyObject* py_newstring = NULL;
    static char *kwlist[] = {"n", "newstring", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_n, &py_newstring)) {
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
    int newstring_len = 0;
    char* newstring = NULL;
    int newstring_is_array = 0;
    if (py_newstring == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument newstring cannot be None");
        return NULL;
    } else {
        PyObject* newstring_bytes = NULL;
        if (PyArray_Check(py_newstring)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* newstring_arr = (PyArrayObject*)py_newstring;
            if (PyArray_TYPE(newstring_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument newstring must be a string array");
                return NULL;
            }
            newstring_len = (int)PyArray_ITEMSIZE(newstring_arr);
            newstring = (char*)PyArray_DATA(newstring_arr);
            newstring_is_array = 1;
        } else if (PyBytes_Check(py_newstring)) {
            newstring_bytes = py_newstring;
            Py_INCREF(newstring_bytes);
        } else if (PyUnicode_Check(py_newstring)) {
            newstring_bytes = PyUnicode_AsUTF8String(py_newstring);
            if (newstring_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument newstring must be str, bytes, or numpy array");
            return NULL;
        }
        if (newstring_bytes != NULL) {
            newstring_len = (int)PyBytes_GET_SIZE(newstring_bytes);
            newstring = (char*)malloc((size_t)newstring_len + 1);
            if (newstring == NULL) {
                Py_DECREF(newstring_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(newstring, PyBytes_AS_STRING(newstring_bytes), (size_t)newstring_len);
            newstring[newstring_len] = '\0';
            Py_DECREF(newstring_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_string_io__set_global_string)(n, newstring, newstring_len);
    if (PyErr_Occurred()) {
        if (!newstring_is_array) free(newstring);
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
    if (!newstring_is_array) free(newstring);
    Py_RETURN_NONE;
}

static PyObject* wrap_string_io_inout_string(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    PyObject* py_stringinout = Py_None;
    static char *kwlist[] = {"n", "stringinout", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_n, &py_stringinout)) {
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
    int stringinout_len = 0;
    char* stringinout = NULL;
    int stringinout_is_array = 0;
    if (py_stringinout == Py_None) {
        stringinout_len = 1024;
        if (stringinout_len <= 0) {
            PyErr_SetString(PyExc_ValueError, "Character length for stringinout must be positive");
            return NULL;
        }
        stringinout = (char*)malloc((size_t)stringinout_len + 1);
        if (stringinout == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        memset(stringinout, ' ', stringinout_len);
        stringinout[stringinout_len] = '\0';
    } else {
        PyObject* stringinout_bytes = NULL;
        if (PyArray_Check(py_stringinout)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* stringinout_arr = (PyArrayObject*)py_stringinout;
            if (PyArray_TYPE(stringinout_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument stringinout must be a string array");
                return NULL;
            }
            stringinout_len = (int)PyArray_ITEMSIZE(stringinout_arr);
            stringinout = (char*)PyArray_DATA(stringinout_arr);
            stringinout_is_array = 1;
        } else if (PyBytes_Check(py_stringinout)) {
            stringinout_bytes = py_stringinout;
            Py_INCREF(stringinout_bytes);
        } else if (PyUnicode_Check(py_stringinout)) {
            stringinout_bytes = PyUnicode_AsUTF8String(py_stringinout);
            if (stringinout_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument stringinout must be str, bytes, or numpy array");
            return NULL;
        }
        if (stringinout_bytes != NULL) {
            stringinout_len = (int)PyBytes_GET_SIZE(stringinout_bytes);
            stringinout = (char*)malloc((size_t)stringinout_len + 1);
            if (stringinout == NULL) {
                Py_DECREF(stringinout_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(stringinout, PyBytes_AS_STRING(stringinout_bytes), (size_t)stringinout_len);
            stringinout[stringinout_len] = '\0';
            Py_DECREF(stringinout_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_string_io__inout_string)(n, stringinout, stringinout_len);
    if (PyErr_Occurred()) {
        if (!stringinout_is_array) free(stringinout);
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
    PyObject* py_stringinout_obj = NULL;
    if (stringinout_is_array) {
        /* Numpy array was modified in place, no return object or free needed */
    } else {
        int stringinout_trim = stringinout_len;
        while (stringinout_trim > 0 && stringinout[stringinout_trim - 1] == ' ') {
            --stringinout_trim;
        }
        py_stringinout_obj = PyBytes_FromStringAndSize(stringinout, stringinout_trim);
        free(stringinout);
        if (py_stringinout_obj == NULL) {
            return NULL;
        }
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_stringinout_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_stringinout_obj != NULL) return py_stringinout_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_stringinout_obj != NULL) Py_DECREF(py_stringinout_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_stringinout_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_stringinout_obj);
    }
    return result_tuple;
}

static PyObject* wrap_string_io_helper_get_global_string(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value_len = 512;
    if (value_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character helper length must be positive");
        return NULL;
    }
    char* buffer = (char*)malloc((size_t)value_len + 1);
    if (buffer == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(buffer, ' ', value_len);
    buffer[value_len] = '\0';
    F90WRAP_F_SYMBOL(f90wrap_string_io__get__global_string)(buffer, value_len);
    int actual_len = value_len;
    while (actual_len > 0 && buffer[actual_len - 1] == ' ') {
        --actual_len;
    }
    PyObject* result = PyBytes_FromStringAndSize(buffer, actual_len);
    free(buffer);
    if (result == NULL) {
        return NULL;
    }
    return result;
}

static PyObject* wrap_string_io_helper_set_global_string(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_value;
    static char *kwlist[] = {"global_string", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_value)) {
        return NULL;
    }
    if (py_value == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument global_string must be str or bytes");
        return NULL;
    }
    PyObject* value_bytes = NULL;
    if (PyBytes_Check(py_value)) {
        value_bytes = py_value;
        Py_INCREF(value_bytes);
    } else if (PyUnicode_Check(py_value)) {
        value_bytes = PyUnicode_AsUTF8String(py_value);
        if (value_bytes == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument global_string must be str or bytes");
        return NULL;
    }
    int value_len = (int)PyBytes_GET_SIZE(value_bytes);
    char* value = (char*)malloc((size_t)value_len + 1);
    if (value == NULL) {
        Py_DECREF(value_bytes);
        PyErr_NoMemory();
        return NULL;
    }
    memcpy(value, PyBytes_AS_STRING(value_bytes), (size_t)value_len);
    value[value_len] = '\0';
    F90WRAP_F_SYMBOL(f90wrap_string_io__set__global_string)(value, value_len);
    free(value);
    Py_DECREF(value_bytes);
    Py_RETURN_NONE;
}

/* Method table for _ExampleStrings_pkg module */
static PyMethodDef _ExampleStrings_pkg_methods[] = {
    {"f90wrap_string_io__func_generate_string", (PyCFunction)wrap_string_io_func_generate_string, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for func_generate_string"},
    {"f90wrap_string_io__func_return_string", (PyCFunction)wrap_string_io_func_return_string, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for func_return_string"},
    {"f90wrap_string_io__generate_string", (PyCFunction)wrap_string_io_generate_string, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for generate_string"},
    {"f90wrap_string_io__return_string", (PyCFunction)wrap_string_io_return_string, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for return_string"},
    {"f90wrap_string_io__set_global_string", (PyCFunction)wrap_string_io_set_global_string, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for set_global_string"},
    {"f90wrap_string_io__inout_string", (PyCFunction)wrap_string_io_inout_string, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        inout_string"},
    {"f90wrap_string_io__get__global_string", (PyCFunction)wrap_string_io_helper_get_global_string, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for global_string"},
    {"f90wrap_string_io__set__global_string", (PyCFunction)wrap_string_io_helper_set_global_string, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for global_string"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _ExampleStrings_pkgmodule = {
    PyModuleDef_HEAD_INIT,
    "ExampleStrings_pkg",
    "Direct-C wrapper for _ExampleStrings_pkg module",
    -1,
    _ExampleStrings_pkg_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__ExampleStrings_pkg(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_ExampleStrings_pkgmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
