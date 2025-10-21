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
extern void F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_in)(char* input, int input_len);
extern void F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_in_array)(int* f90wrap_n0, char* input);
extern void F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_in_array_hardcoded_size)(char* input);
extern void F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_to_string)(char* input, char* output, int input_len, int \
    output_len);
extern void F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_to_string_array)(int* f90wrap_n0, int* f90wrap_n1, char* \
    input, char* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_out)(char* output, int output_len);
extern void F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_out_optional)(char* output, int output_len);
extern void F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_out_optional_array)(int* f90wrap_n0, char* output);

static PyObject* wrap_m_string_test_string_in(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    int input_len = 0;
    char* input = NULL;
    int input_is_array = 0;
    if (py_input == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument input cannot be None");
        return NULL;
    } else {
        PyObject* input_bytes = NULL;
        if (PyArray_Check(py_input)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* input_arr = (PyArrayObject*)py_input;
            if (PyArray_TYPE(input_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument input must be a string array");
                return NULL;
            }
            input_len = (int)PyArray_ITEMSIZE(input_arr);
            input = (char*)PyArray_DATA(input_arr);
            input_is_array = 1;
        } else if (PyBytes_Check(py_input)) {
            input_bytes = py_input;
            Py_INCREF(input_bytes);
        } else if (PyUnicode_Check(py_input)) {
            input_bytes = PyUnicode_AsUTF8String(py_input);
            if (input_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument input must be str, bytes, or numpy array");
            return NULL;
        }
        if (input_bytes != NULL) {
            input_len = (int)PyBytes_GET_SIZE(input_bytes);
            input = (char*)malloc((size_t)input_len + 1);
            if (input == NULL) {
                Py_DECREF(input_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(input, PyBytes_AS_STRING(input_bytes), (size_t)input_len);
            input[input_len] = '\0';
            Py_DECREF(input_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_in)(input, input_len);
    if (PyErr_Occurred()) {
        if (!input_is_array) free(input);
        return NULL;
    }
    
    if (!input_is_array) free(input);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_string_test_string_in_array(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_input = NULL;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    PyArrayObject* input_arr = NULL;
    char* input = NULL;
    /* Extract input array data */
    if (!PyArray_Check(py_input)) {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a NumPy array");
        return NULL;
    }
    input_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_input, NPY_STRING, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (input_arr == NULL) {
        return NULL;
    }
    input = (char*)PyArray_DATA(input_arr);
    int n0_input = (int)PyArray_DIM(input_arr, 0);
    f90wrap_n0_val = n0_input;
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_in_array)(&f90wrap_n0_val, input);
    if (PyErr_Occurred()) {
        Py_XDECREF(input_arr);
        return NULL;
    }
    
    Py_DECREF(input_arr);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_string_test_string_in_array_hardcoded_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    PyArrayObject* input_arr = NULL;
    char* input = NULL;
    /* Extract input array data */
    if (!PyArray_Check(py_input)) {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a NumPy array");
        return NULL;
    }
    input_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_input, NPY_STRING, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (input_arr == NULL) {
        return NULL;
    }
    input = (char*)PyArray_DATA(input_arr);
    int n0_input = (int)PyArray_DIM(input_arr, 0);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_in_array_hardcoded_size)(input);
    if (PyErr_Occurred()) {
        Py_XDECREF(input_arr);
        return NULL;
    }
    
    Py_DECREF(input_arr);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_string_test_string_to_string(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    int input_len = 0;
    char* input = NULL;
    int input_is_array = 0;
    if (py_input == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument input cannot be None");
        return NULL;
    } else {
        PyObject* input_bytes = NULL;
        if (PyArray_Check(py_input)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* input_arr = (PyArrayObject*)py_input;
            if (PyArray_TYPE(input_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument input must be a string array");
                return NULL;
            }
            input_len = (int)PyArray_ITEMSIZE(input_arr);
            input = (char*)PyArray_DATA(input_arr);
            input_is_array = 1;
        } else if (PyBytes_Check(py_input)) {
            input_bytes = py_input;
            Py_INCREF(input_bytes);
        } else if (PyUnicode_Check(py_input)) {
            input_bytes = PyUnicode_AsUTF8String(py_input);
            if (input_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument input must be str, bytes, or numpy array");
            return NULL;
        }
        if (input_bytes != NULL) {
            input_len = (int)PyBytes_GET_SIZE(input_bytes);
            input = (char*)malloc((size_t)input_len + 1);
            if (input == NULL) {
                Py_DECREF(input_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(input, PyBytes_AS_STRING(input_bytes), (size_t)input_len);
            input[input_len] = '\0';
            Py_DECREF(input_bytes);
        }
    }
    int output_len = 1024;
    if (output_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character length for output must be positive");
        return NULL;
    }
    char* output = (char*)malloc((size_t)output_len + 1);
    if (output == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(output, ' ', output_len);
    output[output_len] = '\0';
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_to_string)(input, output, input_len, output_len);
    if (PyErr_Occurred()) {
        if (!input_is_array) free(input);
        free(output);
        return NULL;
    }
    
    int output_trim = output_len;
    while (output_trim > 0 && output[output_trim - 1] == ' ') {
        --output_trim;
    }
    PyObject* py_output_obj = PyBytes_FromStringAndSize(output, output_trim);
    free(output);
    if (py_output_obj == NULL) {
        return NULL;
    }
    if (!input_is_array) free(input);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_output_obj != NULL) return py_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_output_obj != NULL) Py_DECREF(py_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_string_test_string_to_string_array(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_input = NULL;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"input", "output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_input, &py_output)) {
        return NULL;
    }
    
    PyArrayObject* input_arr = NULL;
    char* input = NULL;
    /* Extract input array data */
    if (!PyArray_Check(py_input)) {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a NumPy array");
        return NULL;
    }
    input_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_input, NPY_STRING, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (input_arr == NULL) {
        return NULL;
    }
    input = (char*)PyArray_DATA(input_arr);
    int n0_input = (int)PyArray_DIM(input_arr, 0);
    f90wrap_n0_val = n0_input;
    
    PyArrayObject* output_arr = NULL;
    PyObject* py_output_arr = NULL;
    int output_needs_copyback = 0;
    char* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_STRING, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (char*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    f90wrap_n1_val = n0_output;
    Py_INCREF(py_output);
    py_output_arr = py_output;
    if (PyArray_DATA(output_arr) != PyArray_DATA((PyArrayObject*)py_output) || PyArray_TYPE(output_arr) != \
        PyArray_TYPE((PyArrayObject*)py_output)) {
        output_needs_copyback = 1;
    }
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_to_string_array)(&f90wrap_n0_val, &f90wrap_n1_val, input, output);
    if (PyErr_Occurred()) {
        Py_XDECREF(input_arr);
        Py_XDECREF(py_output_arr);
        return NULL;
    }
    
    Py_DECREF(input_arr);
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

static PyObject* wrap_m_string_test_string_out(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int output_len = 13;
    if (output_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character length for output must be positive");
        return NULL;
    }
    char* output = (char*)malloc((size_t)output_len + 1);
    if (output == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(output, ' ', output_len);
    output[output_len] = '\0';
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_out)(output, output_len);
    if (PyErr_Occurred()) {
        free(output);
        return NULL;
    }
    
    int output_trim = output_len;
    while (output_trim > 0 && output[output_trim - 1] == ' ') {
        --output_trim;
    }
    PyObject* py_output_obj = PyBytes_FromStringAndSize(output, output_trim);
    free(output);
    if (py_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_output_obj != NULL) return py_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_output_obj != NULL) Py_DECREF(py_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_string_test_string_out_optional(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_output = Py_None;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &py_output)) {
        return NULL;
    }
    
    int output_len = 0;
    char* output = NULL;
    int output_is_array = 0;
    if (py_output == Py_None) {
        output_len = 13;
        if (output_len <= 0) {
            PyErr_SetString(PyExc_ValueError, "Character length for output must be positive");
            return NULL;
        }
        output = (char*)malloc((size_t)output_len + 1);
        if (output == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        memset(output, ' ', output_len);
        output[output_len] = '\0';
    } else {
        PyObject* output_bytes = NULL;
        if (PyArray_Check(py_output)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* output_arr = (PyArrayObject*)py_output;
            if (PyArray_TYPE(output_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument output must be a string array");
                return NULL;
            }
            output_len = (int)PyArray_ITEMSIZE(output_arr);
            output = (char*)PyArray_DATA(output_arr);
            output_is_array = 1;
        } else if (PyBytes_Check(py_output)) {
            output_bytes = py_output;
            Py_INCREF(output_bytes);
        } else if (PyUnicode_Check(py_output)) {
            output_bytes = PyUnicode_AsUTF8String(py_output);
            if (output_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument output must be str, bytes, or numpy array");
            return NULL;
        }
        if (output_bytes != NULL) {
            output_len = (int)PyBytes_GET_SIZE(output_bytes);
            output = (char*)malloc((size_t)output_len + 1);
            if (output == NULL) {
                Py_DECREF(output_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(output, PyBytes_AS_STRING(output_bytes), (size_t)output_len);
            output[output_len] = '\0';
            Py_DECREF(output_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_out_optional)(output, output_len);
    if (PyErr_Occurred()) {
        if (!output_is_array) free(output);
        return NULL;
    }
    
    PyObject* py_output_obj = NULL;
    if (output_is_array) {
        /* Numpy array was modified in place, no return object or free needed */
    } else {
        int output_trim = output_len;
        while (output_trim > 0 && output[output_trim - 1] == ' ') {
            --output_trim;
        }
        py_output_obj = PyBytes_FromStringAndSize(output, output_trim);
        free(output);
        if (py_output_obj == NULL) {
            return NULL;
        }
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_output_obj != NULL) return py_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_output_obj != NULL) Py_DECREF(py_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_string_test_string_out_optional_array(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &py_output)) {
        return NULL;
    }
    
    PyArrayObject* output_arr = NULL;
    PyObject* py_output_arr = NULL;
    int output_needs_copyback = 0;
    char* output = NULL;
    if (py_output != NULL && py_output != Py_None) {
        /* Extract output array data */
        if (!PyArray_Check(py_output)) {
            PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
            return NULL;
        }
        output_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_output, NPY_STRING, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (output_arr == NULL) {
            return NULL;
        }
        output = (char*)PyArray_DATA(output_arr);
        int n0_output = (int)PyArray_DIM(output_arr, 0);
        f90wrap_n0_val = n0_output;
        Py_INCREF(py_output);
        py_output_arr = py_output;
        if (PyArray_DATA(output_arr) != PyArray_DATA((PyArrayObject*)py_output) || PyArray_TYPE(output_arr) != \
            PyArray_TYPE((PyArrayObject*)py_output)) {
            output_needs_copyback = 1;
        }
        
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_string_test__string_out_optional_array)(&f90wrap_n0_val, output);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_output_arr);
        return NULL;
    }
    
    if (output_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_output, output_arr) < 0) {
            Py_DECREF(output_arr);
            Py_DECREF(py_output_arr);
            return NULL;
        }
    }
    Py_XDECREF(output_arr);
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
    {"f90wrap_m_string_test__string_in", (PyCFunction)wrap_m_string_test_string_in, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for string_in"},
    {"f90wrap_m_string_test__string_in_array", (PyCFunction)wrap_m_string_test_string_in_array, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for string_in_array"},
    {"f90wrap_m_string_test__string_in_array_hardcoded_size", \
        (PyCFunction)wrap_m_string_test_string_in_array_hardcoded_size, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        string_in_array_hardcoded_size"},
    {"f90wrap_m_string_test__string_to_string", (PyCFunction)wrap_m_string_test_string_to_string, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for string_to_string"},
    {"f90wrap_m_string_test__string_to_string_array", (PyCFunction)wrap_m_string_test_string_to_string_array, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for string_to_string_array"},
    {"f90wrap_m_string_test__string_out", (PyCFunction)wrap_m_string_test_string_out, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for string_out"},
    {"f90wrap_m_string_test__string_out_optional", (PyCFunction)wrap_m_string_test_string_out_optional, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for string_out_optional"},
    {"f90wrap_m_string_test__string_out_optional_array", (PyCFunction)wrap_m_string_test_string_out_optional_array, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for string_out_optional_array"},
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
