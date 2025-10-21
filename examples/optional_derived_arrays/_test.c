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
extern void F90WRAP_F_SYMBOL(f90wrap_io__io_freeform_open)(char* filename, int filename_len);
extern void F90WRAP_F_SYMBOL(f90wrap_io__keyword_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_io__keyword_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_io__keyword__get__key)(int* handle, char* value, int value_len);
extern void F90WRAP_F_SYMBOL(f90wrap_io__keyword__set__key)(int* handle, char* value, int value_len);
extern void F90WRAP_F_SYMBOL(f90wrap_io__keyword__get__typ)(int* handle, char* value, int value_len);
extern void F90WRAP_F_SYMBOL(f90wrap_io__keyword__set__typ)(int* handle, char* value, int value_len);
extern void F90WRAP_F_SYMBOL(f90wrap_io__keyword__get__description)(int* handle, char* value, int value_len);
extern void F90WRAP_F_SYMBOL(f90wrap_io__keyword__set__description)(int* handle, char* value, int value_len);

static PyObject* wrap_io_io_freeform_open(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_filename = NULL;
    static char *kwlist[] = {"filename", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_filename)) {
        return NULL;
    }
    
    int filename_len = 0;
    char* filename = NULL;
    int filename_is_array = 0;
    if (py_filename == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument filename cannot be None");
        return NULL;
    } else {
        PyObject* filename_bytes = NULL;
        if (PyArray_Check(py_filename)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* filename_arr = (PyArrayObject*)py_filename;
            if (PyArray_TYPE(filename_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument filename must be a string array");
                return NULL;
            }
            filename_len = (int)PyArray_ITEMSIZE(filename_arr);
            filename = (char*)PyArray_DATA(filename_arr);
            filename_is_array = 1;
        } else if (PyBytes_Check(py_filename)) {
            filename_bytes = py_filename;
            Py_INCREF(filename_bytes);
        } else if (PyUnicode_Check(py_filename)) {
            filename_bytes = PyUnicode_AsUTF8String(py_filename);
            if (filename_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument filename must be str, bytes, or numpy array");
            return NULL;
        }
        if (filename_bytes != NULL) {
            filename_len = (int)PyBytes_GET_SIZE(filename_bytes);
            filename = (char*)malloc((size_t)filename_len + 1);
            if (filename == NULL) {
                Py_DECREF(filename_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(filename, PyBytes_AS_STRING(filename_bytes), (size_t)filename_len);
            filename[filename_len] = '\0';
            Py_DECREF(filename_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_io__io_freeform_open)(filename, filename_len);
    if (PyErr_Occurred()) {
        if (!filename_is_array) free(filename);
        return NULL;
    }
    
    if (!filename_is_array) free(filename);
    Py_RETURN_NONE;
}

static PyObject* wrap_io_keyword_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_io__keyword_initialise)(this);
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

static PyObject* wrap_io_keyword_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_io__keyword_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_io__keyword_helper_get_key(PyObject* self, PyObject* args, PyObject* kwargs)
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
    int value_len = 10;
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
    F90WRAP_F_SYMBOL(f90wrap_io__keyword__get__key)(this_handle, buffer, value_len);
    int actual_len = value_len;
    while (actual_len > 0 && buffer[actual_len - 1] == ' ') {
        --actual_len;
    }
    PyObject* result = PyBytes_FromStringAndSize(buffer, actual_len);
    free(buffer);
    return result;
}

static PyObject* wrap_io__keyword_helper_set_key(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    PyObject* py_value;
    static char *kwlist[] = {"handle", "key", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_handle, &py_value)) {
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
    if (py_value == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument key must be str or bytes");
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
        PyErr_SetString(PyExc_TypeError, "Argument key must be str or bytes");
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
    F90WRAP_F_SYMBOL(f90wrap_io__keyword__set__key)(this_handle, value, value_len);
    free(value);
    Py_DECREF(value_bytes);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_io__keyword_helper_get_typ(PyObject* self, PyObject* args, PyObject* kwargs)
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
    int value_len = 3;
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
    F90WRAP_F_SYMBOL(f90wrap_io__keyword__get__typ)(this_handle, buffer, value_len);
    int actual_len = value_len;
    while (actual_len > 0 && buffer[actual_len - 1] == ' ') {
        --actual_len;
    }
    PyObject* result = PyBytes_FromStringAndSize(buffer, actual_len);
    free(buffer);
    return result;
}

static PyObject* wrap_io__keyword_helper_set_typ(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    PyObject* py_value;
    static char *kwlist[] = {"handle", "typ", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_handle, &py_value)) {
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
    if (py_value == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument typ must be str or bytes");
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
        PyErr_SetString(PyExc_TypeError, "Argument typ must be str or bytes");
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
    F90WRAP_F_SYMBOL(f90wrap_io__keyword__set__typ)(this_handle, value, value_len);
    free(value);
    Py_DECREF(value_bytes);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_io__keyword_helper_get_description(PyObject* self, PyObject* args, PyObject* kwargs)
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
    int value_len = 10;
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
    F90WRAP_F_SYMBOL(f90wrap_io__keyword__get__description)(this_handle, buffer, value_len);
    int actual_len = value_len;
    while (actual_len > 0 && buffer[actual_len - 1] == ' ') {
        --actual_len;
    }
    PyObject* result = PyBytes_FromStringAndSize(buffer, actual_len);
    free(buffer);
    return result;
}

static PyObject* wrap_io__keyword_helper_set_description(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    PyObject* py_value;
    static char *kwlist[] = {"handle", "description", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_handle, &py_value)) {
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
    if (py_value == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument description must be str or bytes");
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
        PyErr_SetString(PyExc_TypeError, "Argument description must be str or bytes");
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
    F90WRAP_F_SYMBOL(f90wrap_io__keyword__set__description)(this_handle, value, value_len);
    free(value);
    Py_DECREF(value_bytes);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _test module */
static PyMethodDef _test_methods[] = {
    {"f90wrap_io__io_freeform_open", (PyCFunction)wrap_io_io_freeform_open, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        io_freeform_open"},
    {"f90wrap_io__keyword_initialise", (PyCFunction)wrap_io_keyword_initialise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated constructor for keyword"},
    {"f90wrap_io__keyword_finalise", (PyCFunction)wrap_io_keyword_finalise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated destructor for keyword"},
    {"f90wrap_io__keyword__get__key", (PyCFunction)wrap_io__keyword_helper_get_key, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for key"},
    {"f90wrap_io__keyword__set__key", (PyCFunction)wrap_io__keyword_helper_set_key, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for key"},
    {"f90wrap_io__keyword__get__typ", (PyCFunction)wrap_io__keyword_helper_get_typ, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for typ"},
    {"f90wrap_io__keyword__set__typ", (PyCFunction)wrap_io__keyword_helper_set_typ, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for typ"},
    {"f90wrap_io__keyword__get__description", (PyCFunction)wrap_io__keyword_helper_get_description, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for description"},
    {"f90wrap_io__keyword__set__description", (PyCFunction)wrap_io__keyword_helper_set_description, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for description"},
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
