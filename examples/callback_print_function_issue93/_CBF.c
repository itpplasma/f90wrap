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
extern void F90WRAP_F_SYMBOL(f90wrap_caller__test_write_msg)(void);
extern void F90WRAP_F_SYMBOL(f90wrap_caller__test_write_msg_2)(void);
extern void F90WRAP_F_SYMBOL(f90wrap_cback__write_message)(char* msg, int msg_len);

static PyObject* wrap_caller_test_write_msg(PyObject* self, PyObject* args, PyObject* kwargs)
{
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_caller__test_write_msg)();
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* wrap_caller_test_write_msg_2(PyObject* self, PyObject* args, PyObject* kwargs)
{
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_caller__test_write_msg_2)();
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* wrap_cback_write_message(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_msg = NULL;
    static char *kwlist[] = {"msg", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_msg)) {
        return NULL;
    }
    
    int msg_len = 0;
    char* msg = NULL;
    int msg_is_array = 0;
    if (py_msg == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument msg cannot be None");
        return NULL;
    } else {
        PyObject* msg_bytes = NULL;
        if (PyArray_Check(py_msg)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* msg_arr = (PyArrayObject*)py_msg;
            if (PyArray_TYPE(msg_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument msg must be a string array");
                return NULL;
            }
            msg_len = (int)PyArray_ITEMSIZE(msg_arr);
            msg = (char*)PyArray_DATA(msg_arr);
            msg_is_array = 1;
        } else if (PyBytes_Check(py_msg)) {
            msg_bytes = py_msg;
            Py_INCREF(msg_bytes);
        } else if (PyUnicode_Check(py_msg)) {
            msg_bytes = PyUnicode_AsUTF8String(py_msg);
            if (msg_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument msg must be str, bytes, or numpy array");
            return NULL;
        }
        if (msg_bytes != NULL) {
            msg_len = (int)PyBytes_GET_SIZE(msg_bytes);
            msg = (char*)malloc((size_t)msg_len + 1);
            if (msg == NULL) {
                Py_DECREF(msg_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(msg, PyBytes_AS_STRING(msg_bytes), (size_t)msg_len);
            msg[msg_len] = '\0';
            Py_DECREF(msg_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_cback__write_message)(msg, msg_len);
    if (PyErr_Occurred()) {
        if (!msg_is_array) free(msg);
        return NULL;
    }
    
    if (!msg_is_array) free(msg);
    Py_RETURN_NONE;
}

/* Method table for _CBF module */
static PyMethodDef _CBF_methods[] = {
    {"f90wrap_caller__test_write_msg", (PyCFunction)wrap_caller_test_write_msg, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        test_write_msg"},
    {"f90wrap_caller__test_write_msg_2", (PyCFunction)wrap_caller_test_write_msg_2, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for test_write_msg_2"},
    {"f90wrap_cback__write_message", (PyCFunction)wrap_cback_write_message, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        write_message"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _CBFmodule = {
    PyModuleDef_HEAD_INIT,
    "CBF",
    "Direct-C wrapper for _CBF module",
    -1,
    _CBF_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__CBF(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_CBFmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
