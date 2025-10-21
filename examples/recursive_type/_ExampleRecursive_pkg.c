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
extern void F90WRAP_F_SYMBOL(f90wrap_tree__treeallocate)(int* root);
extern void F90WRAP_F_SYMBOL(f90wrap_tree__treedeallocate)(int* root);
extern void F90WRAP_F_SYMBOL(f90wrap_tree__node__get__left)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_tree__node__set__left)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_tree__node__get__right)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_tree__node__set__right)(int* handle, int* value);

static PyObject* wrap_tree_treeallocate(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int root[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_tree__treeallocate)(root);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_root_obj = PyList_New(4);
    if (py_root_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)root[i]);
        if (item == NULL) {
            Py_DECREF(py_root_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_root_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_root_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_root_obj != NULL) return py_root_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_root_obj != NULL) Py_DECREF(py_root_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_root_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_root_obj);
    }
    return result_tuple;
}

static PyObject* wrap_tree_treedeallocate(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_root = NULL;
    static char *kwlist[] = {"root", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_root)) {
        return NULL;
    }
    
    PyObject* root_handle_obj = NULL;
    PyObject* root_sequence = NULL;
    Py_ssize_t root_handle_len = 0;
    if (PyObject_HasAttrString(py_root, "_handle")) {
        root_handle_obj = PyObject_GetAttrString(py_root, "_handle");
        if (root_handle_obj == NULL) {
            return NULL;
        }
        root_sequence = PySequence_Fast(root_handle_obj, "Failed to access handle sequence");
        if (root_sequence == NULL) {
            Py_DECREF(root_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_root)) {
        root_sequence = PySequence_Fast(py_root, "Argument root must be a handle sequence");
        if (root_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument root must be a Fortran derived-type instance");
        return NULL;
    }
    root_handle_len = PySequence_Fast_GET_SIZE(root_sequence);
    if (root_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument root has an invalid handle length");
        Py_DECREF(root_sequence);
        if (root_handle_obj) Py_DECREF(root_handle_obj);
        return NULL;
    }
    int* root = (int*)malloc(sizeof(int) * root_handle_len);
    if (root == NULL) {
        PyErr_NoMemory();
        Py_DECREF(root_sequence);
        if (root_handle_obj) Py_DECREF(root_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < root_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(root_sequence, i);
        if (item == NULL) {
            free(root);
            Py_DECREF(root_sequence);
            if (root_handle_obj) Py_DECREF(root_handle_obj);
            return NULL;
        }
        root[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(root);
            Py_DECREF(root_sequence);
            if (root_handle_obj) Py_DECREF(root_handle_obj);
            return NULL;
        }
    }
    (void)root_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_tree__treedeallocate)(root);
    if (root_sequence) {
        Py_DECREF(root_sequence);
    }
    if (root_handle_obj) {
        Py_DECREF(root_handle_obj);
    }
    free(root);
    Py_RETURN_NONE;
}

static PyObject* wrap_tree__node_helper_get_derived_left(PyObject* self, PyObject* args, PyObject* kwargs)
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
    int handle_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        handle_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    int value_handle[4] = {0};
    F90WRAP_F_SYMBOL(f90wrap_tree__node__get__left)(handle_handle, value_handle);
    PyObject* result = PyList_New(4);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)value_handle[i]);
        if (item == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* wrap_tree__node_helper_set_derived_left(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent = Py_None;
    PyObject* py_value = Py_None;
    static char *kwlist[] = {"handle", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_parent, &py_value)) {
        return NULL;
    }
    
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    Py_DECREF(parent_sequence);
    int value_handle[4] = {0};
    PyObject* value_sequence = PySequence_Fast(py_value, "Value must be a sequence");
    if (value_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t value_len = PySequence_Fast_GET_SIZE(value_sequence);
    if (value_len != 4) {
        Py_DECREF(value_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);
        value_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(value_sequence);
            return NULL;
        }
    }
    Py_DECREF(value_sequence);
    F90WRAP_F_SYMBOL(f90wrap_tree__node__set__left)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_tree__node_helper_get_derived_right(PyObject* self, PyObject* args, PyObject* kwargs)
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
    int handle_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        handle_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    int value_handle[4] = {0};
    F90WRAP_F_SYMBOL(f90wrap_tree__node__get__right)(handle_handle, value_handle);
    PyObject* result = PyList_New(4);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)value_handle[i]);
        if (item == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* wrap_tree__node_helper_set_derived_right(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent = Py_None;
    PyObject* py_value = Py_None;
    static char *kwlist[] = {"handle", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_parent, &py_value)) {
        return NULL;
    }
    
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    Py_DECREF(parent_sequence);
    int value_handle[4] = {0};
    PyObject* value_sequence = PySequence_Fast(py_value, "Value must be a sequence");
    if (value_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t value_len = PySequence_Fast_GET_SIZE(value_sequence);
    if (value_len != 4) {
        Py_DECREF(value_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);
        value_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(value_sequence);
            return NULL;
        }
    }
    Py_DECREF(value_sequence);
    F90WRAP_F_SYMBOL(f90wrap_tree__node__set__right)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

/* Method table for _ExampleRecursive_pkg module */
static PyMethodDef _ExampleRecursive_pkg_methods[] = {
    {"f90wrap_tree__treeallocate", (PyCFunction)wrap_tree_treeallocate, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        treeallocate"},
    {"f90wrap_tree__treedeallocate", (PyCFunction)wrap_tree_treedeallocate, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        treedeallocate"},
    {"f90wrap_tree__node__get__left", (PyCFunction)wrap_tree__node_helper_get_derived_left, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for left"},
    {"f90wrap_tree__node__set__left", (PyCFunction)wrap_tree__node_helper_set_derived_left, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for left"},
    {"f90wrap_tree__node__get__right", (PyCFunction)wrap_tree__node_helper_get_derived_right, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for right"},
    {"f90wrap_tree__node__set__right", (PyCFunction)wrap_tree__node_helper_set_derived_right, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for right"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _ExampleRecursive_pkgmodule = {
    PyModuleDef_HEAD_INIT,
    "ExampleRecursive_pkg",
    "Direct-C wrapper for _ExampleRecursive_pkg module",
    -1,
    _ExampleRecursive_pkg_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__ExampleRecursive_pkg(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_ExampleRecursive_pkgmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
