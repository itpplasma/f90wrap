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
extern void F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__allocate_node)(int* root, int* n_node);
extern void F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__deallocate_node)(int* root);
extern void F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__t_node_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__t_node_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__t_node__array_getitem__node)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__t_node__array_setitem__node)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__t_node__array_len__node)(int* dummy_this, int* length);

static PyObject* wrap_mod_recursive_type_array_allocate_node(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_root = NULL;
    PyObject* py_n_node = NULL;
    int n_node_val = 0;
    PyArrayObject* n_node_scalar_arr = NULL;
    int n_node_scalar_copyback = 0;
    int n_node_scalar_is_array = 0;
    static char *kwlist[] = {"root", "n_node", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_root, &py_n_node)) {
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
    
    int* n_node = &n_node_val;
    if (PyArray_Check(py_n_node)) {
        n_node_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n_node, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_node_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_node_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n_node must have exactly one element");
            Py_DECREF(n_node_scalar_arr);
            return NULL;
        }
        n_node_scalar_is_array = 1;
        n_node = (int*)PyArray_DATA(n_node_scalar_arr);
        n_node_val = n_node[0];
        if (PyArray_DATA(n_node_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n_node) || PyArray_TYPE(n_node_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n_node)) {
            n_node_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n_node)) {
        n_node_val = (int)PyLong_AsLong(py_n_node);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n_node must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__allocate_node)(root, n_node);
    if (PyErr_Occurred()) {
        if (root_sequence) Py_DECREF(root_sequence);
        if (root_handle_obj) Py_DECREF(root_handle_obj);
        free(root);
        return NULL;
    }
    
    if (n_node_scalar_is_array) {
        if (n_node_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n_node, n_node_scalar_arr) < 0) {
                Py_DECREF(n_node_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n_node_scalar_arr);
    }
    if (root_sequence) {
        Py_DECREF(root_sequence);
    }
    if (root_handle_obj) {
        Py_DECREF(root_handle_obj);
    }
    free(root);
    Py_RETURN_NONE;
}

static PyObject* wrap_mod_recursive_type_array_deallocate_node(PyObject* self, PyObject* args, PyObject* kwargs)
{
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
    F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__deallocate_node)(root);
    if (PyErr_Occurred()) {
        if (root_sequence) Py_DECREF(root_sequence);
        if (root_handle_obj) Py_DECREF(root_handle_obj);
        free(root);
        return NULL;
    }
    
    if (root_sequence) {
        Py_DECREF(root_sequence);
    }
    if (root_handle_obj) {
        Py_DECREF(root_handle_obj);
    }
    free(root);
    Py_RETURN_NONE;
}

static PyObject* wrap_mod_recursive_type_array_t_node_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__t_node_initialise)(this);
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

static PyObject* wrap_mod_recursive_type_array_t_node_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__t_node_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_mod_recursive_type_array__t_node_helper_array_getitem_node(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    int index = 0;
    static char *kwlist[] = {"handle", "index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_parent, &index)) {
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
    int handle[4] = {0};
    F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__t_node__array_getitem__node)(parent_handle, &index, handle);
    if (PyErr_Occurred()) {
        Py_DECREF(parent_sequence);
        return NULL;
    }
    Py_DECREF(parent_sequence);
    PyObject* result = PyList_New(4);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)handle[i]);
        if (item == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* wrap_mod_recursive_type_array__t_node_helper_array_setitem_node(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    int index = 0;
    PyObject* py_value;
    static char *kwlist[] = {"handle", "index", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiO", kwlist, &py_parent, &index, &py_value)) {
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
    PyObject* value_handle_obj = NULL;
    PyObject* value_sequence = NULL;
    Py_ssize_t value_handle_len = 0;
    if (PyObject_HasAttrString(py_value, "_handle")) {
        value_handle_obj = PyObject_GetAttrString(py_value, "_handle");
        if (value_handle_obj == NULL) { return NULL; }
        value_sequence = PySequence_Fast(value_handle_obj, "Failed to access handle sequence");
        if (value_sequence == NULL) { Py_DECREF(value_handle_obj); return NULL; }
    } else if (PySequence_Check(py_value)) {
        value_sequence = PySequence_Fast(py_value, "Argument value must be a handle sequence");
        if (value_sequence == NULL) { return NULL; }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument value must be a Fortran derived-type instance");
        return NULL;
    }
    value_handle_len = PySequence_Fast_GET_SIZE(value_sequence);
    if (value_handle_len != 4) {
        Py_DECREF(parent_sequence);
        Py_DECREF(value_sequence);
        if (value_handle_obj) Py_DECREF(value_handle_obj);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    Py_DECREF(parent_sequence);
    int* value = (int*)malloc(sizeof(int) * 4);
    if (value == NULL) {
        PyErr_NoMemory();
        Py_DECREF(value_sequence);
        if (value_handle_obj) Py_DECREF(value_handle_obj);
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);
        if (item == NULL) {
            free(value);
            Py_DECREF(value_sequence);
            if (value_handle_obj) Py_DECREF(value_handle_obj);
            return NULL;
        }
        value[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(value);
            Py_DECREF(value_sequence);
            if (value_handle_obj) Py_DECREF(value_handle_obj);
            return NULL;
        }
    }
    F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__t_node__array_setitem__node)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_mod_recursive_type_array__t_node_helper_array_len_node(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_parent;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_parent)) {
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
    int length = 0;
    F90WRAP_F_SYMBOL(f90wrap_mod_recursive_type_array__t_node__array_len__node)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

/* Method table for _recursive_type_array module */
static PyMethodDef _recursive_type_array_methods[] = {
    {"f90wrap_mod_recursive_type_array__allocate_node", (PyCFunction)wrap_mod_recursive_type_array_allocate_node, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for allocate_node"},
    {"f90wrap_mod_recursive_type_array__deallocate_node", (PyCFunction)wrap_mod_recursive_type_array_deallocate_node, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for deallocate_node"},
    {"f90wrap_mod_recursive_type_array__t_node_initialise", (PyCFunction)wrap_mod_recursive_type_array_t_node_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for t_node"},
    {"f90wrap_mod_recursive_type_array__t_node_finalise", (PyCFunction)wrap_mod_recursive_type_array_t_node_finalise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated destructor for t_node"},
    {"f90wrap_mod_recursive_type_array__t_node__array_getitem__node", \
        (PyCFunction)wrap_mod_recursive_type_array__t_node_helper_array_getitem_node, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for node"},
    {"f90wrap_mod_recursive_type_array__t_node__array_setitem__node", \
        (PyCFunction)wrap_mod_recursive_type_array__t_node_helper_array_setitem_node, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for node"},
    {"f90wrap_mod_recursive_type_array__t_node__array_len__node", \
        (PyCFunction)wrap_mod_recursive_type_array__t_node_helper_array_len_node, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for node"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _recursive_type_arraymodule = {
    PyModuleDef_HEAD_INIT,
    "recursive_type_array",
    "Direct-C wrapper for _recursive_type_array module",
    -1,
    _recursive_type_array_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__recursive_type_array(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_recursive_type_arraymodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
