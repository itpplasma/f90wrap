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
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__recup_point)(int* x);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__type_ptmes_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__type_ptmes_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_type_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_type_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_getitem__xarr)(int* dummy_this, int* index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_setitem__xarr)(int* dummy_this, int* index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_len__xarr)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__type_ptmes__get__y)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__type_ptmes__set__y)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_type__array_getitem__x)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_type__array_setitem__x)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_type__array_len__x)(int* dummy_this, int* length);

static PyObject* wrap_module_calcul_recup_point(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_x = NULL;
    static char *kwlist[] = {"x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_x)) {
        return NULL;
    }
    
    PyObject* x_handle_obj = NULL;
    PyObject* x_sequence = NULL;
    Py_ssize_t x_handle_len = 0;
    if (PyObject_HasAttrString(py_x, "_handle")) {
        x_handle_obj = PyObject_GetAttrString(py_x, "_handle");
        if (x_handle_obj == NULL) {
            return NULL;
        }
        x_sequence = PySequence_Fast(x_handle_obj, "Failed to access handle sequence");
        if (x_sequence == NULL) {
            Py_DECREF(x_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_x)) {
        x_sequence = PySequence_Fast(py_x, "Argument x must be a handle sequence");
        if (x_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument x must be a Fortran derived-type instance");
        return NULL;
    }
    x_handle_len = PySequence_Fast_GET_SIZE(x_sequence);
    if (x_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument x has an invalid handle length");
        Py_DECREF(x_sequence);
        if (x_handle_obj) Py_DECREF(x_handle_obj);
        return NULL;
    }
    int* x = (int*)malloc(sizeof(int) * x_handle_len);
    if (x == NULL) {
        PyErr_NoMemory();
        Py_DECREF(x_sequence);
        if (x_handle_obj) Py_DECREF(x_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < x_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(x_sequence, i);
        if (item == NULL) {
            free(x);
            Py_DECREF(x_sequence);
            if (x_handle_obj) Py_DECREF(x_handle_obj);
            return NULL;
        }
        x[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(x);
            Py_DECREF(x_sequence);
            if (x_handle_obj) Py_DECREF(x_handle_obj);
            return NULL;
        }
    }
    (void)x_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__recup_point)(x);
    if (PyErr_Occurred()) {
        if (x_sequence) Py_DECREF(x_sequence);
        if (x_handle_obj) Py_DECREF(x_handle_obj);
        free(x);
        return NULL;
    }
    
    if (x_sequence) {
        Py_DECREF(x_sequence);
    }
    if (x_handle_obj) {
        Py_DECREF(x_handle_obj);
    }
    free(x);
    Py_RETURN_NONE;
}

static PyObject* wrap_module_calcul_type_ptmes_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__type_ptmes_initialise)(this);
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

static PyObject* wrap_module_calcul_type_ptmes_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__type_ptmes_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_module_calcul_array_type_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_type_initialise)(this);
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

static PyObject* wrap_module_calcul_array_type_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_type_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_module_calcul_helper_array_getitem_xarr(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_getitem__xarr)(parent_handle, &index, handle);
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

static PyObject* wrap_module_calcul_helper_array_setitem_xarr(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_setitem__xarr)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_module_calcul_helper_array_len_xarr(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_len__xarr)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_module_calcul__type_ptmes_helper_get_y(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__type_ptmes__get__y)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_module_calcul__type_ptmes_helper_set_y(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "y", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__type_ptmes__set__y)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_module_calcul__array_type_helper_array_getitem_x(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_type__array_getitem__x)(parent_handle, &index, handle);
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

static PyObject* wrap_module_calcul__array_type_helper_array_setitem_x(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_type__array_setitem__x)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_module_calcul__array_type_helper_array_len_x(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_module_calcul__array_type__array_len__x)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

/* Method table for _arrayderivedtype module */
static PyMethodDef _arrayderivedtype_methods[] = {
    {"f90wrap_module_calcul__recup_point", (PyCFunction)wrap_module_calcul_recup_point, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for recup_point"},
    {"f90wrap_module_calcul__type_ptmes_initialise", (PyCFunction)wrap_module_calcul_type_ptmes_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for type_ptmes"},
    {"f90wrap_module_calcul__type_ptmes_finalise", (PyCFunction)wrap_module_calcul_type_ptmes_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for type_ptmes"},
    {"f90wrap_module_calcul__array_type_initialise", (PyCFunction)wrap_module_calcul_array_type_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for array_type"},
    {"f90wrap_module_calcul__array_type_finalise", (PyCFunction)wrap_module_calcul_array_type_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for array_type"},
    {"f90wrap_module_calcul__array_getitem__xarr", (PyCFunction)wrap_module_calcul_helper_array_getitem_xarr, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for xarr"},
    {"f90wrap_module_calcul__array_setitem__xarr", (PyCFunction)wrap_module_calcul_helper_array_setitem_xarr, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for xarr"},
    {"f90wrap_module_calcul__array_len__xarr", (PyCFunction)wrap_module_calcul_helper_array_len_xarr, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for xarr"},
    {"f90wrap_module_calcul__type_ptmes__get__y", (PyCFunction)wrap_module_calcul__type_ptmes_helper_get_y, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for y"},
    {"f90wrap_module_calcul__type_ptmes__set__y", (PyCFunction)wrap_module_calcul__type_ptmes_helper_set_y, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for y"},
    {"f90wrap_module_calcul__array_type__array_getitem__x", \
        (PyCFunction)wrap_module_calcul__array_type_helper_array_getitem_x, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        x"},
    {"f90wrap_module_calcul__array_type__array_setitem__x", \
        (PyCFunction)wrap_module_calcul__array_type_helper_array_setitem_x, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        x"},
    {"f90wrap_module_calcul__array_type__array_len__x", (PyCFunction)wrap_module_calcul__array_type_helper_array_len_x, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for x"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _arrayderivedtypemodule = {
    PyModuleDef_HEAD_INIT,
    "arrayderivedtype",
    "Direct-C wrapper for _arrayderivedtype module",
    -1,
    _arrayderivedtype_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__arrayderivedtype(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_arrayderivedtypemodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
