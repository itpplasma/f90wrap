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
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_subra0ea)(void);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_typefcc3)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type6ffd)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_typebe6a)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type1fc3)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__get__m_long_subroutine_namebc01)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__set__m_long_subroutine_name860c)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type2f88)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_typebfce)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_typeeaf3)(int* dummy_this, int* nd, \
    int* dtype, int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type4adb)(int* dummy_this, int* \
    index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type97d9)(int* dummy_this, int* \
    index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type5005)(int* dummy_this, int* \
    length);

static PyObject* wrap_m_long_subroutine_name_m_long_subroutine_name_subroutine(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_subra0ea)();
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* wrap_m_long_subroutine_name_m_long_subroutine_name_type_initialise(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_typefcc3)(this);
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

static PyObject* wrap_m_long_subroutine_name_m_long_subroutine_name_type_finalise(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type6ffd)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_long_subroutine_name_m_long_subroutine_name_type_2_initialise(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_typebe6a)(this);
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

static PyObject* wrap_m_long_subroutine_name_m_long_subroutine_name_type_2_finalise(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type1fc3)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_long_subroutine_name_helper_get_m_long_subroutine_name_integer(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__get__m_long_subroutine_namebc01)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_m_long_subroutine_name_helper_set_m_long_subroutine_name_integer(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    int value;
    static char *kwlist[] = {"m_long_subroutine_name_integer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__set__m_long_subroutine_name860c)(&value);
    Py_RETURN_NONE;
}

static PyObject* \
    wrap_m_long_subroutine_name__m_long_subroutine_name_type_helper_get_m_long_subroutine_name_type_integer(PyObject* \
    self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type2f88)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* \
    wrap_m_long_subroutine_name__m_long_subroutine_name_type_helper_set_m_long_subroutine_name_type_integer(PyObject* \
    self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "m_long_subroutine_name_type_integer", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_typebfce)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* \
    wrap_m_long_subroutine_name__m_long_subroutine_name_type_helper_array_m_long_subroutine_name_type_integer_array(PyObject* \
    self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_typeeaf3)(dummy_this, &nd, &dtype, dshape, \
        &handle);
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

static PyObject* \
    wrap_m_long_subroutine_name__m_long_subroutine_name_type_2_helper_array_getitem_m_long_subroutine_name_type_2_type_array(PyObject* \
    self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type4adb)(parent_handle, &index, handle);
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

static PyObject* \
    wrap_m_long_subroutine_name__m_long_subroutine_name_type_2_helper_array_setitem_m_long_subroutine_name_type_2_type_array(PyObject* \
    self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type97d9)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* \
    wrap_m_long_subroutine_name__m_long_subroutine_name_type_2_helper_array_len_m_long_subroutine_name_type_2_type_array(PyObject* \
    self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_long_subroutine_name__m_long_subroutine_name_type5005)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

/* Method table for _pywrapper module */
static PyMethodDef _pywrapper_methods[] = {
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_subra0ea", \
        (PyCFunction)wrap_m_long_subroutine_name_m_long_subroutine_name_subroutine, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for m_long_subroutine_name_subroutine"},
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_typefcc3", \
        (PyCFunction)wrap_m_long_subroutine_name_m_long_subroutine_name_type_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for m_long_subroutine_name_type"},
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_type6ffd", \
        (PyCFunction)wrap_m_long_subroutine_name_m_long_subroutine_name_type_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for m_long_subroutine_name_type"},
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_typebe6a", \
        (PyCFunction)wrap_m_long_subroutine_name_m_long_subroutine_name_type_2_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for m_long_subroutine_name_type_2"},
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_type1fc3", \
        (PyCFunction)wrap_m_long_subroutine_name_m_long_subroutine_name_type_2_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for m_long_subroutine_name_type_2"},
    {"f90wrap_m_long_subroutine_name__get__m_long_subroutine_namebc01", \
        (PyCFunction)wrap_m_long_subroutine_name_helper_get_m_long_subroutine_name_integer, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for m_long_subroutine_name_integer"},
    {"f90wrap_m_long_subroutine_name__set__m_long_subroutine_name860c", \
        (PyCFunction)wrap_m_long_subroutine_name_helper_set_m_long_subroutine_name_integer, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for m_long_subroutine_name_integer"},
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_type2f88", \
        (PyCFunction)wrap_m_long_subroutine_name__m_long_subroutine_name_type_helper_get_m_long_subroutine_name_type_integer, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for m_long_subroutine_name_type_integer"},
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_typebfce", \
        (PyCFunction)wrap_m_long_subroutine_name__m_long_subroutine_name_type_helper_set_m_long_subroutine_name_type_integer, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for m_long_subroutine_name_type_integer"},
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_typeeaf3", \
        (PyCFunction)wrap_m_long_subroutine_name__m_long_subroutine_name_type_helper_array_m_long_subroutine_name_type_integer_array, \
        METH_VARARGS | METH_KEYWORDS, "Array helper for m_long_subroutine_name_type_integer_array"},
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_type4adb", \
        (PyCFunction)wrap_m_long_subroutine_name__m_long_subroutine_name_type_2_helper_array_getitem_m_long_subroutine_name_type_2_type_array, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for m_long_subroutine_name_type_2_type_array"},
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_type97d9", \
        (PyCFunction)wrap_m_long_subroutine_name__m_long_subroutine_name_type_2_helper_array_setitem_m_long_subroutine_name_type_2_type_array, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for m_long_subroutine_name_type_2_type_array"},
    {"f90wrap_m_long_subroutine_name__m_long_subroutine_name_type5005", \
        (PyCFunction)wrap_m_long_subroutine_name__m_long_subroutine_name_type_2_helper_array_len_m_long_subroutine_name_type_2_type_array, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for m_long_subroutine_name_type_2_type_array"},
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
