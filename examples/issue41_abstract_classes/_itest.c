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
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_base__get_value_i)(int* self, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_base__myclass_t_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_base__myclass_t_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_factory__create_myclass)(char* impl_type, int* ret_myobject, int \
    impl_type_len);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_impl2__myclass_impl2_t_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_impl__myclass_impl_t_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_base__get_value__binding__myclass_t)(int* self, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_impl2__get_value__binding__myclass_impl2_t)(int* self, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_impl2__myclass_impl2_destroy__binding__mycla358)(int* self);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_impl__get_value__binding__myclass_impl_t)(int* self, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_impl__myclass_impl_destroy__binding__myclas021a)(int* self);

static PyObject* wrap_myclass_base_get_value_i(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_self = NULL;
    float value_val = 0;
    static char *kwlist[] = {"self", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_self)) {
        return NULL;
    }
    
    PyObject* self_handle_obj = NULL;
    PyObject* self_sequence = NULL;
    Py_ssize_t self_handle_len = 0;
    if (PyObject_HasAttrString(py_self, "_handle")) {
        self_handle_obj = PyObject_GetAttrString(py_self, "_handle");
        if (self_handle_obj == NULL) {
            return NULL;
        }
        self_sequence = PySequence_Fast(self_handle_obj, "Failed to access handle sequence");
        if (self_sequence == NULL) {
            Py_DECREF(self_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_self)) {
        self_sequence = PySequence_Fast(py_self, "Argument self must be a handle sequence");
        if (self_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument self must be a Fortran derived-type instance");
        return NULL;
    }
    self_handle_len = PySequence_Fast_GET_SIZE(self_sequence);
    if (self_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument self has an invalid handle length");
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    int* self_handle = (int*)malloc(sizeof(int) * self_handle_len);
    if (self_handle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(self_sequence, i);
        if (item == NULL) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
        self_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
    }
    (void)self_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_base__get_value_i)(self_handle, &value_val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    PyObject* py_value_obj = Py_BuildValue("d", value_val);
    if (py_value_obj == NULL) {
        return NULL;
    }
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_value_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_value_obj != NULL) return py_value_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_value_obj != NULL) Py_DECREF(py_value_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_value_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_value_obj);
    }
    return result_tuple;
}

static PyObject* wrap_myclass_base_myclass_t_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_base__myclass_t_initialise)(this);
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

static PyObject* wrap_myclass_base_myclass_t_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_myclass_base__myclass_t_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_myclass_factory_create_myclass(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_impl_type = NULL;
    static char *kwlist[] = {"impl_type", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_impl_type)) {
        return NULL;
    }
    
    int impl_type_len = 0;
    char* impl_type = NULL;
    int impl_type_is_array = 0;
    if (py_impl_type == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument impl_type cannot be None");
        return NULL;
    } else {
        PyObject* impl_type_bytes = NULL;
        if (PyArray_Check(py_impl_type)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* impl_type_arr = (PyArrayObject*)py_impl_type;
            if (PyArray_TYPE(impl_type_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument impl_type must be a string array");
                return NULL;
            }
            impl_type_len = (int)PyArray_ITEMSIZE(impl_type_arr);
            impl_type = (char*)PyArray_DATA(impl_type_arr);
            impl_type_is_array = 1;
        } else if (PyBytes_Check(py_impl_type)) {
            impl_type_bytes = py_impl_type;
            Py_INCREF(impl_type_bytes);
        } else if (PyUnicode_Check(py_impl_type)) {
            impl_type_bytes = PyUnicode_AsUTF8String(py_impl_type);
            if (impl_type_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument impl_type must be str, bytes, or numpy array");
            return NULL;
        }
        if (impl_type_bytes != NULL) {
            impl_type_len = (int)PyBytes_GET_SIZE(impl_type_bytes);
            impl_type = (char*)malloc((size_t)impl_type_len + 1);
            if (impl_type == NULL) {
                Py_DECREF(impl_type_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(impl_type, PyBytes_AS_STRING(impl_type_bytes), (size_t)impl_type_len);
            impl_type[impl_type_len] = '\0';
            Py_DECREF(impl_type_bytes);
        }
    }
    int ret_myobject[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_factory__create_myclass)(impl_type, ret_myobject, impl_type_len);
    if (PyErr_Occurred()) {
        if (!impl_type_is_array) free(impl_type);
        return NULL;
    }
    
    PyObject* py_ret_myobject_obj = PyList_New(4);
    if (py_ret_myobject_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_myobject[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_myobject_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_myobject_obj, i, item);
    }
    if (!impl_type_is_array) free(impl_type);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_myobject_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_myobject_obj != NULL) return py_ret_myobject_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_myobject_obj != NULL) Py_DECREF(py_ret_myobject_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_myobject_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_myobject_obj);
    }
    return result_tuple;
}

static PyObject* wrap_myclass_impl2_myclass_impl2_t_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_impl2__myclass_impl2_t_initialise)(this);
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

static PyObject* wrap_myclass_impl_myclass_impl_t_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_impl__myclass_impl_t_initialise)(this);
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

static PyObject* wrap__myclass_base__get_value__binding__myclass_t(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    float value_val = 0;
    static char *kwlist[] = {"self", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_self)) {
        return NULL;
    }
    
    PyObject* self_handle_obj = NULL;
    PyObject* self_sequence = NULL;
    Py_ssize_t self_handle_len = 0;
    if (PyObject_HasAttrString(py_self, "_handle")) {
        self_handle_obj = PyObject_GetAttrString(py_self, "_handle");
        if (self_handle_obj == NULL) {
            return NULL;
        }
        self_sequence = PySequence_Fast(self_handle_obj, "Failed to access handle sequence");
        if (self_sequence == NULL) {
            Py_DECREF(self_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_self)) {
        self_sequence = PySequence_Fast(py_self, "Argument self must be a handle sequence");
        if (self_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument self must be a Fortran derived-type instance");
        return NULL;
    }
    self_handle_len = PySequence_Fast_GET_SIZE(self_sequence);
    if (self_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument self has an invalid handle length");
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    int* self_handle = (int*)malloc(sizeof(int) * self_handle_len);
    if (self_handle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(self_sequence, i);
        if (item == NULL) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
        self_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
    }
    (void)self_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_base__get_value__binding__myclass_t)(self_handle, &value_val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    PyObject* py_value_obj = Py_BuildValue("d", value_val);
    if (py_value_obj == NULL) {
        return NULL;
    }
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_value_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_value_obj != NULL) return py_value_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_value_obj != NULL) Py_DECREF(py_value_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_value_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_value_obj);
    }
    return result_tuple;
}

static PyObject* wrap__myclass_impl2__get_value__binding__myclass_impl2_t(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    float value_val = 0;
    static char *kwlist[] = {"self", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_self)) {
        return NULL;
    }
    
    PyObject* self_handle_obj = NULL;
    PyObject* self_sequence = NULL;
    Py_ssize_t self_handle_len = 0;
    if (PyObject_HasAttrString(py_self, "_handle")) {
        self_handle_obj = PyObject_GetAttrString(py_self, "_handle");
        if (self_handle_obj == NULL) {
            return NULL;
        }
        self_sequence = PySequence_Fast(self_handle_obj, "Failed to access handle sequence");
        if (self_sequence == NULL) {
            Py_DECREF(self_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_self)) {
        self_sequence = PySequence_Fast(py_self, "Argument self must be a handle sequence");
        if (self_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument self must be a Fortran derived-type instance");
        return NULL;
    }
    self_handle_len = PySequence_Fast_GET_SIZE(self_sequence);
    if (self_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument self has an invalid handle length");
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    int* self_handle = (int*)malloc(sizeof(int) * self_handle_len);
    if (self_handle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(self_sequence, i);
        if (item == NULL) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
        self_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
    }
    (void)self_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_impl2__get_value__binding__myclass_impl2_t)(self_handle, &value_val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    PyObject* py_value_obj = Py_BuildValue("d", value_val);
    if (py_value_obj == NULL) {
        return NULL;
    }
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_value_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_value_obj != NULL) return py_value_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_value_obj != NULL) Py_DECREF(py_value_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_value_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_value_obj);
    }
    return result_tuple;
}

static PyObject* wrap__myclass_impl2__myclass_impl2_destroy__binding__mycla358(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    static char *kwlist[] = {"self", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_self)) {
        return NULL;
    }
    
    PyObject* self_handle_obj = NULL;
    PyObject* self_sequence = NULL;
    Py_ssize_t self_handle_len = 0;
    if (PyObject_HasAttrString(py_self, "_handle")) {
        self_handle_obj = PyObject_GetAttrString(py_self, "_handle");
        if (self_handle_obj == NULL) {
            return NULL;
        }
        self_sequence = PySequence_Fast(self_handle_obj, "Failed to access handle sequence");
        if (self_sequence == NULL) {
            Py_DECREF(self_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_self)) {
        self_sequence = PySequence_Fast(py_self, "Argument self must be a handle sequence");
        if (self_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument self must be a Fortran derived-type instance");
        return NULL;
    }
    self_handle_len = PySequence_Fast_GET_SIZE(self_sequence);
    if (self_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument self has an invalid handle length");
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    int* self_handle = (int*)malloc(sizeof(int) * self_handle_len);
    if (self_handle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(self_sequence, i);
        if (item == NULL) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
        self_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
    }
    (void)self_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_impl2__myclass_impl2_destroy__binding__mycla358)(self_handle);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap__myclass_impl__get_value__binding__myclass_impl_t(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    float value_val = 0;
    static char *kwlist[] = {"self", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_self)) {
        return NULL;
    }
    
    PyObject* self_handle_obj = NULL;
    PyObject* self_sequence = NULL;
    Py_ssize_t self_handle_len = 0;
    if (PyObject_HasAttrString(py_self, "_handle")) {
        self_handle_obj = PyObject_GetAttrString(py_self, "_handle");
        if (self_handle_obj == NULL) {
            return NULL;
        }
        self_sequence = PySequence_Fast(self_handle_obj, "Failed to access handle sequence");
        if (self_sequence == NULL) {
            Py_DECREF(self_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_self)) {
        self_sequence = PySequence_Fast(py_self, "Argument self must be a handle sequence");
        if (self_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument self must be a Fortran derived-type instance");
        return NULL;
    }
    self_handle_len = PySequence_Fast_GET_SIZE(self_sequence);
    if (self_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument self has an invalid handle length");
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    int* self_handle = (int*)malloc(sizeof(int) * self_handle_len);
    if (self_handle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(self_sequence, i);
        if (item == NULL) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
        self_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
    }
    (void)self_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_impl__get_value__binding__myclass_impl_t)(self_handle, &value_val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    PyObject* py_value_obj = Py_BuildValue("d", value_val);
    if (py_value_obj == NULL) {
        return NULL;
    }
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_value_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_value_obj != NULL) return py_value_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_value_obj != NULL) Py_DECREF(py_value_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_value_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_value_obj);
    }
    return result_tuple;
}

static PyObject* wrap__myclass_impl__myclass_impl_destroy__binding__myclas021a(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    static char *kwlist[] = {"self", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_self)) {
        return NULL;
    }
    
    PyObject* self_handle_obj = NULL;
    PyObject* self_sequence = NULL;
    Py_ssize_t self_handle_len = 0;
    if (PyObject_HasAttrString(py_self, "_handle")) {
        self_handle_obj = PyObject_GetAttrString(py_self, "_handle");
        if (self_handle_obj == NULL) {
            return NULL;
        }
        self_sequence = PySequence_Fast(self_handle_obj, "Failed to access handle sequence");
        if (self_sequence == NULL) {
            Py_DECREF(self_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_self)) {
        self_sequence = PySequence_Fast(py_self, "Argument self must be a handle sequence");
        if (self_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument self must be a Fortran derived-type instance");
        return NULL;
    }
    self_handle_len = PySequence_Fast_GET_SIZE(self_sequence);
    if (self_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument self has an invalid handle length");
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    int* self_handle = (int*)malloc(sizeof(int) * self_handle_len);
    if (self_handle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(self_sequence, i);
        if (item == NULL) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
        self_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
    }
    (void)self_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_impl__myclass_impl_destroy__binding__myclas021a)(self_handle);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    Py_RETURN_NONE;
}

/* Method table for _itest module */
static PyMethodDef _itest_methods[] = {
    {"f90wrap_myclass_base__get_value_i", (PyCFunction)wrap_myclass_base_get_value_i, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for get_value_i"},
    {"f90wrap_myclass_base__myclass_t_initialise", (PyCFunction)wrap_myclass_base_myclass_t_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for myclass_t"},
    {"f90wrap_myclass_base__myclass_t_finalise", (PyCFunction)wrap_myclass_base_myclass_t_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for myclass_t"},
    {"f90wrap_myclass_factory__create_myclass", (PyCFunction)wrap_myclass_factory_create_myclass, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for create_myclass"},
    {"f90wrap_myclass_impl2__myclass_impl2_t_initialise", (PyCFunction)wrap_myclass_impl2_myclass_impl2_t_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for myclass_impl2_t"},
    {"f90wrap_myclass_impl__myclass_impl_t_initialise", (PyCFunction)wrap_myclass_impl_myclass_impl_t_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for myclass_impl_t"},
    {"f90wrap_myclass_base__get_value__binding__myclass_t", (PyCFunction)wrap__myclass_base__get_value__binding__myclass_t, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for get_value"},
    {"f90wrap_myclass_impl2__get_value__binding__myclass_impl2_t", \
        (PyCFunction)wrap__myclass_impl2__get_value__binding__myclass_impl2_t, METH_VARARGS | METH_KEYWORDS, "Binding alias \
        for get_value"},
    {"f90wrap_myclass_impl2__myclass_impl2_destroy__binding__mycla358", \
        (PyCFunction)wrap__myclass_impl2__myclass_impl2_destroy__binding__mycla358, METH_VARARGS | METH_KEYWORDS, "Binding \
        alias for myclass_impl2_destroy"},
    {"f90wrap_myclass_impl__get_value__binding__myclass_impl_t", \
        (PyCFunction)wrap__myclass_impl__get_value__binding__myclass_impl_t, METH_VARARGS | METH_KEYWORDS, "Binding alias \
        for get_value"},
    {"f90wrap_myclass_impl__myclass_impl_destroy__binding__myclas021a", \
        (PyCFunction)wrap__myclass_impl__myclass_impl_destroy__binding__myclas021a, METH_VARARGS | METH_KEYWORDS, "Binding \
        alias for myclass_impl_destroy"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _itestmodule = {
    PyModuleDef_HEAD_INIT,
    "itest",
    "Direct-C wrapper for _itest module",
    -1,
    _itest_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__itest(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_itestmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
