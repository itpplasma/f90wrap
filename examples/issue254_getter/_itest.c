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
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__initialize_interface)(int* this, int* options);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__optionstype_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__optionstype_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__kimdispersionequation9266)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__kimdispersionequationc882)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersion_horton_module__kimdispersion_horton_ib155)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersion_horton_module__kimdispersion_horton_fa9f5)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__optionstype__get__omega)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__optionstype__set__omega)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersion_horton_module__kimdispersion_horton__6483)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersion_horton_module__kimdispersion_horton__7aeb)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__initialize__binding__5dd3)(int* this, int* options);
extern void F90WRAP_F_SYMBOL(f90wrap_kimdispersion_horton_module__initialize__binding__k2119)(int* this, int* options);

static PyObject* wrap_kimdispersionequation_module_initialize_interface(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    PyObject* py_this = NULL;
    PyObject* py_options = NULL;
    static char *kwlist[] = {"this", "options", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_options)) {
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
    
    PyObject* options_handle_obj = NULL;
    PyObject* options_sequence = NULL;
    Py_ssize_t options_handle_len = 0;
    if (PyObject_HasAttrString(py_options, "_handle")) {
        options_handle_obj = PyObject_GetAttrString(py_options, "_handle");
        if (options_handle_obj == NULL) {
            return NULL;
        }
        options_sequence = PySequence_Fast(options_handle_obj, "Failed to access handle sequence");
        if (options_sequence == NULL) {
            Py_DECREF(options_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_options)) {
        options_sequence = PySequence_Fast(py_options, "Argument options must be a handle sequence");
        if (options_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument options must be a Fortran derived-type instance");
        return NULL;
    }
    options_handle_len = PySequence_Fast_GET_SIZE(options_sequence);
    if (options_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument options has an invalid handle length");
        Py_DECREF(options_sequence);
        if (options_handle_obj) Py_DECREF(options_handle_obj);
        return NULL;
    }
    int* options = (int*)malloc(sizeof(int) * options_handle_len);
    if (options == NULL) {
        PyErr_NoMemory();
        Py_DECREF(options_sequence);
        if (options_handle_obj) Py_DECREF(options_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < options_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(options_sequence, i);
        if (item == NULL) {
            free(options);
            Py_DECREF(options_sequence);
            if (options_handle_obj) Py_DECREF(options_handle_obj);
            return NULL;
        }
        options[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(options);
            Py_DECREF(options_sequence);
            if (options_handle_obj) Py_DECREF(options_handle_obj);
            return NULL;
        }
    }
    (void)options_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__initialize_interface)(this, options);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        if (options_sequence) Py_DECREF(options_sequence);
        if (options_handle_obj) Py_DECREF(options_handle_obj);
        free(options);
        return NULL;
    }
    
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    if (options_sequence) {
        Py_DECREF(options_sequence);
    }
    if (options_handle_obj) {
        Py_DECREF(options_handle_obj);
    }
    free(options);
    Py_RETURN_NONE;
}

static PyObject* wrap_kimdispersionequation_module_optionstype_initialise(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__optionstype_initialise)(this);
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

static PyObject* wrap_kimdispersionequation_module_optionstype_finalise(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__optionstype_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_kimdispersionequation_module_kimdispersionequation_initialise(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__kimdispersionequation9266)(this);
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

static PyObject* wrap_kimdispersionequation_module_kimdispersionequation_finalise(PyObject* self, PyObject* args, \
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
    F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__kimdispersionequationc882)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_kimdispersion_horton_module_kimdispersion_horton_initialise(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_kimdispersion_horton_module__kimdispersion_horton_ib155)(this);
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

static PyObject* wrap_kimdispersion_horton_module_kimdispersion_horton_finalise(PyObject* self, PyObject* args, \
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
    F90WRAP_F_SYMBOL(f90wrap_kimdispersion_horton_module__kimdispersion_horton_fa9f5)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_kimdispersionequation_module__optionstype_helper_get_omega(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    double value;
    F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__optionstype__get__omega)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_kimdispersionequation_module__optionstype_helper_set_omega(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "omega", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od", kwlist, &py_handle, &value)) {
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
    F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__optionstype__set__omega)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_kimdispersion_horton_module__kimdispersion_horton_helper_get_derived_options(PyObject* self, \
    PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_kimdispersion_horton_module__kimdispersion_horton__6483)(handle_handle, value_handle);
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

static PyObject* wrap_kimdispersion_horton_module__kimdispersion_horton_helper_set_derived_options(PyObject* self, \
    PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_kimdispersion_horton_module__kimdispersion_horton__7aeb)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap__kimdispersionequation_module__initialize__binding__5dd3(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_options = NULL;
    static char *kwlist[] = {"this", "options", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_options)) {
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
    
    PyObject* options_handle_obj = NULL;
    PyObject* options_sequence = NULL;
    Py_ssize_t options_handle_len = 0;
    if (PyObject_HasAttrString(py_options, "_handle")) {
        options_handle_obj = PyObject_GetAttrString(py_options, "_handle");
        if (options_handle_obj == NULL) {
            return NULL;
        }
        options_sequence = PySequence_Fast(options_handle_obj, "Failed to access handle sequence");
        if (options_sequence == NULL) {
            Py_DECREF(options_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_options)) {
        options_sequence = PySequence_Fast(py_options, "Argument options must be a handle sequence");
        if (options_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument options must be a Fortran derived-type instance");
        return NULL;
    }
    options_handle_len = PySequence_Fast_GET_SIZE(options_sequence);
    if (options_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument options has an invalid handle length");
        Py_DECREF(options_sequence);
        if (options_handle_obj) Py_DECREF(options_handle_obj);
        return NULL;
    }
    int* options = (int*)malloc(sizeof(int) * options_handle_len);
    if (options == NULL) {
        PyErr_NoMemory();
        Py_DECREF(options_sequence);
        if (options_handle_obj) Py_DECREF(options_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < options_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(options_sequence, i);
        if (item == NULL) {
            free(options);
            Py_DECREF(options_sequence);
            if (options_handle_obj) Py_DECREF(options_handle_obj);
            return NULL;
        }
        options[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(options);
            Py_DECREF(options_sequence);
            if (options_handle_obj) Py_DECREF(options_handle_obj);
            return NULL;
        }
    }
    (void)options_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_kimdispersionequation_module__initialize__binding__5dd3)(this, options);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        if (options_sequence) Py_DECREF(options_sequence);
        if (options_handle_obj) Py_DECREF(options_handle_obj);
        free(options);
        return NULL;
    }
    
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    if (options_sequence) {
        Py_DECREF(options_sequence);
    }
    if (options_handle_obj) {
        Py_DECREF(options_handle_obj);
    }
    free(options);
    Py_RETURN_NONE;
}

static PyObject* wrap__kimdispersion_horton_module__initialize__binding__k2119(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_options = NULL;
    static char *kwlist[] = {"this", "options", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_options)) {
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
    
    PyObject* options_handle_obj = NULL;
    PyObject* options_sequence = NULL;
    Py_ssize_t options_handle_len = 0;
    if (PyObject_HasAttrString(py_options, "_handle")) {
        options_handle_obj = PyObject_GetAttrString(py_options, "_handle");
        if (options_handle_obj == NULL) {
            return NULL;
        }
        options_sequence = PySequence_Fast(options_handle_obj, "Failed to access handle sequence");
        if (options_sequence == NULL) {
            Py_DECREF(options_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_options)) {
        options_sequence = PySequence_Fast(py_options, "Argument options must be a handle sequence");
        if (options_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument options must be a Fortran derived-type instance");
        return NULL;
    }
    options_handle_len = PySequence_Fast_GET_SIZE(options_sequence);
    if (options_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument options has an invalid handle length");
        Py_DECREF(options_sequence);
        if (options_handle_obj) Py_DECREF(options_handle_obj);
        return NULL;
    }
    int* options = (int*)malloc(sizeof(int) * options_handle_len);
    if (options == NULL) {
        PyErr_NoMemory();
        Py_DECREF(options_sequence);
        if (options_handle_obj) Py_DECREF(options_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < options_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(options_sequence, i);
        if (item == NULL) {
            free(options);
            Py_DECREF(options_sequence);
            if (options_handle_obj) Py_DECREF(options_handle_obj);
            return NULL;
        }
        options[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(options);
            Py_DECREF(options_sequence);
            if (options_handle_obj) Py_DECREF(options_handle_obj);
            return NULL;
        }
    }
    (void)options_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_kimdispersion_horton_module__initialize__binding__k2119)(this, options);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        if (options_sequence) Py_DECREF(options_sequence);
        if (options_handle_obj) Py_DECREF(options_handle_obj);
        free(options);
        return NULL;
    }
    
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    if (options_sequence) {
        Py_DECREF(options_sequence);
    }
    if (options_handle_obj) {
        Py_DECREF(options_handle_obj);
    }
    free(options);
    Py_RETURN_NONE;
}

/* Method table for _itest module */
static PyMethodDef _itest_methods[] = {
    {"f90wrap_kimdispersionequation_module__initialize_interface", \
        (PyCFunction)wrap_kimdispersionequation_module_initialize_interface, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        initialize_interface"},
    {"f90wrap_kimdispersionequation_module__optionstype_initialise", \
        (PyCFunction)wrap_kimdispersionequation_module_optionstype_initialise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated constructor for optionstype"},
    {"f90wrap_kimdispersionequation_module__optionstype_finalise", \
        (PyCFunction)wrap_kimdispersionequation_module_optionstype_finalise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated destructor for optionstype"},
    {"f90wrap_kimdispersionequation_module__kimdispersionequation9266", \
        (PyCFunction)wrap_kimdispersionequation_module_kimdispersionequation_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for kimdispersionequation"},
    {"f90wrap_kimdispersionequation_module__kimdispersionequationc882", \
        (PyCFunction)wrap_kimdispersionequation_module_kimdispersionequation_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for kimdispersionequation"},
    {"f90wrap_kimdispersion_horton_module__kimdispersion_horton_ib155", \
        (PyCFunction)wrap_kimdispersion_horton_module_kimdispersion_horton_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for kimdispersion_horton"},
    {"f90wrap_kimdispersion_horton_module__kimdispersion_horton_fa9f5", \
        (PyCFunction)wrap_kimdispersion_horton_module_kimdispersion_horton_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for kimdispersion_horton"},
    {"f90wrap_kimdispersionequation_module__optionstype__get__omega", \
        (PyCFunction)wrap_kimdispersionequation_module__optionstype_helper_get_omega, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for omega"},
    {"f90wrap_kimdispersionequation_module__optionstype__set__omega", \
        (PyCFunction)wrap_kimdispersionequation_module__optionstype_helper_set_omega, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for omega"},
    {"f90wrap_kimdispersion_horton_module__kimdispersion_horton__6483", \
        (PyCFunction)wrap_kimdispersion_horton_module__kimdispersion_horton_helper_get_derived_options, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for options"},
    {"f90wrap_kimdispersion_horton_module__kimdispersion_horton__7aeb", \
        (PyCFunction)wrap_kimdispersion_horton_module__kimdispersion_horton_helper_set_derived_options, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for options"},
    {"f90wrap_kimdispersionequation_module__initialize__binding__5dd3", \
        (PyCFunction)wrap__kimdispersionequation_module__initialize__binding__5dd3, METH_VARARGS | METH_KEYWORDS, "Binding \
        alias for initialize"},
    {"f90wrap_kimdispersion_horton_module__initialize__binding__k2119", \
        (PyCFunction)wrap__kimdispersion_horton_module__initialize__binding__k2119, METH_VARARGS | METH_KEYWORDS, "Binding \
        alias for initialize"},
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
