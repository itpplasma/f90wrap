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
extern void F90WRAP_F_SYMBOL(f90wrap_m_base_type__t_base_type_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_base_type__t_base_type_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_composition__t_composition_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_composition__t_composition_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_inheritance__t_inheritance_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_inheritance__t_inheritance_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_fortran_module__a_subroutine)(int* input);
extern void F90WRAP_F_SYMBOL(f90wrap_m_fortran_module__b_subroutine)(int* input);
extern void F90WRAP_F_SYMBOL(f90wrap_m_fortran_module__c_subroutine)(int* input);
extern void F90WRAP_F_SYMBOL(f90wrap_m_base_type__t_base_type__get__real_number)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_base_type__t_base_type__set__real_number)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_composition__t_composition__get__member)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_composition__t_composition__set__member)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_inheritance__t_inheritance__get__integer_number)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_inheritance__t_inheritance__set__integer_number)(int* handle, int* value);

static PyObject* wrap_m_base_type_t_base_type_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_base_type__t_base_type_initialise)(this);
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

static PyObject* wrap_m_base_type_t_base_type_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_base_type__t_base_type_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_composition_t_composition_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_composition__t_composition_initialise)(this);
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

static PyObject* wrap_m_composition_t_composition_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_composition__t_composition_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_inheritance_t_inheritance_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_inheritance__t_inheritance_initialise)(this);
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

static PyObject* wrap_m_inheritance_t_inheritance_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_inheritance__t_inheritance_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_fortran_module_a_subroutine(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    PyObject* input_handle_obj = NULL;
    PyObject* input_sequence = NULL;
    Py_ssize_t input_handle_len = 0;
    if (PyObject_HasAttrString(py_input, "_handle")) {
        input_handle_obj = PyObject_GetAttrString(py_input, "_handle");
        if (input_handle_obj == NULL) {
            return NULL;
        }
        input_sequence = PySequence_Fast(input_handle_obj, "Failed to access handle sequence");
        if (input_sequence == NULL) {
            Py_DECREF(input_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_input)) {
        input_sequence = PySequence_Fast(py_input, "Argument input must be a handle sequence");
        if (input_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a Fortran derived-type instance");
        return NULL;
    }
    input_handle_len = PySequence_Fast_GET_SIZE(input_sequence);
    if (input_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument input has an invalid handle length");
        Py_DECREF(input_sequence);
        if (input_handle_obj) Py_DECREF(input_handle_obj);
        return NULL;
    }
    int* input = (int*)malloc(sizeof(int) * input_handle_len);
    if (input == NULL) {
        PyErr_NoMemory();
        Py_DECREF(input_sequence);
        if (input_handle_obj) Py_DECREF(input_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < input_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(input_sequence, i);
        if (item == NULL) {
            free(input);
            Py_DECREF(input_sequence);
            if (input_handle_obj) Py_DECREF(input_handle_obj);
            return NULL;
        }
        input[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(input);
            Py_DECREF(input_sequence);
            if (input_handle_obj) Py_DECREF(input_handle_obj);
            return NULL;
        }
    }
    (void)input_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_fortran_module__a_subroutine)(input);
    if (PyErr_Occurred()) {
        if (input_sequence) Py_DECREF(input_sequence);
        if (input_handle_obj) Py_DECREF(input_handle_obj);
        free(input);
        return NULL;
    }
    
    if (input_sequence) {
        Py_DECREF(input_sequence);
    }
    if (input_handle_obj) {
        Py_DECREF(input_handle_obj);
    }
    free(input);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_fortran_module_b_subroutine(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    PyObject* input_handle_obj = NULL;
    PyObject* input_sequence = NULL;
    Py_ssize_t input_handle_len = 0;
    if (PyObject_HasAttrString(py_input, "_handle")) {
        input_handle_obj = PyObject_GetAttrString(py_input, "_handle");
        if (input_handle_obj == NULL) {
            return NULL;
        }
        input_sequence = PySequence_Fast(input_handle_obj, "Failed to access handle sequence");
        if (input_sequence == NULL) {
            Py_DECREF(input_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_input)) {
        input_sequence = PySequence_Fast(py_input, "Argument input must be a handle sequence");
        if (input_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a Fortran derived-type instance");
        return NULL;
    }
    input_handle_len = PySequence_Fast_GET_SIZE(input_sequence);
    if (input_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument input has an invalid handle length");
        Py_DECREF(input_sequence);
        if (input_handle_obj) Py_DECREF(input_handle_obj);
        return NULL;
    }
    int* input = (int*)malloc(sizeof(int) * input_handle_len);
    if (input == NULL) {
        PyErr_NoMemory();
        Py_DECREF(input_sequence);
        if (input_handle_obj) Py_DECREF(input_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < input_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(input_sequence, i);
        if (item == NULL) {
            free(input);
            Py_DECREF(input_sequence);
            if (input_handle_obj) Py_DECREF(input_handle_obj);
            return NULL;
        }
        input[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(input);
            Py_DECREF(input_sequence);
            if (input_handle_obj) Py_DECREF(input_handle_obj);
            return NULL;
        }
    }
    (void)input_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_fortran_module__b_subroutine)(input);
    if (PyErr_Occurred()) {
        if (input_sequence) Py_DECREF(input_sequence);
        if (input_handle_obj) Py_DECREF(input_handle_obj);
        free(input);
        return NULL;
    }
    
    if (input_sequence) {
        Py_DECREF(input_sequence);
    }
    if (input_handle_obj) {
        Py_DECREF(input_handle_obj);
    }
    free(input);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_fortran_module_c_subroutine(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    PyObject* input_handle_obj = NULL;
    PyObject* input_sequence = NULL;
    Py_ssize_t input_handle_len = 0;
    if (PyObject_HasAttrString(py_input, "_handle")) {
        input_handle_obj = PyObject_GetAttrString(py_input, "_handle");
        if (input_handle_obj == NULL) {
            return NULL;
        }
        input_sequence = PySequence_Fast(input_handle_obj, "Failed to access handle sequence");
        if (input_sequence == NULL) {
            Py_DECREF(input_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_input)) {
        input_sequence = PySequence_Fast(py_input, "Argument input must be a handle sequence");
        if (input_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a Fortran derived-type instance");
        return NULL;
    }
    input_handle_len = PySequence_Fast_GET_SIZE(input_sequence);
    if (input_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument input has an invalid handle length");
        Py_DECREF(input_sequence);
        if (input_handle_obj) Py_DECREF(input_handle_obj);
        return NULL;
    }
    int* input = (int*)malloc(sizeof(int) * input_handle_len);
    if (input == NULL) {
        PyErr_NoMemory();
        Py_DECREF(input_sequence);
        if (input_handle_obj) Py_DECREF(input_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < input_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(input_sequence, i);
        if (item == NULL) {
            free(input);
            Py_DECREF(input_sequence);
            if (input_handle_obj) Py_DECREF(input_handle_obj);
            return NULL;
        }
        input[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(input);
            Py_DECREF(input_sequence);
            if (input_handle_obj) Py_DECREF(input_handle_obj);
            return NULL;
        }
    }
    (void)input_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_fortran_module__c_subroutine)(input);
    if (PyErr_Occurred()) {
        if (input_sequence) Py_DECREF(input_sequence);
        if (input_handle_obj) Py_DECREF(input_handle_obj);
        free(input);
        return NULL;
    }
    
    if (input_sequence) {
        Py_DECREF(input_sequence);
    }
    if (input_handle_obj) {
        Py_DECREF(input_handle_obj);
    }
    free(input);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_base_type__t_base_type_helper_get_real_number(PyObject* self, PyObject* args, PyObject* kwargs)
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
    float value;
    F90WRAP_F_SYMBOL(f90wrap_m_base_type__t_base_type__get__real_number)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_base_type__t_base_type_helper_set_real_number(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    float value;
    static char *kwlist[] = {"handle", "real_number", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_m_base_type__t_base_type__set__real_number)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_composition__t_composition_helper_get_derived_member(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_composition__t_composition__get__member)(handle_handle, value_handle);
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

static PyObject* wrap_m_composition__t_composition_helper_set_derived_member(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_composition__t_composition__set__member)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_inheritance__t_inheritance_helper_get_integer_number(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_inheritance__t_inheritance__get__integer_number)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_m_inheritance__t_inheritance_helper_set_integer_number(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "integer_number", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_m_inheritance__t_inheritance__set__integer_number)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _pywrapper module */
static PyMethodDef _pywrapper_methods[] = {
    {"f90wrap_m_base_type__t_base_type_initialise", (PyCFunction)wrap_m_base_type_t_base_type_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for t_base_type"},
    {"f90wrap_m_base_type__t_base_type_finalise", (PyCFunction)wrap_m_base_type_t_base_type_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for t_base_type"},
    {"f90wrap_m_composition__t_composition_initialise", (PyCFunction)wrap_m_composition_t_composition_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for t_composition"},
    {"f90wrap_m_composition__t_composition_finalise", (PyCFunction)wrap_m_composition_t_composition_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for t_composition"},
    {"f90wrap_m_inheritance__t_inheritance_initialise", (PyCFunction)wrap_m_inheritance_t_inheritance_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for t_inheritance"},
    {"f90wrap_m_inheritance__t_inheritance_finalise", (PyCFunction)wrap_m_inheritance_t_inheritance_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for t_inheritance"},
    {"f90wrap_m_fortran_module__a_subroutine", (PyCFunction)wrap_m_fortran_module_a_subroutine, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for a_subroutine"},
    {"f90wrap_m_fortran_module__b_subroutine", (PyCFunction)wrap_m_fortran_module_b_subroutine, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for b_subroutine"},
    {"f90wrap_m_fortran_module__c_subroutine", (PyCFunction)wrap_m_fortran_module_c_subroutine, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for c_subroutine"},
    {"f90wrap_m_base_type__t_base_type__get__real_number", \
        (PyCFunction)wrap_m_base_type__t_base_type_helper_get_real_number, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        real_number"},
    {"f90wrap_m_base_type__t_base_type__set__real_number", \
        (PyCFunction)wrap_m_base_type__t_base_type_helper_set_real_number, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        real_number"},
    {"f90wrap_m_composition__t_composition__get__member", \
        (PyCFunction)wrap_m_composition__t_composition_helper_get_derived_member, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for member"},
    {"f90wrap_m_composition__t_composition__set__member", \
        (PyCFunction)wrap_m_composition__t_composition_helper_set_derived_member, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for member"},
    {"f90wrap_m_inheritance__t_inheritance__get__integer_number", \
        (PyCFunction)wrap_m_inheritance__t_inheritance_helper_get_integer_number, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for integer_number"},
    {"f90wrap_m_inheritance__t_inheritance__set__integer_number", \
        (PyCFunction)wrap_m_inheritance__t_inheritance_helper_set_integer_number, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for integer_number"},
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
