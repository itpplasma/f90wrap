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
extern void F90WRAP_F_SYMBOL(f90wrap_mytype_mod__constructor)(int* ret_obj);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype_mod__plus_b)(int* obj, int* b, int* c);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype_mod__mytype_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype_mod__mytype_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype_mod__mytype__get__a)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype_mod__mytype__set__a)(int* handle, int* value);

static PyObject* wrap_mytype_mod_constructor(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int ret_obj[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_mytype_mod__constructor)(ret_obj);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_ret_obj_obj = PyList_New(4);
    if (py_ret_obj_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_obj[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_obj_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_obj_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_obj_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_obj_obj != NULL) return py_ret_obj_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_obj_obj != NULL) Py_DECREF(py_ret_obj_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_obj_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_obj_obj);
    }
    return result_tuple;
}

static PyObject* wrap_mytype_mod_plus_b(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_obj = NULL;
    PyObject* py_b = NULL;
    int b_val = 0;
    PyArrayObject* b_scalar_arr = NULL;
    int b_scalar_copyback = 0;
    int b_scalar_is_array = 0;
    int c_val = 0;
    static char *kwlist[] = {"obj", "b", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_obj, &py_b)) {
        return NULL;
    }
    
    PyObject* obj_handle_obj = NULL;
    PyObject* obj_sequence = NULL;
    Py_ssize_t obj_handle_len = 0;
    if (PyObject_HasAttrString(py_obj, "_handle")) {
        obj_handle_obj = PyObject_GetAttrString(py_obj, "_handle");
        if (obj_handle_obj == NULL) {
            return NULL;
        }
        obj_sequence = PySequence_Fast(obj_handle_obj, "Failed to access handle sequence");
        if (obj_sequence == NULL) {
            Py_DECREF(obj_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_obj)) {
        obj_sequence = PySequence_Fast(py_obj, "Argument obj must be a handle sequence");
        if (obj_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument obj must be a Fortran derived-type instance");
        return NULL;
    }
    obj_handle_len = PySequence_Fast_GET_SIZE(obj_sequence);
    if (obj_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument obj has an invalid handle length");
        Py_DECREF(obj_sequence);
        if (obj_handle_obj) Py_DECREF(obj_handle_obj);
        return NULL;
    }
    int* obj = (int*)malloc(sizeof(int) * obj_handle_len);
    if (obj == NULL) {
        PyErr_NoMemory();
        Py_DECREF(obj_sequence);
        if (obj_handle_obj) Py_DECREF(obj_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < obj_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(obj_sequence, i);
        if (item == NULL) {
            free(obj);
            Py_DECREF(obj_sequence);
            if (obj_handle_obj) Py_DECREF(obj_handle_obj);
            return NULL;
        }
        obj[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(obj);
            Py_DECREF(obj_sequence);
            if (obj_handle_obj) Py_DECREF(obj_handle_obj);
            return NULL;
        }
    }
    (void)obj_handle_len;  /* suppress unused warnings when unchanged */
    
    int* b = &b_val;
    if (PyArray_Check(py_b)) {
        b_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_b, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (b_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(b_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument b must have exactly one element");
            Py_DECREF(b_scalar_arr);
            return NULL;
        }
        b_scalar_is_array = 1;
        b = (int*)PyArray_DATA(b_scalar_arr);
        b_val = b[0];
        if (PyArray_DATA(b_scalar_arr) != PyArray_DATA((PyArrayObject*)py_b) || PyArray_TYPE(b_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_b)) {
            b_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_b)) {
        b_val = (int)PyLong_AsLong(py_b);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument b must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_mytype_mod__plus_b)(obj, b, &c_val);
    if (PyErr_Occurred()) {
        if (obj_sequence) Py_DECREF(obj_sequence);
        if (obj_handle_obj) Py_DECREF(obj_handle_obj);
        free(obj);
        return NULL;
    }
    
    if (b_scalar_is_array) {
        if (b_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_b, b_scalar_arr) < 0) {
                Py_DECREF(b_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(b_scalar_arr);
    }
    PyObject* py_c_obj = Py_BuildValue("i", c_val);
    if (py_c_obj == NULL) {
        return NULL;
    }
    if (obj_sequence) {
        Py_DECREF(obj_sequence);
    }
    if (obj_handle_obj) {
        Py_DECREF(obj_handle_obj);
    }
    free(obj);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_c_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_c_obj != NULL) return py_c_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_c_obj != NULL) Py_DECREF(py_c_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_c_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_c_obj);
    }
    return result_tuple;
}

static PyObject* wrap_mytype_mod_mytype_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_mytype_mod__mytype_initialise)(this);
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

static PyObject* wrap_mytype_mod_mytype_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_mytype_mod__mytype_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_mytype_mod__mytype_helper_get_a(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_mytype_mod__mytype__get__a)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_mytype_mod__mytype_helper_set_a(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "a", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_mytype_mod__mytype__set__a)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _mytype module */
static PyMethodDef _mytype_methods[] = {
    {"f90wrap_mytype_mod__constructor", (PyCFunction)wrap_mytype_mod_constructor, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        constructor"},
    {"f90wrap_mytype_mod__plus_b", (PyCFunction)wrap_mytype_mod_plus_b, METH_VARARGS | METH_KEYWORDS, "Wrapper for plus_b"},
    {"f90wrap_mytype_mod__mytype_initialise", (PyCFunction)wrap_mytype_mod_mytype_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for mytype"},
    {"f90wrap_mytype_mod__mytype_finalise", (PyCFunction)wrap_mytype_mod_mytype_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for mytype"},
    {"f90wrap_mytype_mod__mytype__get__a", (PyCFunction)wrap_mytype_mod__mytype_helper_get_a, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for a"},
    {"f90wrap_mytype_mod__mytype__set__a", (PyCFunction)wrap_mytype_mod__mytype_helper_set_a, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for a"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _mytypemodule = {
    PyModuleDef_HEAD_INIT,
    "mytype",
    "Direct-C wrapper for _mytype module",
    -1,
    _mytype_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__mytype(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_mytypemodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
