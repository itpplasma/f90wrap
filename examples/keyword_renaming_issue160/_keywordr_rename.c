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
extern void F90WRAP_F_SYMBOL(f90wrap_global__is_)(int* a);
extern void F90WRAP_F_SYMBOL(f90wrap_global__class2_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_global__class2_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_in_)(int* a, int* ret_in_);
extern void F90WRAP_F_SYMBOL(f90wrap_global___get__abc)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_global___set__abc)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_global___get__lambda_)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_global___array__with_)(int* dummy_this, int* nd, int* dtype, int* dshape, long \
    long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_global__class2__get__x)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_global__class2__set__x)(int* handle, int* value);

static PyObject* wrap_global_is_(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_a = NULL;
    int a_val = 0;
    PyArrayObject* a_scalar_arr = NULL;
    int a_scalar_copyback = 0;
    int a_scalar_is_array = 0;
    static char *kwlist[] = {"a", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_a)) {
        return NULL;
    }
    
    int* a = &a_val;
    if (PyArray_Check(py_a)) {
        a_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_a, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (a_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(a_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument a must have exactly one element");
            Py_DECREF(a_scalar_arr);
            return NULL;
        }
        a_scalar_is_array = 1;
        a = (int*)PyArray_DATA(a_scalar_arr);
        a_val = a[0];
        if (PyArray_DATA(a_scalar_arr) != PyArray_DATA((PyArrayObject*)py_a) || PyArray_TYPE(a_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_a)) {
            a_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_a)) {
        a_val = (int)PyLong_AsLong(py_a);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument a must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_global__is_)(a);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (a_scalar_is_array) {
        if (a_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_a, a_scalar_arr) < 0) {
                Py_DECREF(a_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(a_scalar_arr);
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_global_class2_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_global__class2_initialise)(this);
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

static PyObject* wrap_global_class2_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_global__class2_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap__keywordr_rename_in_(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_a = NULL;
    int a_val = 0;
    PyArrayObject* a_scalar_arr = NULL;
    int a_scalar_copyback = 0;
    int a_scalar_is_array = 0;
    int ret_in__val = 0;
    static char *kwlist[] = {"a", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_a)) {
        return NULL;
    }
    
    int* a = &a_val;
    if (PyArray_Check(py_a)) {
        a_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_a, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (a_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(a_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument a must have exactly one element");
            Py_DECREF(a_scalar_arr);
            return NULL;
        }
        a_scalar_is_array = 1;
        a = (int*)PyArray_DATA(a_scalar_arr);
        a_val = a[0];
        if (PyArray_DATA(a_scalar_arr) != PyArray_DATA((PyArrayObject*)py_a) || PyArray_TYPE(a_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_a)) {
            a_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_a)) {
        a_val = (int)PyLong_AsLong(py_a);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument a must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_in_)(a, &ret_in__val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (a_scalar_is_array) {
        if (a_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_a, a_scalar_arr) < 0) {
                Py_DECREF(a_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(a_scalar_arr);
    }
    PyObject* py_ret_in__obj = Py_BuildValue("i", ret_in__val);
    if (py_ret_in__obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_in__obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_in__obj != NULL) return py_ret_in__obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_in__obj != NULL) Py_DECREF(py_ret_in__obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_in__obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_in__obj);
    }
    return result_tuple;
}

static PyObject* wrap_global__helper_get_abc(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_global___get__abc)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_global__helper_set_abc(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    int value;
    static char *kwlist[] = {"abc", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_global___set__abc)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_global__helper_get_lambda_(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_global___get__lambda_)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_global__helper_array_with_(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_global___array__with_)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_global__class2_helper_get_x(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_global__class2__get__x)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_global__class2_helper_set_x(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "x", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_global__class2__set__x)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _keywordr_rename module */
static PyMethodDef _keywordr_rename_methods[] = {
    {"f90wrap_global__is_", (PyCFunction)wrap_global_is_, METH_VARARGS | METH_KEYWORDS, "Wrapper for is_"},
    {"f90wrap_global__class2_initialise", (PyCFunction)wrap_global_class2_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for class2"},
    {"f90wrap_global__class2_finalise", (PyCFunction)wrap_global_class2_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for class2"},
    {"f90wrap_in_", (PyCFunction)wrap__keywordr_rename_in_, METH_VARARGS | METH_KEYWORDS, "Wrapper for in_"},
    {"f90wrap_global___get__abc", (PyCFunction)wrap_global__helper_get_abc, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        abc"},
    {"f90wrap_global___set__abc", (PyCFunction)wrap_global__helper_set_abc, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        abc"},
    {"f90wrap_global___get__lambda_", (PyCFunction)wrap_global__helper_get_lambda_, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for lambda_"},
    {"f90wrap_global___array__with_", (PyCFunction)wrap_global__helper_array_with_, METH_VARARGS | METH_KEYWORDS, "Array \
        helper for with_"},
    {"f90wrap_global__class2__get__x", (PyCFunction)wrap_global__class2_helper_get_x, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for x"},
    {"f90wrap_global__class2__set__x", (PyCFunction)wrap_global__class2_helper_set_x, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for x"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _keywordr_renamemodule = {
    PyModuleDef_HEAD_INIT,
    "keywordr_rename",
    "Direct-C wrapper for _keywordr_rename module",
    -1,
    _keywordr_rename_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__keywordr_rename(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_keywordr_renamemodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
