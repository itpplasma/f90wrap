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
extern void F90WRAP_F_SYMBOL(f90wrap_my_module__allocit)(int* x, int* n, int* m);
extern void F90WRAP_F_SYMBOL(f90wrap_my_module__mytype_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_my_module__mytype_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_my_module__mytype__get__n)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_my_module__mytype__set__n)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_my_module__mytype__get__m)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_my_module__mytype__set__m)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_my_module__mytype__array__y)(int* dummy_this, int* nd, int* dtype, int* dshape, \
    long long* handle);

static PyObject* wrap_my_module_allocit(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_x = NULL;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    PyObject* py_m = NULL;
    int m_val = 0;
    PyArrayObject* m_scalar_arr = NULL;
    int m_scalar_copyback = 0;
    int m_scalar_is_array = 0;
    static char *kwlist[] = {"x", "n", "m", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_x, &py_n, &py_m)) {
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
    
    int* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n must have exactly one element");
            Py_DECREF(n_scalar_arr);
            return NULL;
        }
        n_scalar_is_array = 1;
        n = (int*)PyArray_DATA(n_scalar_arr);
        n_val = n[0];
        if (PyArray_DATA(n_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n) || PyArray_TYPE(n_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n)) {
            n_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n)) {
        n_val = (int)PyLong_AsLong(py_n);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n must be a scalar number or NumPy array");
        return NULL;
    }
    int* m = &m_val;
    if (PyArray_Check(py_m)) {
        m_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_m, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (m_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(m_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument m must have exactly one element");
            Py_DECREF(m_scalar_arr);
            return NULL;
        }
        m_scalar_is_array = 1;
        m = (int*)PyArray_DATA(m_scalar_arr);
        m_val = m[0];
        if (PyArray_DATA(m_scalar_arr) != PyArray_DATA((PyArrayObject*)py_m) || PyArray_TYPE(m_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_m)) {
            m_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_m)) {
        m_val = (int)PyLong_AsLong(py_m);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument m must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_my_module__allocit)(x, n, m);
    if (PyErr_Occurred()) {
        if (x_sequence) Py_DECREF(x_sequence);
        if (x_handle_obj) Py_DECREF(x_handle_obj);
        free(x);
        return NULL;
    }
    
    if (n_scalar_is_array) {
        if (n_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n, n_scalar_arr) < 0) {
                Py_DECREF(n_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n_scalar_arr);
    }
    if (m_scalar_is_array) {
        if (m_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_m, m_scalar_arr) < 0) {
                Py_DECREF(m_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(m_scalar_arr);
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

static PyObject* wrap_my_module_mytype_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_my_module__mytype_initialise)(this);
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

static PyObject* wrap_my_module_mytype_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_my_module__mytype_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_my_module__mytype_helper_get_n(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_my_module__mytype__get__n)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_my_module__mytype_helper_set_n(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "n", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_my_module__mytype__set__n)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_my_module__mytype_helper_get_m(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_my_module__mytype__get__m)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_my_module__mytype_helper_set_m(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "m", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_my_module__mytype__set__m)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_my_module__mytype_helper_array_y(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_my_module__mytype__array__y)(dummy_this, &nd, &dtype, dshape, &handle);
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

/* Method table for _testmodule module */
static PyMethodDef _testmodule_methods[] = {
    {"f90wrap_my_module__allocit", (PyCFunction)wrap_my_module_allocit, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        allocit"},
    {"f90wrap_my_module__mytype_initialise", (PyCFunction)wrap_my_module_mytype_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for mytype"},
    {"f90wrap_my_module__mytype_finalise", (PyCFunction)wrap_my_module_mytype_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for mytype"},
    {"f90wrap_my_module__mytype__get__n", (PyCFunction)wrap_my_module__mytype_helper_get_n, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for n"},
    {"f90wrap_my_module__mytype__set__n", (PyCFunction)wrap_my_module__mytype_helper_set_n, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for n"},
    {"f90wrap_my_module__mytype__get__m", (PyCFunction)wrap_my_module__mytype_helper_get_m, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for m"},
    {"f90wrap_my_module__mytype__set__m", (PyCFunction)wrap_my_module__mytype_helper_set_m, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for m"},
    {"f90wrap_my_module__mytype__array__y", (PyCFunction)wrap_my_module__mytype_helper_array_y, METH_VARARGS | \
        METH_KEYWORDS, "Array helper for y"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _testmodulemodule = {
    PyModuleDef_HEAD_INIT,
    "testmodule",
    "Direct-C wrapper for _testmodule module",
    -1,
    _testmodule_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__testmodule(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_testmodulemodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
