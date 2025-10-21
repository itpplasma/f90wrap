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
extern void F90WRAP_F_SYMBOL(f90wrap_cell__cell_dosomething)(int* cell_, int* num_species, char* species_symbol, int \
    species_symbol_len);
extern void F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell__get__num_species)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell__set__num_species)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell__get__species_symbol)(int* handle, char* value, int value_len);
extern void F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell__set__species_symbol)(int* handle, char* value, int value_len);

static PyObject* wrap_cell_cell_dosomething(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_cell_ = NULL;
    PyObject* py_num_species = NULL;
    int num_species_val = 0;
    PyArrayObject* num_species_scalar_arr = NULL;
    int num_species_scalar_copyback = 0;
    int num_species_scalar_is_array = 0;
    PyObject* py_species_symbol = NULL;
    static char *kwlist[] = {"cell_", "num_species", "species_symbol", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_cell_, &py_num_species, &py_species_symbol)) {
        return NULL;
    }
    
    PyObject* cell__handle_obj = NULL;
    PyObject* cell__sequence = NULL;
    Py_ssize_t cell__handle_len = 0;
    if (PyObject_HasAttrString(py_cell_, "_handle")) {
        cell__handle_obj = PyObject_GetAttrString(py_cell_, "_handle");
        if (cell__handle_obj == NULL) {
            return NULL;
        }
        cell__sequence = PySequence_Fast(cell__handle_obj, "Failed to access handle sequence");
        if (cell__sequence == NULL) {
            Py_DECREF(cell__handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_cell_)) {
        cell__sequence = PySequence_Fast(py_cell_, "Argument cell_ must be a handle sequence");
        if (cell__sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument cell_ must be a Fortran derived-type instance");
        return NULL;
    }
    cell__handle_len = PySequence_Fast_GET_SIZE(cell__sequence);
    if (cell__handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument cell_ has an invalid handle length");
        Py_DECREF(cell__sequence);
        if (cell__handle_obj) Py_DECREF(cell__handle_obj);
        return NULL;
    }
    int* cell_ = (int*)malloc(sizeof(int) * cell__handle_len);
    if (cell_ == NULL) {
        PyErr_NoMemory();
        Py_DECREF(cell__sequence);
        if (cell__handle_obj) Py_DECREF(cell__handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < cell__handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(cell__sequence, i);
        if (item == NULL) {
            free(cell_);
            Py_DECREF(cell__sequence);
            if (cell__handle_obj) Py_DECREF(cell__handle_obj);
            return NULL;
        }
        cell_[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(cell_);
            Py_DECREF(cell__sequence);
            if (cell__handle_obj) Py_DECREF(cell__handle_obj);
            return NULL;
        }
    }
    (void)cell__handle_len;  /* suppress unused warnings when unchanged */
    
    int* num_species = &num_species_val;
    if (PyArray_Check(py_num_species)) {
        num_species_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_num_species, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (num_species_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(num_species_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument num_species must have exactly one element");
            Py_DECREF(num_species_scalar_arr);
            return NULL;
        }
        num_species_scalar_is_array = 1;
        num_species = (int*)PyArray_DATA(num_species_scalar_arr);
        num_species_val = num_species[0];
        if (PyArray_DATA(num_species_scalar_arr) != PyArray_DATA((PyArrayObject*)py_num_species) || \
            PyArray_TYPE(num_species_scalar_arr) != PyArray_TYPE((PyArrayObject*)py_num_species)) {
            num_species_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_num_species)) {
        num_species_val = (int)PyLong_AsLong(py_num_species);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument num_species must be a scalar number or NumPy array");
        return NULL;
    }
    int species_symbol_len = 0;
    char* species_symbol = NULL;
    int species_symbol_is_array = 0;
    if (py_species_symbol == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument species_symbol cannot be None");
        return NULL;
    } else {
        PyObject* species_symbol_bytes = NULL;
        if (PyArray_Check(py_species_symbol)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* species_symbol_arr = (PyArrayObject*)py_species_symbol;
            if (PyArray_TYPE(species_symbol_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument species_symbol must be a string array");
                return NULL;
            }
            species_symbol_len = (int)PyArray_ITEMSIZE(species_symbol_arr);
            species_symbol = (char*)PyArray_DATA(species_symbol_arr);
            species_symbol_is_array = 1;
        } else if (PyBytes_Check(py_species_symbol)) {
            species_symbol_bytes = py_species_symbol;
            Py_INCREF(species_symbol_bytes);
        } else if (PyUnicode_Check(py_species_symbol)) {
            species_symbol_bytes = PyUnicode_AsUTF8String(py_species_symbol);
            if (species_symbol_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument species_symbol must be str, bytes, or numpy array");
            return NULL;
        }
        if (species_symbol_bytes != NULL) {
            species_symbol_len = (int)PyBytes_GET_SIZE(species_symbol_bytes);
            species_symbol = (char*)malloc((size_t)species_symbol_len + 1);
            if (species_symbol == NULL) {
                Py_DECREF(species_symbol_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(species_symbol, PyBytes_AS_STRING(species_symbol_bytes), (size_t)species_symbol_len);
            species_symbol[species_symbol_len] = '\0';
            Py_DECREF(species_symbol_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_cell__cell_dosomething)(cell_, num_species, species_symbol, species_symbol_len);
    if (PyErr_Occurred()) {
        if (cell__sequence) Py_DECREF(cell__sequence);
        if (cell__handle_obj) Py_DECREF(cell__handle_obj);
        free(cell_);
        if (!species_symbol_is_array) free(species_symbol);
        return NULL;
    }
    
    if (num_species_scalar_is_array) {
        if (num_species_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_num_species, num_species_scalar_arr) < 0) {
                Py_DECREF(num_species_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(num_species_scalar_arr);
    }
    if (cell__sequence) {
        Py_DECREF(cell__sequence);
    }
    if (cell__handle_obj) {
        Py_DECREF(cell__handle_obj);
    }
    free(cell_);
    if (!species_symbol_is_array) free(species_symbol);
    Py_RETURN_NONE;
}

static PyObject* wrap_cell_unit_cell_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell_initialise)(this);
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

static PyObject* wrap_cell_unit_cell_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_cell__unit_cell_helper_get_num_species(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell__get__num_species)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_cell__unit_cell_helper_set_num_species(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "num_species", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell__set__num_species)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_cell__unit_cell_helper_get_species_symbol(PyObject* self, PyObject* args, PyObject* kwargs)
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
    int value_len = 8;
    if (value_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character helper length must be positive");
        return NULL;
    }
    char* buffer = (char*)malloc((size_t)value_len + 1);
    if (buffer == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(buffer, ' ', value_len);
    buffer[value_len] = '\0';
    F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell__get__species_symbol)(this_handle, buffer, value_len);
    int actual_len = value_len;
    while (actual_len > 0 && buffer[actual_len - 1] == ' ') {
        --actual_len;
    }
    PyObject* result = PyBytes_FromStringAndSize(buffer, actual_len);
    free(buffer);
    return result;
}

static PyObject* wrap_cell__unit_cell_helper_set_species_symbol(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    PyObject* py_value;
    static char *kwlist[] = {"handle", "species_symbol", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_handle, &py_value)) {
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
    if (py_value == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument species_symbol must be str or bytes");
        return NULL;
    }
    PyObject* value_bytes = NULL;
    if (PyBytes_Check(py_value)) {
        value_bytes = py_value;
        Py_INCREF(value_bytes);
    } else if (PyUnicode_Check(py_value)) {
        value_bytes = PyUnicode_AsUTF8String(py_value);
        if (value_bytes == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument species_symbol must be str or bytes");
        return NULL;
    }
    int value_len = (int)PyBytes_GET_SIZE(value_bytes);
    char* value = (char*)malloc((size_t)value_len + 1);
    if (value == NULL) {
        Py_DECREF(value_bytes);
        PyErr_NoMemory();
        return NULL;
    }
    memcpy(value, PyBytes_AS_STRING(value_bytes), (size_t)value_len);
    value[value_len] = '\0';
    F90WRAP_F_SYMBOL(f90wrap_cell__unit_cell__set__species_symbol)(this_handle, value, value_len);
    free(value);
    Py_DECREF(value_bytes);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _test module */
static PyMethodDef _test_methods[] = {
    {"f90wrap_cell__cell_dosomething", (PyCFunction)wrap_cell_cell_dosomething, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        cell_dosomething"},
    {"f90wrap_cell__unit_cell_initialise", (PyCFunction)wrap_cell_unit_cell_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for unit_cell"},
    {"f90wrap_cell__unit_cell_finalise", (PyCFunction)wrap_cell_unit_cell_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for unit_cell"},
    {"f90wrap_cell__unit_cell__get__num_species", (PyCFunction)wrap_cell__unit_cell_helper_get_num_species, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for num_species"},
    {"f90wrap_cell__unit_cell__set__num_species", (PyCFunction)wrap_cell__unit_cell_helper_set_num_species, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for num_species"},
    {"f90wrap_cell__unit_cell__get__species_symbol", (PyCFunction)wrap_cell__unit_cell_helper_get_species_symbol, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for species_symbol"},
    {"f90wrap_cell__unit_cell__set__species_symbol", (PyCFunction)wrap_cell__unit_cell_helper_set_species_symbol, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for species_symbol"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _testmodule = {
    PyModuleDef_HEAD_INIT,
    "test",
    "Direct-C wrapper for _test module",
    -1,
    _test_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__test(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_testmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
