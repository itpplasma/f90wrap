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
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__return_example_first)(int* first, int* ret_instance);
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__return_example_second)(int* first, int* second, int* ret_instance);
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__return_example_third)(int* first, int* second, int* third, int* \
    ret_instance);
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__example_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__example_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__example__get__first)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__example__set__first)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__example__get__second)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__example__set__second)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__example__get__third)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_class_example__example__set__third)(int* handle, int* value);

static PyObject* wrap_class_example_return_example_first(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_first = NULL;
    int first_val = 0;
    PyArrayObject* first_scalar_arr = NULL;
    int first_scalar_copyback = 0;
    int first_scalar_is_array = 0;
    static char *kwlist[] = {"first", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_first)) {
        return NULL;
    }
    
    int* first = &first_val;
    if (PyArray_Check(py_first)) {
        first_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_first, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (first_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(first_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument first must have exactly one element");
            Py_DECREF(first_scalar_arr);
            return NULL;
        }
        first_scalar_is_array = 1;
        first = (int*)PyArray_DATA(first_scalar_arr);
        first_val = first[0];
        if (PyArray_DATA(first_scalar_arr) != PyArray_DATA((PyArrayObject*)py_first) || PyArray_TYPE(first_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_first)) {
            first_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_first)) {
        first_val = (int)PyLong_AsLong(py_first);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument first must be a scalar number or NumPy array");
        return NULL;
    }
    int ret_instance[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_class_example__return_example_first)(first, ret_instance);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (first_scalar_is_array) {
        if (first_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_first, first_scalar_arr) < 0) {
                Py_DECREF(first_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(first_scalar_arr);
    }
    PyObject* py_ret_instance_obj = PyList_New(4);
    if (py_ret_instance_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_instance[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_instance_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_instance_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_instance_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_instance_obj != NULL) return py_ret_instance_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_instance_obj != NULL) Py_DECREF(py_ret_instance_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_instance_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_instance_obj);
    }
    return result_tuple;
}

static PyObject* wrap_class_example_return_example_second(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_first = NULL;
    int first_val = 0;
    PyArrayObject* first_scalar_arr = NULL;
    int first_scalar_copyback = 0;
    int first_scalar_is_array = 0;
    PyObject* py_second = NULL;
    int second_val = 0;
    PyArrayObject* second_scalar_arr = NULL;
    int second_scalar_copyback = 0;
    int second_scalar_is_array = 0;
    static char *kwlist[] = {"first", "second", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_first, &py_second)) {
        return NULL;
    }
    
    int* first = &first_val;
    if (PyArray_Check(py_first)) {
        first_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_first, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (first_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(first_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument first must have exactly one element");
            Py_DECREF(first_scalar_arr);
            return NULL;
        }
        first_scalar_is_array = 1;
        first = (int*)PyArray_DATA(first_scalar_arr);
        first_val = first[0];
        if (PyArray_DATA(first_scalar_arr) != PyArray_DATA((PyArrayObject*)py_first) || PyArray_TYPE(first_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_first)) {
            first_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_first)) {
        first_val = (int)PyLong_AsLong(py_first);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument first must be a scalar number or NumPy array");
        return NULL;
    }
    int* second = &second_val;
    if (PyArray_Check(py_second)) {
        second_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_second, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (second_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(second_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument second must have exactly one element");
            Py_DECREF(second_scalar_arr);
            return NULL;
        }
        second_scalar_is_array = 1;
        second = (int*)PyArray_DATA(second_scalar_arr);
        second_val = second[0];
        if (PyArray_DATA(second_scalar_arr) != PyArray_DATA((PyArrayObject*)py_second) || PyArray_TYPE(second_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_second)) {
            second_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_second)) {
        second_val = (int)PyLong_AsLong(py_second);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument second must be a scalar number or NumPy array");
        return NULL;
    }
    int ret_instance[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_class_example__return_example_second)(first, second, ret_instance);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (first_scalar_is_array) {
        if (first_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_first, first_scalar_arr) < 0) {
                Py_DECREF(first_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(first_scalar_arr);
    }
    if (second_scalar_is_array) {
        if (second_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_second, second_scalar_arr) < 0) {
                Py_DECREF(second_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(second_scalar_arr);
    }
    PyObject* py_ret_instance_obj = PyList_New(4);
    if (py_ret_instance_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_instance[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_instance_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_instance_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_instance_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_instance_obj != NULL) return py_ret_instance_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_instance_obj != NULL) Py_DECREF(py_ret_instance_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_instance_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_instance_obj);
    }
    return result_tuple;
}

static PyObject* wrap_class_example_return_example_third(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_first = NULL;
    int first_val = 0;
    PyArrayObject* first_scalar_arr = NULL;
    int first_scalar_copyback = 0;
    int first_scalar_is_array = 0;
    PyObject* py_second = NULL;
    int second_val = 0;
    PyArrayObject* second_scalar_arr = NULL;
    int second_scalar_copyback = 0;
    int second_scalar_is_array = 0;
    PyObject* py_third = NULL;
    int third_val = 0;
    PyArrayObject* third_scalar_arr = NULL;
    int third_scalar_copyback = 0;
    int third_scalar_is_array = 0;
    static char *kwlist[] = {"first", "second", "third", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_first, &py_second, &py_third)) {
        return NULL;
    }
    
    int* first = &first_val;
    if (PyArray_Check(py_first)) {
        first_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_first, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (first_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(first_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument first must have exactly one element");
            Py_DECREF(first_scalar_arr);
            return NULL;
        }
        first_scalar_is_array = 1;
        first = (int*)PyArray_DATA(first_scalar_arr);
        first_val = first[0];
        if (PyArray_DATA(first_scalar_arr) != PyArray_DATA((PyArrayObject*)py_first) || PyArray_TYPE(first_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_first)) {
            first_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_first)) {
        first_val = (int)PyLong_AsLong(py_first);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument first must be a scalar number or NumPy array");
        return NULL;
    }
    int* second = &second_val;
    if (PyArray_Check(py_second)) {
        second_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_second, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (second_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(second_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument second must have exactly one element");
            Py_DECREF(second_scalar_arr);
            return NULL;
        }
        second_scalar_is_array = 1;
        second = (int*)PyArray_DATA(second_scalar_arr);
        second_val = second[0];
        if (PyArray_DATA(second_scalar_arr) != PyArray_DATA((PyArrayObject*)py_second) || PyArray_TYPE(second_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_second)) {
            second_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_second)) {
        second_val = (int)PyLong_AsLong(py_second);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument second must be a scalar number or NumPy array");
        return NULL;
    }
    int* third = &third_val;
    if (PyArray_Check(py_third)) {
        third_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_third, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (third_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(third_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument third must have exactly one element");
            Py_DECREF(third_scalar_arr);
            return NULL;
        }
        third_scalar_is_array = 1;
        third = (int*)PyArray_DATA(third_scalar_arr);
        third_val = third[0];
        if (PyArray_DATA(third_scalar_arr) != PyArray_DATA((PyArrayObject*)py_third) || PyArray_TYPE(third_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_third)) {
            third_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_third)) {
        third_val = (int)PyLong_AsLong(py_third);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument third must be a scalar number or NumPy array");
        return NULL;
    }
    int ret_instance[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_class_example__return_example_third)(first, second, third, ret_instance);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (first_scalar_is_array) {
        if (first_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_first, first_scalar_arr) < 0) {
                Py_DECREF(first_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(first_scalar_arr);
    }
    if (second_scalar_is_array) {
        if (second_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_second, second_scalar_arr) < 0) {
                Py_DECREF(second_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(second_scalar_arr);
    }
    if (third_scalar_is_array) {
        if (third_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_third, third_scalar_arr) < 0) {
                Py_DECREF(third_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(third_scalar_arr);
    }
    PyObject* py_ret_instance_obj = PyList_New(4);
    if (py_ret_instance_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_instance[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_instance_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_instance_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_instance_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_instance_obj != NULL) return py_ret_instance_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_instance_obj != NULL) Py_DECREF(py_ret_instance_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_instance_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_instance_obj);
    }
    return result_tuple;
}

static PyObject* wrap_class_example_example_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_class_example__example_initialise)(this);
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

static PyObject* wrap_class_example_example_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_class_example__example_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_class_example__example_helper_get_first(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_class_example__example__get__first)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_class_example__example_helper_set_first(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "first", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_class_example__example__set__first)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_class_example__example_helper_get_second(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_class_example__example__get__second)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_class_example__example_helper_set_second(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "second", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_class_example__example__set__second)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_class_example__example_helper_get_third(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_class_example__example__get__third)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_class_example__example_helper_set_third(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "third", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_class_example__example__set__third)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _example module */
static PyMethodDef _example_methods[] = {
    {"f90wrap_class_example__return_example_first", (PyCFunction)wrap_class_example_return_example_first, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_example_first"},
    {"f90wrap_class_example__return_example_second", (PyCFunction)wrap_class_example_return_example_second, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_example_second"},
    {"f90wrap_class_example__return_example_third", (PyCFunction)wrap_class_example_return_example_third, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_example_third"},
    {"f90wrap_class_example__example_initialise", (PyCFunction)wrap_class_example_example_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for example"},
    {"f90wrap_class_example__example_finalise", (PyCFunction)wrap_class_example_example_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for example"},
    {"f90wrap_class_example__example__get__first", (PyCFunction)wrap_class_example__example_helper_get_first, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for first"},
    {"f90wrap_class_example__example__set__first", (PyCFunction)wrap_class_example__example_helper_set_first, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for first"},
    {"f90wrap_class_example__example__get__second", (PyCFunction)wrap_class_example__example_helper_get_second, METH_VARARGS \
        | METH_KEYWORDS, "Module helper for second"},
    {"f90wrap_class_example__example__set__second", (PyCFunction)wrap_class_example__example_helper_set_second, METH_VARARGS \
        | METH_KEYWORDS, "Module helper for second"},
    {"f90wrap_class_example__example__get__third", (PyCFunction)wrap_class_example__example_helper_get_third, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for third"},
    {"f90wrap_class_example__example__set__third", (PyCFunction)wrap_class_example__example_helper_set_third, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for third"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _examplemodule = {
    PyModuleDef_HEAD_INIT,
    "example",
    "Direct-C wrapper for _example module",
    -1,
    _example_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__example(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_examplemodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
