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
extern void F90WRAP_F_SYMBOL(f90wrap_m_error__str_input)(char* keyword, int keyword_len);
extern void F90WRAP_F_SYMBOL(f90wrap_m_error__auto_raise)(int* ierr, char* errmsg, int errmsg_len);
extern void F90WRAP_F_SYMBOL(f90wrap_m_error__auto_raise_optional)(int* ierr, char* errmsg, int errmsg_len);
extern void F90WRAP_F_SYMBOL(f90wrap_m_error__auto_no_raise)(int* ierr, char* errmsg, int errmsg_len);
extern void F90WRAP_F_SYMBOL(f90wrap_m_error__auto_no_raise_optional)(int* ierr, char* errmsg, int errmsg_len);
extern void F90WRAP_F_SYMBOL(f90wrap_m_error__no_error_var)(int* a_num, char* a_string, int a_string_len);

static PyObject* wrap_m_error_str_input(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_keyword = Py_None;
    static char *kwlist[] = {"keyword", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &py_keyword)) {
        return NULL;
    }
    
    int keyword_len = 0;
    char* keyword = NULL;
    int keyword_is_array = 0;
    if (py_keyword == Py_None) {
        keyword_len = 1024;
        if (keyword_len <= 0) {
            PyErr_SetString(PyExc_ValueError, "Character length for keyword must be positive");
            return NULL;
        }
        keyword = (char*)malloc((size_t)keyword_len + 1);
        if (keyword == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        memset(keyword, ' ', keyword_len);
        keyword[keyword_len] = '\0';
    } else {
        PyObject* keyword_bytes = NULL;
        if (PyArray_Check(py_keyword)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* keyword_arr = (PyArrayObject*)py_keyword;
            if (PyArray_TYPE(keyword_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument keyword must be a string array");
                return NULL;
            }
            keyword_len = (int)PyArray_ITEMSIZE(keyword_arr);
            keyword = (char*)PyArray_DATA(keyword_arr);
            keyword_is_array = 1;
        } else if (PyBytes_Check(py_keyword)) {
            keyword_bytes = py_keyword;
            Py_INCREF(keyword_bytes);
        } else if (PyUnicode_Check(py_keyword)) {
            keyword_bytes = PyUnicode_AsUTF8String(py_keyword);
            if (keyword_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument keyword must be str, bytes, or numpy array");
            return NULL;
        }
        if (keyword_bytes != NULL) {
            keyword_len = (int)PyBytes_GET_SIZE(keyword_bytes);
            keyword = (char*)malloc((size_t)keyword_len + 1);
            if (keyword == NULL) {
                Py_DECREF(keyword_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(keyword, PyBytes_AS_STRING(keyword_bytes), (size_t)keyword_len);
            keyword[keyword_len] = '\0';
            Py_DECREF(keyword_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_error__str_input)(keyword, keyword_len);
    if (PyErr_Occurred()) {
        if (!keyword_is_array) free(keyword);
        return NULL;
    }
    
    if (!keyword_is_array) free(keyword);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_error_auto_raise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int ierr_val = 0;
    int errmsg_len = 1024;
    if (errmsg_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character length for errmsg must be positive");
        return NULL;
    }
    char* errmsg = (char*)malloc((size_t)errmsg_len + 1);
    if (errmsg == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(errmsg, ' ', errmsg_len);
    errmsg[errmsg_len] = '\0';
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_error__auto_raise)(&ierr_val, errmsg, errmsg_len);
    if (PyErr_Occurred()) {
        free(errmsg);
        return NULL;
    }
    
    if (PyErr_Occurred()) {
        free(errmsg);
        return NULL;
    }
    if (ierr_val != 0) {
        f90wrap_abort_(errmsg, errmsg_len);
        free(errmsg);
        return NULL;
    }
    PyObject* py_ierr_obj = Py_BuildValue("i", ierr_val);
    if (py_ierr_obj == NULL) {
        return NULL;
    }
    int errmsg_trim = errmsg_len;
    while (errmsg_trim > 0 && errmsg[errmsg_trim - 1] == ' ') {
        --errmsg_trim;
    }
    PyObject* py_errmsg_obj = PyBytes_FromStringAndSize(errmsg, errmsg_trim);
    free(errmsg);
    if (py_errmsg_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ierr_obj != NULL) result_count++;
    if (py_errmsg_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ierr_obj != NULL) return py_ierr_obj;
        if (py_errmsg_obj != NULL) return py_errmsg_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ierr_obj != NULL) Py_DECREF(py_ierr_obj);
        if (py_errmsg_obj != NULL) Py_DECREF(py_errmsg_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ierr_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ierr_obj);
    }
    if (py_errmsg_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_errmsg_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_error_auto_raise_optional(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_ierr = Py_None;
    int ierr_val = 0;
    PyArrayObject* ierr_scalar_arr = NULL;
    int ierr_scalar_copyback = 0;
    int ierr_scalar_is_array = 0;
    PyObject* py_errmsg = Py_None;
    static char *kwlist[] = {"ierr", "errmsg", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist, &py_ierr, &py_errmsg)) {
        return NULL;
    }
    
    int* ierr = &ierr_val;
    if (py_ierr == Py_None) {
        ierr_val = 0;
    } else {
        if (PyArray_Check(py_ierr)) {
            ierr_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
                py_ierr, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
            if (ierr_scalar_arr == NULL) {
                return NULL;
            }
            if (PyArray_SIZE(ierr_scalar_arr) != 1) {
                PyErr_SetString(PyExc_ValueError, "Argument ierr must have exactly one element");
                Py_DECREF(ierr_scalar_arr);
                return NULL;
            }
            ierr_scalar_is_array = 1;
            ierr = (int*)PyArray_DATA(ierr_scalar_arr);
            ierr_val = ierr[0];
            if (PyArray_DATA(ierr_scalar_arr) != PyArray_DATA((PyArrayObject*)py_ierr) || PyArray_TYPE(ierr_scalar_arr) != \
                PyArray_TYPE((PyArrayObject*)py_ierr)) {
                ierr_scalar_copyback = 1;
            }
        } else if (PyNumber_Check(py_ierr)) {
            ierr_val = (int)PyLong_AsLong(py_ierr);
            if (PyErr_Occurred()) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument ierr must be a scalar number or NumPy array");
            return NULL;
        }
    }
    int errmsg_len = 0;
    char* errmsg = NULL;
    int errmsg_is_array = 0;
    if (py_errmsg == Py_None) {
        errmsg_len = 1024;
        if (errmsg_len <= 0) {
            PyErr_SetString(PyExc_ValueError, "Character length for errmsg must be positive");
            return NULL;
        }
        errmsg = (char*)malloc((size_t)errmsg_len + 1);
        if (errmsg == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        memset(errmsg, ' ', errmsg_len);
        errmsg[errmsg_len] = '\0';
    } else {
        PyObject* errmsg_bytes = NULL;
        if (PyArray_Check(py_errmsg)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* errmsg_arr = (PyArrayObject*)py_errmsg;
            if (PyArray_TYPE(errmsg_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument errmsg must be a string array");
                return NULL;
            }
            errmsg_len = (int)PyArray_ITEMSIZE(errmsg_arr);
            errmsg = (char*)PyArray_DATA(errmsg_arr);
            errmsg_is_array = 1;
        } else if (PyBytes_Check(py_errmsg)) {
            errmsg_bytes = py_errmsg;
            Py_INCREF(errmsg_bytes);
        } else if (PyUnicode_Check(py_errmsg)) {
            errmsg_bytes = PyUnicode_AsUTF8String(py_errmsg);
            if (errmsg_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument errmsg must be str, bytes, or numpy array");
            return NULL;
        }
        if (errmsg_bytes != NULL) {
            errmsg_len = (int)PyBytes_GET_SIZE(errmsg_bytes);
            errmsg = (char*)malloc((size_t)errmsg_len + 1);
            if (errmsg == NULL) {
                Py_DECREF(errmsg_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(errmsg, PyBytes_AS_STRING(errmsg_bytes), (size_t)errmsg_len);
            errmsg[errmsg_len] = '\0';
            Py_DECREF(errmsg_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_error__auto_raise_optional)(ierr, errmsg, errmsg_len);
    if (PyErr_Occurred()) {
        if (!errmsg_is_array) free(errmsg);
        return NULL;
    }
    
    if (PyErr_Occurred()) {
        if (!errmsg_is_array) free(errmsg);
        return NULL;
    }
    if (ierr_val != 0) {
        f90wrap_abort_(errmsg, errmsg_len);
        if (!errmsg_is_array) free(errmsg);
        return NULL;
    }
    if (ierr_scalar_is_array) {
        if (ierr_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_ierr, ierr_scalar_arr) < 0) {
                Py_DECREF(ierr_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(ierr_scalar_arr);
    }
    PyObject* py_ierr_obj = Py_BuildValue("i", ierr_val);
    if (py_ierr_obj == NULL) {
        return NULL;
    }
    PyObject* py_errmsg_obj = NULL;
    if (errmsg_is_array) {
        /* Numpy array was modified in place, no return object or free needed */
    } else {
        int errmsg_trim = errmsg_len;
        while (errmsg_trim > 0 && errmsg[errmsg_trim - 1] == ' ') {
            --errmsg_trim;
        }
        py_errmsg_obj = PyBytes_FromStringAndSize(errmsg, errmsg_trim);
        free(errmsg);
        if (py_errmsg_obj == NULL) {
            return NULL;
        }
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ierr_obj != NULL) result_count++;
    if (py_errmsg_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ierr_obj != NULL) return py_ierr_obj;
        if (py_errmsg_obj != NULL) return py_errmsg_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ierr_obj != NULL) Py_DECREF(py_ierr_obj);
        if (py_errmsg_obj != NULL) Py_DECREF(py_errmsg_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ierr_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ierr_obj);
    }
    if (py_errmsg_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_errmsg_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_error_auto_no_raise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int ierr_val = 0;
    int errmsg_len = 1024;
    if (errmsg_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character length for errmsg must be positive");
        return NULL;
    }
    char* errmsg = (char*)malloc((size_t)errmsg_len + 1);
    if (errmsg == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(errmsg, ' ', errmsg_len);
    errmsg[errmsg_len] = '\0';
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_error__auto_no_raise)(&ierr_val, errmsg, errmsg_len);
    if (PyErr_Occurred()) {
        free(errmsg);
        return NULL;
    }
    
    if (PyErr_Occurred()) {
        free(errmsg);
        return NULL;
    }
    if (ierr_val != 0) {
        f90wrap_abort_(errmsg, errmsg_len);
        free(errmsg);
        return NULL;
    }
    PyObject* py_ierr_obj = Py_BuildValue("i", ierr_val);
    if (py_ierr_obj == NULL) {
        return NULL;
    }
    int errmsg_trim = errmsg_len;
    while (errmsg_trim > 0 && errmsg[errmsg_trim - 1] == ' ') {
        --errmsg_trim;
    }
    PyObject* py_errmsg_obj = PyBytes_FromStringAndSize(errmsg, errmsg_trim);
    free(errmsg);
    if (py_errmsg_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ierr_obj != NULL) result_count++;
    if (py_errmsg_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ierr_obj != NULL) return py_ierr_obj;
        if (py_errmsg_obj != NULL) return py_errmsg_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ierr_obj != NULL) Py_DECREF(py_ierr_obj);
        if (py_errmsg_obj != NULL) Py_DECREF(py_errmsg_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ierr_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ierr_obj);
    }
    if (py_errmsg_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_errmsg_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_error_auto_no_raise_optional(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_ierr = Py_None;
    int ierr_val = 0;
    PyArrayObject* ierr_scalar_arr = NULL;
    int ierr_scalar_copyback = 0;
    int ierr_scalar_is_array = 0;
    PyObject* py_errmsg = Py_None;
    static char *kwlist[] = {"ierr", "errmsg", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist, &py_ierr, &py_errmsg)) {
        return NULL;
    }
    
    int* ierr = &ierr_val;
    if (py_ierr == Py_None) {
        ierr_val = 0;
    } else {
        if (PyArray_Check(py_ierr)) {
            ierr_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
                py_ierr, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
            if (ierr_scalar_arr == NULL) {
                return NULL;
            }
            if (PyArray_SIZE(ierr_scalar_arr) != 1) {
                PyErr_SetString(PyExc_ValueError, "Argument ierr must have exactly one element");
                Py_DECREF(ierr_scalar_arr);
                return NULL;
            }
            ierr_scalar_is_array = 1;
            ierr = (int*)PyArray_DATA(ierr_scalar_arr);
            ierr_val = ierr[0];
            if (PyArray_DATA(ierr_scalar_arr) != PyArray_DATA((PyArrayObject*)py_ierr) || PyArray_TYPE(ierr_scalar_arr) != \
                PyArray_TYPE((PyArrayObject*)py_ierr)) {
                ierr_scalar_copyback = 1;
            }
        } else if (PyNumber_Check(py_ierr)) {
            ierr_val = (int)PyLong_AsLong(py_ierr);
            if (PyErr_Occurred()) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument ierr must be a scalar number or NumPy array");
            return NULL;
        }
    }
    int errmsg_len = 0;
    char* errmsg = NULL;
    int errmsg_is_array = 0;
    if (py_errmsg == Py_None) {
        errmsg_len = 1024;
        if (errmsg_len <= 0) {
            PyErr_SetString(PyExc_ValueError, "Character length for errmsg must be positive");
            return NULL;
        }
        errmsg = (char*)malloc((size_t)errmsg_len + 1);
        if (errmsg == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        memset(errmsg, ' ', errmsg_len);
        errmsg[errmsg_len] = '\0';
    } else {
        PyObject* errmsg_bytes = NULL;
        if (PyArray_Check(py_errmsg)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* errmsg_arr = (PyArrayObject*)py_errmsg;
            if (PyArray_TYPE(errmsg_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument errmsg must be a string array");
                return NULL;
            }
            errmsg_len = (int)PyArray_ITEMSIZE(errmsg_arr);
            errmsg = (char*)PyArray_DATA(errmsg_arr);
            errmsg_is_array = 1;
        } else if (PyBytes_Check(py_errmsg)) {
            errmsg_bytes = py_errmsg;
            Py_INCREF(errmsg_bytes);
        } else if (PyUnicode_Check(py_errmsg)) {
            errmsg_bytes = PyUnicode_AsUTF8String(py_errmsg);
            if (errmsg_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument errmsg must be str, bytes, or numpy array");
            return NULL;
        }
        if (errmsg_bytes != NULL) {
            errmsg_len = (int)PyBytes_GET_SIZE(errmsg_bytes);
            errmsg = (char*)malloc((size_t)errmsg_len + 1);
            if (errmsg == NULL) {
                Py_DECREF(errmsg_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(errmsg, PyBytes_AS_STRING(errmsg_bytes), (size_t)errmsg_len);
            errmsg[errmsg_len] = '\0';
            Py_DECREF(errmsg_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_error__auto_no_raise_optional)(ierr, errmsg, errmsg_len);
    if (PyErr_Occurred()) {
        if (!errmsg_is_array) free(errmsg);
        return NULL;
    }
    
    if (PyErr_Occurred()) {
        if (!errmsg_is_array) free(errmsg);
        return NULL;
    }
    if (ierr_val != 0) {
        f90wrap_abort_(errmsg, errmsg_len);
        if (!errmsg_is_array) free(errmsg);
        return NULL;
    }
    if (ierr_scalar_is_array) {
        if (ierr_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_ierr, ierr_scalar_arr) < 0) {
                Py_DECREF(ierr_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(ierr_scalar_arr);
    }
    PyObject* py_ierr_obj = Py_BuildValue("i", ierr_val);
    if (py_ierr_obj == NULL) {
        return NULL;
    }
    PyObject* py_errmsg_obj = NULL;
    if (errmsg_is_array) {
        /* Numpy array was modified in place, no return object or free needed */
    } else {
        int errmsg_trim = errmsg_len;
        while (errmsg_trim > 0 && errmsg[errmsg_trim - 1] == ' ') {
            --errmsg_trim;
        }
        py_errmsg_obj = PyBytes_FromStringAndSize(errmsg, errmsg_trim);
        free(errmsg);
        if (py_errmsg_obj == NULL) {
            return NULL;
        }
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ierr_obj != NULL) result_count++;
    if (py_errmsg_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ierr_obj != NULL) return py_ierr_obj;
        if (py_errmsg_obj != NULL) return py_errmsg_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ierr_obj != NULL) Py_DECREF(py_ierr_obj);
        if (py_errmsg_obj != NULL) Py_DECREF(py_errmsg_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ierr_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ierr_obj);
    }
    if (py_errmsg_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_errmsg_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_error_no_error_var(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int a_num_val = 0;
    int a_string_len = 1024;
    if (a_string_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "Character length for a_string must be positive");
        return NULL;
    }
    char* a_string = (char*)malloc((size_t)a_string_len + 1);
    if (a_string == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(a_string, ' ', a_string_len);
    a_string[a_string_len] = '\0';
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_error__no_error_var)(&a_num_val, a_string, a_string_len);
    if (PyErr_Occurred()) {
        free(a_string);
        return NULL;
    }
    
    PyObject* py_a_num_obj = Py_BuildValue("i", a_num_val);
    if (py_a_num_obj == NULL) {
        return NULL;
    }
    int a_string_trim = a_string_len;
    while (a_string_trim > 0 && a_string[a_string_trim - 1] == ' ') {
        --a_string_trim;
    }
    PyObject* py_a_string_obj = PyBytes_FromStringAndSize(a_string, a_string_trim);
    free(a_string);
    if (py_a_string_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_a_num_obj != NULL) result_count++;
    if (py_a_string_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_a_num_obj != NULL) return py_a_num_obj;
        if (py_a_string_obj != NULL) return py_a_string_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_a_num_obj != NULL) Py_DECREF(py_a_num_obj);
        if (py_a_string_obj != NULL) Py_DECREF(py_a_string_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_a_num_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_a_num_obj);
    }
    if (py_a_string_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_a_string_obj);
    }
    return result_tuple;
}

/* Method table for _pywrapper module */
static PyMethodDef _pywrapper_methods[] = {
    {"f90wrap_m_error__str_input", (PyCFunction)wrap_m_error_str_input, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        str_input"},
    {"f90wrap_m_error__auto_raise", (PyCFunction)wrap_m_error_auto_raise, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        auto_raise"},
    {"f90wrap_m_error__auto_raise_optional", (PyCFunction)wrap_m_error_auto_raise_optional, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for auto_raise_optional"},
    {"f90wrap_m_error__auto_no_raise", (PyCFunction)wrap_m_error_auto_no_raise, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        auto_no_raise"},
    {"f90wrap_m_error__auto_no_raise_optional", (PyCFunction)wrap_m_error_auto_no_raise_optional, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for auto_no_raise_optional"},
    {"f90wrap_m_error__no_error_var", (PyCFunction)wrap_m_error_no_error_var, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        no_error_var"},
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
