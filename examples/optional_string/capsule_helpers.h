/*
 * f90wrap: F90 to Python interface generator with derived type support
 *
 * Copyright James Kermode 2011-2018
 *
 * This file is part of f90wrap
 * For the latest version see github.com/jameskermode/f90wrap
 *
 * f90wrap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * f90wrap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with f90wrap. If not, see <http://www.gnu.org/licenses/>.
 *
 * If you would like to license the source code under different terms,
 * please contact James Kermode, james.kermode@gmail.com
 */

#ifndef F90WRAP_CAPSULE_HELPERS_H
#define F90WRAP_CAPSULE_HELPERS_H

#include <Python.h>
#include <stdlib.h>
#include <string.h>

/*
 * Shared PyCapsule helper functions for f90wrap-generated C modules.
 * These inline functions reduce code duplication across generated modules.
 */

/* Create a PyCapsule with a destructor that frees the memory */
static inline PyObject* f90wrap_create_capsule(void* ptr, const char* name,
                                                PyCapsule_Destructor destructor) {
    if (ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Cannot create capsule from NULL pointer");
        return NULL;
    }
    return PyCapsule_New(ptr, name, destructor);
}

/* Generic PyCapsule destructor that frees memory */
static inline void f90wrap_capsule_free_destructor(PyObject *capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
    if (ptr != NULL) {
        free(ptr);
    }
}

/* Unwrap a PyCapsule and return the pointer, with type checking */
static inline void* f90wrap_unwrap_capsule(PyObject* obj, const char* expected_name) {
    if (!PyCapsule_CheckExact(obj)) {
        /* Try to extract from custom type object with fortran_ptr member */
        if (Py_TYPE(obj)->tp_name != NULL &&
            strstr(Py_TYPE(obj)->tp_name, expected_name) != NULL) {
            /* Assume the type has a fortran_ptr member at the expected offset */
            typedef struct { PyObject_HEAD void* fortran_ptr; } GenericDerivedType;
            GenericDerivedType* typed_obj = (GenericDerivedType*)obj;
            return typed_obj->fortran_ptr;
        }

        PyErr_Format(PyExc_TypeError, "Expected %s instance or PyCapsule", expected_name);
        return NULL;
    }

    /* Check capsule name matches */
    const char* capsule_name = PyCapsule_GetName(obj);
    if (capsule_name == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyCapsule has no name");
        return NULL;
    }

    /* Allow both exact match and match with "_capsule" suffix */
    size_t expected_len = strlen(expected_name);
    size_t capsule_suffix_len = 8; /* "_capsule" */

    if (strcmp(capsule_name, expected_name) != 0) {
        /* Check if it matches with "_capsule" suffix */
        char expected_with_suffix[256];
        snprintf(expected_with_suffix, sizeof(expected_with_suffix),
                 "%s_capsule", expected_name);

        if (strcmp(capsule_name, expected_with_suffix) != 0) {
            PyErr_Format(PyExc_TypeError, "PyCapsule name mismatch: expected '%s' or '%s', got '%s'",
                         expected_name, expected_with_suffix, capsule_name);
            return NULL;
        }
    }

    void* ptr = PyCapsule_GetPointer(obj, capsule_name);
    if (ptr == NULL && !PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError, "PyCapsule contains NULL pointer for %s", expected_name);
    }

    return ptr;
}

/* Safely extract capsule pointer without error if not a capsule */
static inline void* f90wrap_try_unwrap_capsule(PyObject* obj, const char* expected_name) {
    if (!PyCapsule_CheckExact(obj)) {
        /* Try to extract from custom type object */
        if (Py_TYPE(obj)->tp_name != NULL &&
            strstr(Py_TYPE(obj)->tp_name, expected_name) != NULL) {
            typedef struct { PyObject_HEAD void* fortran_ptr; } GenericDerivedType;
            GenericDerivedType* typed_obj = (GenericDerivedType*)obj;
            return typed_obj->fortran_ptr;
        }
        return NULL;
    }

    const char* capsule_name = PyCapsule_GetName(obj);
    if (capsule_name == NULL) {
        return NULL;
    }

    /* Check name matches */
    char expected_with_suffix[256];
    snprintf(expected_with_suffix, sizeof(expected_with_suffix),
             "%s_capsule", expected_name);

    if (strcmp(capsule_name, expected_name) != 0 &&
        strcmp(capsule_name, expected_with_suffix) != 0) {
        return NULL;
    }

    PyErr_Clear();  /* Clear any previous errors */
    void* ptr = PyCapsule_GetPointer(obj, capsule_name);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return NULL;
    }

    return ptr;
}

/* Clear a capsule pointer to prevent double-free */
static inline int f90wrap_clear_capsule(PyObject* capsule) {
    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "Expected PyCapsule");
        return -1;
    }

    if (PyCapsule_SetPointer(capsule, NULL) != 0) {
        return -1;
    }

    return 0;
}

/* Create a type-specific destructor function using a macro */
#define F90WRAP_DEFINE_CAPSULE_DESTRUCTOR(type_name, destructor_func) \
    static void type_name##_capsule_destructor(PyObject *capsule) { \
        void* ptr = PyCapsule_GetPointer(capsule, #type_name "_capsule"); \
        if (ptr != NULL) { \
            destructor_func(&ptr); \
        } \
    }

/* Simplified macro for types that just need free() */
#define F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(type_name) \
    static void type_name##_capsule_destructor(PyObject *capsule) { \
        f90wrap_capsule_free_destructor(capsule); \
    }

#endif /* F90WRAP_CAPSULE_HELPERS_H */