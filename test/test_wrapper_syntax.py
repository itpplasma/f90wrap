#!/usr/bin/env python3
"""
Test suite to verify that generated C code has correct syntax.

This includes:
- Balanced braces
- Proper function termination
- Correct function_wrapper_end calls
"""

import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


def count_braces(code):
    """Count opening and closing braces in code."""
    open_braces = code.count('{')
    close_braces = code.count('}')
    return open_braces, close_braces


def verify_function_structure(code):
    """Verify each function has proper start and end."""
    # Find all function definitions
    func_pattern = r'static PyObject\* (\w+)\(PyObject \*self, PyObject \*args(?:, PyObject \*kwargs)?\) \{'
    functions = re.findall(func_pattern, code)

    errors = []
    for func_name in functions:
        # Find the function body
        func_start = code.find(f'static PyObject* {func_name}(')
        if func_start == -1:
            continue

        # Find the opening brace for this function
        brace_start = code.find('{', func_start)
        if brace_start == -1:
            errors.append(f"Function {func_name} has no opening brace")
            continue

        # Count braces from this point to find the matching closing brace
        brace_count = 1
        pos = brace_start + 1
        func_end = -1

        while pos < len(code) and brace_count > 0:
            if code[pos] == '{':
                brace_count += 1
            elif code[pos] == '}':
                brace_count -= 1
                if brace_count == 0:
                    func_end = pos
                    break
            pos += 1

        if func_end == -1:
            errors.append(f"Function {func_name} has unbalanced braces")
            continue

        # Extract function body
        func_body = code[brace_start:func_end+1]

        # Check if function ends with proper return statement
        # Look for return statements before the closing brace
        last_statement_match = re.search(r'(return [^;]+;|Py_RETURN_NONE;)\s*\}$', func_body.strip())
        if not last_statement_match:
            errors.append(f"Function {func_name} does not end with a return statement")

    return errors


def test_simple_function_wrapper():
    """Test that a simple function wrapper has balanced braces."""
    # Create minimal AST with one function
    ast = ft.Root()
    ast.modules = []
    ast.procedures = []

    # Add a simple function
    func = ft.Function()
    func.name = 'test_func'
    func.doc = ['Test function']
    func.arguments = []
    func.ret_val = ft.Argument()
    func.ret_val.type = 'integer'
    func.ret_val.name = 'result'

    ast.procedures.append(func)

    # Generate C code
    generator = CWrapperGenerator(ast, 'test_module')
    c_code = generator.generate()

    # Check balanced braces
    open_braces, close_braces = count_braces(c_code)
    assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

    # Check function structure
    errors = verify_function_structure(c_code)
    assert len(errors) == 0, f"Function structure errors: {', '.join(errors)}"

    print("✓ Simple function wrapper has balanced braces")


def test_subroutine_wrapper():
    """Test that a subroutine wrapper has balanced braces."""
    # Create minimal AST with one subroutine
    ast = ft.Root()
    ast.modules = []
    ast.procedures = []

    # Add a simple subroutine
    sub = ft.Subroutine()
    sub.name = 'test_sub'
    sub.doc = ['Test subroutine']
    sub.arguments = []

    ast.procedures.append(sub)

    # Generate C code
    generator = CWrapperGenerator(ast, 'test_module')
    c_code = generator.generate()

    # Check balanced braces
    open_braces, close_braces = count_braces(c_code)
    assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

    # Check function structure
    errors = verify_function_structure(c_code)
    assert len(errors) == 0, f"Function structure errors: {', '.join(errors)}"

    print("✓ Subroutine wrapper has balanced braces")


def test_derived_type_wrappers():
    """Test that derived type constructor/destructor wrappers have balanced braces."""
    # Create AST with a module containing a derived type
    ast = ft.Root()
    ast.procedures = []

    # Create module
    module = ft.Module()
    module.name = 'test_mod'
    module.types = []
    module.routines = []

    # Add a derived type
    dtype = ft.Type()
    dtype.name = 'MyType'
    dtype.mod_name = 'test_mod'
    dtype.elements = []
    dtype.procedures = []

    # Add an element
    elem = ft.Element()
    elem.name = 'value'
    elem.type = 'integer'
    elem.attributes = []
    dtype.elements.append(elem)

    module.types.append(dtype)
    ast.modules = [module]

    # Generate C code
    generator = CWrapperGenerator(ast, 'test_module')
    c_code = generator.generate()

    # Check balanced braces
    open_braces, close_braces = count_braces(c_code)
    assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

    # Check function structure
    errors = verify_function_structure(c_code)
    assert len(errors) == 0, f"Function structure errors: {', '.join(errors)}"

    print("✓ Derived type wrappers have balanced braces")


def test_complex_function_with_arrays():
    """Test function with array arguments has balanced braces."""
    # Create AST with a function that has array arguments
    ast = ft.Root()
    ast.modules = []
    ast.procedures = []

    # Add a function with arrays
    func = ft.Function()
    func.name = 'array_func'
    func.doc = ['Function with arrays']
    func.arguments = []

    # Add array argument
    arg = ft.Argument()
    arg.name = 'data'
    arg.type = 'real'
    arg.attributes = ['dimension(:)', 'intent(in)']
    func.arguments.append(arg)

    func.ret_val = ft.Argument()
    func.ret_val.type = 'real'
    func.ret_val.name = 'result'

    ast.procedures.append(func)

    # Generate C code
    generator = CWrapperGenerator(ast, 'test_module')
    c_code = generator.generate()

    # Check balanced braces
    open_braces, close_braces = count_braces(c_code)
    assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

    # Check function structure
    errors = verify_function_structure(c_code)
    assert len(errors) == 0, f"Function structure errors: {', '.join(errors)}"

    print("✓ Complex function with arrays has balanced braces")


def test_optional_arguments():
    """Test function with optional arguments has balanced braces."""
    # Create AST with a function that has optional arguments
    ast = ft.Root()
    ast.modules = []
    ast.procedures = []

    # Add a function with optional args
    func = ft.Subroutine()
    func.name = 'optional_func'
    func.doc = ['Function with optional args']
    func.arguments = []

    # Add mandatory argument
    arg1 = ft.Argument()
    arg1.name = 'required'
    arg1.type = 'integer'
    arg1.attributes = ['intent(in)']
    func.arguments.append(arg1)

    # Add optional argument
    arg2 = ft.Argument()
    arg2.name = 'optional_val'
    arg2.type = 'real'
    arg2.attributes = ['intent(in)', 'optional']
    func.arguments.append(arg2)

    ast.procedures.append(func)

    # Generate C code
    generator = CWrapperGenerator(ast, 'test_module')
    c_code = generator.generate()

    # Check balanced braces
    open_braces, close_braces = count_braces(c_code)
    assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

    # Check function structure
    errors = verify_function_structure(c_code)
    assert len(errors) == 0, f"Function structure errors: {', '.join(errors)}"

    print("✓ Function with optional arguments has balanced braces")


def test_callback_function():
    """Test function with callback argument has balanced braces."""
    # Create AST with a function that has a callback
    ast = ft.Root()
    ast.modules = []
    ast.procedures = []

    # Add a function with callback
    func = ft.Subroutine()
    func.name = 'callback_func'
    func.doc = ['Function with callback']
    func.arguments = []

    # Add callback argument
    arg = ft.Argument()
    arg.name = 'callback'
    arg.type = 'callback'
    arg.attributes = ['callback']
    func.arguments.append(arg)

    ast.procedures.append(func)

    # Generate C code
    generator = CWrapperGenerator(ast, 'test_module')
    c_code = generator.generate()

    # Check balanced braces
    open_braces, close_braces = count_braces(c_code)
    assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

    # Check function structure
    errors = verify_function_structure(c_code)
    assert len(errors) == 0, f"Function structure errors: {', '.join(errors)}"

    print("✓ Function with callback has balanced braces")


if __name__ == '__main__':
    # Run all tests
    test_simple_function_wrapper()
    test_subroutine_wrapper()
    test_derived_type_wrappers()
    test_complex_function_with_arrays()
    test_optional_arguments()
    test_callback_function()

    print("\n✓✓✓ All syntax tests passed!")