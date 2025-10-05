"""
Unit tests for optional argument handling in cwrapgen.
"""

import unittest
import tempfile
from pathlib import Path

from f90wrap import parser
from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


class TestOptionalArguments(unittest.TestCase):
    """Test optional argument handling in C wrapper generation."""

    def test_optional_scalar_argument(self):
        """Test wrapper generation for procedure with optional scalar argument."""
        # Parse simple Fortran code with optional argument
        code = """
        subroutine test_optional(required, optional_arg, result)
            implicit none
            integer, intent(in) :: required
            integer, intent(in), optional :: optional_arg
            integer, intent(out) :: result

            if (present(optional_arg)) then
                result = required + optional_arg
            else
                result = required * 2
            endif
        end subroutine test_optional
        """

        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Parse the code
            tree = parser.read_files([temp_file])
        finally:
            Path(temp_file).unlink()

        # Generate C wrapper
        gen = CWrapperGenerator(tree, 'test_module')
        c_code = gen.generate()

        # Verify key aspects of the generated code
        self.assertIn('PyArg_ParseTuple', c_code)
        self.assertIn('|', c_code)  # Optional separator in format string
        self.assertIn('optional_arg_present', c_code)
        self.assertIn('optional_arg_present ? &optional_arg : NULL', c_code)

    def test_optional_array_argument(self):
        """Test wrapper generation for procedure with optional array argument."""
        code = """
        subroutine test_optional_array(required, optional_arr, n)
            implicit none
            integer, intent(in) :: n
            real, intent(in) :: required(n)
            real, intent(in), optional :: optional_arr(n)

            if (present(optional_arr)) then
                print *, 'Array is present'
            endif
        end subroutine test_optional_array
        """

        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Parse the code
            tree = parser.read_files([temp_file])
        finally:
            Path(temp_file).unlink()

        # Generate C wrapper
        gen = CWrapperGenerator(tree, 'test_module')
        c_code = gen.generate()

        # Verify optional array handling
        self.assertIn('optional_arr_present', c_code)
        self.assertIn('optional_arr_present ? optional_arr_data : NULL', c_code)
        self.assertIn('py_optional_arr != NULL && py_optional_arr != Py_None', c_code)

    def test_multiple_optional_arguments(self):
        """Test wrapper generation for procedure with multiple optional arguments."""
        code = """
        subroutine test_multiple_optional(a, b, c, d)
            implicit none
            integer, intent(in) :: a
            integer, intent(in), optional :: b
            real, intent(in), optional :: c
            integer, intent(out), optional :: d

            if (present(b)) print *, 'b =', b
            if (present(c)) print *, 'c =', c
            if (present(d)) d = a * 2
        end subroutine test_multiple_optional
        """

        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Parse the code
            tree = parser.read_files([temp_file])
        finally:
            Path(temp_file).unlink()

        # Generate C wrapper
        gen = CWrapperGenerator(tree, 'test_module')
        c_code = gen.generate()

        # Verify all optional arguments are handled
        self.assertIn('b_present', c_code)
        self.assertIn('c_present', c_code)
        self.assertIn('d_present', c_code)

        # Verify format string has only one '|' separator
        lines = c_code.split('\n')
        format_lines = [l for l in lines if 'PyArg_ParseTuple' in l]
        if format_lines:
            # Find the format string
            for line in lines:
                if 'PyArg_ParseTuple' in line:
                    # The next few lines should contain the format string
                    idx = lines.index(line)
                    format_section = '\n'.join(lines[idx:idx+5])
                    # Check that format string contains | separator for optional args
                    self.assertIn('|', format_section,
                                   "Format string should contain | separator for optional arguments")
                    # Count | separators in the format string (should be exactly 1)
                    # Extract just the format string between quotes
                    import re
                    match = re.search(r'"([^"]*)"', format_section)
                    if match:
                        fmt_str = match.group(1)
                        self.assertEqual(fmt_str.count('|'), 1,
                                       f"Format string '{fmt_str}' should have exactly one | separator")

    def test_optional_with_intent_out(self):
        """Test wrapper generation for optional output argument."""
        code = """
        subroutine test_optional_out(input, output)
            implicit none
            integer, intent(in) :: input
            integer, intent(out), optional :: output

            if (present(output)) then
                output = input * 3
            endif
        end subroutine test_optional_out
        """

        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Parse the code
            tree = parser.read_files([temp_file])
        finally:
            Path(temp_file).unlink()

        # Generate C wrapper
        gen = CWrapperGenerator(tree, 'test_module')
        c_code = gen.generate()

        # Optional output arguments still need presence checking
        self.assertIn('output_present', c_code)
        # Fortran call should pass NULL if not present
        self.assertIn('output_present ? &output : NULL', c_code)


if __name__ == '__main__':
    unittest.main()