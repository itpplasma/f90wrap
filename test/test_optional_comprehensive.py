"""
Comprehensive test for optional argument handling combining arrays and scalars.
"""

import unittest
import tempfile
from pathlib import Path

from f90wrap import parser
from f90wrap.cwrapgen import CWrapperGenerator


class TestOptionalComprehensive(unittest.TestCase):
    """Comprehensive tests for optional argument handling."""

    def test_mixed_optional_mandatory_arrays_scalars(self):
        """Test complex mix of mandatory and optional arrays and scalars."""
        code = """
        subroutine complex_optional(mandatory_scalar, optional_scalar, &
                                   mandatory_array, optional_array, n, result)
            implicit none
            integer, intent(in) :: n
            real, intent(in) :: mandatory_scalar
            real, intent(in), optional :: optional_scalar
            real, intent(in) :: mandatory_array(n)
            real, intent(inout), optional :: optional_array(n)
            real, intent(out) :: result

            result = mandatory_scalar + sum(mandatory_array)

            if (present(optional_scalar)) then
                result = result + optional_scalar
            endif

            if (present(optional_array)) then
                optional_array = optional_array * 2.0
                result = result + sum(optional_array)
            endif
        end subroutine complex_optional
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

        # Verify mandatory arguments come before optional in PyArg_ParseTuple
        import re
        match = re.search(r'PyArg_ParseTuple\(args,\s*"([^"]*)"', c_code)
        self.assertIsNotNone(match, "Should find PyArg_ParseTuple")

        format_str = match.group(1)
        # Check format: should have mandatory args before |, then optional args
        self.assertIn('|', format_str, "Should have | separator")
        parts = format_str.split('|')
        # Mandatory: mandatory_scalar (f), mandatory_array (O), n (i)
        # Optional: optional_scalar (f), optional_array (O)
        self.assertEqual(len(parts), 2, "Should have exactly 2 parts (mandatory|optional)")

        # Verify presence flags for optional arguments
        self.assertIn('optional_scalar_present', c_code)
        self.assertIn('optional_array_present', c_code)

        # Verify conditional passing to Fortran
        self.assertIn('optional_scalar_present ? &optional_scalar : NULL', c_code)
        self.assertIn('optional_array_present ? optional_array_data : NULL', c_code)

    def test_all_optional_arguments(self):
        """Test procedure with all arguments optional."""
        code = """
        subroutine all_optional(a, b, c)
            implicit none
            integer, intent(in), optional :: a
            real, intent(in), optional :: b
            character(len=*), intent(in), optional :: c

            if (present(a)) print *, 'a =', a
            if (present(b)) print *, 'b =', b
            if (present(c)) print *, 'c =', c
        end subroutine all_optional
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

        # When all arguments are optional, format should start with |
        import re
        match = re.search(r'PyArg_ParseTuple\(args,\s*"([^"]*)"', c_code)
        if match:
            format_str = match.group(1)
            self.assertTrue(format_str.startswith('|'),
                          f"Format string '{format_str}' should start with | when all args are optional")

        # Verify all presence flags
        self.assertIn('a_present', c_code)
        self.assertIn('b_present', c_code)
        self.assertIn('c_present', c_code)

    def test_optional_character_argument(self):
        """Test optional character/string argument."""
        code = """
        subroutine test_optional_string(required, optional_str)
            implicit none
            integer, intent(in) :: required
            character(len=*), intent(in), optional :: optional_str

            if (present(optional_str)) then
                print *, 'String:', optional_str
            endif
        end subroutine test_optional_string
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

        # Verify optional string argument handling
        self.assertIn('optional_str_present', c_code)
        # For character strings, the pointer is passed differently
        self.assertIn('optional_str_present ? &optional_str : NULL', c_code)


if __name__ == '__main__':
    unittest.main()