"""
Test that demonstrates code reduction achieved by shared capsule utilities.
"""

import unittest
from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


class TestCodeReduction(unittest.TestCase):
    """Test code reduction metrics with shared utilities."""

    def test_code_reduction_metrics(self):
        """Verify that shared utilities reduce generated code size."""
        # Create a module with multiple derived types
        root = ft.Root()
        module = ft.Module('metrics_module')
        module.routines = []
        module.types = []

        # Create 5 derived types to show meaningful reduction
        num_types = 5
        for i in range(1, num_types + 1):
            dtype = ft.Type(f'dtype_{i}')
            dtype.mod_name = 'metrics_module'
            dtype.elements = [
                ft.Element(f'value_{i}', type='real(8)', attributes=[])
            ]
            dtype.procedures = []
            module.types.append(dtype)

            # Add function returning the type
            func = ft.Function(f'create_dtype_{i}', filename='test.f90')
            func.arguments = []
            func.ret_val = ft.Argument(name='result', type=f'type(dtype_{i})')
            func.mod_name = 'metrics_module'
            module.routines.append(func)

            # Add subroutine taking the type
            sub = ft.Subroutine(f'use_dtype_{i}', filename='test.f90')
            arg = ft.Argument(name='obj', type=f'type(dtype_{i})')
            arg.intent = 'in'
            arg.attributes = ['intent(in)']
            sub.arguments = [arg]
            sub.mod_name = 'metrics_module'
            module.routines.append(sub)

        root.modules = [module]

        # Generate code
        gen = CWrapperGenerator(root, 'metrics_module')
        code = gen.generate()

        # Verify key shared utilities are used
        self.assertIn('#include "capsule_helpers.h"', code)

        # Count usage of shared functions
        destructor_macros = code.count('F90WRAP_DEFINE_SIMPLE_DESTRUCTOR')
        create_capsules = code.count('f90wrap_create_capsule')
        unwrap_capsules = code.count('f90wrap_unwrap_capsule')
        clear_capsules = code.count('f90wrap_clear_capsule')

        # Verify expected counts
        self.assertEqual(destructor_macros, num_types,
                         f"Should have {num_types} destructor macros")
        self.assertGreaterEqual(create_capsules, num_types,
                                "Should have at least one create per type")
        self.assertGreaterEqual(unwrap_capsules, num_types,
                                "Should have at least one unwrap per type")

        # Verify no direct PyCapsule API calls (except in macros/comments)
        code_lines = [l.strip() for l in code.split('\n')
                      if not l.strip().startswith('/*') and
                      not l.strip().startswith('*') and
                      not l.strip().startswith('//')]

        for line in code_lines:
            # These should not appear in generated code (only in header)
            if 'PyCapsule_New(' in line:
                self.fail(f"Found direct PyCapsule_New: {line}")
            if 'PyCapsule_CheckExact(' in line and 'f90wrap_' not in line:
                self.fail(f"Found direct PyCapsule_CheckExact: {line}")

        # Calculate approximate code reduction
        # Each destructor: ~15 lines → 1 macro
        # Each create: ~5 lines error handling → 1 call
        # Each unwrap: ~20 lines → 3 lines
        estimated_savings = (
            num_types * 14 +  # Destructor savings
            create_capsules * 4 +  # Create error handling savings
            unwrap_capsules * 17  # Unwrap logic savings
        )

        total_lines = len(code.split('\n'))
        reduction_pct = (estimated_savings / (total_lines + estimated_savings)) * 100

        # Report metrics
        print(f"\nCode Generation Metrics:")
        print(f"========================")
        print(f"Types generated: {num_types}")
        print(f"Total lines: {total_lines}")
        print(f"Destructor macros: {destructor_macros}")
        print(f"Create calls: {create_capsules}")
        print(f"Unwrap calls: {unwrap_capsules}")
        print(f"Clear calls: {clear_capsules}")
        print(f"Estimated lines saved: {estimated_savings}")
        print(f"Reduction percentage: {reduction_pct:.1f}%")

        # Verify significant reduction
        self.assertGreater(reduction_pct, 15,
                           "Should achieve at least 15% code reduction")


if __name__ == '__main__':
    unittest.main()