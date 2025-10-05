"""Test that abstract types are handled correctly in direct C generation."""

import unittest
import tempfile
import os
from pathlib import Path
from f90wrap import parser
from f90wrap.cwrapgen import CWrapperGenerator


class TestAbstractTypeSupport(unittest.TestCase):
    """Test handling of abstract types in Fortran support module generation."""

    def test_abstract_types_no_allocators(self):
        """Test that abstract types don't get allocators/deallocators."""

        # Create test files with abstract types
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write test Fortran files
            base_f90 = os.path.join(tmpdir, 'base.f90')
            with open(base_f90, 'w') as f:
                f.write("""
module m_base
  implicit none
  private

  type, public, abstract :: AbstractBase
    integer :: id
  contains
    procedure :: is_base => base_is_base
  end type AbstractBase
contains
  function base_is_base(this) result(is_base)
    class(AbstractBase), intent(in) :: this
    logical :: is_base
    is_base = .true.
  end function base_is_base
end module m_base
""")

            derived_f90 = os.path.join(tmpdir, 'derived.f90')
            with open(derived_f90, 'w') as f:
                f.write("""
module m_derived
  use m_base, only : AbstractBase
  implicit none
  private

  type, public, abstract, extends(AbstractBase) :: AbstractDerived
    real :: value
  contains
    procedure :: get_value => derived_get_value
  end type AbstractDerived

  type, public, extends(AbstractDerived) :: ConcreteDerived
    integer :: extra
  end type ConcreteDerived

contains
  function derived_get_value(this) result(value)
    class(AbstractDerived), intent(in) :: this
    real :: value
    value = this%value
  end function derived_get_value
end module m_derived
""")

            # Parse the files
            tree = parser.read_files([base_f90, derived_f90])

            # Generate C wrapper
            config = {'kind_map': {}}
            c_generator = CWrapperGenerator(tree, 'test_module', config)

            # Generate Fortran support
            fortran_support = c_generator.generate_fortran_support()

            # Check that abstract types don't have allocators
            self.assertNotIn('f90wrap_abstractbase__allocate', fortran_support.lower())
            self.assertNotIn('f90wrap_abstractbase__deallocate', fortran_support.lower())
            self.assertNotIn('f90wrap_abstractderived__allocate', fortran_support.lower())
            self.assertNotIn('f90wrap_abstractderived__deallocate', fortran_support.lower())

            # Check that concrete type has allocators
            self.assertIn('f90wrap_concretederived__allocate', fortran_support.lower())
            self.assertIn('f90wrap_concretederived__deallocate', fortran_support.lower())

            # Check that abstract types don't have getters/setters
            self.assertNotIn('f90wrap_abstractbase__get__id', fortran_support.lower())
            self.assertNotIn('f90wrap_abstractbase__set__id', fortran_support.lower())
            self.assertNotIn('f90wrap_abstractderived__get__value', fortran_support.lower())
            self.assertNotIn('f90wrap_abstractderived__set__value', fortran_support.lower())

            # Check that concrete type has getters/setters
            self.assertIn('f90wrap_concretederived__get__extra', fortran_support.lower())
            self.assertIn('f90wrap_concretederived__set__extra', fortran_support.lower())

    def test_abstract_types_compilation(self):
        """Test that generated Fortran support module compiles without errors."""

        # Use the actual fortran_oo example
        examples_dir = Path(__file__).parent.parent / 'examples' / 'fortran_oo'
        if not examples_dir.exists():
            self.skipTest(f"Examples directory {examples_dir} not found")

        base_poly = examples_dir / 'base_poly.f90'
        main_oo = examples_dir / 'main-oo.f90'

        if not base_poly.exists() or not main_oo.exists():
            self.skipTest("fortran_oo example files not found")

        # Parse the files
        tree = parser.read_files([str(base_poly), str(main_oo)])

        # Generate C wrapper
        config = {'kind_map': {}}
        c_generator = CWrapperGenerator(tree, 'fortran_oo_test', config)

        # Generate Fortran support
        fortran_support = c_generator.generate_fortran_support()

        # Verify abstract types are handled correctly
        # Abstract types: polygone, rectangle
        # Concrete types: square, circle, ball

        # Check abstract types don't have allocators
        self.assertNotIn('f90wrap_polygone__allocate', fortran_support.lower())
        self.assertNotIn('f90wrap_polygone__deallocate', fortran_support.lower())
        self.assertNotIn('f90wrap_rectangle__allocate', fortran_support.lower())
        self.assertNotIn('f90wrap_rectangle__deallocate', fortran_support.lower())

        # Check concrete types have allocators
        self.assertIn('f90wrap_square__allocate', fortran_support.lower())
        self.assertIn('f90wrap_square__deallocate', fortran_support.lower())
        self.assertIn('f90wrap_circle__allocate', fortran_support.lower())
        self.assertIn('f90wrap_circle__deallocate', fortran_support.lower())
        self.assertIn('f90wrap_ball__allocate', fortran_support.lower())
        self.assertIn('f90wrap_ball__deallocate', fortran_support.lower())

        # Abstract types should not have getters/setters
        self.assertNotIn('f90wrap_rectangle__get__length', fortran_support.lower())
        self.assertNotIn('f90wrap_rectangle__set__length', fortran_support.lower())

        # Concrete types should have getters/setters
        self.assertIn('f90wrap_circle__get__radius', fortran_support.lower())
        self.assertIn('f90wrap_circle__set__radius', fortran_support.lower())


if __name__ == '__main__':
    unittest.main()