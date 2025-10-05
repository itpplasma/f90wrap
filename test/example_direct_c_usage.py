#!/usr/bin/env python3
"""
Example demonstrating use of the direct-C build fixture for f90wrap testing.

This script shows how to compile and test Fortran modules using f90wrap --direct-c
with automatic compilation and linking.
"""

import tempfile
import numpy as np
from pathlib import Path
from test_direct_c_build import DirectCBuilder, build_and_test_module


def example_usage():
    """Demonstrate using the direct-C builder."""

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # Create a DirectCBuilder instance
        builder = DirectCBuilder(work_dir, verbose=True)

        # Create a simple Fortran module for testing
        fortran_file = work_dir / "math_module.f90"
        fortran_file.write_text("""
module math_module
    implicit none

contains

    function compute_pi(n) result(pi_approx)
        integer, intent(in) :: n
        real*8 :: pi_approx
        integer :: i
        real*8 :: term

        ! Use Leibniz formula for π
        pi_approx = 0.0d0
        do i = 0, n-1
            term = 4.0d0 / (2*i + 1)
            if (mod(i, 2) == 0) then
                pi_approx = pi_approx + term
            else
                pi_approx = pi_approx - term
            end if
        end do
    end function compute_pi

    subroutine matrix_multiply(a, b, c, n)
        integer, intent(in) :: n
        real*8, intent(in) :: a(n, n), b(n, n)
        real*8, intent(out) :: c(n, n)
        integer :: i, j, k

        c = 0.0d0
        do i = 1, n
            do j = 1, n
                do k = 1, n
                    c(i, j) = c(i, j) + a(i, k) * b(k, j)
                end do
            end do
        end do
    end subroutine matrix_multiply

    function vector_norm(v, n) result(norm)
        integer, intent(in) :: n
        real*8, intent(in) :: v(n)
        real*8 :: norm
        integer :: i

        norm = 0.0d0
        do i = 1, n
            norm = norm + v(i) * v(i)
        end do
        norm = sqrt(norm)
    end function vector_norm

end module math_module
""")

        # Build and test the module
        print("Building module with f90wrap --direct-c...")
        results = build_and_test_module(builder, [fortran_file], "math_module")

        if results['success']:
            print("Build successful!")
            print(f"Generated files:")
            for name, path in results['generated_files'].items():
                print(f"  {name}: {path}")
            print(f"Extension module: {results['extension_path']}")

            # Test the imported module
            module = results['module']
            print("\nTesting module functions:")
            print(f"Module attributes: {dir(module)}")

            # The module structure may be different depending on f90wrap version
            # Try to find the right attribute
            if hasattr(module, 'Math_Module'):
                math_mod = module.Math_Module
            elif hasattr(module, 'math_module'):
                math_mod = module.math_module
            else:
                # Functions might be directly available
                math_mod = module

            # Test pi computation
            if hasattr(math_mod, 'compute_pi'):
                pi_approx = math_mod.compute_pi(10000)
                print(f"π approximation (10000 terms): {pi_approx:.10f}")
                print(f"Error: {abs(pi_approx - np.pi):.10f}")

            # Test matrix multiplication
            if hasattr(math_mod, 'matrix_multiply'):
                n = 3
                A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
                B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float64)
                C = np.zeros((n, n), dtype=np.float64)
                math_mod.matrix_multiply(A, B, C)
                print(f"\nMatrix multiplication result:")
                print(C)

            # Test vector norm
            if hasattr(math_mod, 'vector_norm'):
                v = np.array([3.0, 4.0, 0.0], dtype=np.float64)
                norm = math_mod.vector_norm(v)
                print(f"\nNorm of [3, 4, 0]: {norm}")

        else:
            print(f"Build failed: {results['error']}")


if __name__ == "__main__":
    example_usage()