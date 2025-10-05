"""
Integration tests for complex real-world scenarios.

Tests that combine multiple features to ensure they work together correctly:
- Module initialization/termination with complex types
- Memory management across features
- Error handling and recovery
- Performance-critical patterns
"""

import unittest
from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


class TestModuleLifecycle(unittest.TestCase):
    """Test complete module lifecycle with initialization and cleanup."""

    def setUp(self):
        self.generator = CWrapperGenerator()

    def test_module_with_allocatable_module_variables(self):
        """Test module with allocatable module-level variables."""
        module = ft.Module(name='global_state')

        # Module variables
        module.elements = [
            ft.Element(name='global_buffer', type='real(8)',
                      attributes=['dimension(:,:)', 'allocatable']),
            ft.Element(name='initialized', type='logical'),
            ft.Element(name='config', type='type(config_t)')
        ]

        # Configuration type
        config_type = ft.Type(name='config_t')
        config_type.elements = [
            ft.Element(name='max_size', type='integer'),
            ft.Element(name='tolerance', type='real(8)'),
            ft.Element(name='log_file', type='character(len=256)')
        ]
        module.types = [config_type]

        # Initialization routine
        init_sub = ft.Subroutine(name='initialize_module')
        init_sub.arguments = [
            ft.Argument(name='buffer_size', type='integer',
                       attributes=['dimension(2)'], intent='in'),
            ft.Argument(name='config_file', type='character(*)', intent='in')
        ]
        module.subroutines.append(init_sub)

        # Cleanup routine
        cleanup_sub = ft.Subroutine(name='cleanup_module')
        cleanup_sub.arguments = []
        module.subroutines.append(cleanup_sub)

        c_code = self.generator.generate_module(module)
        # Fortran support would be generated via self.generator.generate_module_support(module)

        # Check module variable access
        self.assertIn('global_state_get_global_buffer', c_code)
        self.assertIn('global_state_allocate_global_buffer', c_code)
        self.assertIn('global_state_get_initialized', c_code)
        self.assertIn('global_state_get_config', c_code)

        # Check init/cleanup
        self.assertIn('initialize_module_', c_code)
        self.assertIn('cleanup_module_', c_code)


class TestMemoryManagement(unittest.TestCase):
    """Test memory management across different features."""

    def setUp(self):
        self.generator = CWrapperGenerator()

    def test_transfer_allocatable_between_types(self):
        """Test transferring allocatable arrays between derived types."""
        # Source type
        source_type = ft.Type(name='source_t')
        source_type.elements = [
            ft.Element(name='data', type='real(8)',
                      attributes=['dimension(:)', 'allocatable'])
        ]

        # Target type
        target_type = ft.Type(name='target_t')
        target_type.elements = [
            ft.Element(name='data', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='source_ref', type='type(source_t)', attributes=['pointer'])
        ]

        # Transfer subroutine using move_alloc
        transfer_sub = ft.Subroutine(name='transfer_data')
        transfer_sub.arguments = [
            ft.Argument(name='source', type='type(source_t)', intent='inout'),
            ft.Argument(name='target', type='type(target_t)', intent='inout')
        ]

        module = ft.Module(name='memory_transfer')
        module.types = [source_type, target_type]
        module.subroutines = [transfer_sub]

        c_code = self.generator.generate_module(module)

        # Check memory management functions
        self.assertIn('source_t_allocate_data', c_code)
        self.assertIn('source_t_deallocate_data', c_code)
        self.assertIn('target_t_allocate_data', c_code)
        self.assertIn('target_t_set_source_ref', c_code)

    def test_temporary_workspace_management(self):
        """Test efficient temporary workspace allocation."""
        # Workspace type
        workspace_type = ft.Type(name='workspace_t')
        workspace_type.elements = [
            ft.Element(name='temp1', type='real(8)',
                      attributes=['dimension(:,:)', 'allocatable']),
            ft.Element(name='temp2', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='scratch', type='integer',
                      attributes=['dimension(:)', 'allocatable'])
        ]

        # Computation routine using workspace
        compute_sub = ft.Subroutine(name='compute_with_workspace')
        compute_sub.arguments = [
            ft.Argument(name='input', type='real(8)',
                       attributes=['dimension(:,:)'], intent='in'),
            ft.Argument(name='output', type='real(8)',
                       attributes=['dimension(:,:)'], intent='out'),
            ft.Argument(name='workspace', type='type(workspace_t)', intent='inout')
        ]

        module = ft.Module(name='workspace_mgmt')
        module.types = [workspace_type]
        module.subroutines = [compute_sub]

        c_code = self.generator.generate_module(module)

        # Workspace management functions
        self.assertIn('workspace_t_allocate_temp1', c_code)
        self.assertIn('workspace_t_allocate_temp2', c_code)
        self.assertIn('workspace_t_allocate_scratch', c_code)


class TestErrorHandlingPatterns(unittest.TestCase):
    """Test error handling and recovery patterns."""

    def setUp(self):
        self.generator = CWrapperGenerator()

    def test_status_code_pattern(self):
        """Test standard status code error handling pattern."""
        # Status type
        status_type = ft.Type(name='status_t')
        status_type.elements = [
            ft.Element(name='code', type='integer'),
            ft.Element(name='message', type='character(len=256)'),
            ft.Element(name='details', type='character(len=1024)')
        ]

        # Operation with status return
        operation = ft.Subroutine(name='risky_operation')
        operation.arguments = [
            ft.Argument(name='input', type='real(8)',
                       attributes=['dimension(:)'], intent='in'),
            ft.Argument(name='output', type='real(8)',
                       attributes=['dimension(:)'], intent='out'),
            ft.Argument(name='status', type='type(status_t)', intent='out')
        ]

        module = ft.Module(name='error_handling')
        module.types = [status_type]
        module.subroutines = [operation]

        c_code = self.generator.generate_module(module)

        # Check status handling
        self.assertIn('status_t*', c_code)
        self.assertIn('status_t_set_code', c_code)
        self.assertIn('status_t_set_message', c_code)

    def test_optional_error_callback(self):
        """Test optional error callback pattern."""
        # Error handler interface
        error_handler = ft.Interface(name='error_handler')
        handler_proc = ft.Subroutine(name='handle_error')
        handler_proc.arguments = [
            ft.Argument(name='code', type='integer', intent='in'),
            ft.Argument(name='message', type='character(*)', intent='in')
        ]
        error_handler.procedures.append(handler_proc)

        # Function with optional error handler
        func = ft.Function(name='safe_compute', type='real(8)')
        func.arguments = [
            ft.Argument(name='x', type='real(8)', intent='in'),
            ft.Argument(name='error_handler', type='procedure(error_handler)',
                       optional=True)
        ]

        module = ft.Module(name='safe_ops')
        module.interfaces = [error_handler]
        module.functions = [func]

        c_code = self.generator.generate_module(module)

        # Check optional callback handling
        self.assertIn('typedef void (*error_handler_t)', c_code)
        self.assertIn('error_handler_t error_handler', c_code)
        self.assertIn('int* error_handler_present', c_code)


class TestPerformancePatterns(unittest.TestCase):
    """Test performance-critical code patterns."""

    def setUp(self):
        self.generator = CWrapperGenerator()

    def test_structure_of_arrays_pattern(self):
        """Test structure-of-arrays performance pattern."""
        # SoA container
        soa_type = ft.Type(name='particle_soa_t')
        soa_type.elements = [
            ft.Element(name='x', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='y', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='z', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='vx', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='vy', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='vz', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='mass', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='n_particles', type='integer')
        ]

        # Update routine optimized for vectorization
        update_sub = ft.Subroutine(name='update_particles')
        update_sub.arguments = [
            ft.Argument(name='particles', type='type(particle_soa_t)', intent='inout'),
            ft.Argument(name='dt', type='real(8)', intent='in'),
            ft.Argument(name='force_x', type='real(8)',
                       attributes=['dimension(:)'], intent='in'),
            ft.Argument(name='force_y', type='real(8)',
                       attributes=['dimension(:)'], intent='in'),
            ft.Argument(name='force_z', type='real(8)',
                       attributes=['dimension(:)'], intent='in')
        ]

        module = ft.Module(name='particle_sim')
        module.types = [soa_type]
        module.subroutines = [update_sub]

        c_code = self.generator.generate_module(module)

        # Check SoA access patterns
        self.assertIn('particle_soa_t_get_x', c_code)
        self.assertIn('particle_soa_t_get_vx', c_code)
        self.assertIn('particle_soa_t_get_n_particles', c_code)

    def test_contiguous_array_access(self):
        """Test contiguous array access optimization."""
        # Routine requiring contiguous arrays
        proc = ft.Subroutine(name='fast_matmul')
        proc.arguments = [
            ft.Argument(name='a', type='real(8)',
                       attributes=['dimension(:,:)', 'contiguous'], intent='in'),
            ft.Argument(name='b', type='real(8)',
                       attributes=['dimension(:,:)', 'contiguous'], intent='in'),
            ft.Argument(name='c', type='real(8)',
                       attributes=['dimension(:,:)', 'contiguous'], intent='out')
        ]

        module = ft.Module(name='fast_linalg')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)

        # Contiguous arrays can be passed directly
        self.assertIn('double* a', c_code)
        self.assertIn('double* b', c_code)
        self.assertIn('double* c', c_code)
        # Should include shape information
        self.assertIn('a_shape', c_code)
        self.assertIn('b_shape', c_code)
        self.assertIn('c_shape', c_code)


class TestComplexIntegration(unittest.TestCase):
    """Test complete integration scenarios."""

    def setUp(self):
        self.generator = CWrapperGenerator()

    def test_complete_numerical_library(self):
        """Test complete numerical library with multiple interacting components."""
        # Vector type
        vector_type = ft.Type(name='vector_t')
        vector_type.elements = [
            ft.Element(name='data', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='size', type='integer')
        ]

        # Matrix type
        matrix_type = ft.Type(name='matrix_t')
        matrix_type.elements = [
            ft.Element(name='data', type='real(8)',
                      attributes=['dimension(:,:)', 'allocatable']),
            ft.Element(name='rows', type='integer'),
            ft.Element(name='cols', type='integer')
        ]

        # Solver type with callbacks
        solver_type = ft.Type(name='solver_t')
        solver_type.elements = [
            ft.Element(name='tolerance', type='real(8)'),
            ft.Element(name='max_iter', type='integer'),
            ft.Element(name='workspace', type='type(matrix_t)')
        ]

        # Callback for progress monitoring
        progress_callback = ft.Interface(name='progress_callback')
        progress_proc = ft.Subroutine(name='on_progress')
        progress_proc.arguments = [
            ft.Argument(name='iteration', type='integer', intent='in'),
            ft.Argument(name='residual', type='real(8)', intent='in')
        ]
        progress_callback.procedures.append(progress_proc)

        # Main solve routine
        solve_func = ft.Function(name='solve_system', type='integer')
        solve_func.arguments = [
            ft.Argument(name='solver', type='type(solver_t)', intent='inout'),
            ft.Argument(name='matrix', type='type(matrix_t)', intent='in'),
            ft.Argument(name='rhs', type='type(vector_t)', intent='in'),
            ft.Argument(name='solution', type='type(vector_t)', intent='out'),
            ft.Argument(name='callback', type='procedure(progress_callback)',
                       optional=True)
        ]

        # Utility routines
        init_sub = ft.Subroutine(name='init_solver')
        init_sub.arguments = [
            ft.Argument(name='solver', type='type(solver_t)', intent='out'),
            ft.Argument(name='tol', type='real(8)', intent='in', value='1.0d-6'),
            ft.Argument(name='max_iter', type='integer', intent='in', value='1000')
        ]

        cleanup_sub = ft.Subroutine(name='cleanup_solver')
        cleanup_sub.arguments = [
            ft.Argument(name='solver', type='type(solver_t)', intent='inout')
        ]

        module = ft.Module(name='numerical_solver')
        module.types = [vector_type, matrix_type, solver_type]
        module.interfaces = [progress_callback]
        module.functions = [solve_func]
        module.subroutines = [init_sub, cleanup_sub]

        c_code = self.generator.generate_module(module)
        # Fortran support would be generated via self.generator.generate_module_support(module)

        # Verify complete integration
        # Types
        self.assertIn('vector_t', c_code)
        self.assertIn('matrix_t', c_code)
        self.assertIn('solver_t', c_code)

        # Memory management
        self.assertIn('vector_t_allocate_data', c_code)
        self.assertIn('matrix_t_allocate_data', c_code)

        # Callbacks
        self.assertIn('typedef void (*progress_callback_t)', c_code)

        # Main functionality
        self.assertIn('solve_system_', c_code)
        self.assertIn('init_solver_', c_code)
        self.assertIn('cleanup_solver_', c_code)

        # Optional handling
        self.assertIn('callback_present', c_code)

    def test_event_driven_simulation(self):
        """Test event-driven simulation with multiple callbacks and states."""
        # Event type
        event_type = ft.Type(name='event_t')
        event_type.elements = [
            ft.Element(name='time', type='real(8)'),
            ft.Element(name='type', type='integer'),
            ft.Element(name='data', type='real(8)', attributes=['dimension(10)'])
        ]

        # State type
        state_type = ft.Type(name='state_t')
        state_type.elements = [
            ft.Element(name='current_time', type='real(8)'),
            ft.Element(name='variables', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='event_count', type='integer')
        ]

        # Event handler interface
        event_handler = ft.Interface(name='event_handler')
        handler_proc = ft.Subroutine(name='handle')
        handler_proc.arguments = [
            ft.Argument(name='event', type='type(event_t)', intent='in'),
            ft.Argument(name='state', type='type(state_t)', intent='inout')
        ]
        event_handler.procedures.append(handler_proc)

        # Simulation engine
        sim_type = ft.Type(name='simulation_t')
        sim_type.elements = [
            ft.Element(name='state', type='type(state_t)'),
            ft.Element(name='events', type='type(event_t)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='n_events', type='integer'),
            ft.Element(name='handler', type='procedure(event_handler)',
                      attributes=['pointer'])
        ]

        # Run simulation
        run_sub = ft.Subroutine(name='run_simulation')
        run_sub.arguments = [
            ft.Argument(name='sim', type='type(simulation_t)', intent='inout'),
            ft.Argument(name='end_time', type='real(8)', intent='in'),
            ft.Argument(name='output', type='real(8)',
                       attributes=['dimension(:,:)'], intent='out')
        ]

        module = ft.Module(name='event_sim')
        module.types = [event_type, state_type, sim_type]
        module.interfaces = [event_handler]
        module.subroutines = [run_sub]

        c_code = self.generator.generate_module(module)

        # Check event-driven patterns
        self.assertIn('event_t', c_code)
        self.assertIn('state_t', c_code)
        self.assertIn('simulation_t', c_code)
        self.assertIn('event_handler_t', c_code)
        self.assertIn('simulation_t_set_handler', c_code)
        self.assertIn('run_simulation_', c_code)


if __name__ == '__main__':
    unittest.main()