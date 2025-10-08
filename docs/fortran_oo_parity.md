# Fortran OO Parity Remediation Plan

## Diagnostics Summary
- `fortran_oo`: direct-C wrappers still lack helper-visible polymorphic stubs, so generated code cannot `use` the helper interfaces before invoking `circle_obj_name` / `perimeter_8`.
- `issue258_derived_type_attributes`: CLASS allocatables are transferred into TYPE locals inside the direct-C wrappers, triggering ambiguity once helper routines re-read the handles.

## Design Decisions
- Generate a direct-C-only Fortran shim alongside `_module.c` that `use`s `f90wrap_m_geometry` and forwards polymorphic calls. The shim will be emitted automatically by the direct-C generator and does not modify upstream helper sources.
- Attach explicit interface blocks only when helper routines accept polymorphic dummies; this keeps the direct-C files self-contained and avoids editing the legacy Fortran wrappers.
- Preserve CLASS handles by creating temporary wrapper instances rather than transferring into plain TYPE values. This logic will live entirely inside the direct-C generator.

## Next Steps
- [ ] Implement the helper shim emission (Fortran) and matching C call sites in the direct-C generator.
- [ ] Update the CLASS handle transfer logic for direct-C so `issue258_derived_type_attributes` stops downcasting allocatables.
- [ ] Re-run the direct-C sweep and capture the updated pass rate once the shim and handle fixes land.
