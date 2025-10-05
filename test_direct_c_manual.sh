#!/bin/bash
# Manual test script for direct-C examples
# Tests representative examples end-to-end: generate -> compile -> import -> run

set -e  # Exit on error

REPO_ROOT="/home/ert/code/f90wrap"
cd "$REPO_ROOT"

# Select diverse examples covering different features
EXAMPLES=(
    "arrays"                          # Basic array handling
    "strings"                         # String handling
    "derivedtypes"                    # Derived types
    "subroutine_args"                 # Subroutine argument passing
    "kind_map_default"               # Kind mapping
    "arrayderivedtypes"              # Arrays of derived types
    "arrays_fixed"                    # Fixed-size arrays
    "recursive_type"                  # Recursive type definitions
    "auto_raise_error"               # Error handling
    "callback_print_function_issue93" # Callbacks (if supported)
)

PASSED=0
FAILED=0
TOTAL=0

echo "Testing representative examples with direct-C mode"
echo "=================================================================="

for EXAMPLE in "${EXAMPLES[@]}"; do
    TOTAL=$((TOTAL + 1))
    EXAMPLE_DIR="$REPO_ROOT/examples/$EXAMPLE"

    echo ""
    echo "[$TOTAL/${#EXAMPLES[@]}] Testing: $EXAMPLE"
    echo "------------------------------------------------------------------"

    if [ ! -d "$EXAMPLE_DIR" ]; then
        echo "✗ SKIP: Directory not found"
        continue
    fi

    cd "$EXAMPLE_DIR"

    # Clean
    make clean 2>/dev/null || true
    rm -f *.fpp *.o *.mod *.so *_support.f90 _*.c 2>/dev/null || true

    # Find Fortran source files
    F90_FILES=$(ls *.f90 2>/dev/null || echo "")
    if [ -z "$F90_FILES" ]; then
        echo "✗ SKIP: No .f90 files found"
        continue
    fi

    # Preprocess
    echo "  Preprocessing Fortran files..."
    for F90_FILE in $F90_FILES; do
        gfortran -E -x f95-cpp-input -fPIC "$F90_FILE" -o "${F90_FILE%.f90}.fpp"
    done

    FPP_FILES=$(ls *.fpp 2>/dev/null)

    # Determine module name from directory
    MODULE_NAME="${EXAMPLE}_directc"

    # Run f90wrap with --direct-c
    echo "  Running f90wrap --direct-c..."
    KIND_MAP=""
    if [ -f "kind_map" ]; then
        KIND_MAP="-k kind_map"
    fi

    if ! f90wrap --direct-c -m "$MODULE_NAME" $FPP_FILES $KIND_MAP 2>&1 | tee f90wrap.log; then
        echo "✗ FAIL: f90wrap generation failed"
        FAILED=$((FAILED + 1))
        cat f90wrap.log | tail -20
        continue
    fi

    # Compile Fortran sources (handle dependencies: parameters must come first)
    echo "  Compiling Fortran sources..."
    # First compile parameters.f90 if it exists
    if [ -f "parameters.f90" ]; then
        if ! gfortran -c -fPIC "parameters.f90" -o "parameters.o" 2>&1; then
            echo "✗ FAIL: Fortran compilation (parameters) failed"
            FAILED=$((FAILED + 1))
            continue
        fi
    fi
    # Build list of non-parameters files
    OTHER_F90_FILES=""
    for F90_FILE in $F90_FILES; do
        if [ "$F90_FILE" != "parameters.f90" ]; then
            OTHER_F90_FILES="$OTHER_F90_FILES $F90_FILE"
        fi
    done
    # Compile all remaining files one-by-one, letting earlier compilations succeed
    # and later ones fail if dependencies aren't met, then retry failed files
    # This is a simple multi-pass approach
    if [ -n "$OTHER_F90_FILES" ]; then
        REMAINING_FILES="$OTHER_F90_FILES"
        MAX_PASSES=10
        for pass in $(seq 1 $MAX_PASSES); do
            STILL_FAILING=""
            for F90_FILE in $REMAINING_FILES; do
                if [ ! -f "${F90_FILE%.f90}.o" ]; then
                    if gfortran -c -fPIC "$F90_FILE" 2>/dev/null; then
                        : # Success, continue
                    else
                        STILL_FAILING="$STILL_FAILING $F90_FILE"
                    fi
                fi
            done
            REMAINING_FILES="$STILL_FAILING"
            if [ -z "$REMAINING_FILES" ]; then
                break  # All files compiled
            fi
        done
        if [ -n "$REMAINING_FILES" ]; then
            echo "✗ FAIL: Fortran compilation failed for:$REMAINING_FILES"
            # Show error for first failing file
            for F90_FILE in $REMAINING_FILES; do
                gfortran -c -fPIC "$F90_FILE" 2>&1 | head -10
                break
            done
            FAILED=$((FAILED + 1))
            continue
        fi
    fi

    # Compile support module
    SUPPORT_F90="${MODULE_NAME}_support.f90"
    if [ -f "$SUPPORT_F90" ]; then
        echo "  Compiling support module..."
        if ! gfortran -c -fPIC "$SUPPORT_F90" -o "${SUPPORT_F90%.f90}.o" 2>&1; then
            echo "✗ FAIL: Support module compilation failed"
            FAILED=$((FAILED + 1))
            continue
        fi
    fi

    # Compile C wrapper
    C_WRAPPER="_${MODULE_NAME}module.c"
    if [ ! -f "$C_WRAPPER" ]; then
        echo "✗ FAIL: C wrapper not generated"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "  Compiling C wrapper..."
    PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
    NUMPY_INCLUDE=$(python3 -c "import numpy; print(numpy.get_include())")
    F90WRAP_INCLUDE=$(python3 -c "import f90wrap; import os; print(os.path.dirname(f90wrap.__file__))")

    if ! gcc -c -fPIC -Wno-error=implicit-function-declaration -I"$PYTHON_INCLUDE" -I"$NUMPY_INCLUDE" -I"$F90WRAP_INCLUDE" "$C_WRAPPER" -o "${C_WRAPPER%.c}.o" 2>&1; then
        echo "✗ FAIL: C wrapper compilation failed"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Link extension
    echo "  Linking extension module..."
    ALL_OBJECTS=$(ls *.o 2>/dev/null)

    if ! gcc -shared -fPIC $ALL_OBJECTS -o "_${MODULE_NAME}.so" -lgfortran -lm 2>&1; then
        echo "✗ FAIL: Linking failed"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Test import
    echo "  Testing Python import..."
    if ! python3 -c "import sys; sys.path.insert(0, '.'); import ${MODULE_NAME}; print('Import successful')" 2>&1; then
        echo "✗ FAIL: Import failed"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "✓ PASS: $EXAMPLE built and imported successfully with direct-C"
    PASSED=$((PASSED + 1))
done

echo ""
echo "=================================================================="
echo "RESULTS SUMMARY:"
echo "  Total:   $TOTAL"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Success: $(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")%"
echo "=================================================================="

exit $((FAILED > 0 ? 1 : 0))
