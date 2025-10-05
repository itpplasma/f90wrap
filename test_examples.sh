#!/bin/bash

# Test all examples with --direct-c flag

EXAMPLES_DIR="examples"
RESULTS_FILE="example_test_results.txt"

echo "Testing f90wrap --direct-c on all examples" > "$RESULTS_FILE"
echo "=============================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Get list of examples (directories)
for example_dir in "$EXAMPLES_DIR"/*/; do
    example=$(basename "$example_dir")
    
    # Skip if not a directory
    [ ! -d "$example_dir" ] && continue
    
    echo "Testing: $example"
    echo "-----------------------------------" >> "$RESULTS_FILE"
    echo "Example: $example" >> "$RESULTS_FILE"
    
    cd "$example_dir"
    
    # Find .f90 files
    fortran_files=$(find . -maxdepth 1 -name "*.f90" -o -name "*.F90" | tr '\n' ' ')
    
    if [ -z "$fortran_files" ]; then
        echo "  SKIP: No Fortran files found" | tee -a "../../$RESULTS_FILE"
        cd ../..
        continue
    fi
    
    # Check for kind_map
    kind_map_arg=""
    if [ -f "kind_map" ]; then
        kind_map_arg="-k kind_map"
    fi
    
    # Run f90wrap with --direct-c
    if f90wrap --direct-c -m "test_$example" $kind_map_arg $fortran_files > /tmp/f90wrap_output.txt 2>&1; then
        echo "  SUCCESS: Generated C code" | tee -a "../../$RESULTS_FILE"
        
        # Check if .c file was created
        c_file=$(ls -1 test_${example}module.c 2>/dev/null | head -1)
        if [ -n "$c_file" ]; then
            lines=$(wc -l < "$c_file")
            echo "    Generated: $c_file ($lines lines)" | tee -a "../../$RESULTS_FILE"
        fi
    else
        echo "  FAILED" | tee -a "../../$RESULTS_FILE"
        echo "    Error:" >> "../../$RESULTS_FILE"
        tail -20 /tmp/f90wrap_output.txt >> "../../$RESULTS_FILE"
    fi
    
    # Clean up
    rm -f test_${example}module.c test_${example}.py f90wrap_*.f90 .f2py_f2cmap
    
    cd ../..
    echo "" >> "$RESULTS_FILE"
done

echo ""
echo "Results written to: $RESULTS_FILE"
