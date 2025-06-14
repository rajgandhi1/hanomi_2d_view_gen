#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# List of STEP files to process.
# Please ensure these paths are correct.
STEP_FILES=(
    "/Users/apple/Downloads/CHAMFER.stp"
    "/Users/apple/Downloads/C103_25.step"
    "/Users/apple/Downloads/1712.step"
)

# Main directory to store all outputs
MAIN_OUTPUT_DIR="cad_analysis_results"

# --- Script ---

# Create the main output directory if it doesn't exist
mkdir -p "$MAIN_OUTPUT_DIR"

# Loop through each STEP file and process it
for step_file in "${STEP_FILES[@]}"; do
    if [ ! -f "$step_file" ]; then
        echo "Warning: STEP file not found at '$step_file'. Skipping."
        continue
    fi

    echo "-----------------------------------------------------"
    echo "Processing: $step_file"
    echo "-----------------------------------------------------"

    # Get the base name of the file without extension (e.g., "CHAMFER")
    base_name=$(basename "$step_file" | sed 's/\(.*\)\..*/\1/')

    # Define the output directory for the current file
    output_dir="$MAIN_OUTPUT_DIR/$base_name"
    mkdir -p "$output_dir"
    echo "Output will be saved in: $output_dir"

    # Define the path for the generated critical faces JSON file
    critical_faces_json="$output_dir/critical_faces.json"

    # Generate the critical faces JSON file
    echo "Generating critical faces file..."
    python generate_critical_faces.py "$step_file" "$critical_faces_json"

    # Define the path for the profiling report
    profiling_report="$output_dir/profiling_report.txt"

    # Run the main analysis script with profiling
    echo "Running 3D to 2D view optimization with profiling..."
    # The command below runs the analysis with the --profile flag for summarized output.
    # A full report including script output, profiling data, and total time is saved to the report file.
    (time python main.py "$step_file" "$critical_faces_json" "$output_dir" --profile) 2>&1 | tee "$profiling_report"

    echo "Successfully processed $base_name."
    echo "Profiling report saved to: $profiling_report"
    echo ""
done

echo "-----------------------------------------------------"
echo "All files processed."
echo "Check the '$MAIN_OUTPUT_DIR' directory for results."
echo "-----------------------------------------------------" 