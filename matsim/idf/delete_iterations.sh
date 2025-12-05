#!/bin/bash  

# Script to delete iteration folders from it.0 to it.K  
# where K = 10 * int(M/10) - 1 and M is the maximum iteration number  

# Function to display usage  
usage() {  
    echo "Usage: $0 <simulation_output_directory>"  
    echo "Example: $0 /path/to/simulation_output"  
    exit 1  
}  

# Check if path argument is provided  
if [ $# -eq 0 ]; then  
    echo "Error: No directory path provided."  
    usage  
fi  

SIMULATION_OUTPUT_DIR="$1"  

# Check if the directory exists  
if [ ! -d "$SIMULATION_OUTPUT_DIR" ]; then  
    echo "Error: Directory '$SIMULATION_OUTPUT_DIR' not found."  
    exit 1  
fi  

echo "Scanning directory: $SIMULATION_OUTPUT_DIR"  
echo "================================================"  

# Array to store folders to delete  
folders_to_delete=()  

# Statistics  
total_scenarios=0  
scenarios_processed=0  

# Iterate over all entries in the simulation_output directory  
for entry in "$SIMULATION_OUTPUT_DIR"/*; do  
    # Check if it is a directory and has a numeric name  
    if [ -d "$entry" ]; then  
        dir_name=$(basename "$entry")  
        
        # Check if directory name is a number  
        if [[ "$dir_name" =~ ^[0-9]+$ ]]; then  
            ((total_scenarios++))  
            
            # Check if ITERS subdirectory exists  
            iters_dir="$entry/ITERS"  
            if [ ! -d "$iters_dir" ]; then  
                echo "Warning: ITERS directory not found in scenario $dir_name, skipping..."  
                continue  
            fi  
            
            # Find maximum iteration number M  
            max_iter=-1  
            
            for it_folder in "$iters_dir"/it.*; do  
                if [ -d "$it_folder" ]; then  
                    # Extract iteration number from folder name (it.123 -> 123)  
                    it_name=$(basename "$it_folder")  
                    if [[ "$it_name" =~ ^it\.([0-9]+)$ ]]; then  
                        iter_num="${BASH_REMATCH[1]}"  
                        if [ "$iter_num" -gt "$max_iter" ]; then  
                            max_iter=$iter_num  
                        fi  
                    fi  
                fi  
            done  
            
            # If no iteration folders found, skip  
            if [ "$max_iter" -lt 0 ]; then  
                echo "Warning: No iteration folders found in scenario $dir_name, skipping..."  
                continue  
            fi  
            
            # Calculate K = 10 * int(M/10) - 1  
            K=$((10 * (max_iter / 10) - 1))  
            
            echo ""  
            echo "Scenario $dir_name:"  
            echo "  Maximum iteration (M): $max_iter"  
            echo "  Delete up to (K): $K"  
            
            # If K < 0, nothing to delete  
            if [ "$K" -lt 0 ]; then  
                echo "  No folders to delete (K < 0)"  
                continue  
            fi  
            
            ((scenarios_processed++))  
            
            # Find all folders from it.0 to it.K  
            for i in $(seq 0 $K); do  
                it_folder="$iters_dir/it.$i"  
                if [ -d "$it_folder" ]; then  
                    folders_to_delete+=("$it_folder")  
                fi  
            done  
        fi  
    fi  
done  

echo ""  
echo "================================================"  
echo "Scan complete."  
echo "Total scenarios found: $total_scenarios"  
echo "Scenarios with deletable iterations: $scenarios_processed"  
echo ""  

# Check if there are any folders to delete  
if [ ${#folders_to_delete[@]} -eq 0 ]; then  
    echo "No iteration folders found to delete."  
    exit 0  
fi  

# Display summary  
echo "Total folders to delete: ${#folders_to_delete[@]}"  
echo ""  

# Ask if user wants to see the full list  
read -p "Do you want to see the full list of folders to be deleted? (y/n) " -n 1 -r  
echo  
if [[ $REPLY =~ ^[Yy]$ ]]; then  
    echo ""  
    echo "The following folders will be deleted:"  
    echo "----------------------------------------"  
    printf '%s\n' "${folders_to_delete[@]}"  
    echo "----------------------------------------"  
fi  

echo ""  
# Calculate approximate size to be deleted  
echo "Calculating total size to be deleted..."  
total_size=$(du -shc "${folders_to_delete[@]}" 2>/dev/null | tail -1 | cut -f1)  
echo "Total size to be deleted: $total_size"  
echo ""  

# Ask for confirmation  
read -p "Are you sure you want to delete these ${#folders_to_delete[@]} folders? (y/n) " -n 1 -r  
echo  

# If the user confirms, perform deletion  
if [[ $REPLY =~ ^[Yy]$ ]]; then  
    echo ""  
    echo "Deleting folders..."  
    
    deleted_count=0  
    failed_count=0  
    
    # Loop through and delete each folder  
    for folder in "${folders_to_delete[@]}"; do  
        if rm -rf "$folder" 2>/dev/null; then  
            ((deleted_count++))  
            # Show progress every 10 deletions  
            if [ $((deleted_count % 10)) -eq 0 ]; then  
                echo "Progress: $deleted_count/${#folders_to_delete[@]} folders deleted..."  
            fi  
        else  
            ((failed_count++))  
            echo "Error: Failed to delete $folder"  
        fi  
    done  
    
    echo ""  
    echo "================================================"  
    echo "Deletion complete!"  
    echo "Successfully deleted: $deleted_count folders"  
    if [ $failed_count -gt 0 ]; then  
        echo "Failed to delete: $failed_count folders"  
    fi  
else  
    echo ""  
    echo "Operation cancelled."  
fi