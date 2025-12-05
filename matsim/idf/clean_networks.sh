#!/bin/bash  

# Set target directories  
SIMULATION_OUTPUT_DIR="simulation_output"  
NETWORKS_DIR="networks_idf/scenario"  

# Check if the required directories exist  
if [ ! -d "$SIMULATION_OUTPUT_DIR" ]; then  
    echo "Error: Directory '$SIMULATION_OUTPUT_DIR' not found."  
    exit 1  
fi  

if [ ! -d "$NETWORKS_DIR" ]; then  
    echo "Error: Directory '$NETWORKS_DIR' not found."  
    exit 1  
fi  

# Create an array to store the list of files to be deleted  
files_to_delete=()  

echo "Finding matching files..."  

# Iterate over all entries in the simulation_output directory  
for entry in "$SIMULATION_OUTPUT_DIR"/*; do  
    # Check if it is a directory  
    if [ -d "$entry" ]; then  
        # Extract the directory name (the number)  
        num=$(basename "$entry")  

        # Construct the corresponding network file name  
        network_file="$NETWORKS_DIR/network_scenario_${num}.xml.gz"  

        # Check if the network file exists  
        if [ -f "$network_file" ]; then  
            # If the file exists, add it to the list of files to be deleted  
            files_to_delete+=("$network_file")  
        fi  
    fi  
done  

# Check if there are any files to delete  
if [ ${#files_to_delete[@]} -eq 0 ]; then  
    echo "No network files found to delete."  
    exit 0  
fi  

# Display the list of files that will be deleted  
echo ""  
echo "The following files will be deleted:"  
printf '%s\n' "${files_to_delete[@]}"  
echo ""  

# Ask the user for confirmation  
read -p "Are you sure you want to delete these files? (y/n) " -n 1 -r  
echo    # Move to a new line  

# If the user input is 'y' or 'Y', then perform the deletion  
if [[ $REPLY =~ ^[Yy]$ ]]; then  
    echo "Deleting files..."  
    # Loop through and delete each file in the list  
    for file in "${files_to_delete[@]}"; do  
        rm "$file"  
        echo "Deleted: $file"  
    done  
    echo "Deletion complete."  
else  
    echo "Operation cancelled."  
fi