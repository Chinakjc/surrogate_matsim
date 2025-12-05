#!/bin/bash  

# Check number of arguments  
if [ "$#" -ne 2 ]; then  
    echo "Usage: $0 <config_file_path> <networks_folder_path>"  
    echo "Example: $0 /path/to/config.xml /path/to/networks"  
    exit 1  
fi  

CONFIG_FILE="$1"  
NETWORKS_DIR="$2"  

# Check if config file exists  
if [ ! -f "$CONFIG_FILE" ]; then  
    echo "Error: Config file '$CONFIG_FILE' does not exist"  
    exit 1  
fi  

# Check if networks folder exists  
if [ ! -d "$NETWORKS_DIR" ]; then  
    echo "Error: Networks folder '$NETWORKS_DIR' does not exist"  
    exit 1  
fi  

# Create configs folder if it doesn't exist  
CONFIGS_DIR="configs"  
mkdir -p "$CONFIGS_DIR"  

echo "Starting to process network files..."  
echo "Config file: $CONFIG_FILE"  
echo "Networks folder: $NETWORKS_DIR"  
echo "Output folder: $CONFIGS_DIR"  
echo ""  

# Counter  
count=0  

# Iterate through all network*_n.xml.gz files  
for network_file in "$NETWORKS_DIR"/network*_*.xml.gz; do  
    # Check if file exists (prevent errors when no matching files)  
    if [ ! -f "$network_file" ]; then  
        echo "Warning: No matching network files found"  
        continue  
    fi  
    
    # Get filename without path  
    network_filename=$(basename "$network_file")  
    
    # Extract number n using regex  
    if [[ $network_filename =~ network.*_([0-9]+)\.xml\.gz ]]; then  
        n="${BASH_REMATCH[1]}"  
        
        echo "Processing: $network_filename (n=$n)"  
        
        # Create new config filename  
        new_config_file="$CONFIGS_DIR/config_$n.xml"  
        
        # Get network file path  
        network_path="$network_file"  
        
        # Copy and modify config file  
        sed -e "s|<param name=\"inputNetworkFile\" value=\"[^\"]*\" />|<param name=\"inputNetworkFile\" value=\"../$network_path\" />|g" \
            -e "s|<param name=\"outputDirectory\" value=\"[^\"]*\" />|<param name=\"outputDirectory\" value=\"../simulation_outputs/berlin/$n\" />|g" \
            "$CONFIG_FILE" > "$new_config_file"  
        
        echo "  Created: $new_config_file"  
        count=$((count + 1))  
    else  
        echo "Warning: Cannot extract number from filename '$network_filename'"  
    fi  
done  

echo ""  
echo "Done! Processed $count network files"  
echo "All config files saved to '$CONFIGS_DIR' folder"