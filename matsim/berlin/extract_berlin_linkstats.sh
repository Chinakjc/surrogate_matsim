#!/usr/bin/env bash  
set -euo pipefail  

# Default configuration  
SIMULATION_BASE="./output"  
OUTPUT_DIR="./extracted_linkstats"  
ITERATION="last"  
MULTIPLE_OF=10  
VERBOSE=false  

usage() {  
  echo "Usage: extract_berlin_linkstats.sh [OPTIONS]"  
  echo ""  
  echo "Extract linkstats files from Berlin MATSim simulation output."  
  echo "Supports multiple scenarios in berlin_0, berlin_1, berlin_2, ... directories"  
  echo ""  
  echo "OPTIONS:"  
  echo "  -i, --input DIR       Input simulation directory (default: ./output)"  
  echo "  -o, --output DIR      Output directory (default: ./extracted_linkstats)"  
  echo "  -n, --iteration N     Iteration to extract:"  
  echo "                          'last' - last available iteration (default)"  
  echo "                          'all'  - all iterations"  
  echo "                          NUMBER - specific iteration number"  
  echo "  -m, --multiple N      Only process iterations that are multiples of N (default: 10)"  
  echo "  -v, --verbose         Verbose output"  
  echo "  -h, --help            Show this help message"  
  echo ""  
  exit 0  
}  

while [[ $# -gt 0 ]]; do  
  case $1 in  
    -i|--input) SIMULATION_BASE="$2"; shift 2 ;;  
    -o|--output) OUTPUT_DIR="$2"; shift 2 ;;  
    -n|--iteration) ITERATION="$2"; shift 2 ;;  
    -m|--multiple) MULTIPLE_OF="$2"; shift 2 ;;  
    -v|--verbose) VERBOSE=true; shift ;;  
    -h|--help) usage ;;  
    *) echo "Error: Unknown option: $1" >&2; exit 1 ;;  
  esac  
done  

log() {  
  if [[ "$VERBOSE" == true ]]; then  
    echo "[INFO] $*" >&2  
  fi  
}  

# Process a single scenario directory  
process_scenario() {  
  local scenario_dir="$1"  
  local scenario_id="$2"  
  local out_dir="$3"  
  
  log "Processing scenario: $scenario_id from $scenario_dir"  
  
  ITERS_DIR="$scenario_dir/ITERS"  
  if [[ ! -d "$ITERS_DIR" ]]; then  
    echo "  Warning: ITERS directory not found in $scenario_dir, skipping" >&2  
    echo 0  
    return 0  
  fi  

  # Find all iteration directories  
  mapfile -t ALL_ITERS < <(find "$ITERS_DIR" -maxdepth 1 -type d -name 'it.*' -printf '%f\n' \
    | awk -F. 'NF==2 && $2 ~ /^[0-9]+$/ {print $2}' \
    | awk -v mult="$MULTIPLE_OF" '($1 % mult) == 0 {print $1}' \
    | sort -n)  

  if [[ ${#ALL_ITERS[@]} -eq 0 ]]; then  
    echo "  Warning: No iteration directories found in $scenario_dir" >&2  
    echo 0  
    return 0  
  fi  

  log "  Found ${#ALL_ITERS[@]} iteration(s) for scenario $scenario_id"  

  # Determine which iterations to process  
  declare -a ITERS_TO_PROCESS  
  if [[ "$ITERATION" == "last" ]]; then  
    ITERS_TO_PROCESS=("${ALL_ITERS[-1]}")  
  elif [[ "$ITERATION" == "all" ]]; then  
    ITERS_TO_PROCESS=("${ALL_ITERS[@]}")  
  elif [[ "$ITERATION" =~ ^[0-9]+$ ]]; then  
    ITERS_TO_PROCESS=("$ITERATION")  
  fi  

  # Process each iteration  
  local count=0  
  for iter in "${ITERS_TO_PROCESS[@]}"; do  
    IT_DIR="$ITERS_DIR/it.${iter}"  
    
    LINKSTATS_FILE=""  
    for pattern in "${iter}.linkstats.txt.gz" "linkstats.txt.gz" "${iter}.linkstats.csv.gz"; do  
      if [[ -f "$IT_DIR/$pattern" ]]; then  
        LINKSTATS_FILE="$IT_DIR/$pattern"  
        break  
      fi  
    done  

    if [[ -n "$LINKSTATS_FILE" ]]; then  
      OUTPUT_FILE="$out_dir/${iter}.linkstats_berlin_${scenario_id}.txt.gz"  
      cp -f "$LINKSTATS_FILE" "$OUTPUT_FILE"  
      echo "  ✓ Extracted: ${iter}.linkstats_berlin_${scenario_id}.txt.gz" >&2  
      ((count++))  
    else  
      log "  Warning: linkstats file not found in $IT_DIR"  
    fi  
  done  

  # Copy network file if exists (use same naming pattern)  
  for net_pattern in "output_network.xml.gz" "network.xml.gz"; do  
    if [[ -f "$scenario_dir/$net_pattern" ]]; then  
      cp -f "$scenario_dir/$net_pattern" "$out_dir/network_berlin_${scenario_id}.xml.gz"  
      echo "  ✓ Extracted: network_berlin_${scenario_id}.xml.gz" >&2  
      break  
    fi  
  done  

  echo $count  
}  

# Main extraction function  
extract_linkstats() {  
  local base_dir="$1"  
  local out_dir="$2"  
  
  if [[ ! -d "$base_dir" ]]; then  
    echo "Error: Simulation directory not found: $base_dir" >&2  
    exit 1  
  fi  

  mkdir -p "$out_dir"  
  log "Output directory created: $out_dir"  

  # Find all berlin_* directories  
  mapfile -t BERLIN_DIRS < <(find "$base_dir" -maxdepth 1 -type d -name 'berlin_*' | sort -V)  

  if [[ ${#BERLIN_DIRS[@]} -eq 0 ]]; then  
    echo "Error: No berlin_* directories found in $base_dir" >&2  
    echo "Expected directory structure: $base_dir/berlin_0/, berlin_1/, etc." >&2  
    exit 1  
  fi  

  echo "Found ${#BERLIN_DIRS[@]} scenario(s) to process"  
  echo ""  

  local total_files=0  
  for berlin_dir in "${BERLIN_DIRS[@]}"; do  
    # Extract scenario ID (the number after berlin_)  
    scenario_id=$(basename "$berlin_dir" | sed 's/berlin_//')  
    
    echo "Processing scenario: berlin_${scenario_id}"  
    local result=$(process_scenario "$berlin_dir" "$scenario_id" "$out_dir")  
    total_files=$((total_files + result))  
    echo ""  
  done  

  echo "============================================"  
  echo "Extraction complete!"  
  echo "Total scenarios processed: ${#BERLIN_DIRS[@]}"  
  echo "Total linkstats files extracted: $total_files"  
  echo "Output directory: $out_dir"  
  echo "============================================"  
}  

# Main execution  
echo "Berlin MATSim Linkstats Extractor"  
echo "===================================="  
log "Input directory: $SIMULATION_BASE"  
log "Output directory: $OUTPUT_DIR"  
log "Iteration mode: $ITERATION"  
log "Multiple of: $MULTIPLE_OF"  
echo ""  

extract_linkstats "$SIMULATION_BASE" "$OUTPUT_DIR"  