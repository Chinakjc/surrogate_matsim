#!/usr/bin/env bash  
set -euo pipefail  

if [[ $# -ne 2 ]]; then  
  echo "Usage: $0 <ids_txt_file> <output_dir>"  
  exit 1  
fi  

IDS_FILE="$1"  
OUT_DIR="$2"  

# Ensure output directory exists  
mkdir -p "$OUT_DIR"  

# Base directory: simulation_output under the current working directory  
BASE_DIR="./simulation_output"  

if [[ ! -f "$IDS_FILE" ]]; then  
  echo "Error: IDs file not found: $IDS_FILE" >&2  
  exit 1  
fi  

# Read IDs line by line  
while IFS= read -r n || [[ -n "$n" ]]; do  
  # Skip empty lines and comments  
  if [[ -z "${n// }" ]] || [[ "$n" =~ ^# ]]; then  
    continue  
  fi  
  # Accept only non-negative integers  
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then  
    echo "Warning: non-numeric line skipped: '$n'" >&2  
    continue  
  fi  

  RUN_DIR="$BASE_DIR/$n"  
  NET_SRC="$RUN_DIR/output_network.xml.gz"  
  ITERS_DIR="$RUN_DIR/ITERS"  

  if [[ ! -d "$RUN_DIR" ]]; then  
    echo "Warning: directory does not exist: $RUN_DIR, skipping n=$n" >&2  
    continue  
  fi  

  # Copy and rename network  
  if [[ -f "$NET_SRC" ]]; then  
    cp -f "$NET_SRC" "$OUT_DIR/network_idf_${n}.xml.gz"  
    echo "Copied: network_idf_${n}.xml.gz"  
  else  
    echo "Warning: missing network file: $NET_SRC, skipping network n=$n" >&2  
  fi  

  # Process ITERS: find the largest iteration m that is a multiple of 10  
  if [[ ! -d "$ITERS_DIR" ]]; then  
    echo "Warning: missing ITERS directory: $ITERS_DIR, skipping linkstats n=$n" >&2  
    continue  
  fi  

  # Collect m candidates from 'it.m' directories (digits only and multiples of 10)  
  mapfile -t M_CANDIDATES < <(find "$ITERS_DIR" -maxdepth 1 -type d -name 'it.*' -printf '%f\n' \
    | awk -F. 'NF==2 && $2 ~ /^[0-9]+$/ {print $2}' \
    | awk '($1 % 10) == 0 {print $1}' \
    | sort -n)  

  if [[ ${#M_CANDIDATES[@]} -eq 0 ]]; then  
    echo "Warning: no iteration directory 'it.m' that is a multiple of 10 found under $ITERS_DIR, skipping linkstats n=$n" >&2  
    continue  
  fi  

  M="${M_CANDIDATES[-1]}" # Largest multiple-of-10 iteration  
  IT_M_DIR="$ITERS_DIR/it.${M}"  
  LINKSTATS_SRC="$IT_M_DIR/${M}.linkstats.txt.gz"  

  if [[ -f "$LINKSTATS_SRC" ]]; then  
    cp -f "$LINKSTATS_SRC" "$OUT_DIR/${M}.linkstats_idf_${n}.txt.gz"  
    echo "Copied: ${M}.linkstats_idf_${n}.txt.gz"  
  else  
    echo "Warning: missing linkstats file: $LINKSTATS_SRC, skipping n=$n" >&2  
  fi  

done < "$IDS_FILE"  

echo "Done. Output directory: $OUT_DIR"