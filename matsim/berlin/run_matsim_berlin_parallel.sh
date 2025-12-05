#!/usr/bin/env bash  
# Concurrent batch script for running Berlin MATSim configs (enhanced safety version)
# Usage:
#   CONCURRENCY=5 JAVA_XMX=16G LOG_DIR=logs_berlin ENABLE_LOCK=1 AUTO_CONFIRM=0 \
#   OUTPUT_BASE=/output NETWORK_FILE=/network.xml \
#   ./run_matsim_berlin_parallel.sh /path/to/berlin.jar /path/to/config_dir
#
# Arguments:
#   $1: Path to JAR file (required)
#   $2: Config directory (required, only matches config_n.xml)
#
# Environment variables:
#   CONCURRENCY          Maximum concurrency (default 2)
#   JAVA_XMX             Max heap for each JVM (default 8G), e.g., 16G
#   LOG_DIR              Log directory (default: logs)
#   ENABLE_LOCK          Enable mutex lock (1=enable, 0=disable, default 1)
#   LOCK_DIR             Lock directory (default: /tmp/matsim_locks/<config dir name>)
#   AUTO_CONFIRM         Auto confirm (1=continue, no prompt; 0=prompt, default 0)
#
#   Berlin MATSim specific options (optional, to override config file):
#   NETWORK_FILE         Network file path (overrides network.inputFile in config)
#   OUTPUT_BASE          Output base directory (suffix will be added for each config)
#   LINKSTATS_INTERVAL   linkStats write interval
#   LINKSTATS_AVERAGE    linkStats average iteration count

set -euo pipefail  

#-----------------------------
# Arguments and default values
#-----------------------------
JAR_PATH="${1:-}"  
CONFIGS_DIR="${2:-}"  
MAIN_CLASS="org.matsim.project.RunMatsimCli"  

CONCURRENCY="${CONCURRENCY:-2}"  
JAVA_XMX="${JAVA_XMX:-8G}"  
LOG_DIR="${LOG_DIR:-logs}"  
ENABLE_LOCK="${ENABLE_LOCK:-1}"  
AUTO_CONFIRM="${AUTO_CONFIRM:-0}"  

# Berlin MATSim specific options
NETWORK_FILE="${NETWORK_FILE:-}"
OUTPUT_BASE="${OUTPUT_BASE:-}"
LINKSTATS_INTERVAL="${LINKSTATS_INTERVAL:-}"
LINKSTATS_AVERAGE="${LINKSTATS_AVERAGE:-}"

# Lock directory defaults to include the config directory name to avoid
# accidental conflicts between batches with the same config names
LOCK_DIR="${LOCK_DIR:-/tmp/matsim_locks/$(basename "${CONFIGS_DIR:-configs}")}" 

#-----------------------------
# Validate parameters
#-----------------------------
if [[ -z "$JAR_PATH" || -z "$CONFIGS_DIR" ]]; then  
  echo "Usage: $0 <JAR_PATH> <CONFIGS_DIR>" >&2  
  exit 1  
fi  

if [[ ! -f "$JAR_PATH" ]]; then  
  echo "Error: JAR not found: $JAR_PATH" >&2  
  exit 1  
fi  

if [[ ! -d "$CONFIGS_DIR" ]]; then  
  echo "Error: Config directory not found: $CONFIGS_DIR" >&2  
  exit 1  
fi  

mkdir -p "$LOG_DIR"  
mkdir -p "$LOCK_DIR"  

#-----------------------------
# Collect config files (only match config_n.xml; robust handling)
#-----------------------------
mapfile -d '' CONFIG_FILES < <(  
  find "$CONFIGS_DIR" -maxdepth 1 -type f \
    -regextype posix-extended -regex '.*/config_[0-9]+\.xml' \
    -print0 | sort -z  
)  

TOTAL="${#CONFIG_FILES[@]}"  
if (( TOTAL == 0 )); then  
  echo "No config_n.xml files found in $CONFIGS_DIR (e.g., config_1.xml, config_2.xml)"  
  exit 0  
fi  

# Print JAR and main class information
echo "JAR: $JAR_PATH"
echo "Main class: $MAIN_CLASS"
echo

# Show override options (if set)
echo "Override options:"
[[ -n "$NETWORK_FILE" ]] && echo "  Network file (-n): $NETWORK_FILE"
[[ -n "$OUTPUT_BASE" ]] && echo "  Output base (-o): $OUTPUT_BASE (will append config ID)"
[[ -n "$LINKSTATS_INTERVAL" ]] && echo "  LinkStats interval: $LINKSTATS_INTERVAL"
[[ -n "$LINKSTATS_AVERAGE" ]] && echo "  LinkStats average: $LINKSTATS_AVERAGE"
echo

#-----------------------------
# Print list and confirm to continue (safety enhanced)
#-----------------------------
echo "Found $TOTAL configuration files to run (pattern: config_n.xml):"  
idx=1  
for cfg in "${CONFIG_FILES[@]}"; do  
  base="$(basename "$cfg")"  
  echo "  [$idx] $base"  
  ((idx++))  
done  
echo  

if [[ "$AUTO_CONFIRM" == "1" ]]; then  
  echo "AUTO_CONFIRM=1 -> Proceeding without interactive confirmation."  
else  
  if [[ ! -t 0 ]]; then  
    echo "No TTY detected and AUTO_CONFIRM is not set. Aborting for safety."  
    exit 1  
  fi  
  read -r -p "Proceed to run all $TOTAL configurations? [y/N]: " answer  
  case "$answer" in  
    [yY]|[yY][eE][sS]) echo "Confirmed. Starting...";;  
    *) echo "Aborted by user."; exit 0;;  
  esac  
fi  

echo "Total $TOTAL configs; concurrency: $CONCURRENCY; log dir: $LOG_DIR"  
echo  

#-----------------------------
# Helper: detect if `wait -n` is supported
#-----------------------------
supports_wait_n=1  
if ! ( help wait 2>&1 | grep -q -- 'wait:.*-n' ); then  
  supports_wait_n=0  
fi  

#-----------------------------
# Single-task runner function
#-----------------------------
run_one() {  
  local cfg="$1"  
  local base ts log rc jpid  
  base="$(basename "$cfg" .xml)"  
  ts="$(date +%F_%H-%M-%S)"  

  log="$(mktemp -p "$LOG_DIR" "matsim_${base}_${ts}_XXXX.log")"  

  local lock lockfd  
  if [[ "$ENABLE_LOCK" == "1" ]]; then  
    mkdir -p "$LOCK_DIR"  
    lock="${LOCK_DIR}/matsim_${base}.lock"  
    exec {lockfd}> "$lock"  
    if ! flock -n "$lockfd"; then  
      echo "[$(date +%T)] Skip: $base ($cfg) is already running (lock: $lock)"  
      eval "exec ${lockfd}>&-"  
      return 0  
    fi  
  fi  

  echo "[$(date +%T)] Start: $base (runner PID $$) -> $log"  

  # Build Java command arguments (Berlin MATSim CLI format)
  local java_args=()  
  java_args+=("-Xmx${JAVA_XMX}")  
  java_args+=("-cp" "$JAR_PATH")  
  java_args+=("$MAIN_CLASS")  
  java_args+=("-c" "$cfg")  
  
  # Add optional override parameters
  if [[ -n "$NETWORK_FILE" ]]; then  
    java_args+=("-n" "$NETWORK_FILE")  
  fi  
  
  if [[ -n "$OUTPUT_BASE" ]]; then  
    # Create a separate output directory per config (use the n from config_n as suffix)
    local config_id  
    config_id=$(echo "$base" | grep -oP 'config_\K[0-9]+')  
    java_args+=("-o" "${OUTPUT_BASE}_${config_id}")  
  fi  
  
  if [[ -n "$LINKSTATS_INTERVAL" ]]; then  
    java_args+=("--linkstats-interval" "$LINKSTATS_INTERVAL")  
  fi  
  
  if [[ -n "$LINKSTATS_AVERAGE" ]]; then  
    java_args+=("--linkstats-average" "$LINKSTATS_AVERAGE")  
  fi  

  set +e  
  java "${java_args[@]}" >>"$log" 2>&1 &  
  jpid=$!  

  echo "[$(date +%T)] Java PID for $base: $jpid"  

  wait "$jpid"  
  rc=$?  
  set -e  

  if [[ $rc -eq 0 ]]; then  
    echo "[$(date +%T)] Done: $base (java PID $jpid) (OK)"  
  else  
    echo "[$(date +%T)] Failed: $base (java PID $jpid) (exit $rc), log: $log" >&2  
  fi  

  if [[ "$ENABLE_LOCK" == "1" ]]; then  
    eval "exec ${lockfd}>&-"  
  fi  

  return $rc  
}  

#-----------------------------
# Concurrency scheduler
#-----------------------------
set +e  

running=0  
pids=()  

wait_any() {  
  local pid finished=1  
  for pid in "${pids[@]}"; do  
    if ! kill -0 "$pid" 2>/dev/null; then  
      wait "$pid"  
      finished=0  
      local tmp=()  
      for x in "${pids[@]}"; do  
        [[ "$x" == "$pid" ]] || tmp+=("$x")  
      done  
      pids=("${tmp[@]}")  
      break  
    fi  
  done  
  return $finished  
}  

for cfg in "${CONFIG_FILES[@]}"; do  
  echo "Dispatching: $cfg"  
  run_one "$cfg" &  
  pid=$!  
  echo "Launched task for $(basename "$cfg") runner PID: $pid"  
  pids+=("$pid")  
  ((running++))  

  if (( running >= CONCURRENCY )); then  
    if (( supports_wait_n == 1 )); then  
      wait -n  
      ((running--))  
    else  
      while true; do  
        if wait_any; then  
          ((running--))  
          break  
        fi  
        sleep 1  
      done  
    fi  
  fi  
done  

fail=0  
for pid in "${pids[@]}"; do  
  if ! wait "$pid"; then  
    fail=1  
  fi  
done  

set -e  

if (( fail )); then  
  echo  
  echo "Some tasks failed. Please check logs under $LOG_DIR."  
  exit 1  
else  
  echo  
  echo "All tasks completed. Logs are in: $LOG_DIR"  
fi  
# ```## Major changes
#
# ### 1. Command-line argument adaptations
# - Use `-c` instead of `--config-path`
# - Main class fixed to `org.matsim.project.RunMatsimCli`
#
# ### 2. New environment variables (to override config file)
# - `NETWORK_FILE`: path to network file (corresponds to `-n`)
# - `OUTPUT_BASE`: base path for output directories (corresponds to `-o`)
# - `LINKSTATS_INTERVAL`: linkStats write interval
# - `LINKSTATS_AVERAGE`: linkStats average iteration count
#
# ### 3. Smart output directories
# If `OUTPUT_BASE=/output` is set, the script will automatically create separate directories per config:
# - `config_1.xml` -> `/output_1`
# - `config_2.xml` -> `/output_2`
# - `config_10.xml` -> `/output_10`
#
# ## Examples
#
# ### Basic usage
# ```bash
# chmod +x run_matsim_berlin_parallel.sh
#
# # Simple usage (using defaults)
# ./run_matsim_berlin_parallel.sh /path/to/berlin.jar /path/to/configs