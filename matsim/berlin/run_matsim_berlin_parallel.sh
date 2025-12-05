#!/usr/bin/env bash  
# 并发运行柏林 MATSim 配置的批处理脚本（安全增强版）  
# 用法：  
#   CONCURRENCY=5 JAVA_XMX=16G LOG_DIR=logs_berlin ENABLE_LOCK=1 AUTO_CONFIRM=0 \
#   OUTPUT_BASE=/output NETWORK_FILE=/network.xml \
#   ./run_matsim_berlin_parallel.sh /path/to/berlin.jar /path/to/config_dir  
#  
# 参数：  
#   $1: JAR 路径（必填）  
#   $2: 配置目录（必填，仅匹配 config_n.xml）  
#  
# 环境变量：  
#   CONCURRENCY          最大并发数（默认 2）  
#   JAVA_XMX             每个 JVM 的最大堆（默认 8G），例如 16G  
#   LOG_DIR              日志目录（默认 logs）  
#   ENABLE_LOCK          是否启用互斥锁（1=启用，0=禁用，默认 1）  
#   LOCK_DIR             锁目录（默认 /tmp/matsim_locks/<配置目录名>）  
#   AUTO_CONFIRM         是否自动确认（1=继续，不询问；0=询问，默认 0）  
#   
#   柏林 MATSim 特定选项（可选，用于覆盖配置文件）：  
#   NETWORK_FILE         网络文件路径（覆盖 config 中的 network.inputFile）  
#   OUTPUT_BASE          输出目录基础路径（会为每个配置自动添加后缀）  
#   LINKSTATS_INTERVAL   linkStats 写入间隔  
#   LINKSTATS_AVERAGE    linkStats 平均迭代数  

set -euo pipefail  

#-----------------------------  
# 参数与默认值  
#-----------------------------  
JAR_PATH="${1:-}"  
CONFIGS_DIR="${2:-}"  
MAIN_CLASS="org.matsim.project.RunMatsimCli"  

CONCURRENCY="${CONCURRENCY:-2}"  
JAVA_XMX="${JAVA_XMX:-8G}"  
LOG_DIR="${LOG_DIR:-logs}"  
ENABLE_LOCK="${ENABLE_LOCK:-1}"  
AUTO_CONFIRM="${AUTO_CONFIRM:-0}"  

# 柏林 MATSim 特定选项  
NETWORK_FILE="${NETWORK_FILE:-}"  
OUTPUT_BASE="${OUTPUT_BASE:-}"  
LINKSTATS_INTERVAL="${LINKSTATS_INTERVAL:-}"  
LINKSTATS_AVERAGE="${LINKSTATS_AVERAGE:-}"  

# 锁目录默认包含配置目录名，避免跨批次同名 config 误冲突  
LOCK_DIR="${LOCK_DIR:-/tmp/matsim_locks/$(basename "${CONFIGS_DIR:-configs}")}"  

#-----------------------------  
# 校验参数  
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
# 收集配置文件（仅匹配 config_n.xml；健壮处理）  
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

# 先打印 JAR 和主类信息  
echo "JAR: $JAR_PATH"  
echo "Main class: $MAIN_CLASS"  
echo  

# 显示覆盖选项（如果设置了）  
echo "Override options:"  
[[ -n "$NETWORK_FILE" ]] && echo "  Network file (-n): $NETWORK_FILE"  
[[ -n "$OUTPUT_BASE" ]] && echo "  Output base (-o): $OUTPUT_BASE (will append config ID)"  
[[ -n "$LINKSTATS_INTERVAL" ]] && echo "  LinkStats interval: $LINKSTATS_INTERVAL"  
[[ -n "$LINKSTATS_AVERAGE" ]] && echo "  LinkStats average: $LINKSTATS_AVERAGE"  
echo  

#-----------------------------  
# 打印清单并确认继续（增强安全）  
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
# 工具函数：检测是否支持 wait -n  
#-----------------------------  
supports_wait_n=1  
if ! ( help wait 2>&1 | grep -q -- 'wait:.*-n' ); then  
  supports_wait_n=0  
fi  

#-----------------------------  
# 单任务执行函数  
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

  # 构建 Java 命令参数（柏林 MATSim CLI 格式）  
  local java_args=()  
  java_args+=("-Xmx${JAVA_XMX}")  
  java_args+=("-cp" "$JAR_PATH")  
  java_args+=("$MAIN_CLASS")  
  java_args+=("-c" "$cfg")  
  
  # 添加可选覆盖参数  
  if [[ -n "$NETWORK_FILE" ]]; then  
    java_args+=("-n" "$NETWORK_FILE")  
  fi  
  
  if [[ -n "$OUTPUT_BASE" ]]; then  
    # 为每个配置创建独立的输出目录（使用 config_n 的 n 作为后缀）  
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
# 并发调度  
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
# ```## 主要变化说明  

# ### 1. **命令行参数适配**  
# - 使用 `-c` 代替 `--config-path`  
# - 主类固定为 `org.matsim.project.RunMatsimCli`  

# ### 2. **新增环境变量**（用于覆盖配置文件）  
# - `NETWORK_FILE`：网络文件路径（对应 `-n` 参数）  
# - `OUTPUT_BASE`：输出目录基础路径（对应 `-o` 参数）  
# - `LINKSTATS_INTERVAL`：linkStats 写入间隔  
# - `LINKSTATS_AVERAGE`：linkStats 平均迭代数  

# ### 3. **智能输出目录**  
# 如果设置了 `OUTPUT_BASE=/output`，脚本会自动为每个配置创建独立目录：  
# - `config_1.xml` → `/output_1`  
# - `config_2.xml` → `/output_2`  
# - `config_10.xml` → `/output_10`  

# ## 使用示例  

# ### 基础用法  
# ```bash  
# chmod +x run_matsim_berlin_parallel.sh  

# # 最简单的用法（使用默认值）  
# ./run_matsim_berlin_parallel.sh /path/to/berlin.jar /path/to/configs