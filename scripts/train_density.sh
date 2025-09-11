#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="benchmark/protocols/domain_incremental"

# === Preload datasets ===
export PRELOAD_COCO=1
export COCO_SPLIT="val"
export PRELOAD_POSEADAPT=1
export POSEADAPT_SPLIT="density"

# === Config sets ===
DENSITY_O5=(
    density-o5/rtmpose-t_8xb32-10e_poseadapt-density-o5_256x192.py
    density-o5/rtmpose-t_8xb32-10e_poseadapt-density-o5_ewc_256x192.py
    density-o5/rtmpose-t_8xb32-10e_poseadapt-density-o5_lfl_256x192.py
    density-o5/rtmpose-t_8xb32-10e_poseadapt-density-o5_lwf_256x192.py
    density-o5/rtmpose-t_8xb32-10e_poseadapt-density-o5_pt_256x192.py
)

DENSITY_O10=(
    density-o10/rtmpose-t_8xb32-10e_poseadapt-density-o10_256x192.py
    density-o10/rtmpose-t_8xb32-10e_poseadapt-density-o10_ewc_256x192.py
    density-o10/rtmpose-t_8xb32-10e_poseadapt-density-o10_lfl_256x192.py
    density-o10/rtmpose-t_8xb32-10e_poseadapt-density-o10_lwf_256x192.py
    density-o10/rtmpose-t_8xb32-10e_poseadapt-density-o10_pt_256x192.py
)

DENSITY_O20=(
    density-o20/rtmpose-t_8xb32-10e_poseadapt-density-o20_256x192.py
    density-o20/rtmpose-t_8xb32-10e_poseadapt-density-o20_ewc_256x192.py
    density-o20/rtmpose-t_8xb32-10e_poseadapt-density-o20_lfl_256x192.py
    density-o20/rtmpose-t_8xb32-10e_poseadapt-density-o20_lwf_256x192.py
    density-o20/rtmpose-t_8xb32-10e_poseadapt-density-o20_pt_256x192.py
)

MODE_MS=(
    density-ms/rtmpose-t_8xb32-10e_poseadapt-density-ms_256x192.py
    density-ms/rtmpose-t_8xb32-10e_poseadapt-density-ms_ewc_256x192.py
    density-ms/rtmpose-t_8xb32-10e_poseadapt-density-ms_lfl_256x192.py
    density-ms/rtmpose-t_8xb32-10e_poseadapt-density-ms_lwf_256x192.py
    density-ms/rtmpose-t_8xb32-10e_poseadapt-density-ms_pt_256x192.py
)

# === Resource defaults (overridable via env) ===
MEMORY="${MEMORY:-128G}"
TIME="${TIME:-04:00:00}"
PARTITION="${PARTITION:-RTXA6000,RTXA6000-AV,A100-80GB,H100,H200,H200-AV,L40S,L40S-AV}"
CPUS="${CPUS:-128}"

# === Logger ===
log() { echo "$(date +'%Y-%m-%d %H:%M:%S') [INFO] $*"; }

# === Experiment selection ===
case "${1:-all}" in
    o5)   SELECTED=("${DENSITY_O5[@]}") ;;
    o10)  SELECTED=("${DENSITY_O10[@]}") ;;
    o20)  SELECTED=("${DENSITY_O20[@]}") ;;
    ms)   SELECTED=("${MODE_MS[@]}") ;;
    all)  SELECTED=("${DENSITY_O5[@]}" "${DENSITY_O10[@]}" "${DENSITY_O20[@]}" "${MODE_MS[@]}") ;;
    *) echo "Usage: $0 [o5|o10|o20|ms|all]"; exit 1 ;;
esac

# === Run training ===
log "Launching ${#SELECTED[@]} density experiments on partition(s): $PARTITION"
MEMORY=$MEMORY TIME=$TIME PARTITION=$PARTITION CPUS=$CPUS mmtrain "${SELECTED[@]/#/$BASE_DIR/}"
