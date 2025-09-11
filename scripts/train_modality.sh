#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="benchmark/protocols/domain_incremental"

# === Preload datasets ===
export PRELOAD_COCO=1
export COCO_SPLIT="val"
export PRELOAD_POSEADAPT=1
export POSEADAPT_SPLIT="mode"

# === Config sets ===
MODE_G=(
    mode-g/rtmpose-t_8xb32-10e_poseadapt-mode-g_256x192.py
    mode-g/rtmpose-t_8xb32-10e_poseadapt-mode-g_ewc_256x192.py
    mode-g/rtmpose-t_8xb32-10e_poseadapt-mode-g_lfl_256x192.py
    mode-g/rtmpose-t_8xb32-10e_poseadapt-mode-g_lwf_256x192.py
    mode-g/rtmpose-t_8xb32-10e_poseadapt-mode-g_pt_256x192.py
)

MODE_D=(
    mode-d/rtmpose-t_8xb32-10e_poseadapt-mode-d_256x192.py
    mode-d/rtmpose-t_8xb32-10e_poseadapt-mode-d_ewc_256x192.py
    mode-d/rtmpose-t_8xb32-10e_poseadapt-mode-d_lfl_256x192.py
    mode-d/rtmpose-t_8xb32-10e_poseadapt-mode-d_lwf_256x192.py
    mode-d/rtmpose-t_8xb32-10e_poseadapt-mode-d_pt_256x192.py
)

MODE_MS=(
    mode-ms/rtmpose-t_8xb32-10e_poseadapt-mode-ms_256x192.py
    mode-ms/rtmpose-t_8xb32-10e_poseadapt-mode-ms_ewc_256x192.py
    mode-ms/rtmpose-t_8xb32-10e_poseadapt-mode-ms_lfl_256x192.py
    mode-ms/rtmpose-t_8xb32-10e_poseadapt-mode-ms_lwf_256x192.py
    mode-ms/rtmpose-t_8xb32-10e_poseadapt-mode-ms_pt_256x192.py
)

# === Resource defaults (overridable via env) ===
MEMORY="${MEMORY:-64G}"
TIME="${TIME:-04:00:00}"
PARTITION="${PARTITION:-RTXA6000,RTXA6000-AV}"
CPUS="${CPUS:-64}"

# === Logger ===
log() { echo "$(date +'%Y-%m-%d %H:%M:%S') [INFO] $*"; }

# === Experiment selection ===
case "${1:-all}" in
    g)   SELECTED=("${MODE_G[@]}") ;;
    d)  SELECTED=("${MODE_D[@]}") ;;
    ms)   SELECTED=("${MODE_MS[@]}") ;;
    all)  SELECTED=("${MODE_G[@]}" "${MODE_D[@]}" "${MODE_MS[@]}") ;;
    *) echo "Usage: $0 [g|d|ms|all]"; exit 1 ;;
esac

# === Run training ===
log "Launching ${#SELECTED[@]} modality experiments on partition(s): $PARTITION"
MEMORY=$MEMORY TIME=$TIME PARTITION=$PARTITION CPUS=$CPUS mmtrain "${SELECTED[@]/#/$BASE_DIR/}"
