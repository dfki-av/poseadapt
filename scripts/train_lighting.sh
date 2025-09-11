#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="benchmark/protocols/domain_incremental"

# === Preload datasets ===
export PRELOAD_COCO=1
export COCO_SPLIT="val"
export PRELOAD_POSEADAPT=1
export POSEADAPT_SPLIT="lighting"

# === Config sets ===
LIGHTING_LL=(
    lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_256x192.py
    lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_ewc_256x192.py
    lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_ewc-s_256x192.py
    lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_lfl_256x192.py
    lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_lwf_256x192.py
    lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_rwalk_256x192.py
    lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_pt_256x192.py
    lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_si_256x192.py
)

LIGHTING_LLE=(
    lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_256x192.py
    lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_ewc_256x192.py
    lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_ewc-s_256x192.py
    lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_lfl_256x192.py
    lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_lwf_256x192.py
    lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_rwalk_256x192.py
    lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_pt_256x192.py
    lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_si_256x192.py
)

LIGHTING_LLV=(
    lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_256x192.py
    lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_ewc_256x192.py
    lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_ewc-s_256x192.py
    lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_lfl_256x192.py
    lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_lwf_256x192.py
    lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_rwalk_256x192.py
    lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_pt_256x192.py
    lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_si_256x192.py
)

LIGHTING_MS=(
    lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_256x192.py
    lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_ewc_256x192.py
    lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_ewc-s_256x192.py
    lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_lfl_256x192.py
    lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_lwf_256x192.py
    lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_rwalk_256x192.py
    lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_pt_256x192.py
    lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_si_256x192.py
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
    ll)   SELECTED=("${LIGHTING_LL[@]}") ;;
    lle)  SELECTED=("${LIGHTING_LLE[@]}") ;;
    llv)  SELECTED=("${LIGHTING_LLV[@]}") ;;
    ms)   SELECTED=("${LIGHTING_MS[@]}") ;;
    all)  SELECTED=("${LIGHTING_LL[@]}" "${LIGHTING_LLE[@]}" "${LIGHTING_LLV[@]}" "${LIGHTING_MS[@]}") ;;
    *) echo "Usage: $0 [ll|lle|llv|ms|all]"; exit 1 ;;
esac

# === Run training ===
log "Launching ${#SELECTED[@]} lighting experiments on partition(s): $PARTITION"
MEMORY=$MEMORY TIME=$TIME PARTITION=$PARTITION CPUS=$CPUS mmtrain "${SELECTED[@]/#/$BASE_DIR/}"
