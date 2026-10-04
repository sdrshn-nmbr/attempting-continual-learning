#!/usr/bin/env bash
# Cross-model STILL transfer on one 8-GPU node.
# Usage: run.sh setup | calibrate <8b|32b> [k...] | smoke <8b|32b> | evaluate <8b|32b> [ridge|refit] | refit <8b|32b>
#        | train-native 8b | refit-variant <v2|v3|v4> | eval-set <luna|written> <label> <mode...>
set -euo pipefail
ROOT=/mnt/shared/still-transfer
CODE=$ROOT/code/experiments/still_transfer
SOURCE=Qwen/Qwen3-4B-Instruct-2507
CHECKPOINT=/mnt/shared/still-repro/runs-luna/seed17/compactor-1500.pt
CORPUS=/mnt/shared/still-repro/corpus
EVAL_ITEMS=/mnt/shared/still-repro/items-luna/items-eval.jsonl
WRITTEN_ITEMS=/mnt/shared/still-repro/items/items-eval.jsonl
TRAIN_ITEMS=/mnt/shared/still-repro/items-luna/items-train.jsonl
FINEWEB=$ROOT/fineweb/sample/10BT/000_00000.parquet
TORCHRUN="torchrun --standalone --nproc-per-node 8"
log() { echo "[run $(date +%H:%M:%S)] $*"; }
receiver() { case $1 in 8b) echo Qwen/Qwen3-8B ;; 32b) echo Qwen/Qwen3-32B ;; *) log "UNKNOWN_RECEIVER $1"; exit 1 ;; esac; }
mapper_for() {
  case $1 in
    ridge) echo "$ROOT/mappers/8b/mapper.pt" ;;
    v1) echo "$ROOT/refit/8b/mapper.pt" ;;
    v2|v3|v4) echo "$ROOT/refit/8b-$1/mapper.pt" ;;
    native) echo "" ;;
    *) log "UNKNOWN_LABEL $1"; exit 1 ;;
  esac
}
mkdir -p "$ROOT/logs"
cd "$CODE"

case $1 in
  setup)
    pip install -q scipy pytest pyarrow
    for model in "$SOURCE" Qwen/Qwen3-8B Qwen/Qwen3-32B; do hf download "$model" --quiet > /dev/null; done
    hf download HuggingFaceFW/fineweb-edu --repo-type dataset --include "sample/10BT/000_00000.parquet" \
      --local-dir "$ROOT/fineweb" --quiet > /dev/null
    test -f "$FINEWEB" || { log "FINEWEB_MISSING $FINEWEB"; exit 1; }
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider test_kvmap.py
    ;;
  calibrate)
    name=$2; shift 2
    out=$ROOT/mappers/$name
    ks=()
    if [ $# -gt 0 ]; then ks=(--ks "$@"); out=$ROOT/mappers/$name-k$(IFS=-; echo "$*"); fi
    $TORCHRUN calibrate.py --source "$SOURCE" --receiver "$(receiver "$name")" --fineweb "$FINEWEB" --out "$out" "${ks[@]}"
    ;;
  smoke)
    $TORCHRUN evaluate_transfer.py --source "$SOURCE" --receiver "$(receiver "$2")" --checkpoint "$CHECKPOINT" \
      --mapper "$ROOT/mappers/$2/mapper.pt" --corpus "$CORPUS" --items "$EVAL_ITEMS" --out "$ROOT/eval/$2-smoke" \
      --think-off --limit 64
    ;;
  evaluate)
    label=${3:-ridge}
    mapper=$ROOT/mappers/$2/mapper.pt
    modes="none full streaming mapped_full mapped_still"
    if [ "$label" = refit ]; then mapper=$ROOT/refit/$2/mapper.pt; modes="mapped_still"; fi
    $TORCHRUN evaluate_transfer.py --source "$SOURCE" --receiver "$(receiver "$2")" --checkpoint "$CHECKPOINT" \
      --mapper "$mapper" --corpus "$CORPUS" --items "$EVAL_ITEMS" --out "$ROOT/eval/$2-$label" --think-off \
      --modes $modes
    ;;
  refit)
    $TORCHRUN refit.py --source "$SOURCE" --receiver "$(receiver "$2")" --checkpoint "$CHECKPOINT" \
      --mapper "$ROOT/mappers/$2/mapper.pt" --corpus "$CORPUS" --items "$TRAIN_ITEMS" --out "$ROOT/refit/$2" \
      --think-off
    ;;
  train-native)
    cd "$ROOT/code/experiments/still_repro"
    $TORCHRUN train.py --model "$(receiver "$2")" --corpus "$CORPUS" --items "$TRAIN_ITEMS" \
      --out "$ROOT/native/$2" --steps 1500 --seed 17 --data-seed 17 --think-off
    ;;
  refit-variant)
    case $2 in
      v2) init=(--mapper "$ROOT/mappers/8b/mapper.pt" --rank 8) ;;
      v3) init=(--mapper "$ROOT/mappers/8b-k1/mapper.pt") ;;
      v4) init=(--init identity) ;;
      *) log "UNKNOWN_VARIANT $2"; exit 1 ;;
    esac
    $TORCHRUN refit.py --source "$SOURCE" --receiver Qwen/Qwen3-8B --checkpoint "$CHECKPOINT" "${init[@]}" \
      --corpus "$CORPUS" --items "$TRAIN_ITEMS" --out "$ROOT/refit/8b-$2" --think-off
    ;;
  eval-set)
    case $2 in luna) items=$EVAL_ITEMS ;; written) items=$WRITTEN_ITEMS ;; *) log "UNKNOWN_SET $2"; exit 1 ;; esac
    set_name=$2; label=$3; shift 3
    extra=()
    mapper=$(mapper_for "$label")
    if [ -n "$mapper" ]; then extra+=(--checkpoint "$CHECKPOINT" --mapper "$mapper"); fi
    if [ "$label" = native ]; then extra+=(--native-checkpoint "$ROOT/native/8b/compactor-1500.pt"); fi
    $TORCHRUN evaluate_transfer.py --source "$SOURCE" --receiver Qwen/Qwen3-8B --corpus "$CORPUS" --items "$items" \
      --out "$ROOT/eval/8b-$set_name-$label" --think-off --modes "$@" "${extra[@]}"
    ;;
  *)
    log "UNKNOWN_STAGE $1"; exit 1 ;;
esac
log "finished $*"
