#!/usr/bin/env bash
# Cross-model STILL transfer on one 8-GPU node. Usage: run.sh setup | calibrate <8b|32b> | smoke <8b|32b>
# | evaluate <8b|32b> [ridge|refit] | refit <8b|32b>
set -euo pipefail
ROOT=/mnt/shared/still-transfer
CODE=$ROOT/code/experiments/still_transfer
SOURCE=Qwen/Qwen3-4B-Instruct-2507
CHECKPOINT=/mnt/shared/still-repro/runs-luna/seed17/compactor-1500.pt
CORPUS=/mnt/shared/still-repro/corpus
EVAL_ITEMS=/mnt/shared/still-repro/items-luna/items-eval.jsonl
TRAIN_ITEMS=/mnt/shared/still-repro/items-luna/items-train.jsonl
FINEWEB=$ROOT/fineweb/sample/10BT/000_00000.parquet
TORCHRUN="torchrun --standalone --nproc-per-node 8"
log() { echo "[run $(date +%H:%M:%S)] $*"; }
receiver() { case $1 in 8b) echo Qwen/Qwen3-8B ;; 32b) echo Qwen/Qwen3-32B ;; *) log "UNKNOWN_RECEIVER $1"; exit 1 ;; esac; }
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
    $TORCHRUN calibrate.py --source "$SOURCE" --receiver "$(receiver "$2")" --fineweb "$FINEWEB" \
      --out "$ROOT/mappers/$2"
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
  *)
    log "UNKNOWN_STAGE $1"; exit 1 ;;
esac
log "finished $*"
