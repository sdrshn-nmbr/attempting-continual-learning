#!/usr/bin/env bash
# Serve both pilot agents on one 8-GPU node: Qwen3-4B-Instruct-2507 on GPUs 0-1, Qwen3-32B (thinking) on GPUs 2-7.
set -euo pipefail
LOGS="$PILOT_ROOT/logs"
mkdir -p "$LOGS"
log() { echo "[serve $(date +%H:%M:%S)] $*"; }

for model in Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-32B; do
  log "fetching $model"
  hf download "$model" --quiet > /dev/null || { log "DOWNLOAD_FAILED $model"; exit 1; }
done

HIP_VISIBLE_DEVICES=0,1 nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --served-model-name qwen3-4b-instruct-2507 \
  --port 8001 --data-parallel-size 2 --max-model-len 65536 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes > "$LOGS/vllm-4b.log" 2>&1 &
HIP_VISIBLE_DEVICES=2,3,4,5,6,7 nohup vllm serve Qwen/Qwen3-32B --served-model-name qwen3-32b \
  --port 8002 --tensor-parallel-size 2 --data-parallel-size 3 --max-model-len 40960 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 > "$LOGS/vllm-32b.log" 2>&1 &

for port in 8001 8002; do
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null && break
    sleep 10
  done
  curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null || { log "SERVER_NOT_READY port=$port"; exit 1; }
  log "ready on port $port"
done
