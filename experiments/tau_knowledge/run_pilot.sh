#!/usr/bin/env bash
# Run the four pilot cells concurrently against the local vLLM servers. Results land in
# $PILOT_ROOT/tau2-bench/data/simulations/<model>-<condition>/results.json.
set -euo pipefail
TAU2_REF=v1.0.1
TAU2_DIR="$PILOT_ROOT/tau2-bench"
LOGS="$PILOT_ROOT/logs"
mkdir -p "$LOGS"
log() { echo "[pilot $(date +%H:%M:%S)] $*"; }

if [ ! -d "$TAU2_DIR" ]; then
  git clone -q https://github.com/sierra-research/tau2-bench "$TAU2_DIR"
fi
git -C "$TAU2_DIR" fetch -q --tags
git -C "$TAU2_DIR" checkout -q "$TAU2_REF"
(cd "$TAU2_DIR" && uv sync -q --python 3.13 --extra knowledge)
log "tau2 at $(git -C "$TAU2_DIR" rev-parse --short HEAD)"

USER_ARGS='{"reasoning_effort":"low"}'
ARGS_4B='{"api_base":"http://127.0.0.1:8001/v1","temperature":0.7,"top_p":0.8}'
ARGS_32B='{"api_base":"http://127.0.0.1:8002/v1","temperature":0.6,"top_p":0.95}'

cell() {
  local name=$1 llm=$2 llm_args=$3 condition=$4 trials=$5
  (cd "$TAU2_DIR" && uv run tau2 run --domain banking_knowledge --retrieval-config "$condition" \
    --agent-llm "$llm" --agent-llm-args "$llm_args" --user-llm gpt-5.2 --user-llm-args "$USER_ARGS" \
    --num-trials "$trials" --max-concurrency 24 --auto-resume --save-to "$name-$condition") \
    > "$LOGS/$name-$condition.log" 2>&1 && log "done $name-$condition" || log "CELL_FAILED $name-$condition"
}

cell qwen3-4b-instruct-2507 hosted_vllm/qwen3-4b-instruct-2507 "$ARGS_4B" no_knowledge 2 &
cell qwen3-4b-instruct-2507 hosted_vllm/qwen3-4b-instruct-2507 "$ARGS_4B" golden_retrieval 4 &
cell qwen3-32b hosted_vllm/qwen3-32b "$ARGS_32B" no_knowledge 2 &
cell qwen3-32b hosted_vllm/qwen3-32b "$ARGS_32B" golden_retrieval 4 &
wait
log "all cells finished"
