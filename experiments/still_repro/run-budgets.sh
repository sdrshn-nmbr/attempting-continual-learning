set -euo pipefail
ROOT=/mnt/shared/still-budgets
CODE=$ROOT/code/f0b86cb/experiments
MODEL=/mnt/shared/still-repro/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
BUDGETS=$CODE/still_repro/results/head-budgets/optimized-agnostic-164.json
cd $CODE/still_repro
mkdir -p $ROOT/hashes
for s in 17 23 29; do
  torchrun --standalone --nproc-per-node 8 train.py --model $MODEL --corpus /mnt/shared/still-repro/corpus --items /mnt/shared/still-repro/items-luna/items-train.jsonl --out $ROOT/runs/seed$s --budgets $BUDGETS --steps 1500 --seed $s --data-seed 17 > $ROOT/logs/train-seed$s.log 2>&1
  cp $ROOT/runs/seed$s/compactor-1500.sha256 $ROOT/hashes/compactor-seed$s.sha256
  torchrun --standalone --nproc-per-node 8 evaluate.py --model $MODEL --corpus /mnt/shared/still-repro/corpus --items /mnt/shared/still-repro/items-luna/items-eval.jsonl --out $ROOT/eval/seed$s --checkpoint $ROOT/runs/seed$s/compactor-1500.pt --budgets $BUDGETS --modes still > $ROOT/logs/eval-seed$s.log 2>&1
  echo "SEED_DONE $s"
done
torchrun --standalone --nproc-per-node 8 evaluate.py --model $MODEL --corpus /mnt/shared/still-repro/corpus --items /mnt/shared/still-repro/items-luna/items-eval.jsonl --out $ROOT/eval/untrained --budgets $BUDGETS --modes untrained > $ROOT/logs/eval-untrained.log 2>&1
cd $CODE/tau_tokens
torchrun --standalone --nproc-per-node 8 score.py --inputs $ROOT/tau-inputs --model $MODEL --conditions still --budgets $BUDGETS --still-label still-budgets --checkpoint-hashes $ROOT/hashes --checkpoint 17=$ROOT/runs/seed17/compactor-1500.pt --checkpoint 23=$ROOT/runs/seed23/compactor-1500.pt --checkpoint 29=$ROOT/runs/seed29/compactor-1500.pt --out $ROOT/tau-scores > $ROOT/logs/tau-score.log 2>&1
echo ALL_DONE
