#!/usr/bin/env bash
set -euo pipefail

WORKERS="${1:-2}"
ROOT="${2:-/mnt/f/kaggle-orbit-wars-data/bc_feature_rows/official_2p_winners_fight_parallel_${WORKERS}}"
PYTHON="${PYTHON:-/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python}"
REPO="${REPO:-/home/dministrator/kaggle/rl-agent}"
SHARD_SIZE="${SHARD_SIZE:-131072}"
COMPRESSION="${COMPRESSION:-compressed}"
FEATURE_DTYPE="${FEATURE_DTYPE:-float32}"
NOOP_MODE="${NOOP_MODE:-all}"
NOOP_ROW_KEEP_RATIO="${NOOP_ROW_KEEP_RATIO:-1.0}"
SOURCE_FILTER_MODE="${SOURCE_FILTER_MODE:-late}"

mkdir -p "$ROOT"

for i in $(seq 0 "$((WORKERS - 1))"); do
  out="$ROOT/worker_${i}"
  mkdir -p "$out"
  setsid sh -c "cd '$REPO' && exec env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    '$PYTHON' -m jax_agent.build_bc_feature_rows_from_zips \
    --output-dir '$out' \
    --zip-shard-count '$WORKERS' \
    --zip-shard-index '$i' \
    --shard-size '$SHARD_SIZE' \
    --compression '$COMPRESSION' \
    --feature-dtype '$FEATURE_DTYPE' \
    --board-batch-size 1 \
    --noop-mode '$NOOP_MODE' \
    --noop-row-keep-ratio '$NOOP_ROW_KEEP_RATIO' \
    --source-filter-mode '$SOURCE_FILTER_MODE' \
    --learn-players winners \
    --horizon 60 \
    --report-interval 60 \
    >> '$out/build.log' 2>&1" &
  echo "$!" > "$out/pid"
  echo "worker=$i pid=$(cat "$out/pid") output=$out"
done

echo "root=$ROOT"
