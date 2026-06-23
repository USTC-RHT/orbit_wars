#!/usr/bin/env bash
set -euo pipefail

# Fast path:
#   raw replay zips -> owner-canonical feature rows -> player0-view feature flip
#
# This avoids recomputing expensive fight features after state-level mirroring.

REPO_DIR="${REPO_DIR:-/home/dministrator/kaggle/rl-agent}"
CONDA_EXE="${CONDA_EXE:-/home/dministrator/miniconda/bin/conda}"
CONDA_ENV="${CONDA_ENV:-orbit-wars-lab}"

RAW_ROOT="${RAW_ROOT:-/mnt/kaggle/BC_data/raw}"
MANIFEST="${MANIFEST:-}"
ZIP_DIR="${ZIP_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/kaggle/BC_data/selected5_all_games_player0_view_raw_rebuild}"
INTERMEDIATE_DIR="${INTERMEDIATE_DIR:-${OUTPUT_DIR}__owner_canonical}"
LOG_DIR="${LOG_DIR:-/mnt/kaggle/BC_data/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/selected5_raw_rebuild_fast.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/selected5_raw_rebuild_fast.pid}"

SHARD_SIZE="${SHARD_SIZE:-32768}"
BOARD_BATCH_SIZE="${BOARD_BATCH_SIZE:-64}"
ZIP_SHARD_COUNT="${ZIP_SHARD_COUNT:-1}"
ZIP_SHARD_INDEX="${ZIP_SHARD_INDEX:-0}"

if [[ -z "$MANIFEST" ]]; then
  if [[ -f "${RAW_ROOT}/manifests/orbit_2p_manifest.csv" ]]; then
    MANIFEST="${RAW_ROOT}/manifests/orbit_2p_manifest.csv"
  elif [[ -f "${RAW_ROOT}/orbit_2p_manifest.csv" ]]; then
    MANIFEST="${RAW_ROOT}/orbit_2p_manifest.csv"
  else
    MANIFEST="${RAW_ROOT}/manifests/orbit_2p_manifest.csv"
  fi
fi

if [[ -z "$ZIP_DIR" ]]; then
  if [[ -d "${RAW_ROOT}/daily_zips_cli" ]]; then
    ZIP_DIR="${RAW_ROOT}/daily_zips_cli"
  else
    ZIP_DIR="${RAW_ROOT}/daily_zips_cli"
  fi
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "missing REPO_DIR: $REPO_DIR" >&2
  exit 1
fi
if [[ ! -f "$CONDA_EXE" ]]; then
  echo "missing CONDA_EXE: $CONDA_EXE" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "missing MANIFEST: $MANIFEST" >&2
  exit 1
fi
if [[ ! -d "$ZIP_DIR" ]]; then
  echo "missing ZIP_DIR: $ZIP_DIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")" "$(dirname "$INTERMEDIATE_DIR")"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "preprocess already appears to be running: pid=$old_pid"
    echo "pid file: $PID_FILE"
    exit 0
  fi
fi

cat > "${LOG_FILE}.cmd" <<EOF
REPO_DIR=$REPO_DIR
CONDA_ENV=$CONDA_ENV
RAW_ROOT=$RAW_ROOT
MANIFEST=$MANIFEST
ZIP_DIR=$ZIP_DIR
OUTPUT_DIR=$OUTPUT_DIR
INTERMEDIATE_DIR=$INTERMEDIATE_DIR
SHARD_SIZE=$SHARD_SIZE
BOARD_BATCH_SIZE=$BOARD_BATCH_SIZE
ZIP_SHARD_COUNT=$ZIP_SHARD_COUNT
ZIP_SHARD_INDEX=$ZIP_SHARD_INDEX
EOF

nohup bash -lc "
set -euo pipefail
eval \"\$($CONDA_EXE shell.bash hook)\"
conda activate \"$CONDA_ENV\"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd \"$REPO_DIR\"
python -m jax_agent.build_bc_player0_view_feature_rows_from_zips \\
  --manifest \"$MANIFEST\" \\
  --zip-dir \"$ZIP_DIR\" \\
  --output-dir \"$OUTPUT_DIR\" \\
  --intermediate-dir \"$INTERMEDIATE_DIR\" \\
  --zip-shard-count \"$ZIP_SHARD_COUNT\" \\
  --zip-shard-index \"$ZIP_SHARD_INDEX\" \\
  --learn-players all \\
  --learn-player-names \\
    \"Isaiah @ Tufa Labs\" \\
    \"Jake Will\" \\
    \"Ender\" \\
    \"TonyK\" \\
    \"Vadasz & Ascalon\" \\
  --shard-size \"$SHARD_SIZE\" \\
  --compression compressed \\
  --feature-dtype float32 \\
  --board-batch-size \"$BOARD_BATCH_SIZE\" \\
  --noop-mode all \\
  --noop-row-keep-ratio 1.0 \\
  --source-filter-mode late \\
  --horizon 60 \\
  --max-planets 45 \\
  --max-fleets 512 \\
  --ship-bucket-count 8 \\
  --map-grid-size 8 \\
  --max-ships 500.0 \\
  --max-production 5.0 \\
  --episode-steps 500 \\
  --attack-top-k 10 \\
  --defense-top-k 5 \\
  --report-interval 60
" > "$LOG_FILE" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "started selected5 player0-view preprocessing"
echo "pid: $pid"
echo "log: $LOG_FILE"
echo "pid_file: $PID_FILE"
echo "output_dir: $OUTPUT_DIR"
echo "intermediate_dir: $INTERMEDIATE_DIR"
echo
echo "watch:"
echo "  tail -f \"$LOG_FILE\""
