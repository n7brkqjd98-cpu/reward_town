#!/usr/bin/env bash
# Reward Town game: prepare the GPU environment, fetch the Tuxemon art if needed, and run the live game server.
#
#   ./startup.sh                 # play on the GPU (.venv-gpu) at http://127.0.0.1:8765/
#   ./startup.sh --cpu           # run the reward model with the CPU-only .venv instead of .venv-gpu
#   ./startup.sh --port 9000     # serve on another port (default 8765)
set -euo pipefail

SCRIPT=$(readlink -f "$0")
cd "$(dirname "$SCRIPT")"

PORT=8765
HOST=127.0.0.1
DEVICE=cuda
MODEL=models/Skywork-Reward-V2-Qwen3-0.6B
TOWN_JSON=town_demo/game/assets/town.json

usage() { sed -n '2,6p' "$SCRIPT" | sed 's/^# \{0,1\}//'; }

while (($#)); do
  case "$1" in
    --cpu) DEVICE=cpu ;;
    --port)
      [[ $# -ge 2 && $2 =~ ^[0-9]+$ ]] || { echo "error: --port needs a number" >&2; exit 2; }
      PORT=$2
      shift
      ;;
    -h | --help) usage; exit 0 ;;
    *) echo "error: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -f $MODEL/config.json ]] || { echo "error: model not found at $MODEL" >&2; exit 1; }

cuda_ok() {
  .venv-gpu/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

# Builds .venv-gpu with a CUDA torch wheel; tries cu130 first, then cu126 (older drivers).
setup_gpu_env() {
  local cu
  for cu in cu130 cu126; do
    echo "==> Creating .venv-gpu with torch ($cu)"
    rm -rf .venv-gpu
    python3 -m venv .venv-gpu
    .venv-gpu/bin/pip install -q torch --index-url "https://download.pytorch.org/whl/$cu"
    .venv-gpu/bin/pip install -q 'transformers>=4.52,<5' numpy safetensors accelerate
    if cuda_ok; then return 0; fi
    echo "    torch.cuda.is_available() is False with $cu"
  done
  echo "error: no working CUDA torch build; rerun with --cpu" >&2
  exit 1
}

if [[ $DEVICE == cuda ]]; then
  PY=.venv-gpu/bin/python
  if [[ ! -x $PY ]]; then
    setup_gpu_env
  elif ! cuda_ok; then
    echo "error: .venv-gpu exists but CUDA is unavailable; check the driver, delete .venv-gpu to rebuild," \
      "or rerun with --cpu" >&2
    exit 1
  fi
  "$PY" -c "import torch; print('==> GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
else
  PY=.venv/bin/python
  [[ -x $PY ]] || { echo "error: $PY missing; create it as described in README.md" >&2; exit 1; }
fi

if [[ ! -f $TOWN_JSON ]]; then
  echo "==> Fetching Tuxemon assets"
  "$PY" -m town_demo.fetch_assets
fi

echo "==> Starting the Reward Town server on $DEVICE (Ctrl-C to stop)"
exec "$PY" -m town_demo.server --device "$DEVICE" --host "$HOST" --port "$PORT"
