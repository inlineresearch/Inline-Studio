#!/usr/bin/env bash
#
# Launch Inline Core. Friendly flags map onto the engine's environment knobs, so you do not have to
# remember the INLINE_* variables. Run `./webui.sh --help` for the list.
#
#   ./webui.sh                              # loopback, port 8848
#   ./webui.sh --listen --port 9000         # bind all interfaces on 9000
#   ./webui.sh --multi-gpu                  # multi-GPU denoise (auto-detected when 2+ GPUs)
#   ./webui.sh --multi-gpu pipefusion=2     # force a split
#   ./webui.sh --lowvram                    # tight-VRAM profile
#   ./webui.sh --install --extra zimage     # set up the venv with the Z-Image runtime, then exit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="127.0.0.1"
PORT="8848"
EXTRAS="server"
RUN_INSTALL=0

usage() {
  cat <<'EOF'
Usage: ./webui.sh [options]

Networking
  --listen               bind all interfaces (0.0.0.0), so other machines can reach it
  --host ADDR            bind a specific address (default 127.0.0.1)
  --port N               listen on port N (default 8848)

Multi-GPU (split one image's denoise across GPUs)
  --multi-gpu [SPEC]     enable the split; auto-detected with 2+ GPUs. Optional SPEC overrides the
                         degrees, e.g. pipefusion=2 or pipefusion=2,ulysses=2
  --parallel SPEC        alias for --multi-gpu SPEC

Device / memory
  --lowvram              tight-VRAM profile (offload + slicing + int8)
  --cpu                  force the CPU profile
  --profile NAME         set the profile explicitly (gpu-max | lowvram | cpu)
  --vram-budget GB       treat the GPU as having GB of usable VRAM

Paths
  --models-dir PATH      where weights are scanned from (default ./models)
  --data-dir PATH        where runs and takes are written (default ./.inline)

Setup
  --install              create ./.venv (via uv) and install, then exit
  --extra NAME           add an install extra (repeatable): zimage, parallel, server
  -h, --help             show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --listen) HOST="0.0.0.0"; shift ;;
    --host) HOST="${2:?--host needs an address}"; shift 2 ;;
    --port) PORT="${2:?--port needs a number}"; shift 2 ;;
    --multi-gpu|--parallel)
      if [[ $# -ge 2 && "${2:-}" != -* ]]; then
        export INLINE_PARALLEL="$2"; shift 2
      else
        echo "Multi-GPU split is auto-detected with 2+ GPUs; pass e.g. pipefusion=2 to override."
        shift
      fi ;;
    --lowvram) export INLINE_PROFILE="lowvram"; shift ;;
    --cpu) export INLINE_PROFILE="cpu"; shift ;;
    --profile) export INLINE_PROFILE="${2:?--profile needs a name}"; shift 2 ;;
    --vram-budget) export INLINE_VRAM_BUDGET_GB="${2:?--vram-budget needs a number}"; shift 2 ;;
    --models-dir) export INLINE_MODELS_DIR="${2:?--models-dir needs a path}"; shift 2 ;;
    --data-dir) export INLINE_DATA_DIR="${2:?--data-dir needs a path}"; shift 2 ;;
    --install) RUN_INSTALL=1; shift ;;
    --extra) EXTRAS="$EXTRAS,${2:?--extra needs a name}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

export INLINE_HOST="$HOST"
export INLINE_PORT="$PORT"

if [[ "$RUN_INSTALL" -eq 1 ]]; then
  command -v uv >/dev/null 2>&1 || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
  uv venv
  uv pip install -e ".[$EXTRAS]"
  echo "Installed extras: $EXTRAS. Start with: ./webui.sh"
  exit 0
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PY_CMD=(python)
elif [[ -x ".venv/bin/python" ]]; then
  PY_CMD=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PY_CMD=(uv run python)
else
  echo "No .venv found and uv is not installed. Run './webui.sh --install' first." >&2
  exit 1
fi

exec "${PY_CMD[@]}" -m inline_core.server
