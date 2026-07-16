#!/usr/bin/env bash
#
# Launch Inline Studio (Inline Core + the web UI) on one port. Friendly flags map onto the engine's
# environment knobs, so you do not have to remember the INLINE_* variables. This is the one command
# an end user needs: it installs deps (with --install), makes sure a web UI is present (the prebuilt
# inline-studio-frontend package, or a local SPA build), then serves the app. Run `./webui.sh --help`.
#
#   ./webui.sh                              # loopback, port 8848 (UI + API)
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

Web UI
  Served automatically on the same port. On start, if no UI is found we install the prebuilt
  inline-studio-frontend package, else build it from source (npm) when this is the repo checkout,
  else run API-only. Point at a local build with INLINE_FRONTEND_ROOT (or main.py --front-end-root).
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
  # Pull the prebuilt web UI so there's no Node build (best-effort — it may not be published yet).
  uv pip install inline-studio-frontend >/dev/null 2>&1 \
    && echo "Installed the prebuilt web UI (inline-studio-frontend)." \
    || echo "Note: inline-studio-frontend not installed; the UI will build from source or run API-only."
  echo "Installed extras: $EXTRAS. Start with: ./webui.sh"
  exit 0
fi

# Pick the Python interpreter and the matching pip, in priority order.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PY_CMD=(python)
  PIP_INSTALL=(python -m pip install)
elif [[ -x ".venv/bin/python" ]]; then
  PY_CMD=(.venv/bin/python)
  PIP_INSTALL=(.venv/bin/python -m pip install)
elif command -v uv >/dev/null 2>&1; then
  PY_CMD=(uv run python)
  PIP_INSTALL=(uv pip install)
else
  echo "No .venv found and uv is not installed. Run './webui.sh --install' first." >&2
  exit 1
fi

# True when a web UI is resolvable: an explicit --front-end-root, the installed frontend package, or
# a local dist-web/ build (the repo checkout). Sets INLINE_FRONTEND_ROOT for the local-build case.
frontend_available() {
  if [[ -n "${INLINE_FRONTEND_ROOT:-}" && -f "${INLINE_FRONTEND_ROOT}/index.html" ]]; then return 0; fi
  if "${PY_CMD[@]}" - <<'PY' >/dev/null 2>&1; then return 0; fi
import os, sys
try:
    import inline_studio_frontend as f
except ModuleNotFoundError:
    sys.exit(1)
sys.exit(0 if os.path.isfile(os.path.join(os.path.dirname(f.__file__), "static", "index.html")) else 1)
PY
  if [[ -f "../dist-web/index.html" ]]; then
    INLINE_FRONTEND_ROOT="$(cd .. && pwd)/dist-web"; export INLINE_FRONTEND_ROOT; return 0
  fi
  return 1
}

# Make sure a UI is present before serving: try the pip package, then a local npm build, else warn.
ensure_frontend() {
  frontend_available && return 0
  echo "No web UI found — installing the prebuilt package (inline-studio-frontend)…"
  "${PIP_INSTALL[@]}" inline-studio-frontend >/dev/null 2>&1 && frontend_available && return 0
  if [[ -f "../package.json" ]] && command -v npm >/dev/null 2>&1; then
    echo "Building the web UI from source (npm)…"
    ( cd .. && npm ci && npm run build:spa ) && frontend_available && return 0
  fi
  echo "WARNING: no web UI available — serving API only. Install Node to build it, or run" >&2
  echo "         '${PIP_INSTALL[*]} inline-studio-frontend' once it's published." >&2
  return 0
}

ensure_frontend
exec "${PY_CMD[@]}" -m inline_core.server
