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
#   ./webui.sh --install --extra runtime    # set up the venv with the local model runtime, then exit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="127.0.0.1"
PORT="8848"
EXTRAS="server"
RUN_INSTALL=0
DEV_MODE=0
FORCE_REBUILD=0
SMART_MEMORY=0

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
  --lowvram              tight-VRAM profile (slicing + tiling + int8, weights resident)
  --smart-memory         spread a too-big model across VRAM + RAM + CPU: graduated CPU offload
                         (model, or sequential on a tiny GPU) + int8 quant so the offloaded half
                         fits in RAM. Slower per image, but runs where full-resident OOMs.
  --cpu                  force the CPU profile
  --profile NAME         set the profile explicitly (gpu-max | lowvram | cpu)
  --vram-budget GB       treat the GPU as having GB of usable VRAM

Paths
  --models-dir PATH      where weights are scanned from (default ./models)
  --data-dir PATH        where runs and takes are written (default ./.inline)

Setup
  --install              create ./.venv (via uv) and install, then exit
  --extra NAME           add an install extra (repeatable): runtime, parallel, server
  -h, --help             show this help

Development
  --dev                  live-reload dev loop: run Core (API) in the background and the Vite HMR
                         server, then edit src/ and see changes instantly. Open http://localhost:5173
                         (NOT the Core port). Needs the repo checkout + Node/npm.
  --rebuild              force a fresh SPA build (npm run build:spa) from source and serve that local
                         build on the one port. Use after code changes when not running --dev.

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
    --smart-memory)
      # Spread a too-big model across VRAM + RAM + CPU. Force the lowvram profile so the offload +
      # int8 machinery engages. (The expandable allocator that cuts CUDA fragmentation OOMs is now
      # set unconditionally below, so every run — not just smart memory — benefits.)
      SMART_MEMORY=1
      export INLINE_SMART_MEMORY="1"
      export INLINE_PROFILE="${INLINE_PROFILE:-lowvram}"
      shift ;;
    --cpu) export INLINE_PROFILE="cpu"; shift ;;
    --profile) export INLINE_PROFILE="${2:?--profile needs a name}"; shift 2 ;;
    --vram-budget) export INLINE_VRAM_BUDGET_GB="${2:?--vram-budget needs a number}"; shift 2 ;;
    --models-dir) export INLINE_MODELS_DIR="${2:?--models-dir needs a path}"; shift 2 ;;
    --data-dir) export INLINE_DATA_DIR="${2:?--data-dir needs a path}"; shift 2 ;;
    --install) RUN_INSTALL=1; shift ;;
    --extra) EXTRAS="$EXTRAS,${2:?--extra needs a name}"; shift 2 ;;
    --dev) DEV_MODE=1; shift ;;
    --rebuild) FORCE_REBUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

export INLINE_HOST="$HOST"
export INLINE_PORT="$PORT"

# Always use PyTorch's expandable CUDA segments (unless the user set their own config): it lets the
# allocator grow/shrink a single reservation instead of fragmenting into many, which is the
# difference between a small allocation failing "with VRAM still free" and succeeding — a common
# low-VRAM (e.g. T4) OOM. Harmless on CPU-only runs (torch just ignores it).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$RUN_INSTALL" -eq 1 ]]; then
  command -v uv >/dev/null 2>&1 || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
  uv venv
  # Pick the right torch wheel. PyPI's default torch is CPU-only on Windows (Linux wheels bundle
  # CUDA), so installing blind there yields a working install that generates on the CPU ~100x
  # slower, with no error. When an NVIDIA GPU is present, resolve torch from the CUDA index.
  TORCH_INDEX=()
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    TORCH_INDEX=(--extra-index-url https://download.pytorch.org/whl/cu124)
    echo "NVIDIA GPU detected - installing the CUDA build of PyTorch."
  else
    echo "No NVIDIA GPU detected - installing the default (CPU) build of PyTorch."
  fi
  uv pip install "${TORCH_INDEX[@]}" -e ".[$EXTRAS]"
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

# Force a fresh local SPA build and point Core at it (overriding any installed frontend package), so a
# one-port run reflects the current source. Used by --rebuild.
rebuild_frontend() {
  if [[ ! -f "../package.json" ]] || ! command -v npm >/dev/null 2>&1; then
    echo "--rebuild needs the repo checkout and npm; skipping the rebuild." >&2
    return 0
  fi
  echo "Rebuilding the web UI from source (npm run build:spa)…"
  ( cd .. && { [[ -d node_modules ]] || npm ci; } && npm run build:spa )
  INLINE_FRONTEND_ROOT="$(cd .. && pwd)/dist-web"; export INLINE_FRONTEND_ROOT
}

# Live-reload dev loop: Core (API) in the background, Vite (HMR) in the foreground proxying to it. The
# user opens the Vite port (5173), not Core's — edits to src/ hot-reload without a rebuild. Used by --dev.
run_dev() {
  if [[ ! -f "../package.json" ]] || ! command -v npm >/dev/null 2>&1; then
    echo "--dev needs the repo checkout and Node/npm (https://nodejs.org/)." >&2
    exit 1
  fi
  ( cd .. && { [[ -d node_modules ]] || npm ci; } )
  echo "Starting Inline Core (API) on ${HOST}:${PORT} in the background…"
  "${PY_CMD[@]}" -m inline_core.server &
  local core_pid=$!
  trap 'kill "$core_pid" 2>/dev/null || true' EXIT INT TERM
  echo "Starting the Vite dev server (HMR). Open http://localhost:5173  (NOT :${PORT})"
  ( cd .. && INLINE_CORE_URL="http://127.0.0.1:${PORT}" npm run dev:web )
}

# Smart memory needs torchao (int8 weight-only quant). It ships in the runtime extra, but an older
# venv predates it — install it on demand at launch so --smart-memory just works. Best-effort: if the
# install fails, generation still runs, only without int8 (the loader logs and loads full precision).
ensure_smart_memory_deps() {
  "${PY_CMD[@]}" -c "import torchao" >/dev/null 2>&1 && return 0
  echo "Smart memory: installing torchao (int8 quantization)…"
  "${PIP_INSTALL[@]}" torchao >/dev/null 2>&1 \
    && echo "Installed torchao." \
    || echo "WARNING: could not install torchao; smart memory will run without int8 quant." >&2
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

if [[ "$DEV_MODE" -eq 1 ]]; then
  run_dev
  exit 0
fi

if [[ "$FORCE_REBUILD" -eq 1 ]]; then
  rebuild_frontend
fi

if [[ "$SMART_MEMORY" -eq 1 ]]; then
  ensure_smart_memory_deps
fi

ensure_frontend
exec "${PY_CMD[@]}" -m inline_core.server
