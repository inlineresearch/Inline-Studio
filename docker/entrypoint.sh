#!/usr/bin/env bash
# Container entrypoint. PID 1 is `tini -g`, so one SIGTERM reaches the whole process group.
set -euo pipefail

log() { printf '[inline-studio] %s\n' "$*"; }

WORKSPACE="${INLINE_DOCKER_WORKSPACE:-/workspace}"

# Without a real volume here, weights, projects and trained LoRAs die with the container.
if ! grep -q " ${WORKSPACE} " /proc/mounts; then
  log "WARNING: ${WORKSPACE} is not a mounted volume."
  log "WARNING: models, projects and trained LoRAs will be DESTROYED when this container is removed."
  log "WARNING: attach a network volume (RunPod) or a bind mount (docker run -v) at ${WORKSPACE}."
fi

mkdir -p \
  "$INLINE_MODELS_DIR" "$INLINE_DATA_DIR" "$INLINE_EXTENSIONS_DIR" \
  "$INLINE_STUDIO_DATA_DIR" "$INLINE_STUDIO_WORKSPACE_DIR" \
  "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$TRITON_CACHE_DIR"

# Seeded so a fresh volume shows where a downloaded file is meant to land.
for category in diffusion_models vae text_encoders loras controlnet checkpoints \
                clip_vision upscale_models embeddings annotators; do
  mkdir -p "$INLINE_MODELS_DIR/$category"
done

# CUDA 13 needs R580+. Say so here rather than deep inside a model load.
if command -v nvidia-smi >/dev/null 2>&1; then
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  driver_major="${driver%%.*}"
  if [[ "$driver_major" =~ ^[0-9]+$ ]] && (( driver_major < 580 )); then
    log "WARNING: NVIDIA driver ${driver} predates R580, which CUDA 13 needs."
    log "WARNING: generation and training will fail. Redeploy on a host with CUDA 13.0 or newer."
  fi
else
  log "WARNING: no nvidia-smi, so no GPU is visible. Did you start this without GPU access?"
fi

if [ "${ENABLE_JUPYTER:-1}" != "0" ]; then
  token="${JUPYTER_TOKEN:-${JUPYTER_PASSWORD:-}}"
  if [ -z "$token" ]; then
    token="$(/opt/venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(24))')"
    log "JupyterLab token (generated, set JUPYTER_PASSWORD to choose your own): ${token}"
  fi
  /opt/jupyter/bin/jupyter lab \
    --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.token="$token" \
    --ServerApp.root_dir="$WORKSPACE" \
    --ServerApp.trust_xheaders=True \
    --ServerApp.allow_origin='*' \
    --ServerApp.disable_check_xsrf=True \
    >"$WORKSPACE/jupyter.log" 2>&1 &
  log "JupyterLab on :8888, rooted at ${WORKSPACE}, log at ${WORKSPACE}/jupyter.log"
fi

# Never --noauth: the proxy URL is public and this serves the whole volume.
if [ "${ENABLE_FILEBROWSER:-1}" != "0" ]; then
  fb_db="$WORKSPACE/.filebrowser.db"
  fb_user="${FILEBROWSER_USER:-admin}"
  fb_password="${FILEBROWSER_PASSWORD:-}"
  if [ -z "$fb_password" ]; then
    fb_password="$(/opt/venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(12))')"
    log "File browser password (generated, set FILEBROWSER_PASSWORD to choose your own): ${fb_password}"
  fi
  if [ ! -f "$fb_db" ]; then
    filebrowser -d "$fb_db" config init >/dev/null
    filebrowser -d "$fb_db" config set --auth.method=json --root="$WORKSPACE" >/dev/null
    filebrowser -d "$fb_db" users add "$fb_user" "$fb_password" --perm.admin >/dev/null
  else
    filebrowser -d "$fb_db" users update "$fb_user" --password "$fb_password" >/dev/null 2>&1 || true
  fi
  filebrowser -d "$fb_db" -r "$WORKSPACE" -a 0.0.0.0 -p 8080 \
    >"$WORKSPACE/filebrowser.log" 2>&1 &
  log "File browser on :8080 as ${fb_user}, rooted at ${WORKSPACE}"
fi

# exec so the server is tini's direct child and owns the exit code.
log "Starting Inline Studio on ${INLINE_HOST}:${INLINE_PORT}"
exec /opt/venv/bin/inline-studio "$@"
