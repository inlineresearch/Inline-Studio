# syntax=docker/dockerfile:1.7

# Inline Studio in one image: the SPA served by Inline Core, plus JupyterLab and a file browser.
# linux/amd64, CUDA 13.0, Turing through Blackwell. Needs an R580+ driver.

# ------------------------------------------------------------------ stage 1: the SPA
FROM node:22-slim AS web
WORKDIR /build

# NODE_ENV must not be production or npm ci skips vite, which is a devDependency.
ENV HUSKY=0

# Lockfile first so editing src/ does not re-run the install.
COPY package.json package-lock.json ./
RUN npm ci

COPY vite.config.spa.ts tsconfig.json tsconfig.web.json tailwind.config.js postcss.config.js ./
COPY src ./src
RUN npm run build:spa

# -------------------------------------------------------------- stage 2: the runtime
# base, not cudnn-runtime: torch ships cuda-toolkit and cudnn as pip wheels already.
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

# ffmpeg is for ffprobe, which imageio-ffmpeg does not bundle. git is a runtime
# requirement of the Extensions dialog. The rest are opencv and onnxruntime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0t64 \
      libgomp1 \
      libsm6 \
      libxext6 \
      tini \
 && rm -rf /var/lib/apt/lists/*

# Pinned, never :latest. uv also stays for runtime extension installs.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /usr/local/bin/

# The default python dir is under /root at mode 0700. Hardlinks across a cache mount fail.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

# 3.11 to match core/.python-version, not Ubuntu's 3.12.
RUN uv python install 3.11 \
 && uv venv --python 3.11 --python-preference only-managed /opt/venv

WORKDIR /opt/inline-studio/core

# torch first, so .[all] finds it satisfied and never swaps it for a PyPI wheel.
# --index-url is exclusive; --extra-index-url picks the highest version across indexes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.13.0 torchvision==0.28.0

# Stub package so this multi-GB layer is keyed on pyproject.toml, not on the source.
COPY core/pyproject.toml core/README.md ./
RUN mkdir -p src/inline_core && touch src/inline_core/__init__.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-sources '.[all]'

# Not editable: the path hook it installs breaks under any bind mount over this dir.
COPY core/src ./src
COPY core/main.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-sources --no-deps --reinstall .

# bitsandbytes dlopen()s CUDA itself, so libcudart must resolve without importing torch.
RUN <<'SH'
set -eu
/opt/venv/bin/python -c '
import glob, os, site
sp = site.getsitepackages()[0]
dirs = {os.path.dirname(p) for pat in ("nvidia/**/*.so*", "cuda*/**/*.so*")
        for p in glob.glob(os.path.join(sp, pat), recursive=True)}
dirs.add(os.path.join(sp, "torch", "lib"))
open("/etc/ld.so.conf.d/00-torch-cuda.conf", "w").write("\n".join(sorted(dirs)) + "\n")
'
ldconfig
SH

# Fail the build, not the pod, on a CPU wheel or a mismatched torch/torchvision pair.
RUN <<'SH'
set -eu
/opt/venv/bin/python -c '
import torch, torchvision
assert torch.version.cuda and torch.version.cuda.startswith("13"), torch.version.cuda
assert "+cu130" in torch.__version__, torch.__version__
import accelerate, av, bitsandbytes, cv2, diffusers, inline_core, onnxruntime, peft
print("ok", torch.__version__, torchvision.__version__)
'
SH

# Own venv so JupyterLab's pins never compete with diffusers.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.11 --python-preference only-managed /opt/jupyter \
 && uv pip install --python /opt/jupyter/bin/python jupyterlab==4.6.2

ARG FILEBROWSER_VERSION=v2.63.23
RUN curl -fsSL "https://github.com/filebrowser/filebrowser/releases/download/${FILEBROWSER_VERSION}/linux-amd64-filebrowser.tar.gz" \
      | tar -xz -C /usr/local/bin filebrowser \
 && chmod +x /usr/local/bin/filebrowser

# Last, because the UI changes on every commit.
COPY --from=web /build/dist-web /opt/inline-studio/dist-web
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Every writable path is on /workspace so a pod can be redeployed without losing weights.
# HF_HOME has no in-app knob, but the captioner and annotators fall back to that cache.
# FORWARDED_ALLOW_IPS: uvicorn defaults to 127.0.0.1, so behind a TLS proxy it drops
# X-Forwarded-Proto and StaticFiles redirects to http:// from an https:// page.
# HF_HUB_DISABLE_XET=0 overrides Core's default. Xet costs per-chunk progress, but without it
# Hugging Face refuses any file over ~50GB, which is every MiniMax H3 transformer.
ENV INLINE_HOST=0.0.0.0 \
    INLINE_PORT=8848 \
    INLINE_FRONTEND_ROOT=/opt/inline-studio/dist-web \
    INLINE_MODELS_DIR=/workspace/models \
    INLINE_DATA_DIR=/workspace/.inline \
    INLINE_EXTENSIONS_DIR=/workspace/extensions \
    INLINE_STUDIO_DATA_DIR=/workspace/inline-studio \
    INLINE_STUDIO_WORKSPACE_DIR=/workspace/projects \
    HF_HOME=/workspace/huggingface \
    TORCH_HOME=/workspace/torch \
    XDG_CACHE_HOME=/workspace/.cache \
    UV_CACHE_DIR=/workspace/.cache/uv \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    FORWARDED_ALLOW_IPS=* \
    HF_HUB_DISABLE_XET=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    ENABLE_JUPYTER=1 \
    ENABLE_FILEBROWSER=1

WORKDIR /workspace
EXPOSE 8848 8888 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8848/v1/health || exit 1

# -g forwards signals to the process group, so the side services stop with the server.
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/entrypoint.sh"]
