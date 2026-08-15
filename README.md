<h1 align="center">Inline Studio</h1>

<h3 align="center">AI filmmaking on a node canvas</h3>

<p align="center">Inline Studio is a free, open-source app for AI filmmakers. Generate locally on your own GPU and train your own LoRAs on the same node canvas, with the built-in Inline Core engine and hosted fal models. Build a whole visual pipeline from moodboard to final cut, and every render is kept as a versioned, non-destructive take.</p>

<p align="center">
  <a href="LICENSE"><img alt="License: GPLv3" src="https://img.shields.io/badge/License-GPLv3-blue?style=for-the-badge"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="../../releases/latest"><img alt="Latest release" src="https://img.shields.io/badge/Release-v1.2.71-blue?style=for-the-badge"></a>
  <a href="https://discord.gg/cSUS88VdY9"><img alt="Join our Discord" src="https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white&style=for-the-badge"></a>
</p>

![Inline Studio node canvas showing a generative AI film pipeline with frames, takes, and connectors](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/screenshot-dashboard-2.png)

[**New here? Check out our getting started guide →**](https://inlinestudio.art/getting-started)

**Contents:** [What is Inline Studio?](#what-is-inline-studio) · [Get Started](#get-started) ·
[Features](#features) · [LoRA training](#lora-training) · [How it works](#how-it-works) ·
[Two ways to generate](#two-ways-to-generate) · [Inline Core engine](#inline-core-generation-engine)
([Krea 2](#krea-2), [FLUX.2](#flux2), [MiniMax H3](#minimax-h3), [LTX-2.5](#ltx-25), [ControlNet](#controlnet)) ·
[API Nodes](#api-nodes) · [FAQ](#faq) · [Contributing](#contributing)

## What is Inline Studio?

Inline Studio is a free, open-source app for **AI filmmaking on a node canvas**, powered by the built-in **Inline Core** engine (local diffusion models) and hosted [fal](https://fal.ai) models. It gives AI filmmakers a free-form canvas to build a whole visual pipeline, from moodboard to final cut.

- **Non-destructive by default** - every render is kept as a versioned take; generating again adds one, nothing is overwritten.
- **Local diffusion generation engine** - the built-in Inline Core engine runs popular diffusion models locally, on your own GPU, from a single model file, no external server. Currently supported: **Z-Image Turbo**, **Krea 2** (RAW + Turbo), **FLUX.2**, and **MiniMax H3** and **LTX-2.5** for video with sound.
- **Train LoRAs locally** - the Trainer canvas fine-tunes Z-Image, Krea 2, FLUX.2 or MiniMax H3 on your own images, on your own GPU. H3 also trains on short video clips, so a LoRA can learn motion and not just look. With a 4-bit base, Krea 2 trains at 512px inside about 12GB, so a 16GB card can train a LoRA for a 26GB model. See [LoRA training](#lora-training).
- **Hosted models via API Nodes** - reach for closed models with no GPU and no setup for instant creative range; see [API Nodes](#api-nodes).
- **Mix both in the same film** - Inline Studio handles everything around the render: exploring options, keeping what works, and shaping a repeatable process you can iterate on and share.

It runs as a **single process on one port**: the Inline Core engine (Python) serves the web UI _and_ does the generation: `python core/main.py` and open the browser. No desktop install, no separate backend.

**Who it's for:** AI filmmakers, motion artists, and generative creators who want to make AI short films and longer cuts without losing every good version along the way.

## Get Started

The built web UI ships as a Python package, so there is no Node and no build step. You need two
things: [Python 3.11+](https://python.org) and [**uv**](https://github.com/astral-sh/uv), which the
installer uses to create the environment and resolve packages. `--install` stops with
`uv not found` if it is missing.

<details>
<summary><b>Install uv first</b> (one line, no Python needed)</summary>

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or use whatever you already have: `brew install uv`, `pipx install uv`, `pip install uv`. Full
options: [docs.astral.sh/uv/getting-started/installation](https://docs.astral.sh/uv/getting-started/installation/).

</details>

Then **`--install --extra all` is the single command that installs everything** - the engine, the local model runtime, the LoRA trainer, and the UI. On an NVIDIA machine it reads the GPU's compute capability and pulls the CUDA build of PyTorch that has kernels for it, RTX 50-series included.

[**No GPU? Deploy Inline Studio on RunPod →**](https://console.runpod.io/deploy?template=c0qkyaypuv&ref=hs2l4qhc)

**macOS / Linux:**

```bash
git clone https://github.com/inlineresearch/Inline-Studio.git
cd Inline-Studio/core
./webui.sh --install --extra all   # one command: installs everything
./webui.sh                         # then run, on http://127.0.0.1:8848
```

**Windows** (use `webui.bat` - `webui.sh` is a bash script and won't run in PowerShell; you can also double-click it):

```powershell
git clone https://github.com/inlineresearch/Inline-Studio.git
cd Inline-Studio\core
.\webui.bat --install --extra all
.\webui.bat

rem If the CUDA build turns out wrong for your card, name the index yourself:
.\webui.bat --install --extra all --torch-index cu130
```

That's it: `--install` sets up the environment and installs everything once, then `webui.sh` / `webui.bat` runs the app on one port. See **[Command-line options](#command-line-options)** for every flag (`--listen`, `--port`, `--lowvram`, `--multi-gpu`, …).

Everything lands in `core/.venv`, which Inline Studio owns. If you already have another virtualenv or conda env activated in that shell (a ComfyUI one, say), it is left completely untouched - `--install` says so and carries on. Re-running `--install` is safe: an existing `core/.venv` is reused, so adding an extra later is just another `--install --extra NAME`.

Prefer pip over the launcher, or would rather not install uv at all? `pip install -r requirements.txt` (from the repo root) installs the whole app - engine, UI, model runtime, and trainer - from PyPI; then run `inline-studio`. This path uses no uv, but it also does not detect your GPU, so on Windows you may need to name the CUDA build yourself.

### Hardware support

<details>
<summary><b>GPU, CPU, Apple Silicon, and ROCm setup</b></summary>

Honest status - what's actually been run, versus what has a code path but no one has verified:

| Hardware                | Status                                                                                              | Extra steps                                                                                                                                                                                                                                                                                                                                               |
| ----------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NVIDIA, Linux**       | **Tested** - Z-Image Turbo 1024² on a T4 (16 GB); Krea 2 1024² and LoRA training on an L40S (48 GB) | None. `webui.sh --install` picks the CUDA build automatically.                                                                                                                                                                                                                                                                                            |
| **NVIDIA, Windows**     | Supported, needs one step                                                                           | Run `.\webui.bat --install` (the Windows launcher). PyPI's default `torch` is **CPU-only on Windows**, so `--install` reads your GPU's compute capability and pulls the matching CUDA build: `cu130` for RTX 50-series (Blackwell), `cu126` for everything older. Override it with `--torch-index` - see [RTX 50-series](#rtx-50-series-blackwell) below. |
| **Apple Silicon (MPS)** | Code path exists, **untested**                                                                      | None. int8 quantisation doesn't apply on MPS, so a model too big for unified memory won't fit.                                                                                                                                                                                                                                                            |
| **AMD (ROCm), Linux**   | **Untested** - reports welcome                                                                      | Needs a ROCm build of PyTorch - see [AMD (ROCm) setup](#amd-rocm-setup) below.                                                                                                                                                                                                                                                                            |
| **CPU only**            | Works, very slow                                                                                    | `./webui.sh --cpu` (Windows: `.\webui.bat --cpu`)                                                                                                                                                                                                                                                                                                         |

#### RTX 50-series (Blackwell)

RTX 50-series cards (5060/5070/5080/5090 and the RTX PRO Blackwell line) are compute capability **sm_120**, and no PyTorch wheel built for CUDA 12.4 or 12.6 has kernels for them. `--install` handles this: it reads the compute capability off the driver and picks `cu130`, so a plain `.\webui.bat --install --extra all` is all you need.

**Old driver?** CUDA 13 needs driver R580 or newer. If yours predates it, `--install` picks `cu128` for you and says so: cu128 still has `sm_120` but is **frozen at torch 2.11** and will never update, so updating the driver and re-running `--install` is worth doing when you can.

To name an index yourself:

```powershell
.\webui.bat --install --extra all --torch-index cu130

rem Or set it once for the shell, same effect
set INLINE_TORCH_INDEX=cu130
```

`--torch-index` takes a short name (`cu130`, `cu128`, `cu126`), a full index URL, or `cpu` to force the CPU-only build. `webui.sh` takes the same flag. Naming it explicitly also **replaces** an already-installed torch, which a plain re-run will not do, so you rarely need `--recreate`.

Not sure what you have? `.\webui.bat --print-torch-index` prints what the driver reported and which index would be used, and installs nothing. Paste that into a bug report. If the installed build turns out to have no kernels for your card, Core also says so by name at startup rather than leaving you with PyTorch's own `sm_120 is not compatible` warning.

#### AMD (ROCm) setup

Nobody has verified Inline Studio on AMD yet, so treat this as a starting point rather than a supported path. Install everything normally **first**, then replace PyTorch with the ROCm build - doing it in this order means nothing can quietly overwrite your ROCm torch afterwards:

```bash
cd core
./webui.sh --install --extra runtime         # engine + runtime (pulls the default PyPI torch)

# Replace torch with the ROCm build. Pick the index that matches YOUR ROCm version -
# check https://pytorch.org/get-started/locally/ (rocm6.2 shown here as an example).
# --python pins the install to Inline Studio's venv, whatever is activated in your shell.
uv pip install --python .venv/bin/python --force-reinstall \
  --index-url https://download.pytorch.org/whl/rocm6.2 torch

# Verify you actually got a ROCm build (hip should print a version, not None):
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
```

Then run `./webui.sh` as usual.

Three gotchas:

- **Don't run `uv sync` afterwards** - it re-resolves the environment against the lockfile and will pull the PyPI torch back over your ROCm build. Use `uv pip install --python .venv/bin/python` for follow-up installs. The same applies to a hand-picked CUDA index.
- **Don't pass `--recreate`** - it rebuilds `.venv` from scratch and your ROCm torch goes with it. A plain `--install` re-run reuses the venv and is safe.
- ROCm presents itself through `torch.cuda`, so the engine will treat it as a CUDA device and may largely work. But the dtype heuristics key off **NVIDIA** compute capability (`< 8.0` → fp16), which is meaningless on RDNA/CDNA, and the int8 (torchao) path is unverified on ROCm. If it works - or doesn't - [open an issue](https://github.com/inlineresearch/Inline-Studio/issues); that's the fastest way to get AMD properly supported.

**Known limits, so you can judge before installing:**

- **Local model coverage is Z-Image Turbo, Krea 2 and FLUX.2** today. SDXL and others are planned; hosted models via [API Nodes](#api-nodes) need no GPU at all.
- **Krea 2 is a 12.9B model and needs a big card to generate.** The bf16 checkpoint is 26 GB on disk, and generation peaks around 36 GB at 1024 with guidance on, so a 40 GB+ GPU is the practical floor for inference. **Training is cheaper than generating**, because the 4-bit base path puts Krea 2 LoRA training at 512 inside 12 GB - see [Benchmark results](TRAINING.md#benchmark-results). Z-Image remains the low-VRAM path for generation.
- **1024² with Guidance (CFG) above 0 needs more than 16 GB.** CFG runs the prompt and negative prompt together, doubling the denoise. Z-Image Turbo is distilled to run CFG-free - at Guidance 0, 1024² fits in ~11.5 GB. FLUX.2 klein 4B peaks near 17.9 GB at bf16, so 24 GB holds it resident and a 16 GB card runs it quantized instead.

</details>

### From source (for UI development)

<details>
<summary><b>Build the UI and run the engine locally</b></summary>

To hack on the web UI you need [Node.js](https://nodejs.org) 20.11+ as well, and you serve a local SPA build:

```bash
git clone https://github.com/inlineresearch/Inline-Studio.git && cd Inline-Studio

# 1. Build the web UI
npm install
npm run build:spa                        # -> dist-web/

# 2. Set up + run the engine, serving your local build
cd core
uv sync --extra server --extra runtime   # server + the local model runtime (torch/diffusers)
uv run python main.py --front-end-root ../dist-web
```

`uv sync` here manages `core/.venv` as a project environment - it is exact, so it removes anything not in the lockfile (including the `inline-studio-frontend` package a previous `--install` may have added, which doesn't matter when you're serving `--front-end-root ../dist-web`).

Then open **http://127.0.0.1:8848**. Add your [fal.ai API key](https://fal.ai/dashboard/keys) in Settings for hosted models, and set up local generation as in [Two ways to generate](#two-ways-to-generate). The canvas and planning work without any models.

**Hot-reload:** run the engine as above, then in another terminal `npm run dev:web` (Vite serves the UI with HMR and proxies API calls to Core).

</details>

### Command-line options

The friendly launcher (in `core/`) maps flags onto the engine's `INLINE_*` environment knobs: `webui.sh` on macOS/Linux, `webui.bat` on Windows. `core/main.py` takes the same flags when you run the engine directly. `./webui.sh --help` (or `.\webui.bat --help`) lists them all.

<details>
<summary><strong>Show all command-line flags</strong></summary>

| `webui.sh` / `main.py` flag        | Env var                  | What it does                                                                                                                                                                                                           |
| ---------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--listen`                         | `INLINE_HOST=0.0.0.0`    | Bind all interfaces so other machines can reach it                                                                                                                                                                     |
| `--host ADDR`                      | `INLINE_HOST`            | Bind a specific address (default `127.0.0.1`)                                                                                                                                                                          |
| `--port N`                         | `INLINE_PORT`            | Port to serve on (default `8848`)                                                                                                                                                                                      |
| `--models-dir PATH`                | `INLINE_MODELS_DIR`      | Where model weights are scanned from (default `./models`)                                                                                                                                                              |
| `--data-dir PATH`                  | `INLINE_DATA_DIR`        | Where runs + takes are written (default `./.inline`)                                                                                                                                                                   |
| `--lowvram`                        | `INLINE_PROFILE=lowvram` | Tight-VRAM profile (VAE tiling/slicing, attention slicing)                                                                                                                                                             |
| `--cpu`                            | `INLINE_PROFILE=cpu`     | Force CPU generation                                                                                                                                                                                                   |
| `--profile NAME`                   | `INLINE_PROFILE`         | Set the profile explicitly: `gpu-max` \| `lowvram` \| `cpu`                                                                                                                                                            |
| `--vram-budget GB`                 | `INLINE_VRAM_BUDGET_GB`  | Treat the GPU as having GB of usable VRAM                                                                                                                                                                              |
| `--multi-gpu [SPEC]`               | `INLINE_PARALLEL`        | Split one image's denoise across GPUs (e.g. `pipefusion=2`); auto with 2+ GPUs                                                                                                                                         |
| `--front-end-root DIR` _(main.py)_ | `INLINE_FRONTEND_ROOT`   | Serve a local SPA build instead of the installed UI package (dev)                                                                                                                                                      |
| `--rebuild` _(webui.sh)_           | n/a                      | Force a fresh SPA build (`npm run build:spa`) from source and serve it on the one port; use after UI changes when not running `--dev`. Needs the repo checkout + Node/npm                                              |
| `--torch-index WHICH`              | `INLINE_TORCH_INDEX`     | With `--install`, override the PyTorch wheel index picked from your GPU's compute capability. A short name (`cu130`, `cu128`, `cu126`), a full index URL, or `cpu`. Naming it also replaces an already-installed torch |
| `--print-torch-index`              | n/a                      | Print what the GPU probe read and which index would be used, then exit without installing. The one line to paste into a bug report                                                                                     |

</details>

`webui.sh` also has `--install` / `--extra NAME` to set up the venv, plus `--torch-index WHICH` (`INLINE_TORCH_INDEX`) to override the PyTorch wheel index picked from your GPU's compute capability, `--recreate` (rebuild `.venv` from scratch) and `--use-active-env` (install into / run from the environment activated in your shell instead of `.venv`). New to Inline Studio? The [Getting Started guide](https://inlinestudio.art/getting-started) walks you through your first render.

## Features

- Free-form node canvas
- Versioned, non-destructive takes
- Chain frames into a generative pipeline
- Video editing on the canvas
- Local generation, built in
- Multi-reference composition
- Consistent characters from a portable `.char` file
- Train your own LoRAs locally
- API Nodes for hosted models
- Community extensions
- Free & open source (GPL-3.0)

[**Follow our Animated Short Film with LTX 2.3 and GPT Image Generation tutorial →**](https://inlinestudio.art/projects/circuit-race)

## Consistent & portable characters without Lora Training

Getting the same person across shots normally means training a LoRA per character, or re-wiring the same reference images into every node by hand. Instead, build a character once and pick it from a dropdown.

Drop in a photo or two and Inline Studio compiles a **`.char`**: a portable file holding your references and an identity fingerprint. Pick it on a FLUX.2 node and generate. You describe the scene, the references carry the likeness. Nothing is trained and no adapter is installed, and every take comes back with a continuity score out of 100.

![Two reference photos compiled into a portable character file, then the same person generated in an office, a cafe, a park, a street and at a lakeside](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/character_showcase.png)

![How a character is encoded, applied and scored: YuNet detects faces, SFace and DINOv2 build the fingerprint, the references are packed onto the noise sequence, and each take is scored](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/char-flow.png)

[**How characters work, in detail →**](https://inlinestudio.art/characters)

## LoRA training

Train a LoRA on your own images without leaving the app, on your own GPU, with no cloud step. The **Trainer** tab is a second canvas: wire up the nodes, press Start, and watch it run. When the run finishes, the `.safetensors` lands in `models/loras/`, where the LoRA loader node picks it up automatically, so you can generate with it over in the Studio tab straight away.

![Inline Studio Trainer tab showing the LoRA training node graph with a dataset, live logs, and a loss curve](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/lora-trainer.png)

Five nodes, wired left to right, with the hyperparameters behind an Adjust button so the node face stays a status surface:

```
[ Load Dataset ] --> [ Caption ] --> [ Train LoRA ] --> [ Graph ]
                                          |
                                          +--> Resources (VRAM monitor)
```

### Does my card fit?

Peak VRAM at 512px, 12 steps, rank 16, batch 1, gradient checkpointing on:

| Architecture              | 512px peak | 16GB card   |
| ------------------------- | ---------- | ----------- |
| FLUX.2 (klein Base 4B)    | ~8.6GB     | yes         |
| Krea 2 (4-bit base)       | ~11.9GB    | yes         |
| Z-Image                   | ~13.4GB    | yes         |
| MiniMax H3 (4-bit, video) | ~20.6GB    | yes, slowly |
| LTX-2.5 (22B, video)      | ~42GB      | no, 48GB    |

LTX-2.5 is the exception to the row above it. It is a 22B base against MiniMax H3's 33B, but it has no 4-bit training path here, so nothing shrinks it: the base alone is 38GB once loaded and training peaks at 42GB, measured on an L40S. A 48GB card is the floor, and a 24GB one cannot run it at any resolution. Generating is far more forgiving than training here, because LTX streams its own weights: [Benchmark results](TRAINING.md#benchmark-results).

Training is cheaper than generating, and a LoRA trained at 512 applies at any generation resolution. Full per-card matrix, both resolutions and the timings: [Benchmark results](TRAINING.md#benchmark-results).

If you installed with `--extra all` from [Get Started](#get-started), the trainer is ready. Otherwise:

```bash
cd core
./webui.sh --install --extra training   # Windows: .\webui.bat --install --extra training
```

For a worked example, see [`inlineresearch/skin-lora-krea-2-raw`](https://huggingface.co/inlineresearch/skin-lora-krea-2-raw), a photorealistic skin LoRA trained here on the Krea 2 RAW base from the 26 image and caption pairs published as [`inlineresearch/krea2-skin-lora`](https://huggingface.co/datasets/inlineresearch/krea2-skin-lora).

**[TRAINING.md](TRAINING.md) is the full reference:** [which base to train on](TRAINING.md#architecture-and-base-model-modes) · [measured benchmarks](TRAINING.md#benchmark-results) · [datasets and outputs](TRAINING.md#datasets-and-outputs) · [stop and resume](TRAINING.md#stop-and-resume) · [trigger words](TRAINING.md#trigger-words) · [base precision](TRAINING.md#base-precision)

## How it works

Generating a single frame is the easy part. The work that makes an AI film is what comes after: exploring options, keeping what's good, and shaping a repeatable process out of it. Inline Studio is the layer where that happens, organised around one model:

### Export the whole pipeline, not just the final render

From the home screen, **Export** zips a project into one archive. Import it on the other side and you get everything back: the inputs (every imported asset), the outputs (all the generated takes), and the graph that turned one into the other. Whoever opens it can re-run the pipeline exactly and keep iterating.

## Two ways to generate

Pick whatever fits the shot, and mix both in one film. However you render, the frame keeps its full take history, so you never lose a good version.

| How you render                          | What it's like                                                                                                                                                                                                                                                                                                                                                                                                    | What you need                                                                                                                                                                                                             |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Local GPU: Inline Core** _(built in)_ | Drop a **Z-Image Turbo**, **Krea 2**, **FLUX.2**, **MiniMax H3** or **LTX-2.5** node, wire a prompt, hit Run: one node, no loader/sampler wiring. A single `.safetensors` is usually all you bring (a prequantized build can be a diffusers folder); the engine pairs it with a VAE + text-encoder and downloads nothing behind your back. Two or more GPUs? It can split one image's denoise across them (xDiT). | Runs locally on your own GPU. No account, no external server. Low-VRAM friendly: it auto-fits the model to your card (streaming weights, int8, then NF4) so a model too big for full precision still runs, with no flags. |
| **Hosted: API Nodes**                   | Add a Generate node and pick a model: hosted, closed models across image, video, and audio. No GPU, instant range. See [API Nodes](#api-nodes) for the model list and providers.                                                                                                                                                                                                                                  | A provider key (currently [fal](https://fal.ai/dashboard/keys)); it stays on your machine, and you pay per render (each node estimates the price first).                                                                  |

For local generation, either drop a `.safetensors` into `core/models/diffusion_models/`, or add a model node and use its **model popup** (a blinking hint shows up when something's missing) to download the diffusion model, VAE, and text-encoder into `core/models/`, with visible progress. The canvas and planning work with no models at all. See [Krea 2](#krea-2) and [FLUX.2](#flux2) for those models' files and VRAM.

## Inline Core generation engine

Inline Core is a from-scratch generation engine for local rendering. It keeps the open node-graph model (a typed DAG of nodes and edges → immutable "takes"), and Inline Studio drives it as a single process.

![Z-Image Turbo generating locally on the Inline Core engine](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/zit.png)

- **One process, one port** - Inline Studio is a **web SPA** (React) served by Inline Core (a headless Python engine, in `core/`). `core/main.py` runs Core, which serves the built UI and is the app's backend.
- **Core owns the backend** - the browser reaches it over a small typed RPC/WebSocket contract; Core owns the project database, the filesystem, generation, and the ffmpeg timeline. No Electron, no separate Node server, nothing external to stand up.
- **Typed graph, checked before it runs** - named params and type-checked edges, so a bad graph is rejected at submit rather than dying part-way through a denoise.
- **Immutable takes** - regenerating adds a take; nothing is ever overwritten. The take history is the point.
- **Durable runs** - a run survives a restart, and progress streams over a WebSocket.
- **Graph decoupled from GPU work** - the graph is the unit of caching; a batched sampler is the unit of batching, grouping compatible jobs across requests.
- **A single device policy owns all placement** - device, dtype, offload, and attention, so the same graph runs on a 4090, a 6 GB laptop, pure CPU, or split across several GPUs without touching the graph.
- **Bring your own models, no hidden downloads** - a drop-in `models/` layout feeds a typed catalog and versioned node descriptors; nothing is fetched behind your back.

### Krea 2

[Krea 2](https://www.krea.ai/) is a 12.9B single-stream MMDiT, released as two checkpoints that work together: **RAW** is the undistilled base you fine-tune, **Turbo** is an 8-step distilled checkpoint you generate with. A LoRA trained on RAW applies to Turbo unchanged, which is the workflow both nodes are built around.

Both nodes read the ComfyUI-style files from [`Comfy-Org/Krea-2`](https://huggingface.co/Comfy-Org/Krea-2):

```
core/models/
  diffusion_models/  krea2_turbo_bf16.safetensors   <- for the Krea 2 Turbo node
                     krea2_raw_bf16.safetensors     <- for the Krea 2 RAW node (and training)
  text_encoders/     qwen3vl_4b_bf16.safetensors
  vae/               qwen_image_vae_diffusers.safetensors
  loras/             krea2_retroanime.safetensors   <- the official style LoRAs, optional
```

Two things are worth knowing before you download 26 GB twice:

- **Only the `bf16` builds load.** The `fp8_scaled`, `int8_convrot`, `mxfp8` and `nvfp4` files in that repo carry ComfyUI-specific scale tensors that only ComfyUI can read, and the node says so rather than failing deep in a load. Memory saving is the device policy's job instead.
- **The VAE is the diffusers-format one**, fetched from [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image). ComfyUI's `qwen_image_vae.safetensors` holds the same weights in a different module layout that diffusers cannot read. The node's model popup downloads the right file for you.

Nothing here needs a Hugging Face token: every repo involved is public, and Krea's own gated repos are never touched.

### FLUX.2

[FLUX.2](https://bfl.ai/blog/flux-2) from Black Forest Labs is the first family here that is natively **multi-reference**: reference images ride in the denoiser's token sequence, so "the character from image 1 wearing the jacket from image 2" is a first-class capability rather than a workaround.

One node, **FLUX.2**, covers the whole family. Pick a checkpoint in the node's Adjust sidebar and it identifies itself: klein 4B, klein 9B, either Base build, the KV variant, or dev. Steps and guidance default to "from model", so switching a distilled checkpoint for its Base build moves 4 steps at guidance 1.0 to 50 at guidance 4.0 without touching a setting.

- **Reference images** - wire one image to edit it, or several to compose from them. The node numbers them on its face, and the prompt addresses them by position. There is no denoise-strength slider because FLUX.2 has no img2img: a single reference _is_ the edit.
- **klein 4B is Apache 2.0** and the recommended starting point. It renders 1024px in four steps. At bf16 it is 16.1 GB of weights peaking around 17.9 GB, so a 24 GB card holds it resident; on a 16 GB card the fit ladder drops it to int8 or fp16 and it still runs. Its text encoder is the same Qwen3-4B file Z-Image already uses, so if you have run Z-Image you have most of it.
- **dev is 32B.** The fp8 build the model popup offers wants around 32 GB. Bring a prequantized NF4 folder yourself and it fits a 24 GB card instead, because the prompt is encoded first and the text encoder freed before the transformer loads.

Files come from the ungated ComfyUI repacks. The model popup fetches klein 4B, its text encoder and the VAE in one click, and offers the rest of the family as optional extras:

```
core/models/
  diffusion_models/  flux-2-klein-4b.safetensors                      <- the default, Apache 2.0
                     flux-2-klein-base-4b.safetensors                 <- the base build, for LoRA training
                     flux-2-klein-9b-int8-ConvRot-comfyui.safetensors <- klein 9B, int8, ~12 GB
                     flux2_dev_fp8mixed.safetensors                   <- dev, fp8, ~32 GB
  text_encoders/     qwen_3_4b.safetensors                            <- the 4B builds, shared with Z-Image
                     qwen_3_8b.safetensors                            <- klein 9B
                     mistral_3_small_flux2_fp8.safetensors            <- dev
  vae/               flux2-vae.safetensors
```

Each klein size needs its own text encoder, and dev uses Mistral-3 rather than Qwen3.

For dev on a 24 GB card, take the ungated [`diffusers/FLUX.2-dev-bnb-4bit`](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit) instead: an 18.1 GB NF4 transformer beside a 15.4 GB NF4 encoder, far cheaper than the fp8 single file. Clone it into `diffusion_models/` as a folder. The popup does not fetch it, and a diffusers folder is a valid checkpoint anywhere a single file is.

Worth knowing:

- **Prompts are prose, not tags.** FLUX.2 wants natural language, and keyword stuffing works against it. Word order carries weight.
- **Only the Base klein checkpoints take a negative prompt.** The distilled builds run no classifier-free guidance and dev is guidance-distilled, so a negative prompt is logged and ignored there rather than silently pretending to apply.
- **dev and every 9B build are non-commercial.** klein 4B, its Base build, and the VAE are Apache 2.0.

### MiniMax H3

[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is the first video model in Inline Core, and the first model here that generates a soundtrack rather than a silent clip. One transformer denoises the video and its 32 kHz stereo audio in a single pass, so a take is one MP4 with sound in it, ready to drop straight into the timeline.

Four nodes, because the inputs genuinely differ and a node should say what it takes:

- **Text → Video** for a shot from nothing but a prompt.
- **Image → Video** to bring a still you already like into motion.
- **First and Last Frame** to pin the opening frame, the closing frame, or both, and let the model fill in between.
- **Reference → Video** for consistency across shots. Wire up to nine images, three video clips and three audio clips, and address them by position in the prompt. Wiring order is the numbering you see on the node.

Output is 24 fps between 5 and 15 seconds at a 768 pixel short edge. Duration snaps to the frame grid the video decoder works in, so asking for 14.9 seconds renders 14.4 rather than failing. There is no guidance slider and no negative prompt, because the released checkpoints are guidance-distilled and neither exists.

```
core/models/
  diffusion_models/  minimax_h3_fl2va_bf16.safetensors   <- text, image, and first/last frame
                     minimax_h3_ref2va_bf16.safetensors  <- the reference node
  text_encoders/     MiniMax-H3-text-encoder/            <- Qwen3-VL-32B, a folder
                     MiniMax-H3-processor/               <- tokenizer and processor
  vae/               minimax_h3_video_vae_fp16.safetensors
                     minimax_h3_audio_vae_fp32.safetensors
```

Worth knowing before you start a download this size:

- **This is a big model.** 144 GB for the first three nodes, 210 GB with the reference node. A 33B transformer runs beside a 32B conditioner, but not at the same time: the prompt is encoded first, the conditioner then steps off the card, and the denoiser takes it for the whole denoise. Two things make that fit. The modulation weights, 40% of the transformer, are factorised at load and take it from 66.3 GB to 40.3 GB. The video VAE stays resident when the card has room, rather than streaming leaf by leaf. Measured on a 45 GB card: **a 10 second clip at 960x544 takes about 7.2 minutes, peaking at 38.9 GB VRAM and 46.7 GB of system RAM that cannot be reclaimed**, so plan on 64 GB of RAM. Smaller cards stream more and are slower. If that is out of reach, the same model is available as an API node with no setup at all.
- **Canvas is the biggest speed lever.** 960x544 renders about 2.3x faster per step than the trained 1344x768, and the difference is far larger than any other setting.
- **The bf16, `pruned` and `pruned_fp8_scaled` builds load.** The pruned builds ship the modulation branch already reduced and no timestep path, which Inline reads directly; `pruned_fp8_scaled` is 21.0 GB against 66.3 GB for the same model. The `int8_convrot` files still do not load, because their weights are stored rotated and only ComfyUI can undo that. The picker lists what it cannot read with the reason.
- **A smaller file is a smaller download, not a smaller model.** All of them occupy the same memory once loaded, so the choice is bandwidth and disk, not VRAM. **Training needs the bf16 build**: a pruned one has no timestep path to derive the modulation basis from, and it would save nothing anyway, since the base trains at 4-bit whichever file it starts from.
- **LoRAs work.** Every H3 node has a LoRA input, and adapters are fused into each block as it streams, before the factorisation and the quantisation. You can train one in the Trainer tab: see [LoRA training](#lora-training). An H3 LoRA is trained on stills and applies to video, so it carries look and style rather than motion.

### LTX-2.5

[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) is Lightricks' 22B open-weights video model, and the second here that generates a soundtrack rather than a silent clip. It ships as a split pack, one file per component, laid out in exactly the folders Inline Core already scans - so setting it up is dropping files in, not assembling a repo.

Three nodes, because the inputs differ and a node should say what it takes:

- **Text → Video** for a shot from nothing but a prompt.
- **Image → Video** to bring a still you already like into motion.
- **First and Last Frame** to pin the opening and closing frames and let the model fill in between.

Output is 24 fps between 1 and 20 seconds. Duration snaps onto the frame grid the video decoder works in, so 5 seconds renders 121 frames. Width and height snap to a multiple of 64, because the second stage renders at full size from a first stage at half, and both halves have to be legal.

Two modes on every node. **Fast** runs the distilled transformer on its fixed twelve-step schedule with no guidance. **Quality** runs the dev transformer with guidance and refines the second stage with the published distilled LoRA, which is a separate download.

```
core/models/
  diffusion_models/      ltx-2.5-22b-distilled-transformer-bf16.safetensors  <- fast mode
                         ltx-2.5-22b-dev-transformer-bf16.safetensors        <- quality mode, and training
  text_encoders/         gemma4-12b-with-proj-ltx-2.5-bf16.safetensors       <- Gemma 4 12B
  vae/                   ltx-2.5-video-vae-bf16.safetensors
                         ltx-2.5-audio-vae-bf16.safetensors
  latent_upscale_models/ ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
  model_patches/         ltx-2.5-duration-head-bf16.safetensors
```

Worth knowing before you start a download this size:

- **You have to accept the licence first.** The weights are gated on Hugging Face. Open the model page, accept the LTX-2 Community License, and make sure the account you accepted with is the one your token belongs to. Without that every download returns a permission error rather than a file.
- **71 GB for fast mode, 122 GB with quality mode.** The transformer is 42 GB and the text encoder another 26, and they are on the card at the same time while the prompt is encoded. That peak, not either file on its own, is what your card has to hold, and it is why a 48 GB card runs the transformer at half precision rather than full.
- **Measured on a 48 GB L40S:** a 2 second clip at 960x576 takes about **3.8 minutes**, or **7.8 minutes** on the first render while the model loads, peaking at 32 GB VRAM. Longer clips and larger canvases scale from there.
- **LTX streams its own weights, which changes what a small card means.** Most models here either fit or are refused. LTX can stream from system RAM, or from disk through a buffer of about 5 GB, so a card that cannot hold the model is slow rather than excluded. It picks for you from what your machine actually has.
- **The `int8_convrot` builds do not load.** Their weights are stored rotated and only ComfyUI can undo that. The picker lists them with the reason rather than hiding them. The bf16 builds load everywhere; the NVFP4 build loads on Blackwell cards with `ltx-kernels` installed.
- **The distilled and dev transformers are indistinguishable.** Same architecture, same metadata, same byte count. Which is which is recorded when the popup downloads them. If you move or rename one by hand, point the node at it explicitly with the Diffusion model dropdown.
- **LoRAs work.** Every LTX node has a LoRA input, and you can train one in the Trainer tab: see [LoRA training](#lora-training). Training runs against the dev transformer and the adapter then loads in both modes.

### ControlNet

Steer a local render with a pose, depth, or edge map. Wire a control map into a gen node's **Control** input and pick a ControlNet in the node's Adjust sidebar.

- **Control Space** - a 3D pose editor in a node. Pose one or more characters, frame a camera, and render the scene as an OpenPose skeleton or a depth map. No reference photo needed.
- **Apply ControlNet** - turn any image into a control map (OpenPose, Depth-Anything V2, MiDaS depth, or Canny edges). Detector weights download once on first use.
- **Z-Image Turbo** - full ControlNet via the Fun Union model. Use the distilled `-2602-8steps` build; the plain one is blurry at 8 steps.
- **Krea 2** - depth control via the [`Patil/Krea-2-depth-controlnet`](https://huggingface.co/Patil/Krea-2-depth-controlnet) control-LoRA.
- **FLUX.2** - two routes. On any variant, a control map wired into **Control** is used as a reference image, which steers loosely and costs nothing extra. On **dev**, pick the [`alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union) model for tight structural adherence: one union model covering canny, depth, pose, HED, MLSD, scribble and gray, with no mode to select. It needs real headroom, rendering at 512px on a 24 GB card.
- **Control strength** - dial how hard the map is followed, per node.

Drop ControlNet files into `core/models/controlnet/`, or use the node's model popup to download them.

### Community extensions

Install community-built nodes straight from a GitHub repo, from the Extensions dialog or a repo URL.

- **One-click install**, with a live stepper showing download, security review, dependency resolution, and activation.
- **Every install is reviewed.** Code that could replace Inline's PyTorch, hide a payload, or run at install time is blocked outright; subprocesses, sockets, and unknown network hosts need your explicit approval.
- **Extensions can't break your setup.** Their dependencies install into their own folder and can never touch the shared torch/diffusers runtime, and genuine conflicts fail at install with both versions named.
- **Nodes appear on the canvas immediately**, with their own params, model downloads, take history, and Run control. No restart, and no frontend code from the author.
- **Toggle any node on or off**, roll back to a previous version, or uninstall, and see when an update is available.
- **Publish by tagging.** Authors list once in the registry; after that a new tag reaches users with no further PR.

Browse the [**extension registry**](https://github.com/inlineresearch/Inline-Registry), or copy the
[**extension guide**](https://github.com/inlineresearch/Inline-Studio-Extension-Guide) to build your
own: four working nodes, declared model downloads, and a full authoring reference.

<details>
<summary><b>Multi-GPU: split one image across GPUs</b></summary>

Got two or more GPUs? Inline Core can cut a single image's latency by running its **denoise loop** (the expensive, iterative sampling step) collectively across them. This is not "one image per GPU" (independent renders); it's **one image whose sampling is shared by all the GPUs**, so a single render finishes faster.

It's done with [xDiT](https://github.com/xdit-project/xDiT) (`xfuser`), which parallelizes diffusion-transformer inference in an isolated worker group (one process per GPU via `torchrun`, over local IPC). The HTTP server, database, and graph stay single-process; only the denoise distributes, and it sits behind a sampler seam so single-GPU/CPU runs pay no overhead. The split method is chosen from the interconnect Core detects: **PipeFusion** (default, works over plain PCIe) or **Ulysses** (sequence-parallel attention, used when NVLink is present). Turn it on with `./webui.sh --multi-gpu` after `uv pip install -e ".[parallel]"`.

</details>

For the full engineering story (the graph/sampler/device-policy design, the node vocabularies, and the xDiT worker group), see **[core/README.md](core/README.md)** and **[core/CLAUDE.md](core/CLAUDE.md)**.

## API Nodes

**API Nodes** bring hosted, closed models onto the same canvas: no GPU, no setup, instant creative range. Add a Generate node, pick a model, and bring your own provider key (it stays on your machine); you pay the provider per render, and each node estimates the price before you run.

The initial provider is **[fal](https://fal.ai)**, with models across image, video, and audio: **FLUX.2**, **FLUX.2 Edit**, **GPT Image 2**, **Nano Banana**, **Seedance**, **MiniMax H3**, **LTX**, **Sonilo**, and many more. Add your [fal.ai key](https://fal.ai/dashboard/keys) in Settings to use them. More providers will follow behind the same API Node surface.

However you render, the frame keeps its full, non-destructive take history, so you can mix API Nodes and local generation in the same film without ever losing a good version.

### MiniMax H3

H3 (Hailuo 03) is on the canvas as three API nodes, all at 2K, 5 to 15 seconds. The open weights also run locally on the Inline Core engine, as [four nodes with no per-render cost](#minimax-h3) if you have the hardware for them.

- **Text → Video** for a shot from nothing but a prompt.
- **Image → Video** for a still you already like. It has two image dots: wire a start frame on its own, or add an end frame and H3 interpolates between the two.
- **Reference → Video** for consistency across shots. Wire up to nine images, plus reference video and audio, and address them by position in the prompt: "Image 1 is the lead, Image 2 is her dog". Wiring order is the numbering you see on the node.

Video costs $0.26 per second at 2K, so the node's price badge reads about $1.30 for a five second clip. Reference images past the first five and any reference video cost extra on top of that.

![Inline Studio dashboard with recent AI film projects](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/screenshot-dashboard.png)

## FAQ

### Is Inline Studio free?

Yes. Inline Studio is free and open source under the [GPL-3.0 license](LICENSE). There's no paid tier to use the app.

### Do I need a GPU?

Only for **local** generation. The built-in Inline Core engine renders on the GPU of whatever machine runs it (you can also run it on a remote GPU box and open the UI from your laptop). Hosted **fal** models need no GPU at all, and the canvas + planning work with no GPU either.

### Can I train a LoRA locally?

Yes, that is what the [Trainer tab](#lora-training) is for, and it runs entirely on your own GPU with no cloud step. It trains LoRAs for Z-Image, Krea 2, FLUX.2, MiniMax H3 and LTX-2.5. Training is cheaper than generating: FLUX.2 klein Base trains at 512px in about 8.6GB, Z-Image in about 13GB, and Krea 2 with a 4-bit base in about 12GB, so a 16GB card handles all three. See [Benchmark results](TRAINING.md#benchmark-results) for the measured table.

### What models can I run?

See [Two ways to generate](#two-ways-to-generate): Z-Image, [Krea 2](#krea-2), [FLUX.2](#flux2), [MiniMax H3](#minimax-h3) or [LTX-2.5](#ltx-25) locally on your own GPU, or hosted fal models. You can train a LoRA locally for any of the five as well, see [LoRA training](#lora-training). Adding a new local model is a Core change (a model runner), no UI release.

## Contributing

Inline Studio is early and moving fast, any issues, ideas, and pull requests are all welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the checks to run, and how to open a PR. [CLAUDE.md](CLAUDE.md) is the deeper engineering guide: the architecture, the data model, and the conventions to follow. By taking part you agree to our [Code of Conduct](CODE_OF_CONDUCT.md).

Want to help by using it for real? Try the [creator task](task.md): build a short 20-second AI film in Inline Studio and send us your feedback.

## Credits

Inline Core's multi-GPU denoise builds on [**xDiT**](https://github.com/xdit-project/xDiT)'s PipeFusion and Ulysses parallelism.

The LoRA trainer's approach to training on a step-distilled model follows [**ai-toolkit**](https://github.com/ostris/ai-toolkit) by ostris, and the Turbo modes use his training adapters for [Z-Image](https://huggingface.co/ostris/zimage_turbo_training_adapter) and [Krea 2](https://huggingface.co/ostris/krea2_turbo_training_adapter).

Krea 2 support follows the reference implementations in [**diffusers**](https://github.com/huggingface/diffusers) (`Krea2Pipeline` and the Krea 2 DreamBooth LoRA example). Krea 2 is released by [Krea AI](https://www.krea.ai/) under the [Krea AI Community License](https://www.krea.ai/krea-2-licensing); the weights are the user's to obtain and use under that license.

[**FLUX.2**](https://bfl.ai/blog/flux-2) is released by Black Forest Labs. klein 4B, its Base build, and the VAE are Apache 2.0; dev and the 9B builds carry non-commercial terms. The weights are the user's to obtain and use under those.

[**MiniMax H3**](https://huggingface.co/MiniMaxAI/MiniMax-H3) is released by MiniMax under the MiniMax H3 Community License; the weights are the user's to obtain and use under that. Support is built on the [**diffusers**](https://github.com/huggingface/diffusers) integration from its MiniMax-H3 pull request, vendored with provenance until it lands upstream.

## Help shape Inline Studio

Are you an AI filmmaker who wants to help us make this better? We run a **paid trial feedback program**: use Inline Studio on real work, tell us what helps and what gets in your way, and get paid for your time.

Come say hi on our [Discord](https://discord.gg/cSUS88VdY9) and reach out, we'll get you set up.

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/cSUS88VdY9)

## License

Copyright (C) 2026 Inline Studio. Licensed under the [GNU General Public License v3.0](LICENSE): you may use, study, share and modify it, and any work you distribute that builds on it must also be GPL-3.0.

The models you run carry their own licenses, which the GPL does not change: Krea 2 is under the [Krea AI Community License](https://www.krea.ai/krea-2-licensing), Z-Image under Tongyi's terms, and FLUX.2 is split - klein 4B, its Base build, and the VAE are Apache 2.0, while dev and every 9B build are non-commercial. You bring your own weights and use them under those.
