<h1 align="center">Inline Studio</h1>

<h3 align="center">AI filmmaking on a node canvas</h3>

<p align="center">Inline Studio is a free, open-source app for AI filmmakers. Build a whole visual pipeline on a free-form node canvas, from moodboard to final cut, with local diffusion models (the built-in Inline Core engine) and hosted fal models. Train your own LoRAs on the same canvas. Every render is kept as a versioned, non-destructive take.</p>

<p align="center">
  <a href="LICENSE"><img alt="License: GPLv3" src="https://img.shields.io/badge/License-GPLv3-blue?style=for-the-badge"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="../../releases/latest"><img alt="Latest release" src="https://img.shields.io/badge/Release-v1.2.5-blue?style=for-the-badge"></a>
  <a href="https://discord.gg/cSUS88VdY9"><img alt="Join our Discord" src="https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white&style=for-the-badge"></a>
</p>

![Inline Studio node canvas showing a generative AI film pipeline with frames, takes, and connectors](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/screenshot-dashboard-2.png)

[**New here? Check out our getting started guide →**](https://inlinestudio.art/getting-started)

## What is Inline Studio?

Inline Studio is a free, open-source app for **AI filmmaking on a node canvas**, powered by the built-in **Inline Core** engine (local diffusion models) and hosted [fal](https://fal.ai) models. It gives AI filmmakers a free-form canvas to build a whole visual pipeline, from moodboard to final cut.

- **Non-destructive by default** - every render is kept as a versioned take; generating again adds one, nothing is overwritten.
- **Local diffusion generation engine** - the built-in Inline Core engine runs popular diffusion models on your own GPU from a single model file, no external server. Currently supported: **Z-Image Turbo** and **Krea 2** (RAW + Turbo).
- **Hosted models via API Nodes** - reach for closed models with no GPU and no setup for instant creative range; see [API Nodes](#api-nodes).
- **Mix both in the same film** - Inline Studio handles everything around the render: exploring options, keeping what works, and shaping a repeatable process you can iterate on and share.

It runs as a **single process on one port**: the Inline Core engine (Python) serves the web UI _and_ does the generation: `python core/main.py` and open the browser. No desktop install, no separate backend.

**Who it's for:** AI filmmakers, motion artists, and generative creators who want to make AI short films and longer cuts without losing every good version along the way.

## Features

- **Free-form node canvas** - lay out your whole AI film like a mood board that can actually generate. Marquee-select, copy/paste, undo/redo, layers, and text notes all work the way your hands expect.
- **Versioned, non-destructive takes** - every render is kept. Generating again adds a new take; nothing is overwritten. Star the keeper and it flows downstream.
- **Chain frames into a generative pipeline** - wire one frame's output into the next frame's input. Refine a shot, feed it forward, regenerate the source, and everything downstream follows.
- **Video editing on the canvas** - the **Video Director node** is a timeline-in-a-node that assembles your rendered frames into a single cut, with layered audio (the videos' own audio plus your own music/VO), per-input and per-layer volume, an in-node preview to scrub, and high-res export; the **Trim Video/Audio node** lets you drop in a clip, drag the in/out handles over its filmstrip/waveform, and pass just the trimmed segment downstream.
- **Local generation, built in** - the Inline Core engine runs diffusion models on your own GPU. Z-Image Turbo and Krea 2 from single model files, no external server to set up.
- **Train your own LoRAs** - the Trainer tab is a second canvas where the dataset, captioning, training run, and loss curve are all nodes. The finished LoRA drops into `models/loras/` and shows up in the LoRA loader node, ready to generate with. See [LoRA training](#lora-training).
- **API Nodes for hosted models** - run closed models right on the canvas with no GPU. Add a Generate node, pick a model, and bring your own provider key. See [API Nodes](#api-nodes).
- **Community extensions** - install custom nodes from a GitHub repo in one click, security-reviewed and dependency-isolated. Browse the [registry](https://github.com/inlineresearch/Inline-Registry) or [build your own](https://github.com/inlineresearch/Inline-Studio-Extension-Guide).
- **Free & open source (GPL-3.0)** - one process (Python + a browser); runs on macOS, Windows, and Linux.

[**Follow our Animated Short Film with LTX 2.3 and GPT Image Generation tutorial →**](https://inlinestudio.art/projects/circuit-race)

## How it works

Generating a single frame is the easy part. The work that makes an AI film is what comes after: exploring options, keeping what's good, and shaping a repeatable process out of it. Inline Studio is the layer where that happens, organised around one model:

### Export the whole pipeline, not just the final render

From the home screen, **Export** zips a project into one archive. Import it on the other side and you get everything back: the inputs (every imported asset), the outputs (all the generated takes), and the graph that turned one into the other. Whoever opens it can re-run the pipeline exactly and keep iterating.

## Two ways to generate

Pick whatever fits the shot, and mix both in one film. However you render, the frame keeps its full take history, so you never lose a good version.

| How you render                          | What it's like                                                                                                                                                                                                                                                                                                  | What you need                                                                                                                                                                                    |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Local GPU: Inline Core** _(built in)_ | Drop a **Z-Image Turbo** or **Krea 2** node, wire a prompt, hit Run: one node, no loader/sampler wiring. A single `.safetensors` is all you bring; the engine pairs it with a VAE + text-encoder and downloads nothing behind your back. Two or more GPUs? It can split one image's denoise across them (xDiT). | Your own GPU. No account, no external server. Low-VRAM friendly: it auto-fits the model to your card (streaming weights + int8) so a model too big for full precision still runs, with no flags. |
| **Hosted: API Nodes**                   | Add a Generate node and pick a model: hosted, closed models across image, video, and audio. No GPU, instant range. See [API Nodes](#api-nodes) for the model list and providers.                                                                                                                                | A provider key (currently [fal](https://fal.ai/dashboard/keys)); it stays on your machine, and you pay per render (each node estimates the price first).                                         |

For local generation, either drop a `.safetensors` into `core/models/diffusion_models/`, or add a model node and use its **model popup** (a blinking hint shows up when something's missing) to download the diffusion model, VAE, and text-encoder into `core/models/`, with visible progress. The canvas and planning work with no models at all. See [Krea 2](#krea-2) for that model's files and VRAM.

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

The initial provider is **[fal](https://fal.ai)**, with models across image, video, and audio: **GPT Image 2**, **Nano Banana**, **Seedance**, **LTX**, **Sonilo**, and many more. Add your [fal.ai key](https://fal.ai/dashboard/keys) in Settings to use them. More providers will follow behind the same API Node surface.

However you render, the frame keeps its full, non-destructive take history, so you can mix API Nodes and local generation in the same film without ever losing a good version.

![Inline Studio dashboard with recent AI film projects](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/screenshot-dashboard.png)

## Install & run

Inline Studio runs as **one process**: the Inline Core engine serves the web UI _and_ does the generation, on a single port.

### The easy way (no Node build)

The built web UI ships as a Python package, so you only need [Python 3.11+](https://python.org), no Node. With [uv](https://docs.astral.sh/uv/) (or plain `pip`):

**macOS / Linux:**

```bash
git clone https://github.com/inlineresearch/Inline-Studio.git && cd Inline-Studio
cd core
./webui.sh --install --extra runtime --extra training   # venv + engine, model runtime, LoRA trainer, UI
./webui.sh                                               # serve the UI + API on http://127.0.0.1:8848
```

**Windows** (use `webui.bat`, the Windows twin - `webui.sh` is a bash script and won't run in PowerShell):

```powershell
git clone https://github.com/inlineresearch/Inline-Studio.git; cd Inline-Studio\core
.\webui.bat --install --extra runtime --extra training   # venv + engine, model runtime, LoRA trainer, UI
.\webui.bat                                               # serve the UI + API on http://127.0.0.1:8848
```

`webui.sh` (macOS/Linux) and `webui.bat` (Windows) are the one command you need: they install dependencies, make sure the web UI is present (the prebuilt `inline-studio-frontend` package, or a local build), then serve everything. You can also double-click `webui.bat` on Windows. See **[Command-line options](#command-line-options)** for every flag (`--listen`, `--port`, `--lowvram`, `--multi-gpu`, …).

Prefer pip? `pip install -r requirements.txt` (from the repo root) pulls the engine, the prebuilt UI, and the local model runtime from PyPI; then run `inline-studio`.

### Hardware support

<details>
<summary><b>GPU, CPU, Apple Silicon, and ROCm setup</b></summary>

Honest status - what's actually been run, versus what has a code path but no one has verified:

| Hardware                | Status                                                                                              | Extra steps                                                                                                                                                                                                                                                                                                      |
| ----------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NVIDIA, Linux**       | **Tested** - Z-Image Turbo 1024² on a T4 (16 GB); Krea 2 1024² and LoRA training on an L40S (48 GB) | None. `webui.sh --install` picks the CUDA build automatically.                                                                                                                                                                                                                                                   |
| **NVIDIA, Windows**     | Supported, needs one step                                                                           | Run `.\webui.bat --install` (the Windows launcher; it detects the GPU). PyPI's default `torch` is **CPU-only on Windows**, so `--install` pulls the CUDA build for you, or install torch from `https://download.pytorch.org/whl/cu124`. Core warns at startup if it finds an NVIDIA GPU behind a CPU-only torch. |
| **Apple Silicon (MPS)** | Code path exists, **untested**                                                                      | None. int8 quantisation doesn't apply on MPS, so a model too big for unified memory won't fit.                                                                                                                                                                                                                   |
| **AMD (ROCm), Linux**   | **Untested** - reports welcome                                                                      | Needs a ROCm build of PyTorch - see [AMD (ROCm) setup](#amd-rocm-setup) below.                                                                                                                                                                                                                                   |
| **CPU only**            | Works, very slow                                                                                    | `./webui.sh --cpu` (Windows: `.\webui.bat --cpu`)                                                                                                                                                                                                                                                                |

#### AMD (ROCm) setup

Nobody has verified Inline Studio on AMD yet, so treat this as a starting point rather than a supported path. Install everything normally **first**, then replace PyTorch with the ROCm build - doing it in this order means nothing can quietly overwrite your ROCm torch afterwards:

```bash
cd core
uv venv
uv pip install -e ".[runtime,server]"        # engine + runtime (pulls the default PyPI torch)

# Replace torch with the ROCm build. Pick the index that matches YOUR ROCm version -
# check https://pytorch.org/get-started/locally/ (rocm6.2 shown here as an example).
uv pip install --force-reinstall --index-url https://download.pytorch.org/whl/rocm6.2 torch

# Verify you actually got a ROCm build (hip should print a version, not None):
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
```

Then run `./webui.sh` as usual.

Two gotchas:

- **Don't run `uv sync` afterwards** - it re-resolves the environment against the lockfile and will pull the PyPI torch back over your ROCm build. Use `uv pip install` for follow-up installs.
- ROCm presents itself through `torch.cuda`, so the engine will treat it as a CUDA device and may largely work. But the dtype heuristics key off **NVIDIA** compute capability (`< 8.0` → fp16), which is meaningless on RDNA/CDNA, and the int8 (torchao) path is unverified on ROCm. If it works - or doesn't - [open an issue](https://github.com/inlineresearch/Inline-Studio/issues); that's the fastest way to get AMD properly supported.

**Known limits, so you can judge before installing:**

- **Local model coverage is Z-Image Turbo and Krea 2** today. Flux, SDXL and others are planned; hosted models via [API Nodes](#api-nodes) need no GPU at all.
- **Krea 2 is a 12.9B model and needs a big card to generate.** The bf16 checkpoint is 26 GB on disk, and generation peaks around 36 GB at 1024 with guidance on, so a 40 GB+ GPU is the practical floor for inference. **Training is cheaper than generating**, because the 4-bit base path puts Krea 2 LoRA training at 512 inside 12 GB - see [Hardware](#hardware). Z-Image remains the low-VRAM path for generation.
- **1024² with Guidance (CFG) above 0 needs more than 16 GB.** CFG runs the prompt and negative prompt together, doubling the denoise. Z-Image Turbo is distilled to run CFG-free - at Guidance 0, 1024² fits in ~11.5 GB.

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

Then open **http://127.0.0.1:8848**. Add your [fal.ai API key](https://fal.ai/dashboard/keys) in Settings for hosted models, and set up local generation as in [Three ways to generate](#three-ways-to-generate). The canvas and planning work without any models.

**Hot-reload:** run the engine as above, then in another terminal `npm run dev:web` (Vite serves the UI with HMR and proxies API calls to Core).

</details>

### Command-line options

The friendly launcher (in `core/`) maps flags onto the engine's `INLINE_*` environment knobs: `webui.sh` on macOS/Linux, `webui.bat` on Windows. `core/main.py` takes the same flags when you run the engine directly. `./webui.sh --help` (or `.\webui.bat --help`) lists them all.

<details>
<summary><strong>Show all command-line flags</strong></summary>

| `webui.sh` / `main.py` flag        | Env var                  | What it does                                                                                                                                                              |
| ---------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--listen`                         | `INLINE_HOST=0.0.0.0`    | Bind all interfaces so other machines can reach it                                                                                                                        |
| `--host ADDR`                      | `INLINE_HOST`            | Bind a specific address (default `127.0.0.1`)                                                                                                                             |
| `--port N`                         | `INLINE_PORT`            | Port to serve on (default `8848`)                                                                                                                                         |
| `--models-dir PATH`                | `INLINE_MODELS_DIR`      | Where model weights are scanned from (default `./models`)                                                                                                                 |
| `--data-dir PATH`                  | `INLINE_DATA_DIR`        | Where runs + takes are written (default `./.inline`)                                                                                                                      |
| `--lowvram`                        | `INLINE_PROFILE=lowvram` | Tight-VRAM profile (VAE tiling/slicing, attention slicing)                                                                                                                |
| `--cpu`                            | `INLINE_PROFILE=cpu`     | Force CPU generation                                                                                                                                                      |
| `--profile NAME`                   | `INLINE_PROFILE`         | Set the profile explicitly: `gpu-max` \| `lowvram` \| `cpu`                                                                                                               |
| `--vram-budget GB`                 | `INLINE_VRAM_BUDGET_GB`  | Treat the GPU as having GB of usable VRAM                                                                                                                                 |
| `--multi-gpu [SPEC]`               | `INLINE_PARALLEL`        | Split one image's denoise across GPUs (e.g. `pipefusion=2`); auto with 2+ GPUs                                                                                            |
| `--front-end-root DIR` _(main.py)_ | `INLINE_FRONTEND_ROOT`   | Serve a local SPA build instead of the installed UI package (dev)                                                                                                         |
| `--rebuild` _(webui.sh)_           | n/a                      | Force a fresh SPA build (`npm run build:spa`) from source and serve it on the one port; use after UI changes when not running `--dev`. Needs the repo checkout + Node/npm |

</details>

`webui.sh` also has `--install` / `--extra NAME` to set up the venv. New to Inline Studio? The [Getting Started guide](https://inlinestudio.art/getting-started) walks you through your first render.

## LoRA training

Train a LoRA on your own images without leaving the app. The **Trainer** tab is a second canvas: wire up the nodes, press Start, and watch it run. When the run finishes, the `.safetensors` lands in `models/loras/`, where the LoRA loader node picks it up automatically, so you can generate with it over in the Studio tab straight away.

![Inline Studio Trainer tab showing the LoRA training node graph with a dataset, live logs, and a loss curve](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/lora-trainer.png)

### The graph

Five nodes, wired left to right:

- **Load Dataset** picks a training dataset and feeds it downstream. The node face stays a preview (thumbnails, image and caption counts); the images and captions themselves are edited in the side panel.
- **Caption** runs a local captioner over the images that need one, with per-image progress. Captions stay editable afterwards, and a wired dataset overrides the node's own picker.
- **Train LoRA** runs the job. Hyperparameters live behind the Adjust button, off the node face, so the node stays a status surface: a live step counter, the trainer's streaming logs, and a progress bar. The run control is a single chip that reads Start, Stop, or Resume depending on where the run is.
- **Graph** plots the loss curve for whichever run is wired into it, with loss values on the y axis and the step range on the x axis.
- **Resources** is a read-only readout of CPU, RAM, and VRAM as circular gauges. It takes no connections, and you can drop it on the Studio canvas too.

<details>
<summary><b>Datasets, resuming, trigger words, and base model modes</b></summary>

### Datasets and outputs

The sidebar has two tabs. **Datasets** is where you create a dataset, give it a trigger word, add images (drag and drop from your file manager works), and edit captions. **Outputs** lists what training has produced: finished LoRAs with their rank, step count, and resolution, plus any run that stopped early, each with a Resume button.

### Stop and resume

Changing a setting in the Adjust panel stages it behind an **Update** button rather than applying as you type. A checkpoint encodes the rank, LoRA targets and base it was built with, so if the node has a run you could resume, applying asks first and then discards that run's checkpoints. Finished runs' LoRA files are never touched.

Stopping a run flushes a checkpoint before the process exits, so Resume continues from the step it left off instead of starting over. A checkpoint holds the adapter weights, the optimiser state, the RNG state, and the step number, which is what makes a resumed run a continuation rather than a restart. Runs cut short by a crash or a server restart are recovered the same way and show up under Outputs ready to resume.

### Trigger words

A dataset's trigger word is prepended to every caption during training, so the model sees captions in the form `mytoken, a photo of ...`. Put the same token at the front of your prompt to pull the LoRA in. It is worth matching the phrasing of your captions too: if they all say "an oil painting of", a prompt written the same way will hit the trained style far more reliably than the trigger word alone.

### Architecture and base model modes

The Trainer's Adjust panel picks the **architecture** first (Z-Image or Krea 2), then a base within it. Training directly on a step-distilled checkpoint breaks the distillation down (turbo drift), so each architecture offers a way around that.

**Krea 2** avoids the problem outright, which is why it is the recommended path:

- **Krea 2 RAW** trains on the undistilled base. Nothing to fuse, nothing to drift. Put `krea2_raw_bf16.safetensors` in `models/diffusion_models/`, train, then generate with the **Krea 2 Turbo** node - the LoRA carries over unchanged.
- **Krea 2 Turbo + training adapter** exists for people who only hold Turbo. Put [ostris/krea2_turbo_training_adapter](https://huggingface.co/ostris/krea2_turbo_training_adapter) in `models/loras/`, or point `INLINE_KREA2_TRAIN_ADAPTER` at it.

**Z-Image** is distilled either way:

- **Turbo + training adapter** fuses a de-distillation adapter into the base for the duration of training and drops it when the LoRA is saved, which preserves the 8-step speed. Put [ostris/zimage_turbo_training_adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter) in `models/loras/`; any filename containing `adapter` is detected automatically, or point `INLINE_ZIMAGE_TRAIN_ADAPTER` at a specific file. Keep runs short, since the adapter slows the breakdown rather than preventing it.
- **De-Turbo** trains without an adapter and needs no extra download.

</details>

### Install

The trainer's dependencies (PEFT, 8-bit Adam, the captioner) sit behind the `training` extra, so a normal install stays lean:

```bash
cd core
./webui.sh --install --extra runtime --extra training
```

Nothing is downloaded behind your back. Training has no downloader of its own: it reuses whatever is already in `models/diffusion_models/`, `models/vae/` and `models/text_encoders/` for the architecture you pick, which is normally what a generate node's model popup fetched for you. If a file is missing, the run stops and names it.

Two things the model popup does not cover, so you fetch them yourself:

- **Training adapters** for the Turbo base modes: [Z-Image](https://huggingface.co/ostris/zimage_turbo_training_adapter) or [Krea 2](https://huggingface.co/ostris/krea2_turbo_training_adapter), dropped in `models/loras/`.
- **The captioner**, fetched once into the Hugging Face cache the first time you press Auto-caption.

The LoRA a run produces lands in `models/loras/` and shows up in the LoRA loader node straight away, so you can wire it into a generate node and try it without leaving the app.

### Benchmark results

12 steps at rank 16, batch 1, gradient checkpointing on. The number is `torch.cuda.max_memory_allocated`, so leave headroom for the CUDA context and allocator slack.

| Model   | Base mode       | Res  | Base precision | L40S (46GB)   | T4 (15GB)     |
| ------- | --------------- | ---- | -------------- | ------------- | ------------- |
| Z-Image | De-Turbo        | 512  | bf16           | 13.1GB        | 13.4GB        |
| Z-Image | De-Turbo        | 1024 | bf16           | 14.9GB        | out of memory |
| Z-Image | Turbo + adapter | 512  | bf16           | 13.1GB        | 13.4GB        |
| Z-Image | Turbo + adapter | 1024 | bf16           | 14.9GB        | out of memory |
| Krea 2  | RAW             | 512  | bf16           | 30.4GB        | out of memory |
| Krea 2  | RAW             | 512  | **4-bit**      | 11.7GB        | **11.9GB**    |
| Krea 2  | RAW             | 1024 | bf16           | out of memory | out of memory |
| Krea 2  | RAW             | 1024 | **4-bit**      | **27.8GB**    | out of memory |
| Krea 2  | Turbo + adapter | 512  | bf16           | 30.4GB        | out of memory |
| Krea 2  | Turbo + adapter | 512  | **4-bit**      | 11.7GB        | **11.9GB**    |
| Krea 2  | Turbo + adapter | 1024 | bf16           | out of memory | out of memory |
| Krea 2  | Turbo + adapter | 1024 | **4-bit**      | **27.8GB**    | out of memory |

A training adapter is free: it is fused into the base before training starts, so Turbo-plus-adapter and the undistilled base peak identically.

Which card fits what (24GB and 32GB are interpolated, not measured):

| Card | Z-Image 512 | Z-Image 1024 | Krea 2 512 | Krea 2 1024 |
| ---- | ----------- | ------------ | ---------- | ----------- |
| 16GB | yes         | no           | yes, 4-bit | no          |
| 24GB | yes         | yes          | yes        | no          |
| 32GB | yes         | yes          | yes        | yes, 4-bit  |
| 48GB | yes         | yes          | yes        | 4-bit only  |

Fitting and being usable are different questions. Turing has no native bf16, so a T4 runs the same work about 4x slower:

| Configuration                     | L40S | T4   |
| --------------------------------- | ---- | ---- |
| Krea 2 RAW 512, 4-bit             | 192s | 824s |
| Krea 2 Turbo + adapter 512, 4-bit | 219s | 872s |
| Z-Image 512                       | 85s  | 285s |

A 1500-step Krea 2 run is roughly 40 minutes on an L40S and 3 hours on a T4.

**Krea 2 at 512 with the 4-bit base is the configuration to reach for on a small card.** 1024 needs about 32GB and no setting closes that gap: activations scale with image tokens, and gradient checkpointing and memory-efficient attention are already on. Train at 512 instead, since a LoRA trained at 512 applies at any generation resolution.

System RAM matters as well. Checkpoints are read tensor by tensor rather than mapped whole, so Krea 2 trains in about 3GB of host RAM. Without that, Linux refuses to map a file larger than physical RAM when there is no swap, and a 26GB checkpoint cannot be opened on a 16GB machine at all.

### Dataset and adapter options

Three settings shape what the adapter learns rather than what it costs:

- **LoRA scope.** _Full_ adapts the attention and feed-forward layers, which is stronger on short style runs. _Attention only_ is the Krea 2 authors' advice for long runs, where adapting everything starts to cost prompt adherence.
- **Caption dropout** (default 0.05) trains a fraction of steps against an empty caption, so the LoRA still holds when a prompt does not repeat the trigger word verbatim.
- **Flip images** mirrors every image, doubling a small dataset. Both orientations are encoded from pixels rather than by flipping cached latents, so the mirrored copy is exact. Leave it off for anything with text or a deliberate asymmetry.

### Base precision

Krea 2's base is 26GB at bf16, which is what makes it expensive to fine-tune. The Trainer's **Base precision** setting freezes that base at 4-bit (NF4) while the LoRA itself stays full precision - the QLoRA arrangement - so only the frozen base loses fidelity:

- **Auto** (default) sizes the base _plus its activations at your chosen resolution_ against your GPU and picks for you. Weights alone are not enough to decide: a 48GB card holds Krea 2's 26GB base comfortably and then runs out at 1024.
- **Full precision (bf16)** forces the unquantized base.
- **4-bit (NF4)** forces the quantized base.

Z-Image has no 4-bit path and does not need one, so the setting only appears for Krea 2.

To keep the peak down, the VAE and text encoder are loaded first, used to cache latents and captions, then freed before the transformer loads, so the peak is the transformer on its own rather than all three resident at once. If you do hit an out-of-memory error, lower the training resolution before changing anything else.

## FAQ

### Is Inline Studio free?

Yes. Inline Studio is free and open source under the [GPL-3.0 license](LICENSE). There's no paid tier to use the app.

### Do I need a GPU?

Only for **local** generation. The built-in Inline Core engine renders on the GPU of whatever machine runs it (you can also run it on a remote GPU box and open the UI from your laptop). Hosted **fal** models need no GPU at all, and the canvas + planning work with no GPU either.

### What models can I run?

See [Two ways to generate](#two-ways-to-generate): local Z-Image or [Krea 2](#krea-2) on your own GPU, or hosted fal models. Adding a new local model is a Core change (a model runner), no UI release.

## Contributing

Inline Studio is early and moving fast, any issues, ideas, and pull requests are all welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the checks to run, and how to open a PR. [CLAUDE.md](CLAUDE.md) is the deeper engineering guide: the architecture, the data model, and the conventions to follow. By taking part you agree to our [Code of Conduct](CODE_OF_CONDUCT.md).

Want to help by using it for real? Try the [creator task](task.md): build a short 20-second AI film in Inline Studio and send us your feedback.

## Credits

Inline Core's multi-GPU denoise builds on [**xDiT**](https://github.com/xdit-project/xDiT)'s PipeFusion and Ulysses parallelism.

The LoRA trainer's approach to training on a step-distilled model follows [**ai-toolkit**](https://github.com/ostris/ai-toolkit) by ostris, and the Turbo modes use his training adapters for [Z-Image](https://huggingface.co/ostris/zimage_turbo_training_adapter) and [Krea 2](https://huggingface.co/ostris/krea2_turbo_training_adapter).

Krea 2 support follows the reference implementations in [**diffusers**](https://github.com/huggingface/diffusers) (`Krea2Pipeline` and the Krea 2 DreamBooth LoRA example). Krea 2 is released by [Krea AI](https://www.krea.ai/) under the [Krea AI Community License](https://www.krea.ai/krea-2-licensing); the weights are the user's to obtain and use under that license.

## Help shape Inline Studio

Are you an AI filmmaker who wants to help us make this better? We run a **paid trial feedback program**: use Inline Studio on real work, tell us what helps and what gets in your way, and get paid for your time.

Come say hi on our [Discord](https://discord.gg/cSUS88VdY9) and reach out, we'll get you set up.

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/cSUS88VdY9)

## License

Copyright (C) 2026 Inline Studio. Licensed under the [GNU General Public License v3.0](LICENSE): you may use, study, share and modify it, and any work you distribute that builds on it must also be GPL-3.0.

The models you run carry their own licenses, which the GPL does not change: Krea 2 is under the [Krea AI Community License](https://www.krea.ai/krea-2-licensing) and Z-Image under Tongyi's terms. You bring your own weights and use them under those.
