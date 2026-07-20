<h1 align="center">Inline Studio</h1>

<h3 align="center">AI filmmaking on a node canvas</h3>

<p align="center">Inline Studio is a free, open-source app for AI filmmakers. Build a whole visual pipeline on a free-form node canvas, from moodboard to final cut, with local diffusion models (the built-in Inline Core engine) and hosted fal models. Every render is kept as a versioned, non-destructive take.</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="../../releases/latest"><img alt="Latest release" src="https://img.shields.io/badge/Release-v1.2.1-blue?style=for-the-badge"></a>
  <a href="https://discord.gg/cSUS88VdY9"><img alt="Join our Discord" src="https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white&style=for-the-badge"></a>
</p>

![Inline Studio node canvas showing a generative AI film pipeline with frames, takes, and connectors](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/screenshot-dashboard-2.png)

[**New here? Check out our getting started guide →**](https://inlinestudio.art/getting-started)

## What is Inline Studio?

Inline Studio is a free, open-source app for **AI filmmaking on a node canvas**, powered by the built-in **Inline Core** engine (local diffusion models) and hosted [fal](https://fal.ai) models. It gives AI filmmakers a free-form canvas to build a whole visual pipeline, from moodboard to final cut.

- **Non-destructive by default** - every render is kept as a versioned take; generating again adds one, nothing is overwritten.
- **Local diffusion generation engine** - the built-in Inline Core engine runs popular diffusion models on your own GPU from a single model file, no external server. Currently supported: **Z-Image Turbo**.
- **Hosted models via API Nodes** - reach for closed models with no GPU and no setup for instant creative range; see [API Nodes](#api-nodes).
- **Mix both in the same film** - Inline Studio handles everything around the render: exploring options, keeping what works, and shaping a repeatable process you can iterate on and share.

It runs as a **single process on one port**: the Inline Core engine (Python) serves the web UI _and_ does the generation: `python core/main.py` and open the browser. No desktop install, no separate backend.

**Who it's for:** AI filmmakers, motion artists, and generative creators who want to make AI short films and longer cuts without losing every good version along the way.

## Features

- **Free-form node canvas** - lay out your whole AI film like a mood board that can actually generate. Marquee-select, copy/paste, undo/redo, layers, and text notes all work the way your hands expect.
- **Versioned, non-destructive takes** - every render is kept. Generating again adds a new take; nothing is overwritten. Star the keeper and it flows downstream.
- **Chain frames into a generative pipeline** - wire one frame's output into the next frame's input. Refine a shot, feed it forward, regenerate the source, and everything downstream follows.
- **Video editing on the canvas** - the **Video Director node** is a timeline-in-a-node that assembles your rendered frames into a single cut, with layered audio (the videos' own audio plus your own music/VO), per-input and per-layer volume, an in-node preview to scrub, and high-res export; the **Trim Video/Audio node** lets you drop in a clip, drag the in/out handles over its filmstrip/waveform, and pass just the trimmed segment downstream.
- **Local generation, built in** - the Inline Core engine runs diffusion models on your own GPU. Z-Image Turbo from a single model file, no external server to set up.
- **API Nodes for hosted models** - run closed models right on the canvas with no GPU. Add a Generate node, pick a model, and bring your own provider key. See [API Nodes](#api-nodes).
- **Community extensions** - install custom nodes from a GitHub repo in one click, security-reviewed and dependency-isolated. Browse the [registry](https://github.com/inlineresearch/Inline-Registry) or [build your own](https://github.com/inlineresearch/Inline-Studio-Extension-Guide).
- **Free & open source (MIT)** - one process (Python + a browser); runs on macOS, Windows, and Linux.

[**Follow our Animated Short Film with LTX 2.3 and GPT Image Generation tutorial →**](https://inlinestudio.art/projects/circuit-race)

## How it works

Generating a single frame is the easy part. The work that makes an AI film is what comes after: exploring options, keeping what's good, and shaping a repeatable process out of it. Inline Studio is the layer where that happens, organised around one model:

### Export the whole pipeline, not just the final render

From the home screen, **Export** zips a project into one archive. Import it on the other side and you get everything back: the inputs (every imported asset), the outputs (all the generated takes), and the graph that turned one into the other. Whoever opens it can re-run the pipeline exactly and keep iterating.

## Three ways to generate

Pick whatever fits the shot, and mix all three in one film. However you render, the frame keeps its full take history, so you never lose a good version.

| How you render                          | What it's like                                                                                                                                                                                                                                                                                    | What you need                                                                                                                                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Local GPU: Inline Core** _(built in)_ | Drop a **Z-Image Turbo** node, wire a prompt, hit Run: one node, no loader/sampler wiring. A single `.safetensors` is all you bring; the engine pairs it with a VAE + text-encoder and downloads nothing behind your back. Two or more GPUs? It can split one image's denoise across them (xDiT). | Your own GPU. No account, no external server. Low-VRAM friendly: it auto-fits the model to your card (streaming weights + int8) so a model too big for full precision still runs, with no flags. |
| **Hosted: API Nodes**                   | Add a Generate node and pick a model: hosted, closed models across image, video, and audio. No GPU, instant range. See [API Nodes](#api-nodes) for the model list and providers.                                                                                                                  | A provider key (currently [fal](https://fal.ai/dashboard/keys)); it stays on your machine, and you pay per render (each node estimates the price first).                                         |
| **Your own ComfyUI** _(legacy)_         | Point Inline Studio at a running ComfyUI server and drive it from the Generate tab.                                                                                                                                                                                                               | A ComfyUI instance. **Being phased out** in favour of Inline Core; fine for now, but don't build on it.                                                                                          |

For local generation, either drop a Z-Image `.safetensors` into `core/models/diffusion_models/`, or add a Z-Image node and use its **model popup** (a blinking hint shows up when something's missing) to download the diffusion model, VAE, and text-encoder into `core/models/`, with visible progress. The canvas and planning work with no models at all.

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

Like ComfyUI, the built web UI ships as a Python package, so you only need [Python 3.11+](https://python.org), no Node. With [uv](https://docs.astral.sh/uv/) (or plain `pip`):

```bash
git clone https://github.com/inlineresearch/Inline-Studio.git && cd Inline-Studio
cd core
./webui.sh --install --extra runtime     # create the venv, install the engine + model runtime + UI
./webui.sh                               # serve the UI + API on http://127.0.0.1:8848
```

`webui.sh` is the one command you need: it installs dependencies, makes sure the web UI is present (the prebuilt `inline-studio-frontend` package, or a local build), then serves everything. See **[Command-line options](#command-line-options)** for every flag (`--listen`, `--port`, `--lowvram`, `--multi-gpu`, …).

Prefer pip? `pip install -r requirements.txt` (from the repo root) pulls the engine, the prebuilt UI, and the local model runtime from PyPI; then run `inline-studio`.

### Hardware support

<details>
<summary><b>GPU, CPU, Apple Silicon, and ROCm setup</b></summary>

Honest status - what's actually been run, versus what has a code path but no one has verified:

| Hardware                | Status                                           | Extra steps                                                                                                                                                                                                                                |
| ----------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **NVIDIA, Linux**       | **Tested** - Z-Image Turbo 1024² on a T4 (16 GB) | None. `webui.sh --install` picks the CUDA build automatically.                                                                                                                                                                             |
| **NVIDIA, Windows**     | Supported, needs one step                        | PyPI's default `torch` is **CPU-only on Windows**. Use `webui.sh --install` (it detects the GPU), or install torch from `https://download.pytorch.org/whl/cu124`. Core warns at startup if it finds an NVIDIA GPU behind a CPU-only torch. |
| **Apple Silicon (MPS)** | Code path exists, **untested**                   | None. int8 quantisation doesn't apply on MPS, so a model too big for unified memory won't fit.                                                                                                                                             |
| **AMD (ROCm), Linux**   | **Untested** - reports welcome                   | Needs a ROCm build of PyTorch - see [AMD (ROCm) setup](#amd-rocm-setup) below.                                                                                                                                                             |
| **CPU only**            | Works, very slow                                 | `./webui.sh --cpu`                                                                                                                                                                                                                         |

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

- **Local model coverage is Z-Image Turbo only** today. Flux, SDXL and others are planned; hosted models via [API Nodes](#api-nodes) need no GPU at all.
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

The friendly `webui.sh` launcher (in `core/`) maps flags onto the engine's `INLINE_*` environment knobs; `core/main.py` takes the same flags when you run the engine directly. `./webui.sh --help` lists them all.

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

## FAQ

### Is Inline Studio free?

Yes. Inline Studio is free and open source under the [MIT license](LICENSE). There's no paid tier to use the app.

### Do I need a GPU?

Only for **local** generation. The built-in Inline Core engine renders on the GPU of whatever machine runs it (you can also run it on a remote GPU box and open the UI from your laptop). Hosted **fal** models need no GPU at all, and the canvas + planning work with no GPU either.

### What models can I run?

See [Three ways to generate](#three-ways-to-generate): local Z-Image on your own GPU, hosted fal models, or your own ComfyUI. Adding a new local model is a Core change (a model runner), no UI release.

### Does it still use ComfyUI?

Not for the built-in generation: that's all Inline Core now, with nothing external to stand up. You _can_ still connect **your own ComfyUI** server and drive it from the Generate tab, but that path is legacy and **being discontinued** in favour of Inline Core, so don't build anything new on it.

## Contributing

Inline Studio is early and moving fast, any issues, ideas, and pull requests are all welcome. If you're poking at the code, [CLAUDE.md](CLAUDE.md) is the engineering guide: it explains the architecture, the data model, and the conventions to follow.

Want to help by using it for real? Try the [creator task](task.md): build a short 20-second AI film in Inline Studio and send us your feedback.

## Credits

Inline Core's multi-GPU denoise builds on [**xDiT**](https://github.com/xdit-project/xDiT)'s PipeFusion and Ulysses parallelism.

## Help shape Inline Studio

Are you an AI filmmaker who wants to help us make this better? We run a **paid trial feedback program**: use Inline Studio on real work, tell us what helps and what gets in your way, and get paid for your time.

Come say hi on our [Discord](https://discord.gg/cSUS88VdY9) and reach out, we'll get you set up.

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/cSUS88VdY9)

## License

MIT.
