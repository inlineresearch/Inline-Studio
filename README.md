<h1 align="center">Inline Studio</h1>

<h3 align="center">AI filmmaking on a node canvas</h3>

<p align="center">Inline Studio is a free, open-source app for AI filmmakers. Build a whole visual pipeline on a free-form node canvas, from moodboard to final cut, with local diffusion models (the built-in Inline Core engine) and hosted fal models — every render kept as a versioned, non-destructive take.</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"></a>
  <a href="../../releases/latest"><img alt="Platforms: macOS, Windows, Linux" src="https://img.shields.io/badge/Platforms-macOS%20%7C%20Windows%20%7C%20Linux-blue?style=for-the-badge"></a>
  <a href="../../releases/latest"><img alt="Latest release" src="https://img.shields.io/badge/Release-v1.0.38-blue?style=for-the-badge"></a>
  <a href="https://discord.gg/cSUS88VdY9"><img alt="Join our Discord" src="https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white&style=for-the-badge"></a>
</p>

![Inline Studio node canvas showing a generative AI film pipeline with frames, takes, and connectors](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/screenshot-dashboard-2.png)

[**New here? Check out our getting started guide →**](https://inlinestudio.art/getting-started)

## What is Inline Studio?

Inline Studio is a free, open-source app for **AI filmmaking on a node canvas**, powered by the built-in **Inline Core** engine (local diffusion models) and hosted [fal](https://fal.ai) models. It gives AI filmmakers a free-form canvas to build a whole visual pipeline, from moodboard to final cut, where every render is kept as a versioned, non-destructive take. Run **local** models like **Z-Image Turbo** on your own GPU with a single model file, or reach for **fal** for instant creative range: hosted closed models like **GPT Image 2**, **Nano Banana**, **Seedance** & many more, no setup. Mix both in the same film, and Inline Studio handles everything around the render: exploring options, keeping what works, and shaping a repeatable process you can iterate on and share.

It runs as a **single process on one port**: the Inline Core engine (Python) serves the web UI _and_ does the generation — `python core/main.py` and open the browser. No desktop install, no separate backend.

**Who it's for:** AI filmmakers, motion artists, and generative creators who want to make AI short films and longer cuts without losing every good version along the way.

## Features

- **Free-form node canvas** - lay out your whole AI film like a mood board that can actually generate. Marquee-select, copy/paste, undo/redo, layers, and text notes all work the way your hands expect.
- **Versioned, non-destructive takes** - every render is kept. Generating again adds a new take; nothing is overwritten. Star the keeper and it flows downstream.
- **Chain frames into a generative pipeline** - wire one frame's output into the next frame's input. Refine a shot, feed it forward, regenerate the source, and everything downstream follows.
- **Video Director node** - a timeline-in-a-node that assembles your rendered frames into a single cut, with layered audio (the videos' own audio plus your own music/VO), per-input and per-layer volume, an in-node preview to scrub, and high-res export.
- **Trim Video/Audio node** - drop in a clip, drag the in/out handles over its filmstrip/waveform, and pass just the trimmed segment downstream.
- **Local generation, built in** - the Inline Core engine runs diffusion models on your own GPU. Z-Image Turbo from a single model file, no external server to set up.
- **Generate with closed models, no setup** - run hosted models like GPT Image 2, Nano Banana, Seedance, Krea, and LTX right on the canvas. No GPU. Add a Generate node, pick a model, and bring your own fal.ai key.
- **Free & open source (MIT)** - one process (Python + a browser); runs on macOS, Windows, and Linux.

![Inline Studio dashboard with recent AI film projects](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/screenshot-dashboard.png)

|                                                                       Trim Video/Audio node                                                                       |                                                                               Video Director node                                                                               |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Trim Video/Audio node with in/out handles over a clip's waveform](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/1.0.34.1.png) | ![Video Director node assembling rendered frames into one cut with layered audio](https://raw.githubusercontent.com/inlineresearch/Inline-Studio/main/screenshots/1.0.34.2.png) |

[**Follow our Animated Short Film with LTX 2.3 and GPT Image Generation tutorial →**](https://inlinestudio.art/projects/circuit-race)

## How it works

Generating a single frame is the easy part. The work that makes an AI film is what comes after: exploring options, keeping what's good, and shaping a repeatable process out of it. Inline Studio is the layer where that happens, organised around one model:

### Export the whole pipeline, not just the final render

From the home screen, **Export** zips a project into one archive. Import it on the other side and you get everything back: the inputs (every imported asset), the outputs (all the generated takes), and the graph that turned one into the other. Whoever opens it can re-run the pipeline exactly and keep iterating.

## Local generation with Inline Core

Inline Studio ships its own generation engine, **Inline Core** (in `core/`), so you can render on your own GPU with no external server to stand up. The first local model is **Z-Image Turbo** (Alibaba Tongyi) — a fast, distilled diffusion transformer.

- **One model file.** Drop a single Z-Image diffusion `.safetensors` into `core/models/diffusion_models/` and you're ready — the engine loads the transformer from that file and wires up the VAE + text-encoder behind the scenes. Bring your own VAE/text-encoder for a fully offline setup, or let it fetch them once from the reference repo.
- **GPU-first, low-VRAM friendly.** The engine always prefers the GPU and never silently offloads to CPU; on a tight-VRAM card it saves memory with VAE tiling/slicing, attention slicing, and int8 instead.
- **One node.** You see a single **Z-Image Turbo** node — no loader/sampler wiring. Add it, connect a Prompt, hit Run.

Your media, your models, your machine.

## Generate with closed models, no setup

The best closed models are hosted only, and they need no GPU. Alongside local generation, Inline Studio runs hosted models through [fal](https://fal.ai): add a single Generate node, pick a model, and go — no setup, no GPU.

Create a frame, pick a model, and run. Everything else works exactly as it does with a local model: takes, flow links between frames, the Video Director, and export. That means you can mix hosted models and local renders in the same film.

Models available today:

- **Local (Inline Core):** Z-Image Turbo (single-file, GPU) — [see above](#local-generation-with-inline-core)
- **fal · Image:** GPT Image 2, Nano Banana 2, Nano Banana Pro (edit), Krea v2 Large
- **fal · Video:** LTX 2.3 (image to video), Seedance 2.0 (text, image, and reference to video)

It is bring your own key. Add your [fal.ai API key](https://fal.ai/dashboard/keys) in Settings and it stays on your machine, sent only to fal when you generate. You pay fal directly for what you render, and each node shows a rough price estimate before you run it.

## Install & run

Inline Studio runs from source as **one process** — the Inline Core engine serves the web UI _and_ does the generation. You'll need [Node.js](https://nodejs.org) 20.11+ and [Python 3.11+](https://python.org) with [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo> && cd inline-studio

# 1. Build the web UI
npm install
npm run build:spa                        # -> dist-web/

# 2. Set up + run the engine (serves the UI + API on one port)
cd core
uv sync --extra server --extra zimage    # server + the Z-Image runtime (torch/diffusers)
uv run python main.py --front-end-root ../dist-web
# add --listen to bind the network, --port to change from 8848
```

Then open **http://127.0.0.1:8848**. Add your [fal.ai API key](https://fal.ai/dashboard/keys) in Settings for hosted models, and drop a Z-Image `.safetensors` in `core/models/diffusion_models/` for local generation (see [Local generation with Inline Core](#local-generation-with-inline-core)). The canvas and planning work without any models.

**UI development (hot-reload):** run the engine as above, then in another terminal `npm run dev:web` (Vite serves the UI with HMR and proxies API calls to Core).

New to Inline Studio? The [Getting Started guide](https://inlinestudio.art/getting-started) walks you through your first render.

## FAQ

### Is Inline Studio free?

Yes. Inline Studio is free and open source under the [MIT license](LICENSE). There's no paid tier to use the app.

### Do I need a GPU?

Only for **local** generation. The built-in Inline Core engine renders on the GPU of whatever machine runs it (you can also run it on a remote GPU box and open the UI from your laptop). Hosted **fal** models need no GPU at all, and the canvas + planning work with no GPU either.

### What models can I run, and how?

- **Local:** Z-Image Turbo today, on your own GPU. Drop a single diffusion `.safetensors` into `core/models/diffusion_models/` — the engine handles the VAE + text-encoder behind it. Adding another local model is a Core change (a model runner), no UI release.
- **Hosted (fal):** GPT Image 2, Nano Banana, Seedance, Krea, LTX, and more — add a Generate node, pick the model, and bring your own [fal.ai key](https://fal.ai/dashboard/keys) (Settings; it stays server-side).

### Does it still use ComfyUI?

No. Earlier versions embedded ComfyUI; Inline Studio now has its own local engine (Inline Core) instead. There's nothing external to stand up — generation is built in.

## Contributing

Inline Studio is early and moving fast, any issues, ideas, and pull requests are all welcome. If you're poking at the code, [CLAUDE.md](CLAUDE.md) is the engineering guide: it explains the architecture, the data model, and the conventions to follow.

Want to help by using it for real? Try the [creator task](task.md): build a short 20-second AI film in Inline Studio and send us your feedback.

## Help shape Inline Studio

Are you an AI filmmaker who wants to help us make this better? We run a **paid trial feedback program**: use Inline Studio on real work, tell us what helps and what gets in your way, and get paid for your time.

Come say hi on our [Discord](https://discord.gg/cSUS88VdY9) and reach out, we'll get you set up.

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/cSUS88VdY9)

## License

MIT.
