# Inline Core

The generation engine behind Inline. It takes a typed node graph (JSON) and returns immutable renders
("takes"), running image and video models on macOS, Windows, and Linux, from CPU-only boxes and
low-VRAM laptops up to multi-GPU machines that split a single image's sampling across GPUs (via
xDiT). It is Inline Studio's built-in render backend.

First model: Z-Image (Alibaba Tongyi), a 6B rectified-flow diffusion transformer (and a model xDiT
already supports, so the multi-GPU split works on it from the start).

> Status: early, and running end to end against a stub engine. In place and tested: the graph engine,
> the typed `/v1` HTTP + websocket API (durable runs, streamed progress, coalescing), the model-dir
> scan, the device + memory policy (profiles, dtype, offload, int8), the low-level primitive node
> vocabulary. The Z-Image loader is written and validates on a GPU.
> Cross-request batching, single-image multi-GPU (an xDiT worker group behind the sampler seam, with
> the policy and IPC round-trip tested), and out-of-process custom nodes are built as seams but not
> yet running on real hardware.

## Engine design

The two boundaries that matter most: graph orchestration is decoupled from GPU batching (graphs are
the unit of caching, the sampler is the unit of batching, and the multi-GPU split routes through that
same seam), and the device policy is the single owner of placement, so the same graph runs on a 4090,
a 6 GB laptop, pure CPU, or split across several GPUs, without touching the graph.

|              |                                                                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Graph vs GPU | orchestration (cheap, per request) is separate from a batched sampler that groups compatible jobs across requests                                |
| Schema       | typed graph, named params, edges type-checked before the run (a bad graph is rejected at submit)                                                 |
| Devices      | one device/memory policy owns dtype, placement, offload, and attention; no node hardcodes a device, so one graph runs GPU, low-VRAM, or pure CPU |
| Multi-GPU    | one image's denoise can split across GPUs via xDiT (PipeFusion on PCIe, Ulysses on NVLink), in an isolated worker group behind the sampler seam  |
| Custom nodes | run out of process, each pack in its own venv, behind a semver SDK                                                                               |
| Interface    | a headless `/v1` HTTP + websocket API; runs are durable and survive a restart                                                                    |
| Outputs      | immutable takes; regenerating adds a take, never overwrites                                                                                      |
| Models       | drop-in `models/` layout (bring your own, no downloads); a typed catalog feeds versioned node descriptors the UI renders generically             |

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv venv
uv pip install -e ".[server]"     # engine + HTTP/websocket API
uv pip install -e ".[runtime]"    # + torch, diffusers, transformers (for real generation)
uv pip install -e ".[runtime,parallel]"  # + xfuser, to split one image across GPUs (2+ GPUs)
```

## Models

Drop files into the models dir (default `./models`, override with `INLINE_MODELS_DIR`), ComfyUI-style,
by category:

```
models/
  diffusion_models/   z_image_turbo_bf16.safetensors   <- the one file you need for Z-Image
  vae/                ae.safetensors                   <- optional (see below)
  text_encoders/      qwen/   (an HF-format folder: config + tokenizer + weights)  <- optional
  loras/  controlnet/  checkpoints/  ...
```

**Z-Image loads from a single diffusion `.safetensors`.** Drop that one ComfyUI-style file in
`diffusion_models/`; the runner loads the transformer via diffusers `from_single_file`. It also needs
a VAE + text-encoder + tokenizer, resolved from **local files** - `vae/*.safetensors` or a dir, and an
HF-format `text_encoders/` dir.

**Nothing is ever downloaded as a side effect of a render.** Every load runs `local_files_only=True`,
so a missing model is a clear, fast error, not a silent multi-GB fetch. Models arrive by exactly two
paths:

1. **You place files** under `models/` (bring-your-own / fully offline); or
2. **You download them explicitly** from the Z-Image node's model popup in the UI - it lists the
   diffusion model, VAE, and text-encoder, shows which are missing, and downloads them **into
   `models/`** (never the hidden HF cache) with visible progress.

Override the diffusion source with `INLINE_ZIMAGE_MODEL` (a file or a diffusers dir), and the supporting
components with `INLINE_ZIMAGE_VAE` / `INLINE_ZIMAGE_TEXT_ENCODER`. The engine scans the models dir on
start; a node's model pickers list what is present.

## Nodes

`/v1/models` serves each node's descriptor (ports, params, file pickers), so the Inline Studio canvas
renders any node generically - adding a node type needs no UI release.

**High-level model nodes are what the user sees.** Generation is one-click: you drop a single
**Z-Image Turbo** node, wire a Prompt into it, and hit Run. The node hooks up the diffusion model, VAE,
and text-encoder behind the scenes - no loader/sampler wiring.

Underneath, a **low-level primitive vocabulary** exists (`load/diffusion-model`, `load/vae`,
`load/text-encoder`, `encode/text`, `latent/empty`, `sample`, `vae/decode`, `vae/encode`) - the
ComfyUI-equivalent decomposed graph, kept for validation/execution - but these are marked **`hidden`**
and never appear in the add-node menu. Engine handles (`model`, `vae`, `text-encoder`, `conditioning`,
`latent`) are typed sockets between nodes; only media outputs become Frames with take history.

## Multi-GPU: split one image across GPUs

Cut a single image's latency by running its denoise across several GPUs. The denoise loop (the
iterative sampling step) is the expensive part of a render; with two or more GPUs, Inline Core runs
each step's transformer forward collectively across them so one image finishes faster. This is not
"one image per GPU" (independent renders); it is one image whose sampling is shared by all the GPUs.

It is done with xDiT (`xfuser`), which parallelizes diffusion-transformer inference. xfuser runs in
an isolated worker group the engine spawns (one process per GPU via `torchrun`) and talks to over
local IPC, so the HTTP server, database, and graph stay single-process and only the denoise is
distributed. It sits behind the sampler seam (`XFuserBatchedSampler`), so a single-GPU or CPU run
takes the in-process path and pays no overhead.

Two split methods, chosen from the interconnect the engine detects:

- **PipeFusion (default, PCIe):** shards the transformer into a displaced patch pipeline with low,
  depth-independent communication. It needs no NVLink and works over plain PCIe (or Ethernet across
  nodes), so it is the default on a typical multi-GPU box.
- **Ulysses (NVLink):** sequence-parallel attention, used when NVLink is present because it wants
  the higher interconnect bandwidth.

Enabling it:

1. **Install the extra** and have 2+ CUDA GPUs on one machine:
   ```
   uv pip install -e ".[runtime,parallel]"  # xfuser + nvidia-ml-py; torchrun ships with torch
   ```
2. **Run normally.** On the first denoise, the device policy enumerates the GPUs, detects NVLink vs
   PCIe (via `nvidia-ml-py`), and returns a parallel placement when there is more than one GPU. The
   engine then spawns the xfuser worker group (lazily, once, then reuses it) and splits the sampling
   across the GPUs. No graph, API, or per-request change is needed.
3. **Override the split** if you want to pick it by hand, with `INLINE_PARALLEL`:
   ```
   INLINE_PARALLEL=pipefusion=2              # 2 GPUs, PipeFusion
   INLINE_PARALLEL=pipefusion=2,ulysses=2    # 4 GPUs, PipeFusion x Ulysses
   ```
   The degrees multiply to the world size, which must equal the number of GPUs.

The device policy and the worker-group IPC are in place and tested with a stub worker; the real
xfuser denoise lands with the GPU-side Z-Image runner.

## Memory: fit a big model onto a small GPU

When a model is too large to hold full-precision on your GPU - or larger than your system RAM - Inline
Core fits it to the machine automatically, **with no flags**. This is the low-VRAM path, and it aims to
"just run" instead of making you tune profiles.

- **Weights stream straight to the GPU.** Each component loads directly onto the GPU from its
  memory-mapped `.safetensors` - the engine never materializes a full copy in system RAM first. So you
  do **not** need roughly 2× the model size in RAM to load it, and a load can no longer exhaust host RAM
  and take the server down.
- **Auto-fit picks the lightest plan that fits.** The policy sizes the model against your VRAM and
  chooses, in order: full-precision **resident** → **int8 resident** (weight-only quantization halves the
  big weights so they fit on the GPU) → **CPU-offload streaming** (submodules move on/off the GPU). A card
  that can't hold the model full-precision quantizes to int8 on its own - no need to know a flag.
- **Load-time fit check.** A model too large for VRAM + RAM fails fast with a clear message on the node,
  instead of crashing mid-load. The model popup also shows the estimate up front (fits / will run int8 /
  won't fit).
- **Switching models frees the old one.** Loading a different checkpoint evicts the previous weights
  rather than stacking them, so you can move between models without accumulating VRAM.
- **Generation fits too, not just loading.** Prompts encode under `no_grad` so the text-encoder forward
  never retains its activation graph, and the encoder is parked on the CPU during denoise to free its
  VRAM; large images (1024² and up) decode the VAE in tiles instead of one full-frame pass. So the
  memory-heavy steps - encode and decode - stay bounded rather than spiking host RAM or VRAM and taking
  the server down at high resolution.
- **Fragmentation-resistant allocator.** `webui.sh` runs with PyTorch expandable CUDA segments, which
  avoids the "allocation failed with VRAM still free" fragmentation OOMs common on low-VRAM cards.

You can still steer it by hand: `--lowvram` for the tight-VRAM profile, `--smart-memory` to force the
offload machinery, `--vram-budget GB` / `INLINE_VRAM_BUDGET_GB` to cap the assumed VRAM, or
`INLINE_PROFILE` to pin `gpu-max | lowvram | cpu`. `GET /v1/health` reports the live VRAM/RAM budget.

## Run

The easy path is `webui.sh`, which maps friendly flags onto the engine's `INLINE_*` env knobs:

```
./webui.sh                            # loopback, port 8848
./webui.sh --listen --port 9000       # bind all interfaces on 9000
./webui.sh --multi-gpu                # split one image across GPUs (auto with 2+ GPUs)
./webui.sh --lowvram                  # tight-VRAM profile
./webui.sh --install --extra zimage   # set up ./.venv with the Z-Image runtime, then exit
```

`./webui.sh --help` lists every flag (networking, multi-GPU, device/memory profile, paths). The same
flags are available on the Python entrypoint - `python main.py --help` - which is the dev path (it also
takes `--front-end-root DIR` to serve a local SPA build). Or run the server module directly:

```
python main.py --listen --port 9000   # same flags as webui.sh; --front-end-root for a local UI build
python -m inline_core.server          # bare server (INLINE_HOST / INLINE_PORT from the env)
```

Working data (the run database and takes) lives in `INLINE_DATA_DIR` (default `./.inline`).

## API (v1)

- `POST /v1/runs {graph, target}` returns `{runId}` (validated up front; 422 on a bad graph)
- `GET /v1/runs/{id}` returns run state (durable; survives a restart)
- `GET /v1/runs/{id}/events` (websocket): a snapshot, then `progress` / `node_done` / `run_done`
- `DELETE /v1/runs/{id}` cancels
- `GET /v1/models` returns node descriptors + `registryVersion` (ETag-aware)
- `GET /v1/takes/{id}` and `/v1/takes/{id}/bytes`
- `POST /v1/assets` (content-addressed upload) and `GET /v1/health`

## Development

```
ruff check .        # lint
uv run pytest -q    # tests (no GPU needed; model code is import-guarded)
```
