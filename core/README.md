# Inline Core

The generation engine behind Inline. It takes a typed node graph (JSON) and returns immutable renders
("takes"), running image and video models on macOS, Windows, and Linux, from CPU-only boxes and
low-VRAM laptops up to multi-GPU machines that split a single image's sampling across GPUs (via
xDiT). It is the render backend that replaces ComfyUI for Inline.

First model: Z-Image (Alibaba Tongyi), a 6B rectified-flow diffusion transformer (and a model xDiT
already supports, so the multi-GPU split works on it from the start).

> Status: early, and running end to end against a stub engine. In place and tested: the graph engine,
> the typed `/v1` HTTP + websocket API (durable runs, streamed progress, coalescing), the model-dir
> scan, the device + memory policy (profiles, dtype, offload, int8), the low-level primitive node
> vocabulary, and a ComfyUI workflow importer. The Z-Image loader is written and validates on a GPU.
> Cross-request batching, single-image multi-GPU (an xDiT worker group behind the sampler seam, with
> the policy and IPC round-trip tested), and out-of-process custom nodes are built as seams but not
> yet running on real hardware.

## How it differs from ComfyUI's architecture

ComfyUI is a great canvas but a fragile engine. Inline Core keeps the open node-graph model and
rebuilds the engine underneath it.

| | ComfyUI | Inline Core |
| --- | --- | --- |
| Graph vs GPU | runs the denoise loop inline, one request at a time | graph orchestration (cheap, per request) is separate from a batched sampler that groups compatible jobs across requests |
| Schema | positional `widgets_values`, validated at runtime (dies mid-graph) | typed graph, named params, edges type-checked before the run (rejected at submit) |
| Devices | some nodes pin to CPU on a GPU box; Z-Image will not run on CPU | one device/memory policy owns dtype, placement, offload, and attention; no node hardcodes a device, so one graph runs GPU, low-VRAM, or pure CPU |
| Multi-GPU | one image runs on one GPU; splitting a single image needs third-party nodes | one image's denoise can split across GPUs via xDiT (PipeFusion on PCIe, Ulysses on NVLink), in an isolated worker group behind the sampler seam |
| Custom nodes | all load into one interpreter and env, so any node can break the core | run out of process, each pack in its own venv, behind a semver SDK |
| Interface | a web UI driven by graph JSON over a socket; run state is ephemeral | a headless `/v1` HTTP + websocket API; runs are durable and survive a restart |
| Outputs | files you overwrite | immutable takes; regenerating adds a take, never overwrites |
| Models | `models/` dir, dropdowns from a scan | same layout (bring your own, no downloads); a typed catalog feeds versioned node descriptors the UI renders generically |

The two boundaries that matter most: graph orchestration is decoupled from GPU batching (graphs are
the unit of caching, the sampler is the unit of batching, and the multi-GPU split routes through that
same seam), and the device policy is the single owner of placement, so the same graph runs on a 4090,
a 6 GB laptop, pure CPU, or split across several GPUs, without touching the graph.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv venv
uv pip install -e ".[server]"     # engine + HTTP/websocket API
uv pip install -e ".[zimage]"     # + torch, diffusers, transformers (for real generation)
uv pip install -e ".[parallel]"   # + xfuser, for splitting one image across GPUs (needs 2+ GPUs)
```

## Models

Bring your own weights; nothing is downloaded. Drop files into the models dir (default `./models`,
override with `INLINE_MODELS_DIR`), ComfyUI-style, by category:

```
models/
  diffusion_models/   z_image_turbo_bf16.safetensors
  vae/                ae.safetensors
  text_encoders/      qwen3-4b/     (a folder: config + tokenizer + weights)
  loras/  controlnet/  checkpoints/  ...
```

The engine scans this on start; a node's model pickers list what is present.

## Nodes and workflows

The canvas (Storyline) wires **low-level primitive nodes** by typed edges. `/v1/models` serves each
node's descriptor (ports, params, file pickers), so the UI renders any node generically and adding a
node type needs no UI release.

- Loaders: `load/diffusion-model`, `load/vae`, `load/text-encoder`
- Conditioning: `encode/text`
- Latent and sampling: `latent/empty`, `sample`
- VAE: `vae/decode`, `vae/encode`

Engine handles (`model`, `vae`, `text-encoder`, `conditioning`, `latent`) are typed sockets passed
between nodes; only media outputs (`vae/decode`) become Frames with take history, the rest are
ephemeral plumbing. A best-effort ComfyUI importer maps existing workflows onto these nodes.

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
   uv pip install -e ".[parallel]"   # pulls in xfuser and nvidia-ml-py; torchrun ships with torch
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

## Run

The easy path is `webui.sh`, which maps friendly flags onto the engine's `INLINE_*` env knobs:

```
./webui.sh                            # loopback, port 8848
./webui.sh --listen --port 9000       # bind all interfaces on 9000
./webui.sh --multi-gpu                # split one image across GPUs (auto with 2+ GPUs)
./webui.sh --lowvram                  # tight-VRAM profile
./webui.sh --install --extra zimage   # set up ./.venv with the Z-Image runtime, then exit
```

`./webui.sh --help` lists every flag (networking, multi-GPU, device/memory profile, paths). Or run
the server directly:

```
python -m inline_core.server          # serves http://127.0.0.1:8848 (INLINE_HOST / INLINE_PORT)
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
