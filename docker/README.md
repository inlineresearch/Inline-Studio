# Inline Studio

**AI filmmaking on a node canvas.** Generate locally on your own GPU and train your own LoRAs on the
same canvas, with the built-in Inline Core engine and hosted models. Every render is kept as a
versioned take, so you never lose the good version.

Free and open source, GPL-3.0. [inlinestudio.art](https://inlinestudio.art) ·
[GitHub](https://github.com/inlineresearch/Inline-Studio) ·
[Discord](https://discord.gg/cSUS88VdY9)

No GPU of your own? [Deploy it on RunPod in one click](https://console.runpod.io/deploy?template=c0qkyaypuv&ref=hs2l4qhc).

## Quick start

```bash
docker run --gpus all -p 8848:8848 -p 8888:8888 -p 8080:8080 \
  -v "$HOME/inline-workspace:/workspace" \
  --shm-size=16g \
  inlineresearch/inline-studio:latest
```

Open <http://localhost:8848>.

That volume mount is the important part. Everything the app writes lives under `/workspace`: model
weights, projects, takes, trained LoRAs, settings and your fal key. Without it, all of that is
destroyed when the container is removed.

There is a Compose file in the repo at
[`docker/compose.yaml`](https://github.com/inlineresearch/Inline-Studio/blob/main/docker/compose.yaml)
if you prefer that.

## What you need

- **An NVIDIA GPU and the NVIDIA Container Toolkit.** The image is amd64 only.
- **Driver R580 or newer.** The image is built against CUDA 13.0, which has kernels for Turing
  (sm_75, so a T4 works) through Blackwell (sm_120, so a 5090 or an RTX PRO 6000 works). Volta,
  Pascal and Maxwell are not covered by CUDA 13 and will not run.
- **Disk.** The image is about 10GB. Model weights are extra and go on your volume, see below.
- **A generous `--shm-size`.** Docker defaults to 64MB, which the trainer's dataloader workers will
  exhaust. 16g is a safe figure.

## Models are not in the image

Nothing is baked in, which keeps the image small and lets you pull only what you use. On first run,
open a generate or train node in the app and use its download button. Files land in
`/workspace/models`, so they survive a restart and are shared by every container you point at the
same volume.

Rough sizes for the full set behind each node:

| Model           | Download    | Good for                                  |
| --------------- | ----------- | ----------------------------------------- |
| Z-Image Turbo   | about 20GB  | fast local images, fits a 12GB card       |
| FLUX.2 klein 4B | about 24GB  | the cheapest of the four to train         |
| Krea 2          | about 60GB  | highest quality stills, wants a big card  |
| MiniMax H3      | about 144GB | video with a jointly generated soundtrack |

## Ports

| Port | What                                 |
| ---- | ------------------------------------ |
| 8848 | Inline Studio, the app itself        |
| 8888 | JupyterLab, rooted at `/workspace`   |
| 8080 | file browser, rooted at `/workspace` |

JupyterLab and the file browser print a generated password to the container log on every start. Set
`JUPYTER_PASSWORD` and `FILEBROWSER_PASSWORD` to choose your own, or set `ENABLE_JUPYTER=0` and
`ENABLE_FILEBROWSER=0` to turn them off.

## Environment

| Variable                      | Default                    | What it does                                        |
| ----------------------------- | -------------------------- | --------------------------------------------------- |
| `INLINE_MODELS_DIR`           | `/workspace/models`        | where weights are scanned from and downloaded to    |
| `INLINE_DATA_DIR`             | `/workspace/.inline`       | run database and generated takes                    |
| `INLINE_EXTENSIONS_DIR`       | `/workspace/extensions`    | installed community extensions                      |
| `INLINE_STUDIO_DATA_DIR`      | `/workspace/inline-studio` | recents, settings, saved fal key                    |
| `INLINE_STUDIO_WORKSPACE_DIR` | `/workspace/projects`      | your `.inlinestudio` project folders                |
| `HF_HOME`                     | `/workspace/huggingface`   | Hugging Face cache for the captioner and annotators |
| `INLINE_PORT`                 | `8848`                     | port the app serves on                              |
| `INLINE_PROFILE`              | auto                       | `gpu-max`, `lowvram` or `cpu`                       |
| `INLINE_VRAM_BUDGET_GB`       | auto                       | treat the GPU as having this much usable VRAM       |
| `ENABLE_JUPYTER`              | `1`                        | JupyterLab on 8888                                  |
| `ENABLE_FILEBROWSER`          | `1`                        | file browser on 8080                                |
| `JUPYTER_PASSWORD`            | generated                  | JupyterLab token                                    |
| `FILEBROWSER_PASSWORD`        | generated                  | file browser password for user `admin`              |
| `HF_TOKEN`                    | unset                      | needed for gated repos such as FLUX.2 dev           |
| `FAL_KEY`                     | unset                      | fal key for the hosted API nodes                    |

`FAL_KEY` is only read when no key has been saved in the app yet. Once you save one in Settings it
lives on the volume and wins over the environment variable.

## LoRA training

Training is cheaper than generating. A 16GB card trains all three image models at 512px, and a LoRA
trained at 512 applies at any generation resolution.

| Card | Z-Image      | Krea 2                 | FLUX.2       | MiniMax H3  |
| ---- | ------------ | ---------------------- | ------------ | ----------- |
| 16GB | 512          | 512 in 4-bit           | 512 and 1024 | yes, slowly |
| 24GB | 512 and 1024 | 512                    | 512 and 1024 | yes         |
| 48GB | 512 and 1024 | 512, and 1024 in 4-bit | 512 and 1024 | yes         |

Open the Trainer tab, make a dataset, caption it, wire Load Dataset into Caption into Train LoRA, and
run. Finished adapters land in `/workspace/models/loras` and appear in the loader node straight away,
with a download button in the Outputs panel.

Full reference, including measured VRAM and runtimes:
[TRAINING.md](https://github.com/inlineresearch/Inline-Studio/blob/main/TRAINING.md). Walkthroughs per
model: [inlinestudio.art/lora-training](https://inlinestudio.art/lora-training).

## Tags

| Tag                  | What                                             |
| -------------------- | ------------------------------------------------ |
| `latest`             | the most recent stable release                   |
| `1.2.69` and similar | a specific release, pin this for reproducibility |

Every tag is built from the matching GitHub release by CI, so the image and the source always agree.

## Notes

- The app has **no authentication**. Do not publish port 8848 to the open internet.
- Long single requests can time out behind a reverse proxy. Generation streams over a websocket and
  is unaffected, but a very large model download started from the UI may need retrying.
- Multi-GPU generation (xDiT) is not installed in this image. It fails to build on many systems, so
  it stays an opt-in source install.

## License

GPL-3.0-or-later. Model weights carry their own licences.
