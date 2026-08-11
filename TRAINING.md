# Train a LoRA locally on your own GPU

The full reference for Inline Studio's **Trainer**: which base to train on, what it costs in VRAM
on a real card, and every setting that shapes the result. For the short version and a screenshot of
the canvas, see [LoRA training in the README](README.md#lora-training).

Inline Studio trains LoRAs for **Z-Image**, **Krea 2**, **FLUX.2** and **MiniMax H3** on your own
GPU, with no cloud step and nothing uploaded. Training is cheaper than generating: a 16GB card
trains all three image models at 512px, and a LoRA trained at 512 applies at any generation
resolution.

**Contents:** [The graph](#the-graph) · [Datasets and outputs](#datasets-and-outputs) ·
[Stop and resume](#stop-and-resume) · [Trigger words](#trigger-words) ·
[Architecture and base model modes](#architecture-and-base-model-modes) · [Install](#install) ·
[Training on clips](#training-on-clips) · [**Benchmark results**](#benchmark-results) ·
[Dataset and adapter options](#dataset-and-adapter-options) · [Base precision](#base-precision)

## The graph

Five nodes, wired left to right:

- **Load Dataset** picks a training dataset and feeds it downstream. The node face stays a preview (thumbnails, image and caption counts); the images and captions themselves are edited in the side panel.
- **Caption** runs a local captioner over the images that need one, with per-image progress. Captions stay editable afterwards, and a wired dataset overrides the node's own picker.
- **Train LoRA** runs the job. Hyperparameters live behind the Adjust button, off the node face, so the node stays a status surface: a live step counter, the trainer's streaming logs, and a progress bar. The run control is a single chip that reads Start, Stop, or Resume depending on where the run is.
- **Graph** plots the loss curve for whichever run is wired into it, with loss values on the y axis and the step range on the x axis.
- **Resources** is a read-only readout of CPU, RAM, and VRAM as circular gauges. It takes no connections, and you can drop it on the Studio canvas too.

## Datasets and outputs

The sidebar has two tabs. **Datasets** is where you create a dataset, give it a trigger word, add images (drag and drop from your file manager works), and edit captions. **Outputs** lists what training has produced: finished LoRAs with their rank, step count, and resolution, plus any run that stopped early, each with a Resume button.

For a worked example, see [`inlineresearch/skin-lora-krea-2-raw`](https://huggingface.co/inlineresearch/skin-lora-krea-2-raw) - a photorealistic skin LoRA trained here on the Krea 2 RAW base from the 26 image and caption pairs published as [`inlineresearch/krea2-skin-lora`](https://huggingface.co/datasets/inlineresearch/krea2-skin-lora).

## Stop and resume

Changing a setting in the Adjust panel stages it behind an **Update** button rather than applying as you type. A checkpoint encodes the rank, LoRA targets and base it was built with, so if the node has a run you could resume, applying asks first and then discards that run's checkpoints. Finished runs' LoRA files are never touched.

Stopping a run flushes a checkpoint before the process exits, so Resume continues from the step it left off instead of starting over. A checkpoint holds the adapter weights, the optimiser state, the RNG state, and the step number, which is what makes a resumed run a continuation rather than a restart. Runs cut short by a crash or a server restart are recovered the same way and show up under Outputs ready to resume.

## Trigger words

A dataset's trigger word is prepended to every caption during training, so the model sees captions in the form `mytoken, a photo of ...`. Put the same token at the front of your prompt to pull the LoRA in. It is worth matching the phrasing of your captions too: if they all say "an oil painting of", a prompt written the same way will hit the trained style far more reliably than the trigger word alone.

## Architecture and base model modes

The Trainer's Adjust panel picks the **architecture** first (Z-Image, Krea 2, FLUX.2, or MiniMax H3), then a base within it. Training directly on a step-distilled checkpoint breaks the distillation down (turbo drift), so each architecture offers a way around that.

**Krea 2** avoids the problem outright, which is why it is the recommended path:

- **Krea 2 RAW** trains on the undistilled base. Nothing to fuse, nothing to drift. Put `krea2_raw_bf16.safetensors` in `models/diffusion_models/`, train, then generate with the **Krea 2 Turbo** node - the LoRA carries over unchanged.
- **Krea 2 Turbo + training adapter** exists for people who only hold Turbo. Put [ostris/krea2_turbo_training_adapter](https://huggingface.co/ostris/krea2_turbo_training_adapter) in `models/loras/`, or point `INLINE_KREA2_TRAIN_ADAPTER` at it.

**FLUX.2** works like Krea 2, with no adapter to download:

- **FLUX.2 Base** is the only option, and the trainer refuses a distilled checkpoint rather than letting a run produce a bad adapter hours later. Put `flux-2-klein-base-4b.safetensors` in `models/diffusion_models/`, train, then generate with the distilled **klein 4B** checkpoint. The LoRA carries over unchanged.

**MiniMax H3** is the video model, and it trains on **still images**:

- **FL2VA** is the only base, and it is undistilled, so there is no adapter and nothing to drift. Put `minimax_h3_fl2va_bf16.safetensors` in `models/diffusion_models/`, train on stills, then wire the LoRA into any of the four H3 nodes. **It has to be the bf16 file.** The smaller `pruned` and `pruned_fp8_scaled` builds generate but cannot train: they ship no timestep path for the modulation basis to be derived from, and they would save nothing anyway, because the base trains at 4-bit whichever file it starts from. The trainer says so rather than failing part way in. It loads on the Reference to Video node too, which uses a different checkpoint file: the two partitions are the same architecture.
- **Stills or short clips.** Drop images and it learns appearance: look, style, character, lighting. Drop video and it learns motion too. Sound is never learned either way, because the audio rows are empty. See [Training on clips](#training-on-clips).
- **The base is 4-bit, always.** H3 is 40GB after the AdaLN factorisation and 11.7GB after quantisation, so full precision is refused rather than offered and then failing. There is no base-precision control for H3 for the same reason.
- **A 24GB card is comfortable and a 16GB card works, slowly.** The run encodes latents and captions in two passes that never overlap, because H3's fp32 video VAE and its 32B conditioner cannot be resident together. On a card that holds the conditioner it peaks at 20.6GB; on one that does not, the conditioner runs on the CPU and the peak drops to 12.7GB while a step goes from 0.6s to 16s. Either way there is about seven minutes of startup, and 64GB of system RAM for the smaller card. See [Benchmark results](#benchmark-results) for the split. The download is about 124GB before any of that.

**Z-Image** is distilled either way:

- **Turbo + training adapter** fuses a de-distillation adapter into the base for the duration of training and drops it when the LoRA is saved, which preserves the 8-step speed. Put [ostris/zimage_turbo_training_adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter) in `models/loras/`; any filename containing `adapter` is detected automatically, or point `INLINE_ZIMAGE_TRAIN_ADAPTER` at a specific file. Keep runs short, since the adapter slows the breakdown rather than preventing it.
- **De-Turbo** trains without an adapter and needs no extra download.

## Training on clips

The H3 trainer takes video as well as stills. Drop clips into a dataset the same way, set **Clip
length** in the Adjust panel, and each clip trains as a short piece of motion rather than a frame.
Mixed datasets are fine: a still is simply a one-frame clip.

**It costs no extra VRAM.** Measured on an L4, every clip length peaks at the same 20.4GB as a
still, because the high-water mark is the caption pass rather than the training:

| Clip length | Frames | Latent frames | Packed rows at 512px | Peak VRAM |
| ----------- | ------ | ------------- | -------------------- | --------- |
| still       | 1      | 1             | 293                  | 20.55GB   |
| 0.92s       | 22     | 7             | 1,832                | 20.4GB    |
| 1.6s        | 39     | 12            | 3,112                | 20.4GB    |
| 4.5s        | 107    | 32            | 8,232                | 20.4GB    |

Rows are what a longer clip actually buys you, and they cost time rather than memory. That only
holds while the conditioner is resident; on a card too small for it the peak is the training phase
instead, and a long clip will push that up.

**Lengths snap to H3's frame grid.** The VAE encodes `17n + 5` frames at 24fps, so a request lands
on the nearest grid point at or below it. The floor is a whole chunk plus the five-frame head: 22
frames, **0.92 seconds**. Asking for less rounds up rather than being refused, because the VAE has
no way to encode a shorter clip.

**Each clip is trimmed from its start, once.** The window is fixed at precache time so every clip is
encoded exactly once. Sampling a different window each step would mean re-encoding through the VAE
every step, which is the thing the precache exists to avoid. A clip shorter than the grid floor is
refused by name rather than silently padded.

**Captions work the same.** A clip is auto-captioned from its middle frame, which describes the shot
better than the first frame usually does. Write them by hand if you would rather.

Audio is not trained. H3 generates video and its soundtrack jointly, but the trainer packs zero
audio rows, so an adapter changes what a clip looks like and never what it sounds like.

## Install

If you installed with `--extra all` from [Get Started](README.md#get-started), the trainer is already set up - nothing more to do. To add it to a leaner install, its dependencies (PEFT, 8-bit Adam, the captioner) sit behind the `training` extra:

```bash
cd core
./webui.sh --install --extra training   # Windows: .\webui.bat --install --extra training
```

Nothing is downloaded behind your back. Training has no downloader of its own: it reuses whatever is already in `models/diffusion_models/`, `models/vae/` and `models/text_encoders/` for the architecture you pick, which is normally what a generate node's model popup fetched for you. If a file is missing, the run stops and names it.

Two things the model popup does not cover, so you fetch them yourself:

- **Training adapters** for the Turbo base modes: [Z-Image](https://huggingface.co/ostris/zimage_turbo_training_adapter) or [Krea 2](https://huggingface.co/ostris/krea2_turbo_training_adapter), dropped in `models/loras/`.
- **The captioner**, fetched once into the Hugging Face cache the first time you press Auto-caption.

The LoRA a run produces lands in `models/loras/` and shows up in the LoRA loader node straight away, so you can wire it into a generate node and try it without leaving the app.

## Benchmark results

12 steps at rank 16, batch 1, gradient checkpointing on. The number is `torch.cuda.max_memory_allocated`, so leave headroom for the CUDA context and allocator slack.

| Model      | Base mode       | Res  | Base precision | L40S (46GB)   | T4 (15GB)     |
| ---------- | --------------- | ---- | -------------- | ------------- | ------------- |
| Z-Image    | De-Turbo        | 512  | bf16           | 13.1GB        | 13.4GB        |
| Z-Image    | De-Turbo        | 1024 | bf16           | 14.9GB        | out of memory |
| Z-Image    | Turbo + adapter | 512  | bf16           | 13.1GB        | 13.4GB        |
| Z-Image    | Turbo + adapter | 1024 | bf16           | 14.9GB        | out of memory |
| Krea 2     | RAW             | 512  | bf16           | 30.4GB        | out of memory |
| Krea 2     | RAW             | 512  | **4-bit**      | 11.7GB        | **11.9GB**    |
| Krea 2     | RAW             | 1024 | bf16           | out of memory | out of memory |
| Krea 2     | RAW             | 1024 | **4-bit**      | **27.8GB**    | out of memory |
| Krea 2     | Turbo + adapter | 512  | bf16           | 30.4GB        | out of memory |
| Krea 2     | Turbo + adapter | 512  | **4-bit**      | 11.7GB        | **11.9GB**    |
| Krea 2     | Turbo + adapter | 1024 | bf16           | out of memory | out of memory |
| Krea 2     | Turbo + adapter | 1024 | **4-bit**      | **27.8GB**    | out of memory |
| FLUX.2     | Base (klein 4B) | 512  | bf16           | 8.6GB         | not measured  |
| FLUX.2     | Base (klein 4B) | 512  | **4-bit**      | 8.6GB         | not measured  |
| FLUX.2     | Base (klein 4B) | 1024 | bf16           | 9.9GB         | not measured  |
| FLUX.2     | Base (klein 4B) | 1024 | **4-bit**      | 9.9GB         | not measured  |
| MiniMax H3 | FL2VA           | 512  | **4-bit**      | **20.6GB**    | **12.7GB**    |
| MiniMax H3 | FL2VA           | 768  | **4-bit**      | **20.6GB**    | not measured  |
| MiniMax H3 | FL2VA           | 1024 | **4-bit**      | **20.6GB**    | not measured  |
| MiniMax H3 | FL2VA, clips    | 512  | **4-bit**      | **20.4GB**    | not measured  |

**MiniMax H3 costs less on a smaller card, which is not a typo.** The run has three phases that never overlap, and the tallest is not the one doing the learning:

| Phase                      | Peak on an L40S | What is resident                               |
| -------------------------- | --------------- | ---------------------------------------------- |
| Latent caching (video VAE) | 10.8GB          | The fp32 video VAE, then dropped               |
| Caption caching (Qwen3-VL) | 20.5GB          | The 32B conditioner at 4-bit, then dropped     |
| Training                   | 11.7GB          | The 4-bit base, 62GB on disk, plus activations |

The caption pass sets the peak, so on a card that can hold the conditioner the answer is 20.6GB whatever the resolution: 512, 768 and 1024 all read the same, because a 512px still is 310 rows of packed sequence against 630 at 768px and the weights are the cost, not the activations. The AdaLN factorisation is what makes the base figure possible at all, taking the transformer from 62GB on disk to 40GB before quantisation and 11.7GB after. Host RAM stays near 1.1GB during that load, because each block is shrunk as its tensors land.

On a card too small for the conditioner it never goes there at all, so the peak drops to the training phase: **12.7GB, measured on a Tesla T4**. A 16GB card therefore trains H3 where a 24GB card is merely comfortable.

**The bill arrives as time instead.** The conditioner runs on the CPU, and bitsandbytes only quantises on the move to CUDA, so it runs unquantised:

|                         | L40S (46GB) | L4 (24GB) | T4 (16GB, 64GB RAM) |
| ----------------------- | ----------- | --------- | ------------------- |
| Peak VRAM, 512px        | 20.6GB      | 20.55GB   | 12.7GB              |
| Seconds per step, 512px | 0.63        | 1.81      | 16.2                |
| Seconds per step, 768px | 0.77        | 2.73      | not measured        |
| Caption pass, 26 images | 1 min       | 1 min     | 19 min              |

A 1500-step run at 512px is about 16 minutes on the L40S, 45 on the L4, and closer to seven hours on the T4. The L4 holds the conditioner, so it looks like a slower L40S rather than a faster T4: the 9x gap to the T4 is mostly the caption pass being on the wrong processor, not the cards themselves.

**It also wants a lot of system RAM.** The unquantised conditioner pages roughly 63GB through the page cache, and on a 64GB machine that sits at 59GB resident, close enough to the edge that the caption pass is the riskiest part of the run. A T4 with only 16GB of RAM has room in neither VRAM nor RAM and is refused before anything loads, because a host-RAM overrun is killed by the kernel rather than raising.

Narrowing the LoRA will not buy the difference: at rank 16 the whole adapter is 87M parameters, and dropping to attention-only at rank 8 saves 0.4GB out of 13GB. The base is roughly 90 percent of the budget. The fix that would matter is streaming the conditioner to the card in 4-bit slices, the way the generation path already does, which is not built for training yet.

A training adapter is free: it is fused into the base before training starts, so Turbo-plus-adapter and the undistilled base peak identically.

**FLUX.2 is the cheapest of the three to train, and 4-bit does nothing for it.** Both precisions peak at the same number because the peak is not the transformer: klein's base is 7.4GB while its Qwen3-4B text encoder is 7.5GB, so the caption and latent caching pass at the start of the run costs more than training itself does. Dropping the frozen base to 4-bit shrinks a part of the run that was never the high-water mark, and the step gets slower for nothing. Leave base precision on Auto for FLUX.2, which is what it already picks. The rows above are klein Base 4B, the only checkpoint the trainer accepts for this architecture.

Which card fits what (24GB and 32GB are interpolated, not measured, as are the FLUX.2 columns on 16GB: those peaks were measured on an L40S and leave room on a smaller card, but no 16GB run has been done):

| Card | Z-Image 512 | Z-Image 1024 | Krea 2 512 | Krea 2 1024 | FLUX.2 512 | FLUX.2 1024 | MiniMax H3  |
| ---- | ----------- | ------------ | ---------- | ----------- | ---------- | ----------- | ----------- |
| 16GB | yes         | no           | yes, 4-bit | no          | yes        | yes         | yes, slowly |
| 24GB | yes         | yes          | yes        | no          | yes        | yes         | yes         |
| 32GB | yes         | yes          | yes        | yes, 4-bit  | yes        | yes         | yes         |
| 48GB | yes         | yes          | yes        | 4-bit only  | yes        | yes         | yes         |

H3 has one column because resolution barely moves it. The 16GB entry is measured on a T4 with 64GB of RAM, where the conditioner spills to the CPU: it fits in 12.7GB of VRAM but costs 16.2s a step and a 19 minute caption pass. The 24GB entry is interpolated from the 20.6GB peak, not measured on a 24GB card. A 16GB card with only 16GB of RAM is refused up front.

Fitting and being usable are different questions. Turing has no native bf16, so a T4 runs the same work about 4x slower:

| Configuration                     | L40S | T4   |
| --------------------------------- | ---- | ---- |
| Krea 2 RAW 512, 4-bit             | 192s | 824s |
| Krea 2 Turbo + adapter 512, 4-bit | 219s | 872s |
| Z-Image 512                       | 85s  | 285s |

A 1500-step Krea 2 run is roughly 40 minutes on an L40S and 3 hours on a T4.

FLUX.2 is quicker than either. On an L40S, klein Base 4B trains at about 0.3s a step at 512 and 1.0s a step at 1024, so a 1500-step run comes in around 8 minutes at 512 and 25 minutes at 1024. Forcing the 4-bit base costs about 10 percent a step at both resolutions. FLUX.2 has not been timed on a T4.

**MiniMax H3's steps are fast and its startup is not.** On an L40S a step is about 0.63s at 512 and 0.77s at 768, so a 1500-step run is roughly 16 to 19 minutes of actual training. Getting there takes about 7 minutes first: the 62GB checkpoint streams block by block while each one is factorised and quantised, and the two caching passes run before it. Startup is per run and does not scale with steps, so it hurts a short run far more than a long one. On a T4 a step is 16.2s, and the caption pass adds 19 minutes on top, because the conditioner runs unquantised on the CPU there.

**Krea 2 at 512 with the 4-bit base is the configuration to reach for on a small card.** 1024 needs about 32GB and no setting closes that gap: activations scale with image tokens, and gradient checkpointing and memory-efficient attention are already on. Train at 512 instead, since a LoRA trained at 512 applies at any generation resolution.

System RAM matters as well. Checkpoints are read tensor by tensor rather than mapped whole, so Krea 2 trains in about 3GB of host RAM. Without that, Linux refuses to map a file larger than physical RAM when there is no swap, and a 26GB checkpoint cannot be opened on a 16GB machine at all.

## Dataset and adapter options

Three settings shape what the adapter learns rather than what it costs:

- **LoRA scope.** _Full_ adapts the attention and feed-forward layers, which is stronger on short style runs. _Attention only_ is the Krea 2 authors' advice for long runs, where adapting everything starts to cost prompt adherence.
- **Caption dropout** (default 0.05) trains a fraction of steps against an empty caption, so the LoRA still holds when a prompt does not repeat the trigger word verbatim.
- **Flip images** mirrors every image, doubling a small dataset. Both orientations are encoded from pixels rather than by flipping cached latents, so the mirrored copy is exact. Leave it off for anything with text or a deliberate asymmetry.

## Base precision

Krea 2's base is 26GB at bf16, which is what makes it expensive to fine-tune. The Trainer's **Base precision** setting freezes that base at 4-bit (NF4) while the LoRA itself stays full precision - the QLoRA arrangement - so only the frozen base loses fidelity:

- **Auto** (default) sizes the base _plus its activations at your chosen resolution_ against your GPU and picks for you. Weights alone are not enough to decide: a 48GB card holds Krea 2's 26GB base comfortably and then runs out at 1024.
- **Full precision (bf16)** forces the unquantized base.
- **4-bit (NF4)** forces the quantized base.

The setting appears for Krea 2 and FLUX.2, but it only pays off on Krea 2. Z-Image has no 4-bit path and does not need one: it trains in about 15 GB at 1024, so bf16 already fits the cards people have. FLUX.2 has the path and gains nothing from it, because klein 4B is smaller than its own text encoder and the peak sits in the caching pass either way, so Auto leaves it at bf16. See [Benchmark results](#benchmark-results).

To keep the peak down, the VAE and text encoder are loaded first, used to cache latents and captions, then freed before the transformer loads, so the two never stack. Which half then owns the peak depends on the model: for Z-Image and Krea 2 it is the transformer, for FLUX.2 klein it is the caching pass. If you do hit an out-of-memory error, lower the training resolution before changing anything else.

---

Per-model walkthroughs, with the same measured figures written for a first read rather than a
reference: [Krea 2](https://inlinestudio.art/lora-training/krea-2) ·
[Z-Image](https://inlinestudio.art/lora-training/z-image) ·
[FLUX.2](https://inlinestudio.art/lora-training/flux-2). Back to the
[README](README.md), or the [full guide on the site](https://inlinestudio.art/lora-training).
