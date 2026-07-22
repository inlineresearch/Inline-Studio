"""The LoRA training loop: PEFT adapter on the Z-Image transformer, rectified-flow loss, checkpoint/
resume, and a loader-compatible ``.safetensors`` at the end.

Uses ``accelerate`` so the same code runs single-GPU or, under ``accelerate launch --multi_gpu``,
DDP (only the small LoRA grads all-reduce). Progress + samples + checkpoints are reported as JSON
lines (``protocol.py``). A SIGTERM flushes a checkpoint and returns ``None`` (a resumable cancel).

The three Z-Image specifics were validated against the real ``ZImageTransformer2DModel`` +
``ZImagePipeline`` on a GPU (see ``_forward``, ``_TARGET_MODULES``, ``_timesteps`` / the loss):
  1. ``forward`` takes per-image lists + a normalized timestep and returns ``.sample`` (a list),
  2. the LoRA targets are the attention + w1/w2/w3 feed-forward Linears,
  3. the flow-match target is (clean - noise) at timestep (1 - sigma).
"""

from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any

from . import dataset as ds
from . import models, protocol

# Every ZImageTransformerBlock's attention (q/k/v/out) + SwiGLU feed-forward (w1/w2/w3) Linears -
# confirmed against ZImageTransformer2DModel.named_modules() (34 blocks, 238 Linears). Z-Image's
# FeedForward is w1/w2/w3, NOT the diffusers-generic ff.net.*, and `to_out.0` matches only the
# attention output (not the adaLN Linear that also ends in `.0`).
_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]


class _Stop:
    """A SIGTERM latch: the orchestrator's cancel asks the loop to checkpoint and stop."""

    flagged = False

    def __call__(self, *_a: Any) -> None:
        self.flagged = True


def _optimizer(params: list[Any], lr: float) -> Any:
    try:
        import bitsandbytes as bnb

        return bnb.optim.Adam8bit(params, lr=lr)  # 8-bit Adam keeps optimizer state small
    except Exception:  # noqa: BLE001 - bitsandbytes is optional; fall back to torch AdamW
        import torch

        return torch.optim.AdamW(params, lr=lr)


def _timesteps(device: Any, shift: float) -> Any:
    import torch

    # Logit-normal sampling of the noise fraction in (0, 1) - denser near the middle, as flow-match
    # trainers favor - then Z-Image's static resolution shift (scheduler `shift`,
    # use_dynamic_shifting=False) so the training noise levels match the schedule inference visits.
    u = torch.sigmoid(torch.randn((), device=device))
    return shift * u / (1.0 + (shift - 1.0) * u)


def _save_checkpoint(
    accelerator: Any, transformer: Any, optimizer: Any, ckpt_dir: Path, step: int
) -> None:
    """Adapter-only checkpoint: the LoRA weights + optimizer + RNG + step - NOT the frozen ~12GB
    base (which reloads identically from its file every run). ``accelerator.save_state`` would write
    the whole base each time; a rank-8 adapter is a few MB, so checkpointing stays cheap and resume
    is exact. Rank-0 writes; the small files are read by every rank on resume."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    model = accelerator.unwrap_model(transformer)
    adapter = {k: v.detach().to("cpu") for k, v in get_peft_model_state_dict(model).items()}
    save_file(adapter, str(ckpt_dir / "adapter.safetensors"))
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    rng: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state()
    torch.save(rng, ckpt_dir / "rng.pt")
    (ckpt_dir / "step.json").write_text(json.dumps({"step": step}), encoding="utf-8")


def _load_checkpoint(accelerator: Any, transformer: Any, optimizer: Any, resume_from: Path) -> int:
    """Restore an adapter-only checkpoint (see ``_save_checkpoint``) on every rank and return the
    step to continue from."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    model = accelerator.unwrap_model(transformer)
    set_peft_model_state_dict(model, load_file(str(resume_from / "adapter.safetensors")))
    optimizer.load_state_dict(torch.load(resume_from / "optimizer.pt", weights_only=False))
    if (resume_from / "rng.pt").is_file():
        rng = torch.load(resume_from / "rng.pt", weights_only=False)
        torch.set_rng_state(rng["cpu"])
        if torch.cuda.is_available() and "cuda" in rng:
            torch.cuda.set_rng_state(rng["cuda"])
    return _resume_step(resume_from)


def _resume_step(ckpt_dir: Path) -> int:
    meta = ckpt_dir / "step.json"
    if meta.exists():
        try:
            return int(json.loads(meta.read_text(encoding="utf-8"))["step"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0
    return 0


def _save_lora(transformer: Any, output_path: str) -> None:
    """Write the PEFT adapter as safetensors. Its ``base_model.model...lora_A/lora_B`` keys are read
    directly by the loader's fuser (``models/lora.py`` strips the ``base_model.model.`` prefix)."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    state = {
        k: v.detach().to("cpu", dtype=torch.float32).contiguous()
        for k, v in get_peft_model_state_dict(transformer).items()
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, output_path)


def _forward(transformer: Any, noisy: Any, t_norm: Any, embed: Any) -> Any:
    """One prediction from the real ZImageTransformer2DModel, mirroring ZImagePipeline's call.

    The model takes per-image LISTS - latents as (C, F, H, W) with a temporal axis, captions as
    (seq, dim) - a NORMALIZED timestep (1=clean, 0=noise; the model multiplies by t_scale=1000
    itself, so we must not pre-scale), and returns per-image latents in ``.sample`` (a list).
    Returns this single image's (C, H, W) prediction."""
    out = transformer(
        [noisy.unsqueeze(1)],  # (C, H, W) -> [(C, 1, H, W)]
        t_norm.reshape(1),  # (1,) per-image timestep
        [embed],  # [(seq, dim)]
        return_dict=True,
    )
    sample = out.sample if hasattr(out, "sample") else out[0]
    return sample[0].squeeze(1)  # (C, 1, H, W) -> (C, H, W)


def train(manifest: dict[str, Any]) -> str | None:
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig

    hp = manifest["hyperparams"]
    steps = int(hp["steps"])
    save_every = max(1, int(hp.get("saveEvery", 250)))
    resolution = int(hp.get("resolution", 1024))
    ckpt_dir = Path(manifest["checkpointDir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=max(1, int(hp.get("batchSize", 1))))
    device = accelerator.device
    dtype = models.compute_dtype()

    protocol.progress(0, steps, status="loading models")
    comps = models.load_components(manifest["modelsDir"], manifest["baseMode"], str(device), dtype)

    protocol.progress(0, steps, status="caching latents")
    data = ds.precache(manifest["datasetDir"], comps, str(device), dtype, resolution)
    # Precache done: the VAE + text encoder are dead weight for the loop - move them off the GPU so
    # the loop only holds the transformer + LoRA + optimizer (the big low-VRAM win).
    comps.vae.to("cpu")
    comps.text_encoder.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    transformer = comps.transformer
    transformer.requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r=int(hp["rank"]),
            lora_alpha=int(hp.get("alpha") or hp["rank"]),
            lora_dropout=0.0,
            target_modules=_TARGET_MODULES,
        )
    )
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = _optimizer(lora_params, float(hp["learningRate"]))
    transformer, optimizer = accelerator.prepare(transformer, optimizer)

    start = 0
    resume_from = manifest.get("resumeFrom")
    if resume_from and (Path(resume_from) / "adapter.safetensors").exists():
        start = _load_checkpoint(accelerator, transformer, optimizer, Path(resume_from))

    stop = _Stop()
    signal.signal(signal.SIGTERM, stop)

    shift = float(comps.scheduler.config.get("shift", 1.0) or 1.0)
    transformer.train()
    for step in range(start, steps):
        if stop.flagged:
            break
        item = data[step % len(data)]
        clean = item["latent"].to(device, dtype)  # (C, H, W)
        embed = item["embed"].to(device, dtype)  # (seq, dim)
        noise = torch.randn_like(clean)
        sigma = _timesteps(device, shift)  # scalar noise fraction in (0, 1)
        noisy = (1 - sigma) * clean + sigma * noise
        # Z-Image's transformer is trained to predict (clean - noise): the pipeline NEGATES its
        # output before handing it to FlowMatchEuler as the (noise - clean) velocity, so the raw
        # output target is the negation of that. Timestep is 1 - sigma (1=clean, 0=noise).
        target = clean - noise

        with accelerator.accumulate(transformer):
            pred = _forward(transformer, noisy, 1.0 - sigma, embed)
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        done = step + 1
        protocol.progress(done, steps, loss=float(loss.detach().item()), status="training")
        if done % save_every == 0 or done == steps:
            _save_checkpoint(accelerator, transformer, optimizer, ckpt_dir, done)
            if accelerator.is_main_process:
                protocol.checkpoint(str(ckpt_dir))

    if stop.flagged:
        _save_checkpoint(accelerator, transformer, optimizer, ckpt_dir, step)
        return None  # cooperative cancel: a checkpoint exists, so the run is resumable

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_lora(accelerator.unwrap_model(transformer), manifest["outputPath"])
    return manifest["outputPath"]
