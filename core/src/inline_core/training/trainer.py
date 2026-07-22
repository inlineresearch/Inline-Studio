"""The LoRA training loop: PEFT adapter on the Z-Image transformer, rectified-flow loss, checkpoint/
resume, and a loader-compatible ``.safetensors`` at the end.

Uses ``accelerate`` so the same code runs single-GPU or, under ``accelerate launch --multi_gpu``,
DDP (only the small LoRA grads all-reduce). Progress + samples + checkpoints are reported as JSON
lines (``protocol.py``). A SIGTERM flushes a checkpoint and returns ``None`` (a resumable cancel).

NOTE (needs a GPU + Z-Image weights to finalize): three Z-Image specifics must be validated against
the runner + the ai-toolkit reference before this trains a good LoRA -
  1. the transformer ``forward`` kwargs / output attribute (``_forward``),
  2. the LoRA ``target_modules`` for ``ZImageTransformer2DModel`` (``_TARGET_MODULES``),
  3. the flow-match timestep scaling + velocity target (``_timesteps`` / the loss).
These are the "study ai-toolkit" items from the plan, not guesses to ship blind.
"""

from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any

from . import dataset as ds
from . import models, protocol

# Attention + FFN projections a Z-Image LoRA typically targets. VALIDATE against the module tree.
_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]


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


def _timesteps(batch: int, device: Any) -> Any:
    import torch

    # Logit-normal sampling of t in (0, 1) - denser near the middle, as flow-match trainers favor.
    return torch.sigmoid(torch.randn(batch, device=device))


def _save_checkpoint(accelerator: Any, ckpt_dir: Path, step: int) -> None:
    accelerator.save_state(str(ckpt_dir))
    if accelerator.is_main_process:
        (ckpt_dir / "step.json").write_text(json.dumps({"step": step}), encoding="utf-8")


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


def _forward(transformer: Any, noisy: Any, timestep: Any, embeds: Any) -> Any:
    """One transformer prediction. VALIDATE the kwargs/return attr against the Z-Image runner."""
    out = transformer(
        hidden_states=noisy, timestep=timestep, encoder_hidden_states=embeds, return_dict=True
    )
    return out.sample if hasattr(out, "sample") else out[0]


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
    if resume_from and (Path(resume_from) / "step.json").exists():
        accelerator.load_state(resume_from)
        start = _resume_step(Path(resume_from))

    stop = _Stop()
    signal.signal(signal.SIGTERM, stop)

    transformer.train()
    for step in range(start, steps):
        if stop.flagged:
            break
        item = data[step % len(data)]
        latents = item["latent"].unsqueeze(0).to(device, dtype)
        embeds = item["embed"].unsqueeze(0).to(device, dtype)
        noise = torch.randn_like(latents)
        t = _timesteps(latents.shape[0], device)
        t_broadcast = t.view(-1, *([1] * (latents.ndim - 1)))
        noisy = (1 - t_broadcast) * latents + t_broadcast * noise
        target = noise - latents  # rectified-flow velocity target

        with accelerator.accumulate(transformer):
            pred = _forward(transformer, noisy, t * 1000.0, embeds)
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        done = step + 1
        protocol.progress(done, steps, loss=float(loss.detach().item()), status="training")
        if done % save_every == 0 or done == steps:
            _save_checkpoint(accelerator, ckpt_dir, done)
            if accelerator.is_main_process:
                protocol.checkpoint(str(ckpt_dir))

    if stop.flagged:
        _save_checkpoint(accelerator, ckpt_dir, step)
        return None  # cooperative cancel: a checkpoint exists, so the run is resumable

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_lora(accelerator.unwrap_model(transformer), manifest["outputPath"])
    return manifest["outputPath"]
