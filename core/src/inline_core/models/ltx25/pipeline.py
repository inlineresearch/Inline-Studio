"""Building and running the LTX-2.5 pipeline: resolve, size, refuse or plan, then render.

Two things here differ from every other model runner, and both are deliberate.

**LTX streams its own weights, so we choose but do not implement.** `models/offload.py` exists for
models that have no streaming of their own; layering its group offload on top of `ltx_core`'s block
streaming would mean two systems moving the same tensors. The device policy still owns the decision
- `memory.plan_for` turns its verdict into LTX's vocabulary - and the vendored loader owns the
mechanism. `models/prepared.py` is likewise unused: quantisation happens as weights stream, so there
is no separate artifact to cache.

**The text encoder and the transformer are never co-resident.** Gemma 4 is 26 GB beside a 42 GB
transformer, which nothing short of an 80 GB card holds at once, so the prompt is encoded first and
the encoder freed before the transformer loads. That decision has to be made before the load: by the
time an OOM fires there is nothing left to free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...device.policy import DevicePolicy, ModelFootprint
from ...errors import ComponentError
from .. import pipeline_runtime as rt
from . import memory
from . import requirements as reqs

if TYPE_CHECKING:
    from .runner import Request

logger = logging.getLogger(__name__)

_LABEL = "LTX-2.5"

#: NVFP4 needs Blackwell. Loading packed nibbles anywhere else does not raise, it renders noise,
#: so this is checked before the load rather than discovered in the output.
_NVFP4_MIN_CAPABILITY = 10


@dataclass
class Rendered:
    """What a run produced: RGB frames, and the soundtrack denoised alongside them."""

    frames: list[Any]
    waveform: Any = None
    sample_rate: int | None = None


def _require(path: Path | None, what: str) -> Path:
    if path is None:
        raise ComponentError(
            f"{_LABEL} is missing {what}. Download it from the node's model popup."
        )
    return path


def _wired(ref: Any) -> Path | None:
    """A path handed in over a wired model port, if one was."""
    value = getattr(ref, "path", None) if ref is not None else None
    return Path(str(value)) if value else None


def resolve_paths(params: dict[str, Any], build: str, transformer: Any, video_vae: Any,
                  text_encoder: Any) -> dict[str, Path]:
    """Every file this run needs, wired handle first, then dropdown, then the default name."""
    return {
        "transformer": _wired(transformer)
        or _require(
            reqs.resolve_transformer(build, params.get("model")),
            f"the {build} transformer",
        ),
        "text_encoder": _wired(text_encoder)
        or _require(
            reqs.resolve("text_encoders", reqs.TEXT_ENCODER_FILE, params.get("text_encoder")),
            "the Gemma 4 text encoder",
        ),
        "video_vae": _wired(video_vae)
        or _require(reqs.resolve("vae", reqs.VIDEO_VAE_FILE, params.get("vae")), "the video VAE"),
        "audio_vae": _require(reqs.resolve("vae", reqs.AUDIO_VAE_FILE), "the audio VAE"),
        "upscaler": _require(
            reqs.resolve("latent_upscale_models", reqs.SPATIAL_UPSCALER_FILE,
                         params.get("upscaler")),
            "the spatial upscaler",
        ),
    }


def _capability() -> int:
    """The accelerator's compute-capability major, or 0 off-GPU."""
    import torch

    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.get_device_capability()[0])


def _plan(policy: DevicePolicy, paths: dict[str, Path], build: str) -> memory.Ltx25Plan:
    """Size the model, ask the policy, and translate its verdict into LTX's vocabulary."""
    sizes = reqs.footprint_bytes(
        build, transformer=paths["transformer"], video_vae=paths["video_vae"]
    )
    # The encoder is freed before the transformer loads, so it is not part of the peak the ladder
    # sizes against - the same staged accounting the model popup's estimate uses.
    policy.set_footprint(
        ModelFootprint(
            diffusion_bytes=sizes["diffusion_bytes"],
            text_encoder_bytes=0,
            vae_bytes=sizes["vae_bytes"],
        )
    )
    fit = policy.fit_estimate()
    prequantised = reqs.inspect_file(paths["transformer"]).quantisation == "nvfp4"
    if prequantised and _capability() < _NVFP4_MIN_CAPABILITY:
        raise ComponentError(
            f"{_LABEL}: this is the NVFP4 build, which needs a Blackwell card and the ltx-kernels "
            "package. Use the bf16 transformer instead."
        )
    plan = memory.plan_for(
        fit_plan=fit.plan if fit else "offload",
        model_bytes=sizes["diffusion_bytes"],
        total_vram_bytes=int((policy.vram_budget_mb() or 0) * 1024**2),
        free_ram_bytes=int((policy.free_ram_mb() or 0) * 1024**2),
        prequantised=prequantised,
        kernels_available=memory.kernels_available(),
    )
    if plan is None:
        raise ComponentError(
            f"{_LABEL} will not run on this machine. Even streaming from disk it needs about 5 GB "
            "of VRAM for the working buffer."
            + (f" {fit.note}" if fit and fit.note else "")
        )
    logger.info("%s: %s (%s)", _LABEL, plan.note, rt.device_report(policy))
    return plan


def load_pipeline(
    policy: DevicePolicy,
    *,
    params: dict[str, Any],
    request: Request,
    transformer: Any = None,
    video_vae: Any = None,
    text_encoder: Any = None,
    loras: tuple[Any, ...] = (),
) -> Any:
    """The pipeline for this request, built against a plan the policy chose."""
    from .vendor.ltx_pipelines.utils.model_paths import ModelPaths

    paths = resolve_paths(params, request.build, transformer, video_vae, text_encoder)
    plan = _plan(policy, paths, request.build)

    model_paths = ModelPaths.from_split(
        transformer_path=str(paths["transformer"]),
        text_encoder_path=str(paths["text_encoder"]),
        video_vae_path=str(paths["video_vae"]),
        audio_vae_path=str(paths["audio_vae"]) if request.generate_audio else None,
    )
    common: dict[str, Any] = {
        "model_paths": model_paths,
        "spatial_upsampler_path": str(paths["upscaler"]),
        "loras": [_lora(ref) for ref in loras],
        "quantization": memory.quantization_policy(plan, str(paths["transformer"])),
        "offload_mode": memory.offload_mode(plan),
    }
    if request.mode == "quality":
        return _quality_pipeline(common)
    from .vendor.ltx_pipelines.distilled import DistilledPipeline

    return DistilledPipeline(**common)


def _quality_pipeline(common: dict[str, Any]) -> Any:
    """The guided two-stage pipeline, which refines stage 2 with the published distilled LoRA."""
    from .vendor.ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

    lora = _require(
        reqs.resolve("loras", reqs.DISTILLED_LORA_FILE),
        "the distilled LoRA, which quality mode uses to refine its second stage",
    )
    return TI2VidTwoStagesPipeline(**common, distilled_lora=str(lora))


def _lora(ref: Any) -> Any:
    """One of our `LoraRef`s as the vendored path-and-strength tuple."""
    from .vendor.ltx_core.loader import LoraPathStrengthAndSDOps

    path = reqs.resolve("loras", str(getattr(ref, "file", "")))
    if path is None:
        raise ComponentError(f"{_LABEL} could not find the LoRA {getattr(ref, 'file', '')!r}.")
    return LoraPathStrengthAndSDOps(str(path), float(getattr(ref, "strength", 1.0)))


def render(
    pipe: Any,
    request: Request,
    call: dict[str, Any],
    *,
    on_step: Any = None,
    cancel_check: Any = None,
) -> Rendered:
    """Run the pipeline and materialise its output.

    The pipeline returns a frame **iterator**, not a list: decode happens as it is consumed, which
    is what keeps a 20-second clip from existing as one tensor. Draining it here is where decode
    actually costs its time, so the cancel check runs per chunk rather than only between stages.
    """
    rt.attach_step_progress(pipe, on_step)
    frames_iter, audio, sample_rate, _tiling = pipe(**call)

    frames: list[Any] = []
    for chunk in frames_iter:
        if cancel_check is not None:
            cancel_check()
        frames.extend(chunk if isinstance(chunk, list) else [chunk])

    waveform = getattr(audio, "waveform", None) if request.generate_audio else None
    rt.free_vram()
    return Rendered(frames=frames, waveform=waveform, sample_rate=sample_rate)
