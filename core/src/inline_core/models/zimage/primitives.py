"""Diffusers-backed runners for the decomposed Z-Image primitive nodes.

This is the C2 companion to ``graph/primitives.py``: it turns the descriptor-only ``encode/text``,
``latent/empty``, ``sample``, ``vae/decode`` and ``vae/encode`` nodes into runnable nodes, so a
graph wired purely from primitives (``load/*`` -> ``encode/text`` / ``latent/empty`` -> ``sample``
-> ``vae/decode``) reaches parity with the monolithic ``alibaba/z-image-turbo`` node.

torch + diffusers are imported at module top **on purpose** (like ``runner.py``): an absent
``zimage`` extra makes this import raise, and ``server.bootstrap`` skips it best-effort so a
torch-less engine still boots. When it is skipped, the five nodes stay descriptor-only + hidden
(registered that way by ``graph/primitives.py``); when it registers, they are re-registered visible
with these runners.

The maths here mirrors diffusers' ``ZImagePipeline`` / ``ZImageImg2ImgPipeline`` exactly (prompt
encoding, latent shape, the flow-match denoise loop with its ``(1000 - t) / 1000`` timestep
transform and negated velocity prediction, CFG, and the VAE scale/shift), so the components produce
the same tensors the pipeline would. Placement (device/dtype/offload) comes only from the
``DevicePolicy``; the concrete denoise runs behind the batched-sampler seam, never inline.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch
from diffusers.image_processor import VaeImageProcessor

from ...components.conditioning import Conditioning, Latents
from ...components.interfaces import VAE, Denoiser, Sampler, Scheduler, StepCallback, StepInfo
from ...device.policy import DevicePolicy, Placement, Profile
from ...device.types import DType
from ...errors import CancelledError, ComponentError
from ...graph.loader_runners import ComponentRef
from ...graph.primitives import EMPTY_LATENT, ENCODE_TEXT, SAMPLE, VAE_DECODE, VAE_ENCODE
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase, ProgressEvent
from ...runtime.store import TakeStore
from ...sampling.batch import InlineBatchedSampler, SampleJob
from ...takes import AssetRef
from .. import loaders

_ARCH = "z-image"
_SEED_MAX = 2**31 - 1
_MAX_SEQUENCE_LENGTH = 512
# Z-Image constants (verified against the reference transformer/VAE configs): the transformer's
# in_channels == the VAE latent channels, and the VAE downsamples by 8 (4 block_out_channels ->
# 2**(4-1)). The pipeline works in a *2 packed grid, so the effective spatial factor is 16.
_LATENT_CHANNELS = 16
_VAE_SPATIAL_FACTOR = 16  # vae_scale_factor (8) * 2, matching prepare_latents' divisor


# --- copied flow-match helpers (kept tiny + local so this module owns its maths) ----------------


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Resolution-dependent sigma shift ``mu`` (copied from diffusers' flux/z-image pipeline). Only
    used when the scheduler enables dynamic shifting; harmless (ignored) otherwise."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _default_sigmas(steps: int) -> list[float]:
    """Z-Image's default sigma schedule: ``linspace(1.0, 1/steps, steps)`` (copied from the pipeline
    ``get_default_z_image_sigmas``)."""
    return [float(x) for x in torch.linspace(1.0, 1 / steps, steps)]


# --- placement helpers (device policy owns placement; we only translate its answer) --------------


def _torch_dtype(placement: Placement) -> Any:
    return {
        DType.FP16: torch.float16,
        DType.BF16: torch.bfloat16,
        DType.FP32: torch.float32,
    }.get(placement.dtype, torch.bfloat16)


def _run_device(policy: DevicePolicy, placement: Placement) -> str:
    """Where a component runs: the policy's device, unless it asked to offload / is CPU-only. We do
    not build an accelerate offload hook for the decomposed path - the policy only offloads when the
    user opts in (``INLINE_ALLOW_CPU_OFFLOAD``), and then CPU execution is the safe fallback."""
    if placement.offload or policy.profile is Profile.CPU:
        return "cpu"
    return str(placement.device)


def _placed(policy: DevicePolicy, role: str) -> tuple[Any, str]:
    """(torch dtype, device string) for a component role, straight from the policy."""
    placement = policy.placement(role)
    return _torch_dtype(placement), _run_device(policy, placement)


def _component_ref(inputs: dict[str, list[Any]], port: str, kind: str) -> ComponentRef:
    """The required ``ComponentRef`` wired into ``port``, guarding its kind (a mis-wired handle is
    already blocked by the graph validator; the runner guards too)."""
    ref = _first(inputs.get(port))
    if isinstance(ref, ComponentRef) and ref.kind == kind:
        return ref
    raise ComponentError(f"'{port}' input must be a loadable {kind} handle.")


def _first(values: list[Any] | None) -> Any:
    return values[0] if values else None


def _resolve_seed(raw: Any) -> int:
    """A fixed non-negative seed passes through; -1 (or anything invalid) -> a fresh random seed."""
    try:
        seed = int(raw)
    except (TypeError, ValueError):
        seed = -1
    return seed if seed >= 0 else random.randint(0, _SEED_MAX)


# --- concrete Z-Image components -----------------------------------------------------------------


class ZImageConditioning(Conditioning):
    """The opaque conditioning a Z-Image ``encode/text`` node emits: the per-item prompt embeddings
    (masked ``hidden_states[-2]`` of the Qwen3 encoder). ``sample`` combines a positive and an
    optional negative one into a guided conditioning the denoiser reads for CFG."""

    def __init__(
        self,
        positive: list[torch.Tensor],
        *,
        negative: list[torch.Tensor] | None = None,
        guidance: float = 0.0,
    ) -> None:
        self.positive = positive
        self.negative = negative
        self.guidance = guidance


class ZImageScheduler(Scheduler):
    """Wraps diffusers' ``FlowMatchEulerDiscreteScheduler``. Owns the flow-match timestep schedule
    (default Z-Image sigmas + the ``mu`` shift) and the ``x_t -> x_{t-1}`` step. ``image_seq_len``
    and ``device`` are baked in by the ``sample`` runner (which has the latent), so the ABC
    ``timesteps`` call needs nothing more."""

    def __init__(self, scheduler: Any, *, image_seq_len: int, device: str) -> None:
        self._s = scheduler
        self._image_seq_len = image_seq_len
        self._device = device
        self._planned_steps: int | None = None

    def _plan(self, steps: int) -> None:
        if self._planned_steps == steps:
            return
        mu = _calculate_shift(
            self._image_seq_len,
            self._s.config.get("base_image_seq_len", 256),
            self._s.config.get("max_image_seq_len", 4096),
            self._s.config.get("base_shift", 0.5),
            self._s.config.get("max_shift", 1.15),
        )
        self._s.set_timesteps(sigmas=_default_sigmas(steps), device=self._device, mu=mu)
        self._s.set_begin_index(0)
        self._planned_steps = steps

    def timesteps(self, steps: int) -> Sequence[torch.Tensor]:
        self._plan(steps)
        return list(self._s.timesteps)

    def scale_model_input(self, latents: Latents, t: torch.Tensor) -> Latents:
        # Flow-match euler does not pre-scale the model input; identity keeps the ABC contract.
        return latents

    def initial_latent(
        self, latents: torch.Tensor, steps: int, noise: torch.Tensor
    ) -> torch.Tensor:
        """Noise the input latent to the first timestep via flow-match ``scale_noise``. For an empty
        (zeros) latent at sigma≈1 this returns pure noise - matching text-to-image; for a
        VAE-encoded image latent it returns the partially-noised start - matching img2img."""
        self._plan(steps)
        batch = latents.shape[0]
        t0 = self._s.timesteps[:1].to(latents.device).repeat(batch)
        return self._s.scale_noise(latents, t0, noise)

    def step(self, model_output: torch.Tensor, t: torch.Tensor, latents: Latents) -> Latents:
        out = self._s.step(
            model_output.to(torch.float32), t, latents.tensor, return_dict=False
        )[0]
        return Latents(out)


class ZImageDenoiser(Denoiser):
    """Wraps the ``ZImageTransformer2DModel``. One forward = one velocity prediction, mirroring the
    pipeline loop: the ``(1000 - t) / 1000`` timestep transform, the packed 5D ``unsqueeze(2)`` +
    per-item list input, optional classifier-free guidance, and the final velocity negation."""

    def __init__(self, transformer: Any) -> None:
        self._transformer = transformer

    def predict(
        self,
        latents: Latents,
        timestep: torch.Tensor,
        conditioning: Conditioning,
        ctx: ExecutionContext,
    ) -> torch.Tensor:
        if not isinstance(conditioning, ZImageConditioning):
            raise ComponentError("Sample needs Z-Image conditioning on its positive input.")
        transformer = self._transformer
        dtype = transformer.dtype
        lat = latents.tensor
        batch = lat.shape[0]
        t = timestep.to(lat.device).expand(batch)
        t = (1000 - t) / 1000

        apply_cfg = conditioning.negative is not None and conditioning.guidance > 1.0
        if apply_cfg:
            assert conditioning.negative is not None
            latent_input = lat.to(dtype).repeat(2, 1, 1, 1)
            embeds = conditioning.positive + conditioning.negative
            t = t.repeat(2)
        else:
            latent_input = lat.to(dtype)
            embeds = conditioning.positive

        latent_input = latent_input.unsqueeze(2)  # (B, C, 1, H, W): the packed temporal axis
        latent_list = list(latent_input.unbind(dim=0))
        out = transformer(latent_list, t, embeds, return_dict=False)[0]

        if apply_cfg:
            scale = conditioning.guidance
            pos_out = out[:batch]
            neg_out = out[batch:]
            preds = [
                pos_out[j].float() + scale * (pos_out[j].float() - neg_out[j].float())
                for j in range(batch)
            ]
            noise_pred = torch.stack(preds, dim=0)
        else:
            noise_pred = torch.stack([o.float() for o in out], dim=0)
        return (-noise_pred.squeeze(2)).to(torch.float32)


class ZImageSampler(Sampler):
    """The stepping loop, denoiser-agnostic. Streams a tick per step and honours cancellation
    between steps - the concrete work stays here, off the graph executor (which never samples
    inline)."""

    def sample(
        self,
        denoiser: Denoiser,
        scheduler: Scheduler,
        latents: Latents,
        conditioning: Conditioning,
        *,
        steps: int,
        ctx: ExecutionContext,
        on_step: StepCallback | None = None,
    ) -> Latents:
        timesteps = scheduler.timesteps(steps)
        total = len(timesteps)
        current = latents
        for i, t in enumerate(timesteps):
            if ctx.cancel.cancelled:
                raise CancelledError("Run cancelled.")
            current = scheduler.scale_model_input(current, t)
            noise_pred = denoiser.predict(current, t, conditioning, ctx)
            current = scheduler.step(noise_pred, t, current)
            if on_step is not None:
                on_step(StepInfo(step=i + 1, total=total))
        return current


class ZImageVAE(VAE):
    """Wraps ``AutoencoderKL`` with the Z-Image scale/shift (read from the VAE config, not
    hardcoded): decode does ``latents / scaling + shift``; encode inverts it."""

    def __init__(self, vae: Any, device: str) -> None:
        self._vae = vae
        self._device = device
        self._processor: Any = VaeImageProcessor(vae_scale_factor=_VAE_SPATIAL_FACTOR)

    def encode(self, image: torch.Tensor, ctx: ExecutionContext) -> Latents:
        vae = self._vae
        image = image.to(device=self._device, dtype=vae.dtype)
        with torch.no_grad():
            dist = vae.encode(image).latent_dist
            latent = dist.sample()
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        return Latents(latent)

    def decode(self, latents: Latents, ctx: ExecutionContext) -> torch.Tensor:
        vae = self._vae
        lat = latents.tensor.to(device=self._device, dtype=vae.dtype)
        lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
        with torch.no_grad():
            return vae.decode(lat, return_dict=False)[0]

    def to_pil(self, pixels: torch.Tensor) -> Any:
        return self._processor.postprocess(pixels, output_type="pil")[0]

    def preprocess(self, pil_image: Any) -> torch.Tensor:
        return self._processor.preprocess(pil_image)


# --- runners -------------------------------------------------------------------------------------


class EncodeTextRunner(NodeRunner):
    """``encode/text``: (text encoder handle, prompt) -> Z-Image conditioning. Mirrors the pipeline
    ``_encode_prompt``: chat-template the prompt, tokenize to a fixed length, take the Qwen3
    ``hidden_states[-2]`` and keep only the unmasked tokens per item."""

    produces_takes = False

    def __init__(self, policy: DevicePolicy) -> None:
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        ref = _component_ref(inputs, "text_encoder", "text_encoder")
        prompt = str(_first(inputs.get("prompt")) or "").strip()
        if not prompt:
            raise ComponentError("Encode Text needs a prompt.")
        dtype, device = _placed(self._policy, "text_encoder")
        ctx.emitter.emit(_progress(ctx, node, Phase.ENCODE, 0.0, status="Encoding prompt…"))
        text_encoder, tokenizer = loaders.load_text_encoder(_ARCH, ref.file, dtype)
        text_encoder.to(device)
        embeds = _encode_prompt(text_encoder, tokenizer, prompt, device)
        ctx.emitter.emit(_progress(ctx, node, Phase.ENCODE, 1.0, status="Encoded"))
        return NodeResult(outputs={"conditioning": ZImageConditioning(embeds)})


class EmptyLatentRunner(NodeRunner):
    """``latent/empty``: a zeros latent of the right shape (``sample`` adds the seeded noise, so
    this stays a pure, cheap, seed-free canvas - the ComfyUI ``Empty Latent`` model)."""

    produces_takes = False

    def __init__(self, policy: DevicePolicy) -> None:
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        params = {**EMPTY_LATENT.defaults(), **node.params}
        width = int(params["width"])
        height = int(params["height"])
        batch = max(1, int(params["batch"]))
        latent_h = 2 * (height // _VAE_SPATIAL_FACTOR)
        latent_w = 2 * (width // _VAE_SPATIAL_FACTOR)
        shape = (batch, _LATENT_CHANNELS, latent_h, latent_w)
        tensor = torch.zeros(shape, dtype=torch.float32)
        return NodeResult(outputs={"latent": Latents(tensor)})


class SampleRunner(NodeRunner):
    """``sample``: (model, positive, [negative], latent) + params -> a denoised latent. Loads the
    transformer + flow-match scheduler, seeds the initial noise, and submits one ``SampleJob``
    through the batched-sampler seam (never denoising inline)."""

    produces_takes = False

    def __init__(self, policy: DevicePolicy) -> None:
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        model_ref = _component_ref(inputs, "model", "diffusion")
        positive = _require_conditioning(inputs, "positive")
        negative = _optional_conditioning(inputs, "negative")
        latent_in = _first(inputs.get("latent"))
        if not isinstance(latent_in, Latents):
            raise ComponentError("Sample needs a latent input.")

        params = {**SAMPLE.defaults(), **node.params}
        steps = max(1, int(params["steps"]))
        cfg = float(params["cfg"])
        seed = _resolve_seed(params.get("seed"))
        # sampler / scheduler selects: only euler + simple (the default flow-match euler schedule)
        # are reproduced faithfully today. Other choices (dpmpp_2m / heun, karras) run the nearest
        # equivalent - the same flow-match euler loop - rather than crashing; a bespoke schedule per
        # option lands with the broader sampler work.

        dtype, device = _placed(self._policy, "denoiser")
        ctx.emitter.emit(_progress(ctx, node, Phase.LOADING, 0.0, status="Loading model…"))
        transformer = loaders.load_diffusion(_ARCH, model_ref.file, dtype)
        transformer.to(device)
        diff_scheduler = loaders.load_scheduler(_ARCH)

        latent_tensor = latent_in.tensor.to(device=device, dtype=torch.float32)
        batch = latent_tensor.shape[0]
        image_seq_len = (latent_tensor.shape[2] // 2) * (latent_tensor.shape[3] // 2)
        scheduler = ZImageScheduler(diff_scheduler, image_seq_len=image_seq_len, device=device)

        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            latent_tensor.shape,
            generator=generator,
            device=torch.device(device),
            dtype=torch.float32,
        )
        start = scheduler.initial_latent(latent_tensor, steps, noise)

        conditioning = ZImageConditioning(
            positive=_match_batch(positive.positive, batch),
            negative=_match_batch(negative.positive, batch) if negative is not None else None,
            guidance=cfg,
        )

        def on_step(info: StepInfo) -> None:
            ctx.emitter.emit(
                _progress(
                    ctx,
                    node,
                    Phase.SAMPLE,
                    info.step / info.total,
                    step=info.step,
                    step_count=info.total,
                    status=f"Step {info.step}/{info.total}",
                )
            )

        job = SampleJob(
            denoiser=ZImageDenoiser(transformer),
            scheduler=scheduler,
            sampler=ZImageSampler(),
            latents=Latents(start),
            conditioning=conditioning,
            steps=steps,
            on_step=on_step,
        )
        result = InlineBatchedSampler().submit(job, ctx)
        return NodeResult(outputs={"latent": result})


class VaeDecodeRunner(NodeRunner):
    """``vae/decode``: (vae, latent) -> a saved IMAGE take. The only primitive producing media."""

    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy) -> None:
        self._store = store
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        ref = _component_ref(inputs, "vae", "vae")
        latent = _first(inputs.get("latent"))
        if not isinstance(latent, Latents):
            raise ComponentError("VAE Decode needs a latent input.")
        dtype, device = _placed(self._policy, "vae")
        ctx.emitter.emit(_progress(ctx, node, Phase.DECODE, 0.0, status="Decoding…"))
        vae = ZImageVAE(loaders.load_vae(_ARCH, ref.file, dtype), device)
        pixels = vae.decode(latent, ctx)
        image = vae.to_pil(pixels)
        ctx.emitter.emit(_progress(ctx, node, Phase.SAVE, 1.0, status="Saving…"))
        take = self._store.save(ctx.run_id, node.id, image, {"vae": ref.file})
        return NodeResult(outputs={"image": take}, takes=[take])


class VaeEncodeRunner(NodeRunner):
    """``vae/encode``: (vae, image) -> a latent (the img2img entry). No take."""

    produces_takes = False

    def __init__(self, policy: DevicePolicy) -> None:
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        ref = _component_ref(inputs, "vae", "vae")
        image_ref = _first(inputs.get("image"))
        dtype, device = _placed(self._policy, "vae")
        vae = ZImageVAE(loaders.load_vae(_ARCH, ref.file, dtype), device)
        pixels = vae.preprocess(_load_image(image_ref))
        latents = vae.encode(pixels, ctx)
        return NodeResult(outputs={"latent": latents})


# --- shared runner helpers -----------------------------------------------------------------------


def _optional_conditioning(
    inputs: dict[str, list[Any]], port: str
) -> ZImageConditioning | None:
    value = _first(inputs.get(port))
    if value is None:
        return None
    if isinstance(value, ZImageConditioning):
        return value
    raise ComponentError(f"Sample '{port}' input is not Z-Image conditioning.")


def _require_conditioning(inputs: dict[str, list[Any]], port: str) -> ZImageConditioning:
    value = _optional_conditioning(inputs, port)
    if value is None:
        raise ComponentError(f"Sample needs a '{port}' conditioning input.")
    return value


def _match_batch(embeds: list[torch.Tensor], batch: int) -> list[torch.Tensor]:
    """Fan a per-prompt embedding list out to the latent batch (repeat a single prompt; cycle a
    shorter list), so one Encode Text feeds a batched Empty Latent - like num_images_per_prompt."""
    if len(embeds) == batch:
        return embeds
    if not embeds:
        raise ComponentError("Empty conditioning: the text encoder produced no embeddings.")
    return [embeds[i % len(embeds)] for i in range(batch)]


def _encode_prompt(
    text_encoder: Any, tokenizer: Any, prompt: str, device: str
) -> list[torch.Tensor]:
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    text_inputs = tokenizer(
        [templated],
        padding="max_length",
        max_length=_MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    masks = text_inputs.attention_mask.to(device).bool()
    with torch.no_grad():
        hidden = text_encoder(
            input_ids=input_ids, attention_mask=masks, output_hidden_states=True
        ).hidden_states[-2]
    return [hidden[i][masks[i]] for i in range(len(hidden))]


def _load_image(ref: Any) -> Any:
    from PIL import Image

    if isinstance(ref, AssetRef) and ref.ref == "path" and ref.path:
        return Image.open(ref.path).convert("RGB")
    raise ComponentError("VAE Encode needs a readable image path input.")


def _progress(
    ctx: ExecutionContext,
    node: Node,
    phase: Phase,
    fraction: float,
    *,
    step: int | None = None,
    step_count: int | None = None,
    status: str = "",
) -> ProgressEvent:
    return ProgressEvent(
        run_id=ctx.run_id,
        node_id=node.id,
        phase=phase,
        fraction=fraction,
        step=step,
        step_count=step_count,
        status=status,
    )


def register_zimage_primitives(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Re-register the five decomposed primitives **visible** with their diffusers-backed runners.

    Overwrites the hidden, runnerless descriptors ``graph/primitives.py`` registered on a torch-less
    boot (``Registry.register`` replaces by type). Called best-effort by ``server.bootstrap`` so an
    absent ``zimage`` extra leaves the nodes descriptor-only + hidden and the engine still boots."""
    registry.register(replace(ENCODE_TEXT, hidden=False), EncodeTextRunner(policy))
    registry.register(replace(EMPTY_LATENT, hidden=False), EmptyLatentRunner(policy))
    registry.register(replace(SAMPLE, hidden=False), SampleRunner(policy))
    registry.register(replace(VAE_DECODE, hidden=False), VaeDecodeRunner(store, policy))
    registry.register(replace(VAE_ENCODE, hidden=False), VaeEncodeRunner(policy))
