"""FLUX.2 runner: prompt (+ any number of reference images) -> one rendered take.

**One node for the whole family.** ``black-forest-labs/flux-2`` covers klein 4B/9B, their base
builds, the KV variant and dev, because none of them differ in anything the descriptor expresses -
the ports are identical and only the pipeline class, sampler defaults and encoder change. The picked
checkpoint is identified from its own tensor shapes (see ``variants.py``) and everything else
follows from that, so a later FLUX.2 build (or FLUX.3) is a row in the variant table.

Two consequences worth knowing when reading this file:

- ``steps`` and ``guidance`` default to sentinels meaning "ask the checkpoint", so switching a
  distilled build for its base build in the dropdown moves 4 steps / guidance 1 to 50 / 4 without
  the user touching the settings panel.
- FLUX.2 has no img2img denoise strength. A reference image is conditioning that rides in the token
  sequence for the whole denoise, so one reference is an edit and several are a composition. The
  prompt addresses them by position ("the jacket from image 2"), which is why order is preserved.

torch + diffusers import at module top on purpose: an absent ``runtime`` extra makes this import
raise and ``server.bootstrap`` skips the model, so the engine still boots.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import torch

from ...device.policy import DevicePolicy, ModelFootprint, Profile, Quantization
from ...errors import CancelledError, ComponentError
from ...graph.descriptor import NodeDescriptor, Option, ParamField, Port, Widget
from ...graph.loader_runners import LoraRef
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...media import MediaKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase
from ...runtime.store import TakeStore
from .. import loaders
from .. import pipeline_runtime as rt
from ..sampling import SamplingFamily, apply_sampling, sampling_param_fields
from . import embeds
from . import requirements as reqs
from . import variants as V

_LABEL = "FLUX.2"
_NODE_TYPE = "black-forest-labs/flux-2"

#: ``steps`` / ``guidance`` values meaning "take the checkpoint's own default". Mirrors the seed
#: field's -1 = random: one sentinel the user can see and override.
_AUTO_STEPS = 0
_AUTO_GUIDANCE = -1.0

logger = logging.getLogger("inline_core.flux2")


FLUX2 = NodeDescriptor(
    type=_NODE_TYPE,
    title="FLUX.2",
    category="Generate",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        # Optional component handles from load/* subnodes - wire one to override the dropdown.
        Port("model", "Diffusion model", PortKind.MODEL, required=False),
        Port("vae", "VAE", PortKind.VAE, required=False),
        Port("text_encoder", "Text encoder", PortKind.TEXT_ENCODER, required=False),
        Port("lora", "LoRA", PortKind.LORA, required=False),
        # A list port: wire several images to compose from them. One is an edit of that image.
        Port("image", "Reference image(s)", PortKind.IMAGE_LIST, required=False),
        # A pose/depth/canny map from Apply ControlNet or Control Space. FLUX.2 has no separate
        # ControlNet input: the map is appended as a reference, which is what the family's own
        # in-context conditioning is for. Name it in the prompt to steer with it.
        Port("control_image", "Structure map", PortKind.CONTROL, required=False),
    ),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        # Only an undistilled klein checkpoint runs real CFG; on dev and the distilled builds this
        # is ignored (dev is guidance-distilled and its pipeline has no negative path at all).
        ParamField("negative_prompt", "Negative prompt (base models only)", Widget.TEXTAREA, ""),
        ParamField("width", "Width", Widget.NUMBER, 1024, min=256, max=2048, step=64),
        ParamField("height", "Height", Widget.NUMBER, 1024, min=256, max=2048, step=64),
        ParamField(
            "steps", "Steps (0 = from model)", Widget.NUMBER, _AUTO_STEPS, min=0, max=100, step=1
        ),
        ParamField(
            "guidance", "Guidance (-1 = from model)", Widget.NUMBER, _AUTO_GUIDANCE,
            min=-1.0, max=20.0, step=0.5,
        ),
        # FLUX.2 is flow-match, so these tune the FlowMatchEuler scheduler - see models/sampling.py.
        *sampling_param_fields(SamplingFamily.FLOW_MATCH),
        ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
        # "" = identify the picked checkpoint from its tensor shapes. The explicit options exist for
        # a file whose name hides whether it is a base or distilled build.
        ParamField(
            "variant", "Model variant", Widget.SELECT, "",
            options=(Option("", "Auto (detect from file)"), *(
                Option(v.key, f"FLUX.2 {v.label}") for v in V.VARIANTS
            )),
            advanced=True,
        ),
        ParamField(
            "model", "Diffusion file (auto)", Widget.SELECT, "",
            options_from="diffusion_models", advanced=True,
        ),
        ParamField(
            "text_encoder", "Text-encoder file (auto)", Widget.SELECT, "",
            options_from="text_encoders", advanced=True,
        ),
        ParamField(
            "vae", "VAE file (auto)", Widget.SELECT, "", options_from="vae", advanced=True,
        ),
    ),
)


def register_flux2(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register the FLUX.2 node and its runner. Called best-effort by server.bootstrap."""
    registry.register(FLUX2, Flux2Runner(store, policy))


class Flux2Runner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy) -> None:
        self._store = store
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        prompt = rt.first_str(inputs.get("prompt"))
        if not prompt:
            raise ComponentError(f"{_LABEL} needs a prompt.")
        params = {**FLUX2.defaults(), **node.params}
        width, height = _snap(int(params["width"])), _snap(int(params["height"]))
        seed = rt.resolve_seed(params.get("seed"))
        sampler, scheduler = str(params["sampler"]), str(params["scheduler"])

        # Wired component handles from load/* subnodes override the dropdowns.
        model_ref = rt.component_ref(inputs, "model", "diffusion", _LABEL)
        vae_ref = rt.component_ref(inputs, "vae", "vae", _LABEL)
        te_ref = rt.component_ref(inputs, "text_encoder", "text_encoder", _LABEL)
        loras = rt.lora_stack(inputs, _LABEL)
        wired = {ref.kind for ref in (model_ref, vae_ref, te_ref) if ref is not None}

        # No hidden downloads: a required component that is neither wired nor on disk fails fast.
        missing = [
            c.label
            for c in reqs.flux2_requirements(params)
            if not c.present and not c.optional and c.id not in wired
        ]
        if missing:
            raise ComponentError(
                f"{_LABEL} models missing: "
                + ", ".join(missing)
                + ". Download them from the node's model popup (the hint on the node)."
            )

        source = model_ref.file if model_ref else str(reqs.resolve_diffusion(params))
        variant, config = _identify(source, params)
        vae_file = vae_ref.file if vae_ref else rt.path_or_none(reqs.resolve_vae(params))
        te_file = te_ref.file if te_ref else rt.path_or_none(reqs.resolve_text_encoder(params))
        if not vae_file or not te_file:
            raise ComponentError(
                f"{_LABEL} needs a local VAE and text-encoder file. Download them from the node's "
                "model popup."
            )

        steps = _resolve_steps(params, variant)
        guidance = _resolve_guidance(params, variant)
        negative = _resolve_negative(params, variant)

        # References ride in the token sequence for the whole denoise, so they are conditioning
        # rather than a starting point - there is no img2img strength to apply. A wired structure
        # map is just one more reference; FLUX.2 conditions on it the same way.
        refs = list(inputs.get("image") or [])
        control = rt.first(inputs.get("control_image"))
        if control is not None:
            refs.append(control)
        images = [rt.load_image(ref, _LABEL) for ref in refs]

        # Size-aware placement: hand the policy the on-disk sizes so it fits dtype/quant/offload to
        # THIS GPU, then refuse an impossible load up front rather than OOM-killing the server.
        self._policy.set_footprint(
            ModelFootprint(**reqs.footprint_bytes(source, vae_file, te_file))
        )
        fit = self._policy.fit_estimate()
        if fit is not None and not fit.fits:
            raise ComponentError(rt.wont_fit_message(fit))

        logger.info(
            "%s (%s) run: %dx%d, %d steps, guidance=%.1f, %d reference(s) | %s",
            _LABEL, variant.label, width, height, steps, guidance, len(images),
            rt.device_report(self._policy),
        )
        rt.reset_peak_vram()
        rt.raise_if_cancelled(ctx)  # bail before a multi-GB load if already cancelled
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.LOADING, 0.0, status="Loading model…"))
        try:
            pipe = _load_pipeline(
                self._policy,
                variant=variant,
                source=source,
                config=config,
                vae=vae_file,
                text=te_file,
                quant=self._policy.quantization(),
                loras=loras,
                cancel_check=lambda: rt.raise_if_cancelled(ctx),
            )
        except CancelledError:
            rt.free_vram()  # a cancelled load must return whatever VRAM it placed
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, len(images))) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, len(images), host=True)) from error

        placement = self._policy.placement("denoiser")
        on_cpu = placement.offload or self._policy.profile is Profile.CPU
        gen_device = "cpu" if on_cpu else str(placement.device)
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        def on_step_end(_pipe: Any, step: int, _t: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            if ctx.cancel.cancelled:
                raise CancelledError("Run cancelled.")
            done = step + 1
            ctx.emitter.emit(
                rt.progress_event(
                    ctx, node, Phase.SAMPLE, done / steps,
                    step=done, step_count=steps, status=f"Step {done}/{steps}",
                )
            )
            return kwargs

        call: dict[str, Any] = dict(
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            output_type="pil",
            callback_on_step_end=on_step_end,
            max_sequence_length=512,
            text_encoder_out_layers=variant.text_encoder_layers,
        )
        # The KV pipeline is distilled-only and takes no guidance argument at all.
        if variant.pipeline != "klein-kv":
            call["guidance_scale"] = guidance
        if images:
            call["image"] = images
        call.update(
            embeds.prompt_kwargs(
                pipe,
                self._policy,
                prompt=prompt,
                negative=negative,
                text_encoder_file=te_file,
                layers=variant.text_encoder_layers,
            )
        )

        # Rebuild the scheduler for the chosen sampler/scheduler from the pipe's pristine base
        # config (immutable across cache hits).
        base_config = getattr(pipe, "_inline_base_scheduler_config", None)
        if base_config is not None:
            sigmas = apply_sampling(
                pipe, base_config, SamplingFamily.FLOW_MATCH, sampler, scheduler, steps
            )
            if sigmas is not None:
                call["sigmas"] = sigmas

        logger.info(
            "%s sampling %d steps on %s (sampler=%s, scheduler=%s)…",
            _LABEL, steps, gen_device, sampler, scheduler,
        )
        rt.raise_if_cancelled(ctx)  # cancelled during load? don't start the denoise
        sample_start = time.perf_counter()
        try:
            with rt.text_encoder_detached(pipe, "prompt_embeds" in call):
                image = pipe(**call).images[0]
        except CancelledError:
            rt.free_vram()  # release partial-denoise activations so the next run isn't starved
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, len(images))) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, len(images), host=True)) from error
        elapsed = time.perf_counter() - sample_start
        peak_gb = rt.peak_vram_gb()
        peak_note = f", peak VRAM {peak_gb:.1f}GB" if peak_gb else ""
        logger.info(
            "%s sampled %dx%d in %.1fs (%.2fs/step)%s | %s",
            _LABEL, width, height, elapsed, elapsed / steps, peak_note,
            rt.device_report(self._policy),
        )
        rt.free_vram()  # return fragmented free blocks to the driver (keeps the model resident)

        save_status = "Saving…" + (f" (peak VRAM {peak_gb:.1f}GB)" if peak_gb else "")
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.SAVE, 1.0, status=save_status))
        take = self._store.save(
            ctx.run_id,
            node.id,
            image,
            {
                "model": source,
                "variant": variant.key,
                "prompt": prompt,
                "negative_prompt": negative or "",
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "sampler": sampler,
                "scheduler": scheduler,
                "seed": seed,
                **({"references": len(images)} if images else {}),
                **(
                    {"loras": [{"file": lo.file, "strength": lo.strength} for lo in loras]}
                    if loras
                    else {}
                ),
            },
        )
        return NodeResult(outputs={"image": take}, takes=[take])


# --- resolving what the checkpoint wants ----------------------------------------------------------


def _snap(value: int) -> int:
    """FLUX.2 packs latents 2x2 on top of an 8x VAE, so both sides must be multiples of 16. The
    param step is 64, but a recipe or a "match aspect" write can land elsewhere."""
    return max(64, (value // 16) * 16)


def _identify(source: str, params: dict[str, Any]) -> tuple[V.Flux2Variant, dict[str, Any]]:
    """The variant and transformer geometry for the picked checkpoint.

    An explicit ``variant`` param wins over detection, but the geometry always comes from the file:
    a mislabeled variant should still load with the right shapes.
    """
    shapes = _shapes(source)
    config = V.derive_transformer_config(shapes) if shapes else None
    if config is None:
        raise ComponentError(
            f"'{source}' is not a FLUX.2 diffusion model. Pick a FLUX.2 checkpoint in the "
            "Diffusion file dropdown, or download one from the node's model popup."
        )
    variant = V.get(str(params.get("variant") or "")) or V.detect(source, shapes)
    if variant is None:
        raise ComponentError(
            "Could not tell which FLUX.2 variant this checkpoint is. Pick one explicitly in the "
            "Model variant dropdown."
        )
    return variant, config


def _shapes(source: str) -> dict[str, list[int]] | None:
    from pathlib import Path

    if Path(source).suffix.lower() == ".gguf":
        return _gguf_shapes(source)
    try:
        from ..checkpoint import CheckpointReader

        return CheckpointReader(source).shapes()
    except Exception:  # noqa: BLE001 - an unreadable file is reported as "not FLUX.2" by the caller
        return None


def _gguf_shapes(source: str) -> dict[str, list[int]] | None:
    """Tensor shapes from a GGUF checkpoint, so the same geometry derivation works for those too.
    GGUF stores dimensions fastest-varying first, the reverse of safetensors, so they are flipped
    back here."""
    try:
        import gguf  # pyright: ignore[reportMissingImports] - optional dependency, guarded

        reader: Any = gguf.GGUFReader(source)
        tensors: list[Any] = list(reader.tensors)
        return {str(t.name): [int(d) for d in reversed(list(t.shape))] for t in tensors}
    except Exception as error:  # noqa: BLE001 - surfaced as a node error with an install hint
        logger.warning("Could not read GGUF header for %s: %s", source, error)
        return None


def _resolve_steps(params: dict[str, Any], variant: V.Flux2Variant) -> int:
    steps = int(params.get("steps", _AUTO_STEPS))
    return variant.steps if steps <= _AUTO_STEPS else steps


def _resolve_guidance(params: dict[str, Any], variant: V.Flux2Variant) -> float:
    guidance = float(params.get("guidance", _AUTO_GUIDANCE))
    return variant.guidance if guidance < 0 else guidance


def _resolve_negative(params: dict[str, Any], variant: V.Flux2Variant) -> str | None:
    """A negative prompt, but only where it does something. A distilled checkpoint runs no CFG and
    dev's pipeline has no negative path, so we log and drop it rather than pretending it applied."""
    negative = str(params.get("negative_prompt") or "").strip()
    if not negative:
        return None
    if not variant.supports_negative_prompt:
        logger.info(
            "Ignoring the negative prompt: FLUX.2 %s runs no classifier-free guidance. Describe "
            "what you want instead, or use a Base checkpoint.",
            variant.label,
        )
        return None
    return negative


def _oom(width: int, height: int, references: int, *, host: bool = False) -> str:
    message = rt.oom_message(width, height, host=host)
    if references:
        message += (
            f" This render also carried {references} reference image(s); each one adds tokens to "
            "every step, so removing one is often the cheapest fix."
        )
    return message


# --- pipeline build -------------------------------------------------------------------------------


def _load_pipeline(
    policy: DevicePolicy,
    *,
    variant: V.Flux2Variant,
    source: str,
    config: dict[str, Any],
    vae: str,
    text: str,
    quant: Quantization = Quantization.NONE,
    loras: tuple[LoraRef, ...] = (),
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    key = rt.PipelineKey(
        arch=variant.arch,
        diffusion=source,
        vae=vae,
        text_encoder=text,
        # One pipeline shape per variant: references are a call argument, not a different class,
        # so t2i and editing share a build. Only the KV/base split changes the class or its config.
        variant=variant.key,
        quant=quant.value,
        loras=loaders.lora_cache_key(loras),
    )
    with rt.PIPELINES.lock:
        cached = rt.PIPELINES.get(key)
        if cached is not None:
            logger.info("Pipeline cache hit (%s) - reusing loaded weights", source)
            return cached
        if cancel_check is not None:
            cancel_check()  # bail before the disk read if the run was cancelled while queued
        started = time.perf_counter()
        logger.info(
            "Loading %s %s pipeline: source=%s | %s",
            _LABEL, variant.label, source, rt.device_report(policy),
        )
        # Free any *other* model still resident before loading this one, so switching checkpoints
        # doesn't stack VRAM.
        rt.PIPELINES.evict_stale(key)
        placement = policy.placement("denoiser")
        # Resident placement streams weights straight to the GPU; the offload path loads to CPU so
        # accelerate can install its hooks before placing.
        pipe = loaders.assemble_flux2_pipeline(
            arch=variant.arch,
            pipeline=variant.pipeline,
            distilled=variant.distilled,
            diffusion_file=source,
            config=config,
            vae_file=vae,
            text_encoder_file=text,
            dtype=rt.torch_dtype(placement),
            quant=quant,
            vae_dtype=rt.torch_dtype(policy.placement("vae")),
            device=None if placement.offload else str(placement.device),
            loras=loras,
            cancel_check=cancel_check,
        )
        rt.configure_pipeline(pipe, policy)
        rt.capture_base_scheduler_config(pipe)
        rt.PIPELINES.put(key, pipe)
        logger.info("%s pipeline ready in %.1fs", _LABEL, time.perf_counter() - started)
        return pipe
