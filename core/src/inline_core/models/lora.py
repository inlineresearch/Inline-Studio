"""Fuse LoRA weights into an already-loaded model, in stack order.

Fusing (rather than keeping adapters live) is deliberate: ``loaders.py`` caches components by a key
that includes the stack, so a fused model *is* the cached artifact and there is no adapter state to
get out of sync with that key.

Handles both key conventions we see in the wild - ComfyUI/kohya (``lora_down``/``lora_up`` + an
optional ``alpha``) and diffusers/peft (``lora_A``/``lora_B``). Every LoRA key must resolve to a
module: a partial match silently applies some of the LoRA and no-ops the rest, which degrades output
without erroring, so an unmatched key is a hard failure.

Known gap: fully underscore-flattened kohya paths (``lora_unet_layers_0_attention_to_q``) are
ambiguous - the reverse mapping cannot tell a path separator from an underscore inside a name like
``to_q``. Those raise rather than mis-apply. Every Z-Image LoRA seen so far uses dotted paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ComponentError

if TYPE_CHECKING:
    from ..graph.loader_runners import LoraRef

_DOWN = ("lora_down.weight", "lora_A.weight", "lora_A.default.weight")
_UP = ("lora_up.weight", "lora_B.weight", "lora_B.default.weight")
# Prefixes checkpoints put in front of the module path; stripped when matching against the model.
_PREFIXES = ("diffusion_model.", "transformer.", "lora_unet_", "lora_te_", "base_model.model.")


def fuse_loras(model: Any, loras: tuple[LoraRef, ...]) -> None:
    """Merge each LoRA into ``model``'s weights in order. No-op for an empty stack."""
    for lora in loras:
        _fuse_one(model, lora.file, lora.strength)


def _fuse_one(model: Any, path: str, strength: float) -> None:
    import torch
    from safetensors.torch import load_file

    try:
        state = load_file(path)
    except Exception as exc:  # noqa: BLE001
        raise ComponentError(f"Could not read LoRA {path!r}: {exc}") from exc

    modules = _linear_modules(model)
    pairs, alphas = _group(state)
    if not pairs:
        raise ComponentError(f"LoRA {path!r} contains no recognisable lora_down/lora_up pairs.")

    unmatched: list[str] = []
    fused = 0
    for stem, (down, up) in sorted(pairs.items()):
        target = _match(stem, modules)
        if target is None:
            unmatched.append(stem)
            continue
        with torch.no_grad():
            weight = target.weight
            delta = _delta(up, down, weight)
            scale = strength * _alpha_scale(alphas.get(stem), down.shape[0])
            weight.add_((delta * scale).to(device=weight.device, dtype=weight.dtype))
        fused += 1

    if unmatched:
        sample = ", ".join(unmatched[:3])
        raise ComponentError(
            f"LoRA {path!r} does not match this model: {len(unmatched)} of {len(pairs)} layers "
            f"have no target (e.g. {sample}). It was probably trained for a different architecture."
        )
    if fused == 0:
        raise ComponentError(f"LoRA {path!r} matched no layers in this model.")


def _delta(up: Any, down: Any, weight: Any) -> Any:
    """``up @ down``, shaped to the target weight. Conv LoRAs flatten the spatial dims."""
    dtype = _fuse_dtype(up)
    delta = up.to("cpu", dtype=dtype).flatten(1) @ down.to("cpu", dtype=dtype).flatten(1)
    return delta.reshape(weight.shape)


def _fuse_dtype(tensor: Any) -> Any:
    import torch

    # fp16 matmuls of low-rank factors lose precision; fuse in fp32 and cast on the way in.
    return torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype


def _alpha_scale(alpha: Any, rank: int) -> float:
    """kohya-style checkpoints scale by ``alpha / rank``; without an alpha the factors are raw."""
    if alpha is None or rank == 0:
        return 1.0
    return float(alpha.item() if hasattr(alpha, "item") else alpha) / float(rank)


def _group(state: dict[str, Any]) -> tuple[dict[str, tuple[Any, Any]], dict[str, Any]]:
    downs: dict[str, Any] = {}
    ups: dict[str, Any] = {}
    alphas: dict[str, Any] = {}
    for key, value in state.items():
        for suffix in _DOWN:
            if key.endswith("." + suffix):
                downs[key[: -len(suffix) - 1]] = value
        for suffix in _UP:
            if key.endswith("." + suffix):
                ups[key[: -len(suffix) - 1]] = value
        if key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = value
    return {k: (downs[k], ups[k]) for k in downs if k in ups}, alphas


def _linear_modules(model: Any) -> dict[str, Any]:
    import torch

    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear | torch.nn.Conv2d | torch.nn.Conv3d)
    }


def _match(stem: str, modules: dict[str, Any]) -> Any:
    """Resolve a checkpoint's module path to a module, tolerating the usual prefixes and the
    underscore-flattened paths kohya emits."""
    for candidate in _candidates(stem):
        hit = modules.get(candidate)
        if hit is not None:
            return hit
    return None


def _candidates(stem: str) -> list[str]:
    out = [stem]
    for prefix in _PREFIXES:
        if stem.startswith(prefix):
            out.append(stem[len(prefix) :])
    # kohya flattens the whole path with underscores; the dotted form is the model's own naming.
    out.extend([c.replace("_", ".") for c in list(out) if "." not in c])
    return out
