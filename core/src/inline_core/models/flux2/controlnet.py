"""The FLUX.2 Fun ControlNet Union: structural control for the dev checkpoint.

`alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` is a **VACE-style side branch**, not an inline
adapter. Four extra double-stream blocks run in parallel with the base transformer's first four
even-numbered blocks, and each contributes a residual back into the main image stream:

    control_img_in(control_context) -> c
    c = before_proj(c) + hidden_states          (block 0 only; before_proj is zero-init)
    for each control block:  c = block(c);  hint = after_proj(c)
    base block i (i in 0, 2, 4, 6):  hidden_states += hints[i // 2] * scale

It is a genuine *union*: there is no mode index. Canny, depth, pose, HED, MLSD, scribble, gray and
inpainting all arrive as the same VAE-encoded ``control_context``, and the model infers the modality
from the image.

Ported rather than imported: upstream this lives in ``videox_fun``, which would drag a whole video
stack into the shared runtime extra. Only the control branch is ported - the blocks themselves are
diffusers' own ``Flux2TransformerBlock``, whose keys the checkpoint matches exactly - and it hooks
onto a stock ``Flux2Transformer2DModel``, so diffusers' denoise is untouched.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ...errors import ComponentError

#: Which base double blocks receive a hint. The checkpoint carries four control blocks and the
#: reference config maps them onto every other base block.
CONTROL_LAYERS: tuple[int, ...] = (0, 2, 4, 6)

#: Width of the packed control context: a 128-channel control latent, a 128-channel inpaint latent
#: and a 4-channel mask, concatenated. Matches ``control_in_dim`` in the reference config.
CONTROL_IN_DIM = 260


class Flux2ControlBranch(nn.Module):
    """The side branch: an input projection plus the control blocks and their residual taps."""

    def __init__(self, inner_dim: int, num_attention_heads: int, attention_head_dim: int,
                 mlp_ratio: float = 3.0, eps: float = 1e-6, layers: int = 4) -> None:
        super().__init__()
        from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock

        self.control_img_in = nn.Linear(CONTROL_IN_DIM, inner_dim)
        self.control_transformer_blocks = nn.ModuleList(
            Flux2TransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                eps=eps,
            )
            for _ in range(layers)
        )
        # Zero-init projections, so an untrained branch is exactly a no-op on the base model.
        self.before_proj = nn.Linear(inner_dim, inner_dim)
        self.after_proj = nn.ModuleList(nn.Linear(inner_dim, inner_dim) for _ in range(layers))
        for layer in (self.before_proj, *self.after_proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def hints(self, hidden_states: torch.Tensor, control_context: torch.Tensor,
              **block_kwargs: Any) -> list[torch.Tensor]:
        """One residual per control layer, in ``CONTROL_LAYERS`` order."""
        # Not ``.to(self.control_img_in.weight.dtype)``: once the branch is NF4 that weight is a
        # bitsandbytes Params4bit with **uint8** storage, and casting the context to uint8 destroys
        # it silently - the render completes and returns noise. The caller builds the context in the
        # compute dtype already.
        c = self.control_img_in(control_context)
        c = self.before_proj(c) + hidden_states
        encoder_hidden_states = block_kwargs.pop("encoder_hidden_states")
        out: list[torch.Tensor] = []
        for block, tap in zip(self.control_transformer_blocks, self.after_proj, strict=True):
            # The branch carries its own text stream, exactly as the reference does.
            encoder_hidden_states, c = block(
                hidden_states=c, encoder_hidden_states=encoder_hidden_states, **block_kwargs
            )
            out.append(tap(c))
        return out


def load_control_branch(file: str, config: dict[str, Any], dtype: Any,
                        device: str | None = None, quant: Any = None) -> Flux2ControlBranch:
    """Build the branch from a single ``.safetensors`` and the base transformer's geometry.

    Loaded to the CPU and quantized before it moves, so its 8.2 GB never has to sit on the card
    beside an already-resident transformer - which on a 24 GB board is the difference between
    fitting and not.
    """
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    with init_empty_weights():
        branch = Flux2ControlBranch(
            inner_dim=heads * head_dim,
            num_attention_heads=heads,
            attention_head_dim=head_dim,
            mlp_ratio=float(config.get("mlp_ratio", 3.0)),
            eps=float(config.get("eps", 1e-6)),
        )
    state = _remap(load_file(file, device="cpu"))
    missing = set(branch.state_dict()) - set(state)
    if missing:
        raise ComponentError(
            f"'{file}' is not a FLUX.2 Fun ControlNet Union ({len(missing)} tensors missing, e.g. "
            f"{sorted(missing)[0]}). The 'lite' variants use a different geometry."
        )
    branch.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True, assign=True)
    state.clear()
    branch = branch.eval()

    from ...device.policy import Quantization
    from .. import loaders

    if quant is Quantization.NF4:
        # Swap on the CPU, then move: bitsandbytes quantizes during the move, so the full-size
        # weights never touch the card.
        loaders._swap_to_4bit(branch)
        if device:
            branch.to(device)
    else:
        if device:
            branch.to(device)
        loaders._quantize_in_place(branch, quant)  # torchao; no-op unless INT8
    return branch


def _remap(state: dict[str, Any]) -> dict[str, Any]:
    """Checkpoint keys -> this module's layout.

    Upstream keeps ``before_proj``/``after_proj`` inside each control block; here they are separate
    module lists so the blocks stay stock diffusers ``Flux2TransformerBlock``s that a plain
    ``load_state_dict`` can fill.
    """
    out: dict[str, Any] = {}
    for key, value in state.items():
        if ".before_proj." in key:
            out["before_proj." + key.split(".before_proj.")[1]] = value
        elif ".after_proj." in key:
            index = key.split(".")[1]
            out[f"after_proj.{index}." + key.split(".after_proj.")[1]] = value
        else:
            out[key] = value
    return out


def attach(transformer: Any, branch: Flux2ControlBranch, control_context: torch.Tensor,
           scale: float = 0.75) -> list[Any]:
    """Hook the branch onto a stock transformer; returns handles the caller must remove.

    Hooks rather than a subclass: diffusers owns the denoise, and a forked ``forward`` would rot
    silently the next time that file changes. The first base block's inputs are the branch's inputs
    too, so a pre-hook there captures them and computes every hint in one pass.
    """
    blocks = transformer.transformer_blocks
    if len(blocks) <= max(CONTROL_LAYERS):
        raise ComponentError(
            f"This ControlNet expects at least {max(CONTROL_LAYERS) + 1} double blocks, but the "
            f"checkpoint has {len(blocks)}. It is built for FLUX.2 dev."
        )
    state: dict[str, list[torch.Tensor]] = {}

    def compute(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        passthrough = {
            k: kwargs[k]
            for k in ("encoder_hidden_states", "temb_mod_img", "temb_mod_txt", "image_rotary_emb")
            if k in kwargs
        }
        state["hints"] = branch.hints(hidden, control_context, **passthrough)

    def add(index: int):
        def hook(_module: Any, _args: Any, output: tuple[torch.Tensor, torch.Tensor]):
            encoder_hidden_states, hidden = output
            hint = state["hints"][CONTROL_LAYERS.index(index)]
            return encoder_hidden_states, hidden + hint.to(hidden.dtype) * scale

        return hook

    handles = [blocks[0].register_forward_pre_hook(compute, with_kwargs=True)]
    handles += [blocks[i].register_forward_hook(add(i)) for i in CONTROL_LAYERS]
    return handles


def build_context(pipe: Any, control_image: Any, height: int, width: int, dtype: Any,
                  device: str, reference_tokens: int = 0) -> torch.Tensor:
    """Pack a control map into the 260-channel context the branch consumes.

    Layout, in this order: the VAE-encoded control map (128), a mask channel set (4), and an inpaint
    latent (128). We only drive the control path, so the mask and inpaint slots are the zeros the
    reference implementation uses when neither is supplied - the union model reads them as "no
    inpainting requested" rather than as missing input.

    The control latent is normalized by the VAE's running batch-norm statistics, the same way every
    other FLUX.2 latent is; skipping that trains the branch's expectations off-manifold.

    ``reference_tokens`` pads the context so it lines up with the sequence when reference images are
    also wired: those tokens get no control signal.
    """
    import torch.nn.functional as functional
    from diffusers import Flux2KleinPipeline as Packing

    processed = pipe.image_processor.preprocess(control_image, height=height, width=width)
    processed = processed.to(device=device, dtype=dtype)
    latents = pipe.vae.encode(processed)[0].mode()

    # Zeros mean "no inpainting": the reference builds an all-ones mask then inverts it.
    mask = torch.zeros(
        (latents.shape[0], 1, *latents.shape[-2:]), device=latents.device, dtype=latents.dtype
    )
    mask = functional.interpolate(mask, size=latents.shape[-2:], mode="nearest")
    mask = Packing._pack_latents(Packing._patchify_latents(mask))
    inpaint = Packing._pack_latents(Packing._patchify_latents(torch.zeros_like(latents)))

    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps)
    control = Packing._patchify_latents(latents)
    control = (control - mean) / std.to(latents.device, latents.dtype)
    control = Packing._pack_latents(control)

    context = torch.cat([control, mask, inpaint], dim=2)
    if reference_tokens:
        pad = torch.zeros(
            (context.shape[0], reference_tokens, context.shape[2]),
            device=context.device, dtype=context.dtype,
        )
        context = torch.cat([context, pad], dim=1)
    return context
