"""MiniMax H3's checkpoint-to-diffusers key plan.

The released checkpoints are written for MiniMax's own implementation; the vendored port is a
diffusers model. Every tensor maps by renaming except three, and those three are the ones that fail
silently rather than loudly:

* **The fused QKV** (``[21504, 5376]``, i.e. 3 x 56 heads x 128) splits into ``to_q``/``to_k``/
  ``to_v``. Row order differs by publisher: ``MiniMaxAI`` ships it per-head interleaved and
  ``Comfy-Org`` ships the same data de-interleaved, verified by hashing rows 0-127, 128-255 and
  256-383 of one against rows 0, 7168 and 14336 of the other. So the layout is declared per source
  and ``keymap``'s detector measures the real one before slicing.
* **The gated FFN** (``mlp.fc1``, ``[28672, 5376]``, i.e. 2 x 14336) becomes ``ff.net.0.proj``
  with its halves exchanged. diffusers' ``SwiGLU.forward`` does ``value, gate = proj(x).chunk(2)``
  and returns ``value * silu(gate)``: it reads ``[value; gate]``, the reference stores the other.
* **``rope.inv_freq``** is shipped and also recomputed by the port. The config does not state
  ``rope_theta``, so rather than trusting a default this is asserted against what the port built:
  every shipped element solves to theta = 10000.0, which is the port's default, and a future
  checkpoint that disagrees becomes a load error instead of geometry that drifts across the frame.

535 source tensors become 639 targets: the 52 attention blocks (50 plus 2 refiner) each turn one
fused QKV into three.
"""

from __future__ import annotations

from ..keymap import AssertEqual, Drop, KeyPlan, Rename, RowLayout, Split, SwapHalves

#: Bumping this invalidates every prepared artifact built by the old plan.
PLAN_VERSION = "minimax-h3.keys.1"

NUM_BLOCKS = 50
NUM_REFINER_BLOCKS = 2

#: Geometry from FL2VA/transformer/config.json, not inferred from tensor shapes.
NUM_HEADS = 56
HEAD_DIM = 128

#: Where each publisher's fused QKV rows sit. Measured, not assumed - see the module docstring.
SOURCE_LAYOUTS = {
    "comfy-org": RowLayout.CONTIGUOUS,
    "minimaxai": RowLayout.INTERLEAVED,
}

#: Top-level tensors, reference name -> diffusers name. The stems are paired with `.weight`/`.bias`
#: below, except the norms, which have no bias.
_TOP_LEVEL = {
    "video_patch_proj": "proj_in",
    "audio_patch_proj": "audio_proj_in",
    "condition_proj": "context_embedder",
    "time_embedder.proj_in": "time_embedder.linear_1",
    "time_embedder.proj_out": "time_embedder.linear_2",
    "final_layer.video_out": "proj_out",
    "final_layer.audio_out": "audio_proj_out",
    "final_layer.adaln_proj.linear": "norm_out.linear",
}

_WEIGHT_ONLY = {
    "final_layer.norm.weight": "norm_out.norm.weight",
    "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
}

#: Per-block renames that need no transform, reference stem -> diffusers stem.
_BLOCK_RENAMES = {
    "attn.out_proj.weight": "attn.to_out.0.weight",
    "attn.q_norm.weight": "attn.norm_q.weight",
    "attn.k_norm.weight": "attn.norm_k.weight",
    "mlp.fc2.weight": "ff.net.2.weight",
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "adaln_proj.linear.weight": "adaln_proj.linear.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
}

#: The refiner blocks have no AdaLN branch of their own.
_REFINER_SKIP = {"adaln_proj.linear.weight", "adaln_proj.linear.bias"}


def build_plan(
    source: str = "comfy-org",
    *,
    num_blocks: int = NUM_BLOCKS,
    num_refiner_blocks: int = NUM_REFINER_BLOCKS,
    head_dim: int = HEAD_DIM,
    pruned: bool = False,
    sidecars: tuple[str, ...] = (),
) -> KeyPlan:
    """The plan for a publisher's layout. ``source`` selects how the fused QKV rows are arranged.

    ``pruned`` is the published rank-8 build: it ships ``adaln_t_table`` in place of the whole
    timestep path, so the two ``time_embedder`` projections are simply not in the file.

    ``sidecars`` are the quantisation tensors an fp8 build carries beside each weight. They are
    consumed while streaming and dropped here, so the coverage check still accounts for every key.

    The counts are arguments so a round-trip test can exercise the same code at a size that fits in
    memory; the defaults are the released geometry.
    """
    try:
        layout = SOURCE_LAYOUTS[source]
    except KeyError:
        raise ValueError(
            f"Unknown checkpoint source {source!r}; expected one of {sorted(SOURCE_LAYOUTS)}."
        ) from None

    actions: dict[str, object] = {}
    top_level = {
        stem: target
        for stem, target in _TOP_LEVEL.items()
        if not (pruned and stem.startswith("time_embedder."))
    }
    if pruned:
        actions["adaln_t_table"] = Drop("read before streaming, to rebuild the timestep path")
    for key in sidecars:
        actions[key] = Drop("a quantisation scale, applied to its weight while streaming")
    for stem, target in top_level.items():
        for suffix in ("weight", "bias"):
            actions[f"{stem}.{suffix}"] = Rename(f"{target}.{suffix}")
    for key, target in _WEIGHT_ONLY.items():
        actions[key] = Rename(target)
    # Shipped and recomputed; asserted rather than loaded. See the module docstring.
    actions["rope.inv_freq"] = AssertEqual("rope.inv_freq")

    for prefix, target_prefix, count in (
        ("blocks", "transformer_blocks", num_blocks),
        ("token_refiner.blocks", "token_refiner.refiner_blocks", num_refiner_blocks),
    ):
        refiner = prefix.startswith("token_refiner")
        for index in range(count):
            src = f"{prefix}.{index}"
            dst = f"{target_prefix}.{index}"
            actions[f"{src}.attn.qkv_proj.weight"] = Split(
                (
                    f"{dst}.attn.to_q.weight",
                    f"{dst}.attn.to_k.weight",
                    f"{dst}.attn.to_v.weight",
                ),
                layout=layout,
                head_dim=head_dim,
            )
            actions[f"{src}.mlp.fc1.weight"] = SwapHalves(f"{dst}.ff.net.0.proj.weight")
            for stem, target in _BLOCK_RENAMES.items():
                if refiner and stem in _REFINER_SKIP:
                    continue
                actions[f"{src}.{stem}"] = Rename(f"{dst}.{target}")

    suffix = source + ("+pruned" if pruned else "") + ("+fp8" if sidecars else "")
    return KeyPlan(version=f"{PLAN_VERSION}+{suffix}", actions=actions)  # type: ignore[arg-type]


def self_computed_targets(*, pruned: bool = False) -> set[str]:
    """Targets the port builds itself, which ``check_coverage`` must not demand be filled."""
    if not pruned:
        return {"rope.inv_freq"}
    return {"rope.inv_freq", *_PRUNED_SELF_COMPUTED}


#: What ``adaln.tabulate`` creates in place of the timestep path. No checkpoint fills these.
_PRUNED_SELF_COMPUTED = ("time_embedder.table", "time_embedder.linear_1.weight")
