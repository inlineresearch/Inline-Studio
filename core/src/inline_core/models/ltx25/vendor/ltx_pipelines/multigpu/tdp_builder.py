"""Tiled data parallel transformer builder.
Wrapping builder that produces a transformer model with tiled data parallelism applied.
"""

from __future__ import annotations

from typing import Generic

import torch
import torch.distributed as dist

from inline_core.models.ltx25.vendor.ltx_core.loader.primitives import ModelBuilderProtocol
from inline_core.models.ltx25.vendor.ltx_core.loader.registry import Registry
from inline_core.models.ltx25.vendor.ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from inline_core.models.ltx25.vendor.ltx_core.model.model_protocol import LTXModelProtocol
from inline_core.models.ltx25.vendor.ltx_core.multigpu.transformer.tiled_data_parallel import (
    TiledDataParallelModelWrapper,
)
from inline_core.models.ltx25.vendor.ltx_core.tiling import TileCountConfig
from inline_core.models.ltx25.vendor.ltx_core.tools import VideoLatentTools
from inline_core.models.ltx25.vendor.ltx_pipelines.multigpu.delegating_builder import DelegatingBuilder, InnerModelT
from inline_core.models.ltx25.vendor.ltx_pipelines.multigpu.weight_tracker import TransformerWeightTracker


class TiledDataParallelBuilder(DelegatingBuilder[InnerModelT], Generic[InnerModelT]):
    """Builder conforming to :class:`ModelBuilderProtocol` that wraps with
    :class:`TiledDataParallelModelWrapper`.
    Requires ``video_tools`` as a keyword argument to :meth:`build` so the
    wrapper can compute the tile for this rank.
    The underlying model must accept ``(video, audio, perturbations)`` and return
    ``(denoised_video, denoised_audio)`` — i.e. conform to the ``X0Model`` forward
    signature used by the LTX transformer.
    """

    def __init__(
        self,
        inner: ModelBuilderProtocol[LTXModelProtocol],
        group: dist.ProcessGroup,
        tiling: TileCountConfig,
        registry: Registry,
        tracker: TransformerWeightTracker,
        normalize_positions: bool = True,
    ) -> None:
        if not isinstance(inner, Builder):
            raise TypeError(f"TiledDataParallelBuilder wraps a SingleGPUModelBuilder, got {type(inner).__name__}")
        cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._inner = inner.with_registry(registry).with_lora_load_device(cuda_device)
        self._tracker = tracker
        self._group = group
        self._tiling = tiling
        self._normalize_positions = normalize_positions

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        # Weight tracker keeps registry tensors GPU-resident and rebinds them across builds.
        return True

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        video_tools: VideoLatentTools | None = None,
        **_kwargs: object,
    ) -> TiledDataParallelModelWrapper:
        if video_tools is None:
            raise ValueError("TiledDataParallelBuilder.build() requires video_tools")
        model = self._tracker.build(self._inner, device=device, dtype=dtype, **_kwargs)
        return TiledDataParallelModelWrapper(
            model,
            video_tools=video_tools,
            tiling=self._tiling,
            group=self._group,
            normalize_positions=self._normalize_positions,
        )
