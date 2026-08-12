"""Multi-GPU utilities for VAE decoding."""

from inline_core.models.ltx25.vendor.ltx_core.multigpu.vae.distributed_decoder import DistributedVideoDecoder

__all__ = ["DistributedVideoDecoder"]
