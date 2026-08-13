"""Latent upsampler model components."""

from inline_core.models.ltx25.vendor.ltx_core.model.upsampler.model import LatentUpsampler, upsample_video
from inline_core.models.ltx25.vendor.ltx_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator

__all__ = [
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "upsample_video",
]
