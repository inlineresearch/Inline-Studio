"""DurationHead: predicts shot duration from Connector token outputs."""

from inline_core.models.ltx25.vendor.ltx_core.duration_head.duration_head import AttentionPooler, DurationHead
from inline_core.models.ltx25.vendor.ltx_core.duration_head.model_configurator import DURATION_HEAD_KEY_OPS, DurationHeadConfigurator

__all__ = [
    "DURATION_HEAD_KEY_OPS",
    "AttentionPooler",
    "DurationHead",
    "DurationHeadConfigurator",
]
