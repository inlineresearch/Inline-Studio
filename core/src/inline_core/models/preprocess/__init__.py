"""ControlNet preprocessing nodes (Apply ControlNet: image -> pose/depth/canny control map)."""

from .runner import CONTROL_APPLY, register_control_apply

__all__ = ["CONTROL_APPLY", "register_control_apply"]
