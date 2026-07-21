"""Node behaviour, without a running server. Run with `PYTHONPATH=python pytest`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inline_ext_demo.basic import Brightness, Invert
from inline_ext_demo.gradient import Gradient
from inline_ext_demo.upscale import Upscale


def _node(**params: Any) -> Any:
    return SimpleNamespace(params=params)


def _run(runner: Any, image: np.ndarray, **params: Any) -> np.ndarray:
    result = runner.run(_node(**params), {"image": [image]}, None)
    return np.asarray(result.outputs["image"])


def test_invert_flips_the_image_at_full_amount() -> None:
    image = np.array([[0, 128, 255]], dtype=np.uint8)
    assert _run(Invert(), image, amount=1.0).tolist() == [[255, 127, 0]]


def test_invert_at_zero_is_a_no_op() -> None:
    image = np.array([[10, 200]], dtype=np.uint8)
    assert _run(Invert(), image, amount=0.0).tolist() == image.tolist()


def test_invert_uses_its_declared_default() -> None:
    """Params come from the descriptor's defaults merged under the node's saved values."""
    image = np.array([[0]], dtype=np.uint8)
    assert _run(Invert(), image).tolist() == [[255]]


def test_brightness_scales_and_clips() -> None:
    image = np.array([[10, 200]], dtype=np.uint8)
    assert _run(Brightness(), image, scale=2.0).tolist() == [[20, 255]]


def test_upscale_reports_a_missing_weight_file_clearly() -> None:
    image = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(FileNotFoundError, match="model popup"):
        _run(Upscale(), image, model="definitely-not-installed.pth")


def test_the_entry_point_registers_exactly_what_the_manifest_declares() -> None:
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).parent.parent / "inline-extension.json").read_text(encoding="utf-8")
    )
    declared = {node["type"] for node in manifest["nodes"]}
    implemented = {
        cls.__inline_descriptor__.type for cls in (Gradient, Invert, Brightness, Upscale)
    }
    assert implemented == declared
