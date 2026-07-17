"""Z-Image reference-output harness (PLAN phase 1). Run on a GPU box:

    PYTHONPATH=src python scripts/reference.py --update   # capture the reference
    PYTHONPATH=src python scripts/reference.py            # diff a fresh render vs the reference

Fixed prompt + seed + steps, so the only variable is the engine. Exits non-zero if MSE exceeds the
threshold, which makes it a regression gate once the reference is committed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROMPT = "a red fox in snow, cinematic lighting, highly detailed"
_SEED = 42
_STEPS = 8
_SIZE = (1024, 1024)
_REF = Path(__file__).resolve().parent.parent / "tests" / "reference" / "z-image-turbo.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="write the reference")
    parser.add_argument("--threshold", type=float, default=5.0, help="max mean-squared error")
    parser.add_argument("--dit", default="z_image_turbo_bf16.safetensors")
    parser.add_argument("--vae", default="ae.safetensors")
    parser.add_argument("--text-encoder", default="qwen3-4b")
    args = parser.parse_args()

    import numpy as np
    from inline_core.config import models_dir
    from inline_core.device.auto import AutoDevicePolicy
    from inline_core.models.zimage.model import ZImageModel
    from PIL import Image

    model = ZImageModel(AutoDevicePolicy(), models_dir())
    image = model.generate(
        selection=(args.dit, args.vae, args.text_encoder),
        prompt=_PROMPT,
        negative_prompt=None,
        steps=_STEPS,
        guidance_scale=0.0,
        width=_SIZE[0],
        height=_SIZE[1],
        seed=_SEED,
        on_step=lambda *step_args: step_args[-1],
    )

    if args.update or not _REF.exists():
        _REF.parent.mkdir(parents=True, exist_ok=True)
        image.save(_REF)
        print(f"Wrote reference: {_REF}")
        return 0

    reference = np.asarray(Image.open(_REF).convert("RGB"), dtype=np.float64)
    rendered = np.asarray(image.convert("RGB"), dtype=np.float64)
    mse = float(((reference - rendered) ** 2).mean())
    print(f"MSE vs reference: {mse:.3f} (threshold {args.threshold})")
    return 0 if mse <= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
