"""Install-time probe: which torch is installed, and does it have kernels for this card?

Run as ``python -m inline_core.device.probe`` by both launchers. They cannot share the
compatibility rule any other way, and restating it in bash and batch is how the two drift.

Always prints one JSON object and always exits 0, so a shell branches on ``status`` rather than
parsing stderr. A **nonzero exit means the probe itself failed**, which callers must treat as
unknown and never as covered.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from .detect import arch_list_covers

#: A local version tag we recognise as a pytorch.org build, e.g. ``2.13.0+cu130`` or ``2.9.0+cpu``.
#: Only these are safe to replace automatically: a ROCm build, a nightly or a hand-built wheel will
#: also fail the arch check, and silently reinstalling over someone's deliberate choice is worse
#: than the wrong wheel.
_REPLACEABLE_LOCAL = re.compile(r"\+(?:cpu|cu\d+)$")


def probe() -> dict[str, Any]:
    """What the launcher needs to decide whether to replace torch. Never raises."""
    out: dict[str, Any] = {
        "status": "unknown",
        "torch": None,
        "cuda": None,
        "archList": [],
        "capability": None,
        "replaceable": False,
    }
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch yet is a normal first install
        out["status"] = "no-torch"
        return out

    try:
        out["torch"] = str(torch.__version__)
        out["cuda"] = getattr(torch.version, "cuda", None)
        out["replaceable"] = bool(_REPLACEABLE_LOCAL.search(out["torch"]))
        # HIP reports gfx arches through the sm_ call, so the rule does not apply. Never
        # auto-replace a ROCm build.
        if getattr(torch.version, "hip", None):
            out["status"] = "rocm"
            out["replaceable"] = False
            return out
        arches = [str(a) for a in torch.cuda.get_arch_list() if str(a).startswith("sm_")]
        out["archList"] = arches
        if not arches:
            out["status"] = "cpu-only"
            return out
        if torch.cuda.device_count() < 1:
            out["status"] = "no-gpu"
            return out
        major, minor = torch.cuda.get_device_capability(0)
        out["capability"] = [major, minor]
        out["status"] = "covered" if arch_list_covers(arches, major, minor) else "uncovered"
    except Exception:  # noqa: BLE001 - a broken torch is exactly what we are here to detect
        out["status"] = "unknown"
    return out


def main() -> int:
    json.dump(probe(), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
