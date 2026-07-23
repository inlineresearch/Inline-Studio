"""LoRA training subpackage - runs as a **subprocess**, never imported by the server.

The Studio orchestrator (``studio/training.py``) launches ``python -m inline_core.training
<manifest>``; this package trains a LoRA and reports progress as JSON lines (``protocol.py``). Heavy
deps (torch/diffusers/peft) live behind the ``training`` optional-deps extra and import lazily in
``trainer.py`` / ``dataset.py`` / ``caption.py`` - importing this package is cheap + torch-free, so
the server never pulls the training stack in.
"""

from __future__ import annotations
