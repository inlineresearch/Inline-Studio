"""Suite-wide isolation: nothing under test may write into the checkout.

``config.extensions_dir()`` falls back to ``./extensions`` so a dev checkout works with no env set.
That default is process-wide, so any test that builds the server app (``create_app`` ->
``register_models`` -> ``load_extensions``) materialises ``extensions/.cache`` in whatever directory
pytest was started from. Two tests do it today, ``test_lora_download`` and ``test_studio_rpc``, and
neither mentions extensions at all.

Beyond littering, it made ``test_every_channel_uses_the_installers_root_not_the_environment`` pass
on a clean tree and fail on every run afterwards, because the directory it asserts against was
created by the run before it. Pointing the default at a tmp dir per test removes the ordering
dependency and keeps that guard meaningful.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_extensions_root(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the process-wide extensions default somewhere disposable."""
    monkeypatch.setenv("INLINE_EXTENSIONS_DIR", str(tmp_path_factory.mktemp("extensions-default")))
