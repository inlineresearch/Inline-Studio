"""Core serves the Inline Studio SPA on its own port: frontend resolution + the static mount, with
the /v1 API still winning. No torch/models needed."""

from __future__ import annotations

from fastapi.testclient import TestClient

from inline_core.server.app import create_app
from inline_core.server.frontend import resolve_frontend_root


def _spa(tmp_path):
    """A minimal built-SPA dir: index.html + a hashed asset."""
    root = tmp_path / "dist-web"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>Inline Studio</title><div id=r></div>")
    (root / "assets" / "index-abc123.js").write_text("console.log('spa')")
    return root


def test_resolve_prefers_env_dir(monkeypatch, tmp_path):
    root = _spa(tmp_path)
    monkeypatch.setenv("INLINE_FRONTEND_ROOT", str(root))
    assert resolve_frontend_root() == str(root)


def test_resolve_none_when_dir_has_no_index(monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("INLINE_FRONTEND_ROOT", str(empty))
    assert resolve_frontend_root() is None


def test_resolve_none_when_unset_and_package_absent(monkeypatch):
    monkeypatch.delenv("INLINE_FRONTEND_ROOT", raising=False)
    # inline_studio_frontend isn't installed in this env, so it resolves to None (API only).
    assert resolve_frontend_root() is None


def test_serves_spa_index_and_assets(tmp_path):
    root = _spa(tmp_path)
    app = create_app(frontend_root=str(root))
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "Inline Studio" in index.text
        asset = client.get("/assets/index-abc123.js")
        assert asset.status_code == 200
        assert "spa" in asset.text


def test_v1_routes_still_win_over_static_mount(tmp_path):
    root = _spa(tmp_path)
    app = create_app(frontend_root=str(root))
    with TestClient(app) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True


def test_no_mount_when_frontend_root_is_none(tmp_path):
    app = create_app(frontend_root=None)
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/").status_code == 404  # API only, no SPA
