from __future__ import annotations

import pytest
from inline_core.config import server_host, server_port


def test_server_host_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INLINE_HOST", raising=False)
    assert server_host() == "127.0.0.1"


def test_server_host_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_HOST", "0.0.0.0")
    assert server_host() == "0.0.0.0"


def test_server_port_defaults_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INLINE_PORT", raising=False)
    assert server_port() == 8848
    monkeypatch.setenv("INLINE_PORT", "9000")
    assert server_port() == 9000


def test_server_port_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_PORT", "not-a-port")
    assert server_port() == 8848
