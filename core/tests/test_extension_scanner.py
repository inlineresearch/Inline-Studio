"""The security scanner: every blocking rule must fire, and a clean extension must not be gated.

The second half matters as much as the first. A scanner that flags ordinary extensions trains users
to confirm reflexively, and a consent prompt nobody reads protects nobody.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.extensions.manifest import ExtensionNode, Manifest
from inline_core.extensions.scanner import ScanReport, Severity, load_report, scan


def _write(tmp_path: Path, code: str, name: str = "nodes.py") -> Path:
    source = tmp_path / "src"
    source.mkdir(parents=True, exist_ok=True)
    (source / name).write_text(code, encoding="utf-8")
    return source


def _manifest(requirements: tuple[str, ...] = (), license_: str = "MIT") -> Manifest:
    return Manifest(
        schema=1,
        id="acme-tools",
        name="Acme Tools",
        version="1.0.0",
        core_compat=">=1.2",
        python_requires=">=3.11",
        requirements=requirements,
        entry="inline_ext_acme_tools:register",
        nodes=(ExtensionNode(type="acme/invert"),),
        license=license_,
    )


def _rules(report: ScanReport) -> set[str]:
    return {f.rule for f in report.findings}


# --- clean extensions are not gated ---------------------------------------------------------------


CLEAN = '''
"""An ordinary image node."""
import numpy as np
from inline_core.extensions.api import inline_node
from inline_core.graph.runners import NodeResult, NodeRunner


@inline_node(type="acme/invert", title="Invert", category="Image")
class Invert(NodeRunner):
    def run(self, node, inputs, ctx) -> NodeResult:
        image = inputs["image"][0]
        return NodeResult(outputs={"image": np.asarray(255 - image)})
'''


def test_a_clean_extension_produces_no_gate(tmp_path: Path) -> None:
    report = scan(_write(tmp_path, CLEAN), _manifest())
    assert not report.blocked
    assert not report.needs_consent
    assert report.consent_rules() == []


def test_downloading_weights_from_hugging_face_is_informational(tmp_path: Path) -> None:
    """The consent-fatigue guard: nearly every useful extension does this, so it must not prompt."""
    code = """
from huggingface_hub import hf_hub_download

def fetch():
    return hf_hub_download("acme/weights", "model.safetensors")
"""
    report = scan(_write(tmp_path, code), _manifest())
    assert not report.blocked
    assert not report.needs_consent
    assert "model-download" in _rules(report)


def test_egress_to_an_allowlisted_host_is_informational(tmp_path: Path) -> None:
    code = """
import requests

def fetch():
    return requests.get("https://huggingface.co/acme/weights/resolve/main/model.safetensors")
"""
    report = scan(_write(tmp_path, code), _manifest())
    assert not report.needs_consent
    assert "network-egress-known" in _rules(report)


# --- CRITICAL: each must block --------------------------------------------------------------------


def test_requiring_torch_is_blocked(tmp_path: Path) -> None:
    report = scan(_write(tmp_path, CLEAN), _manifest(requirements=("torch>=2.0",)))
    assert report.blocked
    assert "host-dependency-override" in _rules(report)


@pytest.mark.parametrize("protected", ["torch==2.5.1", "diffusers>=0.36", "NumPy", "transformers"])
def test_every_shared_runtime_package_is_blocked(tmp_path: Path, protected: str) -> None:
    report = scan(_write(tmp_path, CLEAN), _manifest(requirements=(protected,)))
    assert report.blocked, f"{protected} must not be installable by an extension"


def test_obfuscated_execution_is_blocked(tmp_path: Path) -> None:
    code = """
import base64

exec(base64.b64decode("cHJpbnQoJ2hpJyk="))
"""
    report = scan(_write(tmp_path, code), _manifest())
    assert report.blocked
    assert "obfuscated-execution" in _rules(report)


def test_obfuscated_execution_is_caught_through_nesting(tmp_path: Path) -> None:
    """The payload is rarely the direct argument in real samples."""
    code = """
import base64, zlib

exec(zlib.decompress(base64.b64decode(PAYLOAD)).decode("utf-8"))
"""
    report = scan(_write(tmp_path, code), _manifest())
    assert report.blocked
    assert "obfuscated-execution" in _rules(report)


def test_exec_of_a_non_decoder_call_is_not_a_critical_block(tmp_path: Path) -> None:
    """CRITICAL blocks with no override, so a false positive is unrecoverable for the user.
    `loads` is a generic name - only a call traceable to a real decoding module counts."""
    code = """
import json

exec(json.loads(config))
"""
    report = scan(_write(tmp_path, code), _manifest())
    assert not report.blocked
    assert "obfuscated-execution" not in _rules(report)
    assert "dynamic-execution" in _rules(report), "still worth consent, just not a block"


def test_decoder_imported_by_name_is_still_caught(tmp_path: Path) -> None:
    """Tightening the rule must not open the obvious evasion."""
    code = """
from base64 import b64decode

exec(b64decode(PAYLOAD))
"""
    report = scan(_write(tmp_path, code), _manifest())
    assert report.blocked
    assert "obfuscated-execution" in _rules(report)


def test_setup_py_is_blocked(tmp_path: Path) -> None:
    source = _write(tmp_path, CLEAN)
    (source / "setup.py").write_text("from setuptools import setup\nsetup()\n", encoding="utf-8")
    report = scan(source, _manifest())
    assert report.blocked
    assert "install-time-code-execution" in _rules(report)


def test_bundled_native_runtime_is_blocked(tmp_path: Path) -> None:
    """Two libtorch copies in one process segfault rather than raising, and that reads as a Core
    bug rather than an extension bug."""
    source = _write(tmp_path, CLEAN)
    (source / "libtorch_cpu.so").write_bytes(b"\x7fELF")
    report = scan(source, _manifest())
    assert report.blocked
    assert "bundled-native-runtime" in _rules(report)


# --- HIGH / MEDIUM: consent, not a block ----------------------------------------------------------


def test_unresolvable_egress_requires_consent(tmp_path: Path) -> None:
    """A URL built at runtime cannot be reviewed, so it must not be waved through."""
    code = """
import requests

def send(cfg, payload):
    return requests.post(f"{cfg.endpoint}/collect", json=payload)
"""
    report = scan(_write(tmp_path, code), _manifest())
    assert not report.blocked
    assert report.needs_consent
    assert "network-egress" in report.consent_rules()


def test_egress_to_an_unknown_host_requires_consent(tmp_path: Path) -> None:
    code = """
import requests

requests.post("https://telemetry.example.com/collect", json={})
"""
    report = scan(_write(tmp_path, code), _manifest())
    assert report.needs_consent
    finding = next(f for f in report.findings if f.rule == "network-egress")
    assert "telemetry.example.com" in finding.message


@pytest.mark.parametrize(
    ("code", "rule"),
    [
        ("import subprocess\nsubprocess.run(['ls'])", "subprocess"),
        ("import os\nos.system('ls')", "subprocess"),
        ("import pickle\npickle.loads(data)", "pickle-deserialization"),
        ("import ctypes\nctypes.CDLL('libc.so.6')", "ctypes"),
        ("import socket\ns = socket.socket()", "raw-socket"),
        ("eval(user_input)", "dynamic-execution"),
    ],
)
def test_capabilities_require_consent(tmp_path: Path, code: str, rule: str) -> None:
    report = scan(_write(tmp_path, code), _manifest())
    assert not report.blocked, f"{rule} has legitimate uses and must not hard-block"
    assert report.needs_consent
    assert rule in _rules(report)


def test_compiled_binaries_are_flagged_but_do_not_block(tmp_path: Path) -> None:
    source = _write(tmp_path, CLEAN)
    (source / "_speedups.cpython-311-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
    report = scan(source, _manifest())
    assert not report.blocked
    assert "compiled-binary" in _rules(report)


def test_unparseable_source_is_reported_not_ignored(tmp_path: Path) -> None:
    report = scan(_write(tmp_path, "def broken(:\n"), _manifest())
    assert "unparseable-source" in _rules(report)


def test_missing_license_is_informational_only(tmp_path: Path) -> None:
    report = scan(_write(tmp_path, CLEAN), _manifest(license_=""))
    assert not report.blocked
    assert not report.needs_consent
    assert "no-license" in _rules(report)


# --- report mechanics -----------------------------------------------------------------------------


def test_findings_are_ordered_most_severe_first(tmp_path: Path) -> None:
    code = """
import base64, subprocess
subprocess.run(['ls'])
exec(base64.b64decode(BLOB))
"""
    report = scan(_write(tmp_path, code), _manifest(license_=""))
    severities = [f.severity for f in report.findings]
    assert severities[0] is Severity.CRITICAL
    assert severities == sorted(severities, key=lambda s: [*Severity].index(s))


def test_blocked_takes_precedence_over_consent(tmp_path: Path) -> None:
    """A blocked install must never render as a consent prompt the user can click past."""
    code = "import base64\nimport subprocess\nsubprocess.run(['ls'])\nexec(base64.b64decode(B))"
    report = scan(_write(tmp_path, code), _manifest())
    assert report.blocked
    assert not report.needs_consent


def test_scan_ignores_vendored_and_vcs_directories(tmp_path: Path) -> None:
    """site/ holds resolved third-party code, which is governed by the lockfile, not this scan."""
    source = _write(tmp_path, CLEAN)
    for folder in ("site", ".git", "__pycache__"):
        (source / folder).mkdir()
        (source / folder / "evil.py").write_text("import base64\nexec(base64.b64decode(X))\n")
    report = scan(source, _manifest())
    assert not report.blocked


def test_report_round_trips_through_json(tmp_path: Path) -> None:
    import json

    report = scan(_write(tmp_path, "import subprocess\nsubprocess.run(['ls'])"), _manifest())
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(report.to_json()), encoding="utf-8")

    restored = load_report(path)
    assert restored.needs_consent == report.needs_consent
    assert restored.consent_rules() == report.consent_rules()


def test_load_report_tolerates_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "scan.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_report(path).findings == []
