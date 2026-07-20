"""The install-time security scan, shared verbatim with registry CI.

This informs a decision; it does not contain anything. In-process Python cannot be sandboxed, so the
real defences are review, provenance, and consent - never imply a clean scan means "safe".

CRITICAL blocks outright (no legitimate use in an extension). HIGH/MEDIUM needs typed consent.
LOW never gates: egress to a known model host stays LOW so the consent prompt stays rare enough
that users actually read it.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .constraints import protected_requirements
from .manifest import Manifest, as_array, as_object


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def blocks(self) -> bool:
        return self is Severity.CRITICAL

    @property
    def needs_consent(self) -> bool:
        return self in (Severity.HIGH, Severity.MEDIUM)


#: Hosts an extension may reach without prompting: model weights and release artifacts.
EGRESS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "huggingface.co",
        "hf.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.hf.co",
        "raw.githubusercontent.com",
        "github.com",
        "objects.githubusercontent.com",
    }
)

#: Callables that reach the network. Value is True when the call carries an explicit URL argument
#: we can try to resolve statically.
_NETWORK_CALLS: dict[str, bool] = {
    "urlopen": True,
    "urlretrieve": True,
    "get": True,
    "post": True,
    "put": True,
    "delete": True,
    "head": True,
    "patch": True,
    "request": True,
    "hf_hub_download": False,
    "snapshot_download": False,
}
_NETWORK_MODULES = {"requests", "httpx", "urllib", "urllib3", "aiohttp", "socket"}
_HF_CALLS = {"hf_hub_download", "snapshot_download", "load_dataset"}

_EXEC_NAMES = {"exec", "eval", "compile"}
_DECODERS = {"b64decode", "b85decode", "a85decode", "unhexlify", "decompress", "loads"}
_SUBPROCESS = {"subprocess", "os", "pty", "commands"}
_SUBPROCESS_CALLS = {"system", "popen", "spawn", "spawnl", "spawnv", "execv", "execve", "fork"}


@dataclass(frozen=True)
class Finding:
    severity: Severity
    rule: str
    message: str
    file: str = ""
    line: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "severity": self.severity.value,
            "rule": self.rule,
            "message": self.message,
            "file": self.file,
            "line": self.line,
        }


@dataclass
class ScanReport:
    findings: list[Finding]

    @property
    def blocked(self) -> bool:
        return any(f.severity.blocks for f in self.findings)

    @property
    def needs_consent(self) -> bool:
        return not self.blocked and any(f.severity.needs_consent for f in self.findings)

    def consent_rules(self) -> list[str]:
        """Rule ids the user must accept. Recorded in state.json so re-installing the same version
        does not re-prompt, while a new version - with possibly new behaviour - does."""
        return sorted({f.rule for f in self.findings if f.severity.needs_consent})

    def by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity is severity]

    def to_json(self) -> dict[str, object]:
        return {
            "blocked": self.blocked,
            "needsConsent": self.needs_consent,
            "consentRules": self.consent_rules(),
            "findings": [f.to_json() for f in self.findings],
        }


def scan(source: Path, manifest: Manifest | None = None) -> ScanReport:
    """Scan an extension checkout. ``manifest`` enables the requirements and packaging checks."""
    findings: list[Finding] = []
    if manifest is not None:
        findings.extend(_scan_manifest(manifest))
    findings.extend(_scan_packaging(source))
    for path in sorted(source.rglob("*.py")):
        if _skip(path, source):
            continue
        findings.extend(_scan_python(path, source))
    findings.extend(_scan_binaries(source))
    findings.sort(key=lambda f: (_ORDER[f.severity], f.file, f.line))
    return ScanReport(findings)


_ORDER = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}


def _skip(path: Path, source: Path) -> bool:
    parts = path.relative_to(source).parts
    return any(part in {".git", "__pycache__", ".venv", "site", "node_modules"} for part in parts)


# --- manifest + packaging -------------------------------------------------------------------------


def _scan_manifest(manifest: Manifest) -> list[Finding]:
    findings: list[Finding] = []
    for requirement in protected_requirements(manifest.requirements):
        findings.append(
            Finding(
                Severity.CRITICAL,
                "host-dependency-override",
                f"requires {requirement!r}, which is part of the shared Inline runtime. "
                "Extensions must depend on the installed torch/diffusers/transformers, never "
                "install their own - two copies in one process corrupts the CUDA context.",
                file="inline-extension.json",
            )
        )
    if not manifest.license:
        findings.append(
            Finding(
                Severity.LOW,
                "no-license",
                "the manifest declares no license",
                file="inline-extension.json",
            )
        )
    return findings


def _scan_packaging(source: Path) -> list[Finding]:
    """Installing must never execute author code.

    Dependencies install from a fully pinned lock with ``--no-deps``, so a ``setup.py`` in the repo
    is not run by us - but its presence means the repo expects to be built, and a build backend
    runs arbitrary code. Flag it rather than silently relying on our install shape.
    """
    findings: list[Finding] = []
    setup_py = source / "setup.py"
    if setup_py.is_file():
        findings.append(
            Finding(
                Severity.CRITICAL,
                "install-time-code-execution",
                "ships a setup.py, which executes arbitrary code at build time. Extensions are "
                "loaded from source and must not require a build step.",
                file="setup.py",
            )
        )
    return findings


def _scan_binaries(source: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or _skip(path, source):
            continue
        name = path.name.lower()
        relative = str(path.relative_to(source))
        if name.startswith("libtorch") or name.startswith("libcud") or name.startswith("nvidia"):
            findings.append(
                Finding(
                    Severity.CRITICAL,
                    "bundled-native-runtime",
                    f"ships {relative!r}. A second copy of the CUDA/torch native runtime in one "
                    "process segfaults rather than raising, and looks like an Inline Core bug.",
                    file=relative,
                )
            )
        elif path.suffix.lower() in {".so", ".pyd", ".dylib"}:
            findings.append(
                Finding(
                    Severity.MEDIUM,
                    "compiled-binary",
                    f"ships the compiled binary {relative!r}, which cannot be reviewed as source",
                    file=relative,
                )
            )
    return findings


# --- python source --------------------------------------------------------------------------------


def _scan_python(path: Path, source: Path) -> list[Finding]:
    relative = str(path.relative_to(source))
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, SyntaxError, ValueError) as error:
        return [
            Finding(
                Severity.MEDIUM,
                "unparseable-source",
                f"could not be parsed for review ({error})",
                file=relative,
            )
        ]
    visitor = _Visitor(relative)
    visitor.visit(tree)
    return visitor.findings


class _Visitor(ast.NodeVisitor):
    def __init__(self, relative: str) -> None:
        self.file = relative
        self.findings: list[Finding] = []
        self._imported_decoders: set[str] = set()

    def _add(self, severity: Severity, rule: str, message: str, node: ast.AST) -> None:
        self.findings.append(
            Finding(severity, rule, message, file=self.file, line=getattr(node, "lineno", 0))
        )

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast API
        for alias in node.names:
            self._note_import(alias.name.split(".")[0], node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast API
        module = (node.module or "").split(".")[0]
        self._note_import(module, node)
        for alias in node.names:
            if alias.name in _DECODERS:
                self._imported_decoders.add(alias.asname or alias.name)
        self.generic_visit(node)

    def _note_import(self, module: str, node: ast.AST) -> None:
        if module in {"base64", "binascii", "zlib", "marshal", "pickle"}:
            self._imported_decoders.add(module)
        if module == "ctypes":
            self._add(
                Severity.HIGH,
                "ctypes",
                "imports ctypes, which can call arbitrary native code and bypass this review",
                node,
            )
        elif module == "socket":
            self._add(
                Severity.HIGH,
                "raw-socket",
                "opens raw sockets, whose destination cannot be checked statically",
                node,
            )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        name = _call_name(node.func)
        qualifier = _call_qualifier(node.func)

        if name in _EXEC_NAMES:
            self._check_exec(node, name)
        elif qualifier in _SUBPROCESS and name in _SUBPROCESS_CALLS:
            self._add(
                Severity.HIGH,
                "subprocess",
                f"runs external commands via {qualifier}.{name}()",
                node,
            )
        elif qualifier == "subprocess" or name in {"Popen", "run"} and qualifier == "subprocess":
            self._add(Severity.HIGH, "subprocess", "runs external commands via subprocess", node)
        elif name in {"loads", "load"} and qualifier == "pickle":
            self._add(
                Severity.HIGH,
                "pickle-deserialization",
                "deserializes pickle data, which executes arbitrary code on load",
                node,
            )
        elif name in _NETWORK_CALLS and (qualifier in _NETWORK_MODULES or name in _HF_CALLS):
            self._check_egress(node, name, qualifier)

        self.generic_visit(node)

    def _check_exec(self, node: ast.Call, name: str) -> None:
        """``exec`` over a decoded blob is the signature of a hidden payload - and unlike a plain
        ``exec``, it has no legitimate explanation in an extension, so it blocks."""
        if node.args and _wraps_decoder(node.args[0], self._imported_decoders):
            self._add(
                Severity.CRITICAL,
                "obfuscated-execution",
                f"calls {name}() on a decoded or decompressed payload, hiding what it runs",
                node,
            )
        elif name != "compile":
            self._add(
                Severity.HIGH,
                "dynamic-execution",
                f"calls {name}() on a value computed at runtime",
                node,
            )

    def _check_egress(self, node: ast.Call, name: str, qualifier: str) -> None:
        if name in _HF_CALLS:
            self._add(
                Severity.LOW,
                "model-download",
                f"downloads model files via {name}()",
                node,
            )
            return
        host = _literal_host(node)
        if host is None:
            self._add(
                Severity.HIGH,
                "network-egress",
                f"{qualifier}.{name}() sends a request to a URL built at runtime, so its "
                "destination cannot be determined by review",
                node,
            )
        elif host in EGRESS_ALLOWLIST:
            self._add(Severity.LOW, "network-egress-known", f"contacts {host}", node)
        else:
            self._add(
                Severity.HIGH,
                "network-egress",
                f"contacts {host}, which is not a recognized model or release host",
                node,
            )


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _call_qualifier(func: ast.expr) -> str:
    """The leftmost name of a dotted call, e.g. ``requests`` in ``requests.adapters.get()``."""
    if isinstance(func, ast.Attribute):
        node: ast.expr = func.value
        while isinstance(node, ast.Attribute):
            node = node.value
        if isinstance(node, ast.Name):
            return node.id
    return ""


def _wraps_decoder(node: ast.expr, imported: set[str]) -> bool:
    """True when the expression decodes or decompresses something anywhere inside it.

    The decoder must be traceable to an actual decoding module (``base64.b64decode``) or to a name
    imported from one (``from base64 import b64decode``). Matching on the bare call name alone
    would make ``exec(json.loads(cfg))`` a CRITICAL finding - and CRITICAL blocks with no override,
    so a false positive here is unrecoverable for the user.
    """
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = _call_name(child.func)
        if name not in _DECODERS:
            continue
        qualifier = _call_qualifier(child.func)
        if qualifier in imported or (not qualifier and name in imported):
            return True
    return False


def _literal_host(node: ast.Call) -> str | None:
    """The host of a statically known URL argument, or None when it isn't a literal.

    An f-string or a variable returns None on purpose: an unresolvable destination is exactly the
    case that must stay HIGH rather than being waved through.
    """
    from urllib.parse import urlparse

    candidates = list(node.args) + [kw.value for kw in node.keywords if kw.arg in {"url", "uri"}]
    for candidate in candidates:
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
            parsed = urlparse(candidate.value)
            if parsed.scheme in {"http", "https"} and parsed.hostname:
                return parsed.hostname.lower()
    return None


def load_report(path: Path) -> ScanReport:
    """Re-read a persisted scan, so a resumed install does not re-scan."""
    try:
        raw = as_object(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return ScanReport([])
    findings: list[Finding] = []
    for item in as_array(raw.get("findings")) or [] if raw is not None else []:
        entry = as_object(item)
        if entry is None:
            continue
        try:
            severity = Severity(str(entry.get("severity")))
        except ValueError:
            continue
        findings.append(
            Finding(
                severity=severity,
                rule=str(entry.get("rule", "")),
                message=str(entry.get("message", "")),
                file=str(entry.get("file", "")),
                line=_int(entry.get("line")),
            )
        )
    return ScanReport(findings)


def _int(value: object) -> int:
    try:
        return int(value)  # pyright: ignore[reportArgumentType] - guarded by the except
    except (TypeError, ValueError):
        return 0
