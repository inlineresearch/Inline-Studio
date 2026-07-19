"""``inline-extension.json``: the extension manifest, its typed form, and its validator.

Hand-written rather than jsonschema-backed to keep Core's base deps at numpy + psutil. Validation
reports every problem at once, each message naming its JSON path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from ..models.catalog import CATEGORIES

#: `json.loads` returns `Any`; these narrow it once at the boundary so the rest of the module is
#: fully typed under pyright strict rather than sprinkled with per-line suppressions.
JsonObject = dict[str, Any]
JsonArray = list[Any]


def as_object(value: object) -> JsonObject | None:
    return cast(JsonObject, value) if isinstance(value, dict) else None


def as_array(value: object) -> JsonArray | None:
    return cast(JsonArray, value) if isinstance(value, list) else None

#: Extension ids name a directory, the `ext:<id>:*` RPC prefix, and (with `-` as `_`) the mandatory
#: top-level Python package, so they stay conservative.
EXTENSION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
#: `module.path:callable`
ENTRY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
#: Node types are namespaced like the built-ins (`alibaba/z-image-turbo`).
NODE_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*$")

MANIFEST_FILENAME = "inline-extension.json"
SCHEMA_VERSION = 1


class ManifestError(ValueError):
    """One or more manifest problems. ``problems`` is the full list, one per JSON path."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def package_name(extension_id: str) -> str:
    """The top-level Python package an extension **must** ship. Enforced, not conventional: it is
    what makes module ownership unambiguous, so no two extensions can collide on their own code."""
    return "inline_ext_" + extension_id.replace("-", "_")


@dataclass(frozen=True)
class ModelRequirement:
    """One weight file a node needs on disk, in the shape the existing model-download popup
    already consumes (see ``models/requirements.py``)."""

    id: str
    label: str
    category: str
    repo: str
    repo_file: str
    filename: str
    optional: bool = False
    approx_bytes: int | None = None


@dataclass(frozen=True)
class ExtensionNode:
    """One node an extension provides. Title, ports, and params come from the ``@inline_node``
    decorator - the manifest only declares what exists, so the two can never disagree."""

    #: Declared ahead of import so collisions are caught in preflight, before any extension code
    #: runs. The registrar rejects a runner whose type is not listed here.
    type: str
    models: tuple[ModelRequirement, ...] = ()
    default_enabled: bool = True


@dataclass(frozen=True)
class Prebuilt:
    """An author-built ``site/`` tarball for one platform+interpreter. When one matches, install
    skips resolution entirely - the Node-Packer path reduced to "an extension with nothing to
    resolve"."""

    platform: str
    python: str
    url: str
    sha256: str


@dataclass(frozen=True)
class Manifest:
    schema: int
    id: str
    name: str
    version: str
    core_compat: str
    python_requires: str
    requirements: tuple[str, ...]
    #: One entry point for the whole extension: ``module.path:register``. Because everything is
    #: imported at once, enabling a node later never needs a restart.
    entry: str
    nodes: tuple[ExtensionNode, ...]
    isolation: Literal["shared", "process"] = "shared"
    prebuilt: tuple[Prebuilt, ...] = ()
    ui: str | None = None
    author: str = ""
    homepage: str = ""
    license: str = ""
    description: str = ""

    @property
    def package(self) -> str:
        return package_name(self.id)

    def node(self, node_type: str) -> ExtensionNode | None:
        return next((n for n in self.nodes if n.type == node_type), None)

    def node_types(self) -> tuple[str, ...]:
        return tuple(node.type for node in self.nodes)


# --- validation ---------------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Accumulates problems so one pass reports them all."""

    problems: list[str] = field(default_factory=lambda: [])

    def bad(self, path: str, message: str) -> None:
        self.problems.append(f"{path}: {message}")

    def text(self, obj: JsonObject, key: str, path: str, *, required: bool = True) -> str:
        value = obj.get(key)
        if value is None:
            if required:
                self.bad(f"{path}.{key}", "is required")
            return ""
        if not isinstance(value, str) or not value.strip():
            self.bad(f"{path}.{key}", "must be a non-empty string")
            return ""
        return value


def parse_manifest(raw: object, *, expect_id: str | None = None) -> Manifest:
    """Validate a decoded manifest. ``expect_id`` (the extension directory name) must match ``id`` -
    the two are the same identity and a mismatch would desynchronize the on-disk layout."""
    ctx = _Ctx()
    obj = as_object(raw)
    if obj is None:
        raise ManifestError(["$: manifest must be a JSON object"])

    schema = obj.get("schema")
    if schema != SCHEMA_VERSION:
        ctx.bad("$.schema", f"must be {SCHEMA_VERSION} (got {schema!r})")

    extension_id = ctx.text(obj, "id", "$")
    if extension_id and not EXTENSION_ID_RE.match(extension_id):
        ctx.bad("$.id", "must be lowercase letters, digits and dashes (2-64 chars)")
    if extension_id and expect_id is not None and extension_id != expect_id:
        ctx.bad("$.id", f"must equal the extension directory name {expect_id!r}")

    name = ctx.text(obj, "name", "$")
    version = ctx.text(obj, "version", "$")
    core_compat = ctx.text(obj, "coreCompat", "$")
    python_requires = obj.get("pythonRequires") or ">=3.11"
    if not isinstance(python_requires, str):
        ctx.bad("$.pythonRequires", "must be a string")
        python_requires = ">=3.11"

    isolation = obj.get("isolation", "shared")
    if isolation not in ("shared", "process"):
        ctx.bad("$.isolation", "must be 'shared' or 'process'")
        isolation = "shared"

    entry = ctx.text(obj, "entry", "$")
    if entry:
        if not ENTRY_RE.match(entry):
            ctx.bad("$.entry", "must look like 'module.path:callable'")
        elif extension_id and not entry.startswith(package_name(extension_id)):
            # The ownership model rests on this: an extension may only import from its own
            # mandatory top-level package, so no two can shadow each other's code.
            ctx.bad("$.entry", f"must live inside the extension's package "
                               f"{package_name(extension_id)!r}")

    requirements = _str_list(ctx, obj.get("requirements", []), "$.requirements")
    prebuilt = _prebuilt(ctx, obj.get("prebuilt", []))
    nodes = _nodes(ctx, obj.get("nodes"))
    ui = _ui(ctx, obj.get("ui"), "$")

    if ctx.problems:
        raise ManifestError(ctx.problems)

    return Manifest(
        schema=SCHEMA_VERSION,
        id=extension_id,
        name=name,
        version=version,
        core_compat=core_compat,
        python_requires=python_requires,
        requirements=requirements,
        entry=entry,
        nodes=nodes,
        isolation="process" if isolation == "process" else "shared",
        prebuilt=prebuilt,
        ui=ui,
        author=_opt_str(obj.get("author")),
        homepage=_opt_str(obj.get("homepage")),
        license=_opt_str(obj.get("license")),
        description=_opt_str(obj.get("description")),
    )


def load_manifest(source_dir: Path, *, expect_id: str | None = None) -> Manifest:
    """Read + validate ``<source_dir>/inline-extension.json``."""
    path = source_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise ManifestError([f"$: {MANIFEST_FILENAME} not found at the repository root"])
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError([f"$: could not read {MANIFEST_FILENAME} ({error})"]) from error
    return parse_manifest(raw, expect_id=expect_id)


def _nodes(ctx: _Ctx, raw: object) -> tuple[ExtensionNode, ...]:
    entries = as_array(raw)
    if entries is None or not entries:
        ctx.bad("$.nodes", "must be a non-empty array")
        return ()
    items: list[ExtensionNode] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        path = f"$.nodes[{index}]"
        node = as_object(entry)
        if node is None:
            ctx.bad(path, "must be an object")
            continue
        node_type = ctx.text(node, "type", path)
        if node_type and not NODE_TYPE_RE.match(node_type):
            ctx.bad(f"{path}.type", f"node type {node_type!r} must look like 'owner/name'")
        if node_type and node_type in seen:
            ctx.bad(f"{path}.type", f"node type {node_type!r} is declared twice")
        seen.add(node_type)

        items.append(
            ExtensionNode(
                type=node_type,
                models=_models(ctx, node.get("models", []), f"{path}.models"),
                default_enabled=bool(node.get("defaultEnabled", True)),
            )
        )
    return tuple(items)


def _ui(ctx: _Ctx, raw: object, path: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        ctx.bad(f"{path}.ui", "must be a non-empty string when present")
        return None
    if raw.startswith("/") or ".." in Path(raw).parts:
        ctx.bad(f"{path}.ui", "must be a relative path inside the repository")
        return None
    return raw


def _models(ctx: _Ctx, raw: object, path: str) -> tuple[ModelRequirement, ...]:
    entries = as_array(raw)
    if entries is None:
        ctx.bad(path, "must be an array")
        return ()
    items: list[ModelRequirement] = []
    for index, entry in enumerate(entries):
        at = f"{path}[{index}]"
        model = as_object(entry)
        if model is None:
            ctx.bad(at, "must be an object")
            continue
        category = ctx.text(model, "category", at)
        if category and category not in CATEGORIES:
            # Rejected, never rewritten: the category picks both the folder and the options_from
            # dropdown, so a wrong one downloads fine then reports the model missing.
            ctx.bad(
                f"{at}.category",
                f"must be one of {', '.join(CATEGORIES)} (got {category!r})",
            )
        filename = model.get("filename") or model.get("repoFile")
        if isinstance(filename, str) and ("/" in filename or ".." in filename):
            ctx.bad(f"{at}.filename", "must be a bare filename, not a path")
            filename = None
        approx = model.get("approxBytes")
        if approx is not None and not isinstance(approx, int):
            ctx.bad(f"{at}.approxBytes", "must be an integer when present")
            approx = None
        items.append(
            ModelRequirement(
                id=ctx.text(model, "id", at),
                label=ctx.text(model, "label", at),
                category=category,
                repo=ctx.text(model, "repo", at),
                repo_file=ctx.text(model, "repoFile", at),
                filename=filename if isinstance(filename, str) else "",
                optional=bool(model.get("optional", False)),
                approx_bytes=approx if isinstance(approx, int) else None,
            )
        )
    return tuple(items)


def _prebuilt(ctx: _Ctx, raw: object) -> tuple[Prebuilt, ...]:
    entries = as_array(raw)
    if entries is None:
        ctx.bad("$.prebuilt", "must be an array")
        return ()
    items: list[Prebuilt] = []
    for index, entry in enumerate(entries):
        at = f"$.prebuilt[{index}]"
        built = as_object(entry)
        if built is None:
            ctx.bad(at, "must be an object")
            continue
        url = ctx.text(built, "url", at)
        if url and not url.startswith("https://"):
            ctx.bad(f"{at}.url", "must be https")
        digest = ctx.text(built, "sha256", at)
        if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
            ctx.bad(f"{at}.sha256", "must be a 64-character hex sha256")
        items.append(
            Prebuilt(
                platform=ctx.text(built, "platform", at),
                python=ctx.text(built, "python", at),
                url=url,
                sha256=digest,
            )
        )
    return tuple(items)


def _str_list(ctx: _Ctx, raw: object, path: str) -> tuple[str, ...]:
    entries = as_array(raw)
    if entries is None:
        ctx.bad(path, "must be an array of strings")
        return ()
    out: list[str] = []
    for index, item in enumerate(entries):
        if not isinstance(item, str) or not item.strip():
            ctx.bad(f"{path}[{index}]", "must be a non-empty string")
            continue
        out.append(item)
    return tuple(out)


def _opt_str(value: object) -> str:
    return value if isinstance(value, str) else ""
