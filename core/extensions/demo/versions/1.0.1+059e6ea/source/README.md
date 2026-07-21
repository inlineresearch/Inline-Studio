# Inline Demo Extension

A reference extension for [Inline Studio](https://github.com/inlineresearch/Inline-Studio). Copy this
repo as the starting point for your own.

It ships four nodes, each independently toggleable by the user:

| Node              | Shows                                                               |
| ----------------- | ------------------------------------------------------------------- |
| `demo/gradient`   | A **runnable** node: Run control, take history, saved output        |
| `demo/invert`     | The minimum: a decorated `NodeRunner` that transforms its input     |
| `demo/brightness` | A second node from the same entry point                             |
| `demo/upscale`    | Declaring weights + an `options_from` model picker (off by default) |

## Install it

Extensions → **Available** → Install. Or **Install from URL** with this repo's URL and
`latest`, which resolves to the newest release tag.

## Layout

```
inline-extension.json          the manifest (identity, nodes, dependencies, models)
python/inline_ext_demo/        your code - the package name is mandatory (see below)
  __init__.py                  the single register() entry point
  basic.py                     nodes with no extra dependencies
  upscale.py                   a node that needs a weight file
tests/test_nodes.py            plain pytest
```

**The package name is not a convention, it's enforced.** It must be `inline_ext_<your id>` with
dashes as underscores: `id: "acme-tools"` → `python/inline_ext_acme_tools/`. That's what guarantees
two extensions can never shadow each other's modules.

## Writing a node

```python
@inline_node(
    type="demo/invert",              # must be listed in the manifest's `nodes`
    title="Invert",
    category="Image",                # groups it in the add-node menu
    icon="wand",
    inputs=(Port("image", "Image", PortKind.IMAGE, required=True),),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(ParamField("amount", "Amount", Widget.NUMBER, 1.0, min=0.0, max=1.0),),
)
class Invert(NodeRunner):
    produces_takes = False

    def run(self, node, inputs, ctx) -> NodeResult:
        ...

```

Then hand every node to the registrar from your package's single entry point:

```python
# python/inline_ext_demo/__init__.py
def register(reg: ExtensionRegistrar) -> None:
    reg.nodes(Invert, Brightness, Upscale)
```

The canvas builds the whole node UI from each descriptor: ports, the settings sidebar, take
history. There is no frontend code to write.

Declare every node in the manifest's `nodes` array. Users toggle nodes individually, and because
your code is imported once, switching one on never needs a restart. Give a node
`"defaultEnabled": false` to ship it off by default.

## Making a node runnable

A node only gets the **Run** control, a preview, and take history when it declares a media output.
Without `output_kind` it is plumbing: it has no Run button and executes only when something
downstream of it runs. That is why `demo/invert` has no Run control but `demo/gradient` does.

```python
@inline_node(
    type="demo/gradient",
    title="Gradient",
    category="Generate",
    output_kind=MediaKind.IMAGE,        # <- this is what puts Run on the node
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(ParamField("width", "Width", Widget.NUMBER, 512),),
)
class Gradient(NodeRunner):
    produces_takes = True               # <- and this is what gives it take history

    def run(self, node, inputs, ctx) -> NodeResult:
        rgb = ...                        # a uint8 HxWx3 numpy array
        take = ctx.takes.save(ctx.run_id, node.id, rgb, node.params)
        return NodeResult(outputs={"image": rgb}, takes=[take])
```

`ctx.takes` is the take store. Guard it with `if ctx.takes is None` so your node still works when
it is executed as part of someone else's graph rather than as a run target.

See `python/inline_ext_demo/gradient.py` for the complete node.

## Dependencies

List them in `requirements`:

```json
"requirements": ["einops>=0.7,<0.9"]
```

They install into your extension's own folder. **You cannot install `torch`, `diffusers`,
`transformers`, `numpy`, or anything else in Inline's runtime.** Import those directly, they're
already there. Declaring one is a hard install failure, not a warning: two copies of torch in one
process is a segfault, not a version conflict.

If your dependency needs a version of a shared package that Inline doesn't have, the install fails
with both versions named. Align with the host, or say so in your README.

## Models

Declare weights on the node that needs them and Inline handles the rest: the download popup,
progress, and the `options_from` dropdown all work with no code:

```json
"nodes": [
  { "type": "demo/upscale", "models": [
  { "id": "realesrgan-x4", "label": "Real-ESRGAN x4", "category": "upscale_models",
    "repo": "ai-forever/Real-ESRGAN", "repoFile": "RealESRGAN_x4.pth",
      "filename": "RealESRGAN_x4.pth", "approxBytes": 67040989 } ] }
]
```

`category` must be one of Inline's model folders (`diffusion_models`, `vae`, `text_encoders`,
`loras`, `controlnet`, `checkpoints`, `clip_vision`, `upscale_models`, `embeddings`). Never download
weights yourself at import or run time.

## Reference

Every value below is what Core actually accepts. Anything not listed is not supported.

<details>
<summary><b>Icons</b> - the <code>icon=</code> string on <code>@inline_node</code></summary>

| `icon`        | Glyph  | Suits                   |
| ------------- | ------ | ----------------------- |
| `"wand"`      | wand   | Generating nodes        |
| `"box"`       | box    | Loaders, model plumbing |
| `"type"`      | type   | Text and prompt nodes   |
| `"image"`     | image  | Image transforms        |
| anything else | square | Fallback, no warning    |

The set is small on purpose: icons are React components in the app, so an extension cannot ship its
own. An unrecognised string silently falls back to the square, which is why a typo produces no
error. Adding one is a Core change (`coreGlyph` in `GraphNode.tsx`).

</details>

<details>
<summary><b>Port kinds</b> - the <code>kind=</code> on every <code>Port</code></summary>

Media kinds cross the wire as files and can back a Frame:

| `PortKind`   | Wire value | Socket colour |
| ------------ | ---------- | ------------- |
| `IMAGE`      | `image`    | green         |
| `IMAGE_LIST` | `image[]`  | green         |
| `VIDEO`      | `video`    | cyan          |
| `AUDIO`      | `audio`    | violet        |
| `TEXT`       | `text`     | amber         |
| `MASK`       | `mask`     | pink          |

Engine handles are opaque objects passed between nodes, never a take:

| `PortKind`     | Wire value     | Socket colour |
| -------------- | -------------- | ------------- |
| `MODEL`        | `model`        | red           |
| `VAE`          | `vae`          | orange        |
| `TEXT_ENCODER` | `text-encoder` | yellow        |
| `CONDITIONING` | `conditioning` | purple        |
| `LATENT`       | `latent`       | blue          |

```python
Port("image", "Image", PortKind.IMAGE, required=True)
#     id      label     kind            required (default False)
```

Edges are type-checked **before** a run, so bad wiring is rejected at submit rather than mid-render.
The only widening rule is `image` → `image[]`. **Custom port kinds are rejected** - the registrar
raises if `kind` isn't one of the above, because edge validation has to stay decidable without
running extension code.

`id` is what you look up in `run()`. `inputs["image"]` is a **list**, since `image[]` accepts several
edges; a single-valued port is `inputs["image"][0]`.

</details>

<details>
<summary><b>Param types</b> - the <code>Widget</code> on every <code>ParamField</code></summary>

| `Widget`   | Renders as         | Extra fields honoured       |
| ---------- | ------------------ | --------------------------- |
| `TEXT`     | one-line input     | -                           |
| `TEXTAREA` | multi-line input   | -                           |
| `NUMBER`   | number field       | `min`, `max`, `step`        |
| `BOOLEAN`  | checkbox           | -                           |
| `SELECT`   | dropdown           | `options` or `options_from` |
| `SEED`     | number + randomise | `min`, `max`                |

```python
ParamField("scale", "Scale", Widget.NUMBER, 2.0, min=1.0, max=4.0, step=0.5)
#           key      label    widget         default
```

| Field                  | Meaning                                                              |
| ---------------------- | -------------------------------------------------------------------- |
| `key`                  | What you read in `run()`, via `node.params["key"]`                   |
| `label`                | Shown in the settings sidebar                                        |
| `widget`               | One of the above                                                     |
| `default`              | Used until the user changes it                                       |
| `min` / `max` / `step` | `NUMBER` and `SEED` only                                             |
| `options`              | `tuple[Option, ...]` - a fixed dropdown                              |
| `options_from`         | A model folder name - Core fills the dropdown from what is installed |
| `advanced`             | Hides it behind the sidebar's advanced section                       |

`options_from` accepts exactly these folders: `diffusion_models`, `checkpoints`, `vae`,
`text_encoders`, `loras`, `clip_vision`, `controlnet`, `upscale_models`, `embeddings`. The dropdown
repopulates whenever a file lands there, so a model you declare appears without a restart.

**Always merge defaults**, because a node stores only the params the user actually changed:

```python
params = {**MyNode.__inline_descriptor__.defaults(), **node.params}
```

</details>

<details>
<summary><b>The descriptor</b> - every argument to <code>@inline_node</code></summary>

| Argument      | Required | Meaning                                                                  |
| ------------- | -------- | ------------------------------------------------------------------------ |
| `type`        | yes      | `owner/name`, and must also appear in the manifest's `nodes`             |
| `title`       | yes      | Shown on the node badge and in the add-node menu                         |
| `category`    | yes      | Groups it in the add-node menu (`Generate`, `Image`, …)                  |
| `inputs`      | no       | `tuple[Port, ...]`, drawn on the left                                    |
| `outputs`     | no       | `tuple[Port, ...]`, drawn on the right                                   |
| `params`      | no       | `tuple[ParamField, ...]`, shown in the settings sidebar                  |
| `output_kind` | no       | `MediaKind.IMAGE` / `VIDEO` / `AUDIO`. **Set this to get a Run control** |
| `icon`        | no       | See Icons                                                                |
| `hidden`      | no       | Registered and runnable, but never offered in the add-node menu          |

`source` is not settable: the registrar stamps `ext:<your-id>`, so provenance cannot be spoofed.

</details>

<details>
<summary><b>The runner</b> - what <code>run()</code> receives and returns</summary>

```python
class MyNode(NodeRunner):
    produces_takes = False   # True when you save takes

    def run(self, node, inputs, ctx) -> NodeResult:
        ...
```

| Handle                 | What it is                                                               |
| ---------------------- | ------------------------------------------------------------------------ |
| `node.id`              | This node's id in the graph                                              |
| `node.params`          | Only what the user changed - merge your defaults                         |
| `inputs[port_id]`      | A **list** of upstream values for that port                              |
| `ctx.run_id`           | The current run                                                          |
| `ctx.takes`            | The take store, or `None`. Guard it                                      |
| `ctx.policy`           | Device policy. Ask `ctx.policy.placement(role)`; never hardcode a device |
| `ctx.cancel.cancelled` | Check inside long loops so Cancel works                                  |
| `ctx.emitter`          | Progress events                                                          |

Return `NodeResult(outputs={port_id: value}, takes=[...])`. The `outputs` keys must be your declared
output port ids.

</details>

<details>
<summary><b>The registrar</b> - what <code>register(reg)</code> can do</summary>

| Call                              | Purpose                                                                  |
| --------------------------------- | ------------------------------------------------------------------------ |
| `reg.nodes(A, B, C)`              | Register decorated runner classes                                        |
| `reg.node(Cls, runner=instance)`  | Register one, with a pre-built instance                                  |
| `reg.models(node_type, provider)` | Declare model requirements in code rather than the manifest              |
| `reg.rpc_channel("method", fn)`   | A backend call, forced to `ext:<your-id>:method`                         |
| `reg.emit("event", payload)`      | Push to connected clients on `ext:<your-id>:event`                       |
| `reg.takes`                       | The take store                                                           |
| `reg.device`                      | The device policy                                                        |
| `reg.data_dir`                    | Your private scratch directory - writing elsewhere is a CRITICAL finding |

</details>

<details>
<summary><b>Engineering rules</b> - the ones that will fail your extension</summary>

- **The device policy owns placement.** Never pick a device, dtype, or offload yourself; ask
  `ctx.policy.placement(role)`. This is what keeps one graph portable across a 4090, a 6 GB laptop,
  and pure CPU.
- **Takes are immutable.** Regenerating adds a take; never overwrite one.
- **Never download at import or run time.** Declare weights in the manifest and let the model popup
  fetch them - that popup is the only place Core touches the network for models.
- **Import heavy dependencies lazily**, inside `run()` rather than at module top. Your whole
  extension is imported at boot, so a top-level `import torch`-scale cost is paid by every user on
  every start, including those who never enable your node.
- **Check `ctx.cancel.cancelled`** inside any loop that runs longer than a moment, or Cancel appears
  to hang.
- **Node types are namespaced** `owner/name`, and colliding with a Core node type is refused at
  install. You cannot replace a Core node.

</details>

## Security review

Every install is scanned, and the same scan runs in registry CI.

**Blocked outright:** declaring a host package (`torch`, …); `exec`/`eval` over a decoded or
compressed payload; shipping a `setup.py`; bundling `libtorch`/CUDA binaries.

**Needs the user to type your extension's name:** `subprocess`, raw sockets, `ctypes`,
`pickle.loads`, `eval`, and network calls to a host that isn't a known model host, including any
URL built at runtime, since it can't be checked by reading the code.

**Just noted:** downloading from Hugging Face or GitHub, shipping a compiled `.so`, no license.

Keep to the first two lists empty and users install without a warning.

## Publishing

**List once, then just tag.** The registry names your repository, not a version - Inline Studio
resolves your newest release tag at install and when checking for updates. You never open another
registry PR to ship a release.

1. Open a PR against
   [inlineresearch/Inline-Registry](https://github.com/inlineresearch/Inline-Registry) adding
   `registry/<your-id>.json`. One time only.
2. To release: bump `version` in `inline-extension.json`, commit, and tag that same version.

```bash
git tag "v$(python -c 'import json;print(json.load(open("inline-extension.json"))["version"])')"
git push --tags
```

Tags must look like `v1.2.3` or `1.2.3`. Anything else (`nightly`, `release`) is ignored, so a
moving tag can never be served as a release. **Prereleases are skipped when floating** -
`v2.0.0-rc.1` is not offered to users even though it sorts above `v1.10.0`; someone who wants it
installs that tag by name.

Keep the manifest `version` in step with the tag: the tag decides what gets installed, the manifest
`version` is what the UI displays.

## Testing

```bash
pip install -e /path/to/Inline-Studio/core
PYTHONPATH=python pytest
```
