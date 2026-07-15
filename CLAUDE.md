# Inline Studio — Engineering Guide

Inline Studio is an **experimentation layer for visual artists**: a free-form node canvas for
building generative pipelines frame-by-frame. The **Inline Core** engine (`core/`) does the actual
image/video generation behind each frame — diffusion models run locally (Z-Image Turbo and others),
plus closed models via fal.ai.

> **Naming:** the project is **Inline Studio** — that is the only name. The npm package is
> `inline-studio`. **Do not use the old "Storyline" codename** anywhere new (docs, identifiers, UI
> strings). Some legacy `STORYLINE_*` env vars and `.storyline` paths still exist in code and are
> being renamed to `inline-studio` — treat them as deprecated, don't add more.
>
> Inline Studio is the **single repo**: it holds the UI client (`src/`) **and** the **Inline Core**
> Python generation engine (`core/`, brought in via `git subtree`). One process serves both —
> `cd core && python main.py --front-end-root ../dist-web` runs Core and serves the built UI on one
> port.

> Read this file before changing code. It defines the architecture and the non-negotiable rules.

## Mental model (everything is organised around this)

```
Project → Sequence → Frame → Take[]
```

- **Project** — a portable `.inlinestudio` folder (see Storage below).
- **Sequence / Scene** — an ordered group of frames.
- **Frame** — the atomic unit. **A Frame is a _slot with a history of takes_, never a single file.**
  Its inputs are library assets _or_ another frame's output (the refine/flow link).
- **Take** — one immutable render of a frame. Generating again adds a new take; nothing is
  overwritten. The frame points at its `heroTakeId` (the chosen take), which flows downstream.
- **Moodboard ↔ Timeline** — a frame is either pinned on the free-form canvas or surfaced in the
  Timeline panel. Same frame, different surface.

If you're tempted to treat a frame as a file, stop — the take history is the core value Comfy lacks.
(Note: the domain was renamed shot → frame; some older migrations still reference `shot_*` tables.)

## Architecture

Inline Studio is a **web SPA** (React, `src/`) served by **Inline Core** (the Python engine, `core/`)
on a single port. One process: `core/main.py` runs Core, which serves the built UI _and_ is the app's
backend. (The former Electron desktop app + Node web server were **retired** — the whole backend was
ported to Python. If you find a reference to `electron/`, `server/`, `window.inlineStudio`, or a
preload bridge, it's stale.)

- **Renderer** (`src/renderer/`) — all React UI. Reaches the backend only through `studio()`
  (`lib/studio.ts`), an injected HTTP/WebSocket client (`lib/webClient.ts`) pointed at Core on the
  same origin: every `InlineStudioApi` call is a `POST /rpc {channel, args}`; events stream over the
  `/events` WebSocket; media loads from `/media/*`; asset uploads `POST /upload`. Never imports Node.
- **Shared** (`src/shared/`) — domain types + the `InlineStudioApi` contract (`ipc.ts`) that the
  renderer and Core both honor. **This is the frozen wire protocol** — change it in lockstep on both
  sides.
- **Inline Core** — the Python backend, in this repo under **`core/`** (`core/src/inline_core/`).
  Owns the project SQLite DB, filesystem, generation, and the ffmpeg timeline. Studio's former backend
  lives under `inline_core/studio/` (`store`, `frames`, `moodboard`, `assets`, `generation`, `fal`,
  `timeline`) + the `/rpc`+`/events`+`/media`+`/upload` routes in `inline_core/server/`. Set it up
  with `cd core && uv sync --extra zimage --extra server`.

Fal node definitions stay studio-side (`src/shared/nodes/`): the browser builds each fal request and
Core relays it to `queue.fal.run` with the API key server-side. Core nodes (e.g. Z-Image) run through
Core's own graph engine.

### Directory map (TS side)

```
src/
  shared/
    types.ts            domain types (Project/Sequence/Frame/Take/MoodboardItem/...)
    ipc.ts              IpcChannels + InlineStudioApi (the wire contract)
    coreNodes.ts        the Core node-descriptor contract (served at /v1/models)
    nodes/              fal model defs (NodeDef: resolveEndpoint/buildRequest/parseOutputs)
    result.ts           Result<T> = Ok | Err
  renderer/
    web/index.html, web/main.tsx   the SPA entry (mounts App with the web client + media resolver)
    App.tsx
    lib/                studio.ts (backend seam), webClient.ts (HTTP/WS), mount.tsx, media.ts
    store/              Zustand stores (moodboardStore, frameStore, generationStore, ...)
    views/              feature-foldered screens (ProjectLauncher, Workspace, Moodboard, Library, ...)
    components/         shared UI
vite.config.spa.ts      builds the SPA -> dist-web/ (the inline_studio_frontend PyPI wheel payload)
```

### Storage — a portable project folder (owned by Core)

```
MyFilm.inlinestudio/
  project.db   (SQLite — source of truth; "save" is implicit)
  assets/      (imported library media, by id)
  takes/       (generated outputs, by take id)
  thumbs/      (director previews / cached media)
  exports/     (hero-take folder exports)
```

Recents, settings, and the fal API key are app-global under Core's data dir
(`~/.inline-studio-server`). The browser has no folder picker, so new projects are created under
`~/InlineStudioProjects` (`INLINE_STUDIO_WORKSPACE_DIR`).

### Generation & timeline (on Core)

- **Core nodes** (Z-Image Turbo, …) — the browser calls `generation:runWorkflow(itemId)`; Core builds
  the graph from the canvas closure, runs it through its own engine, saves takes, streams progress.
- **Fal nodes** — the browser builds `{endpoint, body, outputKind}` from the NodeDef and calls
  `generation:run`; Core's relay submits/polls `queue.fal.run` (key server-side), downloads, saves.
- **Director timelines** — resolved from canvas connectors and rendered with ffmpeg
  (`inline_core/studio/timeline/`), progress over `/events`. Folder export copies hero takes.

The embedded-ComfyUI webview and `comfy.*` channels were desktop-only and are retired; Core returns a
clear "not available" for them.

## Code standards (non-negotiable)

- **TypeScript strict.** No implicit `any`, no `as any` to silence errors. `npm run typecheck`.
- **Typed contract only.** Channels live in `src/shared/ipc.ts`; the web client (`webClient.ts`)
  implements `InlineStudioApi` generically from `IpcChannels`; every call returns `Result<T>`. Core
  implements the same channels in Python — change the contract in lockstep on both sides.
- **Renderer is browser-only.** No Node/Electron imports (ESLint-enforced). All "trusted" work
  (filesystem, DB, generation, ffmpeg) is Core's, reached over `/rpc`; validate payloads in Core.
- **State.** Zustand stores are small and feature-scoped. Components render; stores + `studio()` do
  work.
- **Backend logic lives in Core**, not the renderer. Fal node _definitions_ stay studio-side
  (`src/shared/nodes/`); their execution is Core's fal relay.
- **Files & naming.** Components `PascalCase.tsx`, hooks `useX.ts`, one component per file,
  feature-foldered views. Keep files under ~300 lines without a good reason.
- **Icons, never emoji.** Never use emoji in the UI (no 🎬/🎵/✂/🔊 as glyphs). Use crisp,
  consistent line SVG icons (Lucide-style: `viewBox="0 0 24 24"`, `fill="none"`,
  `stroke="currentColor"`) that inherit color/size via `currentColor` + a size class. Follow the
  existing icon components (`src/renderer/components/icons`, `CanvasToolbar` icons, `DirectorNode`'s
  `VolumeIcon`); reuse or add to those rather than dropping in an emoji.
- **Tests (Vitest).** Cover the logic that matters: Comfy input/workflow resolution, frame-input and
  hero-take resolution, DB migrations. UI is verified by running the app — don't chase view coverage.
- **Commits.** Conventional Commits (`feat:`, `fix:`, `chore:`), small and scoped. `lint` +
  `typecheck` run on pre-commit (husky + lint-staged).

## Node UI style (the canvas node family — non-negotiable)

Every node on the moodboard canvas reads as **one card design**. New nodes (fal models, Inline Core
nodes, anything) MUST match it — the fal Generate node (`nodes/GenNode.tsx`) and the Inline Core node
(`nodes/GraphNode.tsx`) are the reference. The shared parts live in `nodes/NodeBadge.tsx`; reuse them,
don't re-invent:

- **Card chrome:** wrap in `NodeFrame` with `padded={false}` + `subtleSelect` (quiet `zinc-600`
  selection border, not the loud accent). Do your own layout inside.
- **Floating title:** a `NodeBadgeRow` + `NodeBadge` pinned above the card — an icon glyph + the node
  title (+ optional `tone="info"` badges like a price). Icons are Lucide-style stroked glyphs from
  `NodeBadge.tsx` (`WandIcon`, `BoxIcon`, …); **never emoji**. Map a node's icon string to a glyph.
- **Body:** an edge-to-edge output preview on a `bg-black` `flex-1` area (`object-cover` media), with a
  busy overlay = a status pill (top-left) + a 1px bottom progress bar in emerald.
- **Params live OFF the node face.** The face shows no param widgets. A footer **Adjust** button
  (`AdjustIcon`, tagged `data-gen-settings-toggle`) opens a right-hand settings **sidebar** that renders
  the params (`GenerateSettingsPanel` for fal, `CoreSettingsPanel` for Core; both keyed in
  `generationStore`, mutually exclusive in the right gutter). This keeps generation one-click.
- **Footer bar:** `border-t border-border bg-surface/90`, a small left label, and a right cluster with
  the Run control (`PlayIcon`, emerald) + the Adjust button.
- **Handles:** `group !h-3 !w-3 !border-2 !border-surface`, colored per port kind, evenly spaced down
  the edge, with a hover chip naming the port.

## Commands

```
npm run dev:web     # Vite dev server (HMR), proxying /rpc,/events,/media,/upload,/v1 to Core
npm run build:spa   # build the SPA -> dist-web/ (served by Core; the PyPI wheel payload)
npm run typecheck   # tsc on the web project (renderer + shared)
npm run lint        # eslint, zero warnings allowed
npm run test        # vitest
```

Run the whole app on one port (UI + API):

```
npm run build:spa                                       # -> dist-web/
cd core && uv run python main.py --front-end-root ../dist-web   # add --listen for LAN
```

See `core/CLAUDE.md` for the engine internals (nodes, models, device policy).

## Where to add things

- New backend call → add the channel + `InlineStudioApi` signature in `src/shared/ipc.ts`, then
  implement the handler in Core (`inline_core/studio/handlers.py`, backed by the domain modules).
  The web client forwards it automatically.
- New screen → `src/renderer/views/<Feature>/`, plus a store in `src/renderer/store/` if it owns state.
- New canvas node type → a component in `src/renderer/views/Moodboard/nodes/` registered in
  `MoodboardPanel`'s `nodeTypes`, plus any `MoodboardItemType` in `src/shared/types.ts`. **Follow the
  "Node UI style" section above** — match the shared card design (badge, `subtleSelect`, footer
  Run+Adjust, params in a sidebar).
- New fal model → a `NodeDef` file in `src/shared/nodes/` appended to `NODE_DEFS` (the rest is
  data-driven off it — it appears in the Add-node picker automatically).
- New generation-engine behaviour → Core (`inline_core/`); new domain entity → `src/shared/types.ts`
  - the Python schema (`inline_core/studio/schema.py`, bump `SCHEMA_VERSION` + migration).
