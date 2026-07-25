# Contributing to Inline Studio

Thanks for helping. Issues, ideas and pull requests are all welcome. Come talk to us on
[Discord](https://discord.gg/cSUS88VdY9) if you want to sanity-check an idea before you build it.

## Setup

The UI client (`src/`) and the Inline Core engine (`core/`) live in one repo, served on one port.

```bash
npm install
npm run dev:web                                   # Vite dev server for the UI

cd core
uv sync --extra runtime --extra server            # the engine + generation runtime
uv run python main.py --front-end-root ../dist-web # run Core, serving a built UI
```

[CLAUDE.md](CLAUDE.md) and [core/CLAUDE.md](core/CLAUDE.md) are the engineering guides: architecture,
the data model, and the conventions. Read the relevant one before a larger change.

## Before you open a PR

Run the same checks CI does. Zero warnings, all green:

```bash
npm run typecheck && npm run lint && npm run test   # UI + shared
cd core && ruff check . && uv run pytest -q          # engine (no GPU needed)
```

Keep changes small and focused. Match the style of the code around you.

## Commits and sign-off

Use [Conventional Commits](https://www.conventionalcommits.org) (`feat:`, `fix:`, `chore:`), and sign
off every commit:

```bash
git commit -s -m "fix: ..."
```

The `-s` adds a `Signed-off-by` line, your agreement to the
[Developer Certificate of Origin](https://developercertificate.org): a short statement that you wrote
the change, or have the right to submit it, under the project's license.

## License

Inline Studio is GPL-3.0. Contributions are accepted under the same license.
