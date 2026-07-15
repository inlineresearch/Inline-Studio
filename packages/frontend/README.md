# inline-studio-frontend

Prebuilt Inline Studio web UI (SPA), packaged so **Inline Core** can serve it on a single port —
mirroring ComfyUI's `comfyui-frontend-package`. End users `pip install inline-core` (which depends on
this), then run `python main.py`; no Node build on their machine.

## What's in the wheel

The payload is the built SPA under `inline_studio_frontend/static/` (`index.html` + hashed `assets/`).
Core resolves this dir via `inline_core.server.frontend.resolve_frontend_root` and mounts it.

## Publishing (release only)

```bash
# in the Inline Studio repo
npm run build:spa                       # -> dist-web/
cp -r dist-web/* packages/frontend/inline_studio_frontend/static/
cd packages/frontend
python -m build && twine upload dist/*  # publish to PyPI
```

## Local development — don't republish

Point Core at a local build instead:

```bash
npm run build:spa
python main.py --front-end-root ../Inline-Studio/dist-web   # in the Inline Core repo
# or, for HMR:
npm run dev:web        # Vite dev server on :5173, proxying the backend to Core
```
