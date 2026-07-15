"""The Studio app-backend, ported to Python (Part B of the migration).

This is the former Electron/TypeScript ``electron/main`` backend — projects, the project SQLite DB,
frames/takes, the moodboard, settings — reimplemented so Inline Core is the single backend the web
SPA talks to over ``/rpc`` (see ``inline_core.server.rpc``). Ported domain by domain; until the
storage layer lands, the ``/rpc`` bridge still proxies un-ported channels to the Node backend.
"""
