"""Community extensions: installable node extensions from a git repo.

A **extension** is one repo, installed at one pinned commit, holding N independently toggleable
**modules**, each owning node types, optional model requirements, and optional UI.

Authors import only ``inline_core.extensions.api``.

Dependencies get private resolution with install-time conflict detection - not runtime isolation.
See ``importer`` for why the distinction matters.
"""
