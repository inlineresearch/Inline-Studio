"""Vendored LTX-2 model code, taken verbatim from Lightricks' release repository.

    upstream: https://github.com/Lightricks/LTX-2
    branch:   main
    commit:   fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
    taken:    2026-08-12
    packages: packages/ltx-core/src/ltx_core, packages/ltx-pipelines/src/ltx_pipelines

LTX-2.5 has no diffusers release: the Diffusers-packaged variant only loads on diffusers `main`,
and Core pins `diffusers==0.39.0` because Z-Image, Krea 2, FLUX.2 and MiniMax H3 all depend on that
pin. So there is nothing to depend on, and the choice is to vendor or to pin a moving branch.

**The only edit made to these files is import rewriting**, applied mechanically so a re-sync stays a
diff a person can read. Re-running it after a fresh copy reproduces this tree exactly:

    find . -name '*.py' -print0 | xargs -0 sed -i -E \\
      -e 's/^([[:space:]]*)from ltx_core([. ])/\\1from inline_core.models.ltx25.vendor.ltx_core\\2/' \\
      -e 's/^([[:space:]]*)from ltx_pipelines([. ])/\\1from inline_core.models.ltx25.vendor.ltx_pipelines\\2/' \\
      -e 's/^([[:space:]]*)import ltx_core([. ]|$)/\\1import inline_core.models.ltx25.vendor.ltx_core\\2/' \\
      -e 's/^([[:space:]]*)import ltx_pipelines([. ]|$)/\\1import inline_core.models.ltx25.vendor.ltx_pipelines\\2/'

It rewrites import statements only. Docstring references to `ltx_core.*` are prose and were left
alone, so they still name the upstream module a reader would search for.

Nothing was deleted. The HDR/EXR path (`utils/media_io/exr.py`) and the multi-GPU runners
(`*_mgpu.py`, `multigpu/`) are unused by our runner but kept, because dropping them would mean
editing `media_io/__init__.py` and `ltx_pipelines/__init__.py`, and a partial tree is the thing that
makes the next re-sync unreviewable. The cost is two dependencies, `openimageio` (a 7 MB wheel) and
`cloudpickle`, which is cheaper than a permanent local patch.

`core/CLAUDE.md`'s file-length and comment-density rules do not apply inside this directory, and
`pyproject.toml` excludes `models/*/vendor/` from ruff and pyright for the same reason: this is
third-party code, and editing it to satisfy our linters would destroy the property above.

Importing this package requires the `runtime` extra. That is deliberate and matches every other
model runner: an absent extra makes the import raise and `server/bootstrap.py` skips the model, so a
torch-less install still boots.
"""
