"""A sweep written as one self-contained HTML file.

Images are embedded rather than linked so the file survives being moved or sent to someone, and the
filename is timestamped so a run never overwrites the one before it - comparing runs over time is
the point.
"""

from __future__ import annotations

import base64
import html
import io
import json
import time
from pathlib import Path
from typing import Any

#: Thumbnailed before embedding: a sweep is dozens of renders, and a report nobody can open is not
#: a report.
THUMB = 512

#: The main score is a face match. Wardrobe is measured separately, and a polished page is
#: exactly where that stops being obvious.
CAVEAT = (
    "The score column is face consistency. Wardrobe is measured separately and only where the "
    "clothes are in frame; body shape is not scored. Every threshold here is uncalibrated: read "
    "these as a comparison between reference sets, not a verdict."
)


def embed_image(path: str) -> str:
    """One image as a data URI, thumbnailed. Public so a caller can embed the references too."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB, THUMB))
            buffer = io.BytesIO()
            im.save(buffer, format="WEBP", quality=80)
        return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001 - a missing render must not lose the whole report
        return ""


def report_name(character: str, target: str, when: float | None = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(when or time.time()))
    return f"{Path(character).stem}-{target}-{stamp}.html"


def build_report(result: dict[str, Any], out: Path) -> Path:
    """Write the run to `out`, returning the file. Never overwrites: the name carries the clock."""
    out.mkdir(parents=True, exist_ok=True)
    name = report_name(str(result.get("character") or "character"), str(result.get("target")))
    target = out / name
    target.write_text(_render(result), encoding="utf-8")
    return target


def _render(result: dict[str, Any]) -> str:
    esc = html.escape
    refs = _reference_rows(result)
    prompts = sorted({str(c["prompt"]) for c in result.get("cells") or []})
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{esc(str(result.get("character")))} reference sweep</title>
<style>
 body {{ font: 13px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
        background: #101012; color: #e4e4e7; }}
 h1, h2 {{ font-weight: 600; }} h1 {{ font-size: 1.4rem; }}
 h2 {{ font-size: 1rem; margin-top: 2rem; }}
 .caveat {{ border-left: 3px solid #f59e0b; padding: .6rem .9rem; background: #1c1917;
            color: #fcd34d; }}
 table {{ border-collapse: collapse; width: 100%; margin-top: .6rem; }}
 th, td {{ text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #27272a; }}
 th {{ color: #a1a1aa; font-weight: 500; }}
 td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 .keep {{ color: #6ee7b7; }} .drop {{ color: #fca5a5; }} .neutral {{ color: #a1a1aa; }}
 img {{ display: block; border-radius: 4px; }}
 .strip {{ display: flex; gap: .5rem; flex-wrap: wrap; margin: .4rem 0 1.2rem; }}
 .cell {{ font-size: 11px; color: #a1a1aa; }}
 code {{ color: #d4d4d8; }}
</style></head><body>
<h1>{esc(str(result.get("character")))} &middot; reference sweep</h1>
<p class="caveat">{esc(CAVEAT)}</p>
<p><code>{esc(str(result.get("target")))}</code> &middot; {result.get("refs")} references &middot;
   {result.get("done")}/{result.get("total")} renders &middot;
   gallery {esc(_gallery(result))}</p>

<h2>What each reference was worth</h2>
<table><tr><th>Reference</th><th></th><th class="num">Delta</th><th class="num">Spread</th>
<th class="num">Pairs</th><th class="num">Framing shift</th><th class="num">Wardrobe</th>
<th>Verdict</th></tr>
{refs}
</table>
<p class="cell">Delta is score(all references) minus score(without this one), differenced
inside each prompt and seed before averaging. Framing shift is how much the face area moved
between those two renders: a large shift means the delta reports composition as much as
identity. Wardrobe is the same paired difference measured on the clothes, over the cells where
the clothes were in frame - a reference that costs face points and pays wardrobe points is doing
its job, which is what <code>keeps-the-wardrobe</code> means.</p>

{_confirmations(result)}

<h2>Per prompt</h2>
{_per_prompt(result, prompts)}

<h2>Reference sets, confidence adjusted</h2>
{_combinations(result)}

<h2>The renders</h2>
{_renders(result, prompts)}

<script type="application/json" id="run">{json.dumps(result)}</script>
</body></html>
"""


def _gallery(result: dict[str, Any]) -> str:
    return str(result.get("gallery") or "refs")


def _reference_rows(result: dict[str, Any]) -> str:
    rows = []
    thumbs = result.get("referenceThumbs") or {}
    for c in result.get("contributions") or []:
        css = {"keep": "keep", "consider-removing": "drop"}.get(str(c["verdict"]), "neutral")
        thumb = thumbs.get(str(c["index"])) or ""
        img = f'<img src="{thumb}" width="56" height="56" alt="">' if thumb else ""
        rows.append(
            f'<tr><td>{img}</td><td>reference {c["index"]}</td>'
            f'<td class="num">{c["delta"]:+.1f}</td><td class="num">{c["spread"]:.1f}</td>'
            f'<td class="num">{c["pairs"]}</td>'
            f'<td class="num">{c.get("framingShift", 0):+.2f}pp</td>'
            f'<td class="num">{_wardrobe_cell(c)}</td>'
            f'<td class="{css}">{html.escape(str(c["verdict"]))}</td></tr>'
        )
    return "\n".join(rows)


def _wardrobe_cell(contribution: dict[str, Any]) -> str:
    """A dash where the clothes were never in frame, rather than a zero that reads as a verdict."""
    value = contribution.get("wardrobeDelta")
    return f"{value:+.1f}" if isinstance(value, (int, float)) else "-"


def _confirmations(result: dict[str, Any]) -> str:
    rows = result.get("confirmations") or []
    if not rows:
        return ""
    body = "".join(
        f'<tr><td>reference {c["index"]}</td>'
        f'<td class="num">{c["initial"]:+.1f}</td>'
        f'<td class="num">{c["retest"]:+.1f}</td>'
        f'<td class="num">{c["spread"]:.1f}</td>'
        f'<td class="num">{c["pairs"]}</td>'
        f'<td class="num">{c["pooled"]:+.1f}</td>'
        f'<td class="num">{c["pooledPairs"]}</td>'
        f'<td class="{"drop" if c["verdict"] == "removal confirmed" else "neutral"}">'
        f'{html.escape(str(c["verdict"]))}</td></tr>'
        for c in rows
    )
    return (
        "<h2>Re-tested on fresh seeds</h2>"
        '<table><tr><th>Reference</th><th class="num">Sweep</th><th class="num">Re-test</th>'
        '<th class="num">Spread</th><th class="num">Pairs</th><th class="num">Pooled</th>'
        f'<th class="num">Pairs</th><th>Verdict</th></tr>{body}</table>'
        '<p class="cell">Every reference the sweep flagged for removal was rendered again against '
        "the full set on seeds the sweep never used, and scored the same way. Act on the pooled "
        "column: it rests on every pair taken together. A re-test that disagrees with the sweep "
        "means the flag came from too few pairs, not that the reference changed.</p>"
    )


def _per_prompt(result: dict[str, Any], prompts: list[str]) -> str:
    head = "".join(f"<th>{html.escape(p[:38])}</th>" for p in prompts)
    rows = []
    for c in result.get("contributions") or []:
        per = c.get("perPrompt") or {}
        cells = "".join(
            f'<td class="num">{per[p]:+.1f}</td>' if p in per else '<td class="num">-</td>'
            for p in prompts
        )
        rows.append(f'<tr><td>reference {c["index"]}</td>{cells}</tr>')
    return (
        f'<table><tr><th>Reference</th>{head}</tr>{"".join(rows)}</table>'
        '<p class="cell">Read down a column, never across: a close-up and a wide shot sit on '
        "different scales, so a reference that helps one can lose on the other.</p>"
    )


def _combinations(result: dict[str, Any]) -> str:
    rows = "".join(
        f'<tr><td>{html.escape(str(c["key"]))}</td><td class="num">{c["adjusted"]:.1f}</td>'
        f'<td class="num">{c["mean"]:.1f}</td><td class="num">{c["spread"]:.1f}</td>'
        f'<td class="num">{c["tested"]}</td></tr>'
        for c in result.get("combinations") or []
    )
    return (
        '<table><tr><th>Set</th><th class="num">Adjusted</th><th class="num">Mean</th>'
        f'<th class="num">Spread</th><th class="num">Cells</th></tr>{rows}</table>'
        '<p class="cell">Adjusted is the mean minus one spread, so a set that was tested twice and '
        "got lucky does not outrank one measured many times. Never ranked on a best cell.</p>"
    )


def _renders(result: dict[str, Any], prompts: list[str]) -> str:
    out = []
    for prompt in prompts:
        cells = [c for c in (result.get("cells") or []) if c["prompt"] == prompt]
        cells.sort(key=lambda c: (c["combination"] != "full", c["combination"], c["seed"]))
        figures = []
        for c in cells:
            if not c.get("path"):
                continue
            shown = "unmeasured" if c["score"] is None else f"{c['score']:.1f}"
            src = embed_image(str(c["path"]))
            figures.append(
                f'<figure style="margin:0"><img src="{src}" width="200" alt="">'
                f'<figcaption class="cell">{html.escape(str(c["combination"]))} &middot; '
                f'seed {c["seed"]} &middot; {shown}</figcaption></figure>'
            )
        strip = "".join(figures)
        head = f'<h3 class="cell">{html.escape(prompt)}</h3>'
        out.append(f"{head}<div class='strip'>{strip}</div>")
    return "\n".join(out)
