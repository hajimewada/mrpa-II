#!/usr/bin/env python3
"""Generate an HTML index for MRPA-II utterance visualizations."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Entry:
    utterance_id: str
    transcript: str
    html_relative: str
    png_relative: str
    frames: int
    scales: int
    boundary_candidates: int
    nucleus_candidates: int
    boundary_excluded: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a browsable HTML index for MRPA-II aligned results."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
        help="Subset result root, for example .../mrpa-II/basic5000.",
    )
    parser.add_argument(
        "--transcript-file",
        type=Path,
        required=True,
        help="JSUT transcript_utf8.txt for the selected subset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML file. Default: <result-root>/index.html",
    )
    parser.add_argument(
        "--title",
        default="MRPA-II utterance analysis",
        help="Page title.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Include only utterances containing a .complete marker.",
    )
    return parser


def load_transcripts(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Transcript file does not exist: {path}")

    transcripts: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"Invalid transcript line {line_number}: {line}")
        utterance_id, text = line.split(":", 1)
        utterance_id = utterance_id.strip()
        text = text.strip()
        if not utterance_id or not text:
            raise ValueError(f"Invalid transcript line {line_number}: {line}")
        transcripts[utterance_id] = text

    return transcripts


def require_single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one file matching {pattern!r} under "
            f"{directory}; found {len(matches)}"
        )
    return matches[0]


def read_entry(
    utterance_dir: Path,
    result_root: Path,
    transcripts: dict[str, str],
) -> Entry:
    utterance_id = utterance_dir.name
    transcript = transcripts.get(utterance_id, "Transcript unavailable")

    aligned_html = utterance_dir / f"{utterance_id}_aligned.html"
    aligned_png = utterance_dir / f"{utterance_id}_aligned.png"

    if not aligned_html.is_file():
        raise FileNotFoundError(f"Aligned HTML does not exist: {aligned_html}")
    if not aligned_png.is_file():
        raise FileNotFoundError(f"Aligned PNG does not exist: {aligned_png}")

    profile_path = require_single(
        utterance_dir / "scale_profile",
        "*_scale_profile.npz",
    )
    boundary_path = require_single(
        utterance_dir / "boundary",
        "*_boundary_candidates.npz",
    )
    nucleus_path = require_single(
        utterance_dir / "nucleus",
        "*_nucleus_candidates.npz",
    )
    nucleus_metadata_path = require_single(
        utterance_dir / "nucleus",
        "*_nucleus_candidates_metadata.json",
    )

    with np.load(profile_path, allow_pickle=False) as archive:
        magnitude = np.asarray(archive["magnitude"])
        if magnitude.ndim != 2:
            raise ValueError(f"Invalid magnitude shape in {profile_path}")
        scales, frames = magnitude.shape

    with np.load(boundary_path, allow_pickle=False) as archive:
        boundary_candidates = int(
            np.asarray(archive["candidate_frame_indices"]).size
        )

    with np.load(nucleus_path, allow_pickle=False) as archive:
        nucleus_candidates = int(
            np.asarray(archive["candidate_frame_indices"]).size
        )

    metadata = json.loads(
        nucleus_metadata_path.read_text(encoding="utf-8")
    )
    boundary_excluded = int(
        metadata["nucleus_summary"]["boundary_excluded_candidate_count"]
    )

    return Entry(
        utterance_id=utterance_id,
        transcript=transcript,
        html_relative=aligned_html.relative_to(result_root).as_posix(),
        png_relative=aligned_png.relative_to(result_root).as_posix(),
        frames=frames,
        scales=scales,
        boundary_candidates=boundary_candidates,
        nucleus_candidates=nucleus_candidates,
        boundary_excluded=boundary_excluded,
    )


def render_page(title: str, entries: list[Entry], skipped: list[str]) -> str:
    cards: list[str] = []

    for entry in entries:
        safe_id = html.escape(entry.utterance_id)
        safe_text = html.escape(entry.transcript)
        safe_html_path = html.escape(entry.html_relative, quote=True)
        safe_png_path = html.escape(entry.png_relative, quote=True)

        search_text = html.escape(
            f"{entry.utterance_id} {entry.transcript}".casefold(),
            quote=True,
        )

        cards.append(
            f"""
      <article class="card" data-search="{search_text}">
        <a class="preview" href="{safe_html_path}">
          <img src="{safe_png_path}" alt="{safe_id} aligned MRPA-II visualization" loading="lazy">
        </a>
        <div class="content">
          <h2><a href="{safe_html_path}">{safe_id}</a></h2>
          <p class="transcript" lang="ja">{safe_text}</p>
          <dl>
            <div><dt>Frames</dt><dd>{entry.frames}</dd></div>
            <div><dt>Scales</dt><dd>{entry.scales}</dd></div>
            <div><dt>Boundary</dt><dd>{entry.boundary_candidates}</dd></div>
            <div><dt>Nucleus</dt><dd>{entry.nucleus_candidates}</dd></div>
            <div><dt>Excluded</dt><dd>{entry.boundary_excluded}</dd></div>
          </dl>
        </div>
      </article>"""
        )

    skipped_section = ""
    if skipped:
        skipped_items = "".join(
            f"<li>{html.escape(item)}</li>" for item in skipped
        )
        skipped_section = f"""
    <details>
      <summary>Skipped utterances ({len(skipped)})</summary>
      <ul>{skipped_items}</ul>
    </details>"""

    safe_title = html.escape(title)
    cards_html = "\n".join(cards)

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: system-ui, -apple-system, "Noto Sans CJK JP", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: Canvas; color: CanvasText; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 0.8rem 1rem;
      border-bottom: 1px solid color-mix(in srgb, CanvasText 18%, transparent);
      background: color-mix(in srgb, Canvas 94%, transparent);
      backdrop-filter: blur(8px);
    }}
    .header-inner {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ margin: 0 0 0.6rem; font-size: 1.25rem; }}
    .controls {{ display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }}
    input {{
      flex: 1 1 18rem;
      min-width: 0;
      padding: 0.6rem 0.75rem;
      border: 1px solid color-mix(in srgb, CanvasText 30%, transparent);
      border-radius: 0.5rem;
      background: Canvas;
      color: CanvasText;
    }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 1rem; }}
    #grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 1rem; }}
    .card {{
      overflow: hidden;
      border: 1px solid color-mix(in srgb, CanvasText 18%, transparent);
      border-radius: 0.75rem;
      background: color-mix(in srgb, Canvas 96%, CanvasText 4%);
    }}
    .preview {{ display: block; background: white; }}
    .preview img {{ display: block; width: 100%; aspect-ratio: 5 / 4; object-fit: contain; }}
    .content {{ padding: 0.8rem; }}
    h2 {{ margin: 0 0 0.5rem; font-size: 1rem; }}
    a {{ color: LinkText; }}
    .transcript {{ margin: 0 0 0.7rem; line-height: 1.6; min-height: 3.2em; }}
    dl {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.35rem; margin: 0; }}
    dl div {{ text-align: center; padding: 0.35rem 0.15rem; border-radius: 0.35rem; background: color-mix(in srgb, CanvasText 7%, transparent); }}
    dt {{ font-size: 0.65rem; opacity: 0.72; }}
    dd {{ margin: 0.1rem 0 0; font-weight: 650; font-size: 0.85rem; }}
    details {{ margin-top: 1rem; }}
    .hidden {{ display: none; }}
    @media (max-width: 420px) {{
      main {{ padding: 0.65rem; }}
      #grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>{safe_title}</h1>
      <div class="controls">
        <input id="search" type="search" placeholder="発話IDまたは本文を検索" aria-label="発話を検索">
        <span id="count" aria-live="polite">{len(entries)} utterances</span>
      </div>
    </div>
  </header>
  <main>
    <section id="grid">{cards_html}
    </section>
    {skipped_section}
  </main>
  <script>
    const root = document;
    const search = root.querySelector('#search');
    const cards = Array.from(root.querySelectorAll('.card'));
    const count = root.querySelector('#count');

    function update() {{
      const query = search.value.trim().toLocaleLowerCase('ja');
      let visible = 0;
      for (const card of cards) {{
        const matches = !query || card.dataset.search.includes(query);
        card.classList.toggle('hidden', !matches);
        if (matches) visible += 1;
      }}
      count.textContent = `${{visible}} / ${{cards.length}} utterances`;
    }}

    search.addEventListener('input', update);
  </script>
</body>
</html>
"""


def main() -> int:
    args = build_parser().parse_args()
    result_root = args.result_root.expanduser().resolve()
    transcript_file = args.transcript_file.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else result_root / "index.html"
    )

    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {result_root}")

    transcripts = load_transcripts(transcript_file)
    entries: list[Entry] = []
    skipped: list[str] = []

    for utterance_dir in sorted(result_root.iterdir()):
        if not utterance_dir.is_dir():
            continue
        if args.require_complete and not (utterance_dir / ".complete").is_file():
            skipped.append(f"{utterance_dir.name}: missing .complete")
            continue

        try:
            entries.append(read_entry(utterance_dir, result_root, transcripts))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            skipped.append(f"{utterance_dir.name}: {error}")

    if not entries:
        raise ValueError("No aligned utterance visualizations were found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_page(args.title, entries, skipped),
        encoding="utf-8",
    )

    print(f"saved: {output_path}")
    print(f"included: {len(entries)}")
    print(f"skipped: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
