#!/usr/bin/env python3
"""Visualize one MRPA-II utterance with text-to-phone alignment.

Requirements:
- numpy
- matplotlib
- pyopenjtalk
- a Japanese-capable system font
- JSUT transcript_utf8.txt
- an optional HTK/HTS-style .lab file for forced alignment

When a .lab file is available, OpenJTalk morphemes are aligned to the
forced-alignment phone sequence with global sequence alignment. Japanese
morpheme labels are placed below their corresponding phone intervals.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import numpy as np


SILENCE_PHONES = {"sil", "pau"}
JAPANESE_FONT_HINTS = (
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "Noto Serif CJK JP",
    "Noto Serif JP",
    "IPAexGothic",
    "IPAexMincho",
    "IPAGothic",
    "IPAMincho",
    "Yu Gothic",
    "Yu Mincho",
    "TakaoGothic",
    "TakaoMincho",
)


@dataclass(frozen=True)
class PhoneSegment:
    begin: float
    end: float
    phone: str


@dataclass(frozen=True)
class TextToken:
    surface: str
    phones: tuple[str, ...]


@dataclass(frozen=True)
class AlignedToken:
    surface: str
    begin: float
    end: float
    expected_phone_count: int
    matched_phone_count: int

    @property
    def confidence(self) -> float:
        if self.expected_phone_count == 0:
            return 0.0
        return self.matched_phone_count / self.expected_phone_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize MRPA-II F0, ScaleProfile, Boundary, and Nucleus "
            "outputs with JSUT text and optional phone alignment."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Per-utterance output directory created by process_jsut.sh.",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path.home() / "projects/mrpa/data/jsut/raw/jsut_ver1.1",
        help="JSUT v1.1 corpus root.",
    )
    parser.add_argument(
        "--transcript-file",
        type=Path,
        help="Optional transcript_utf8.txt path.",
    )
    parser.add_argument(
        "--lab-file",
        type=Path,
        help="Optional HTK/HTS-style .lab file.",
    )
    parser.add_argument(
        "--font-file",
        type=Path,
        help="Optional Japanese TrueType/OpenType font file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: <input-dir>/<utterance-id>_aligned.png",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        help="Self-contained HTML path. Default: PNG path with .html suffix.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="PNG resolution. Default: 120.",
    )
    parser.add_argument(
        "--label-stride",
        type=int,
        default=5,
        help="Label every Nth Boundary and Nucleus candidate. Default: 5.",
    )
    parser.add_argument(
        "--minimum-token-confidence",
        type=float,
        default=0.70,
        help="Minimum token phone-alignment confidence in [0, 1]. Default: 0.70.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser


def require_single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one file matching {pattern!r} under "
            f"{directory}; found {len(matches)}"
        )
    return matches[0]


def load_array(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=False))


def validate_frame_vector(
    name: str,
    values: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size != frame_count:
        raise ValueError(
            f"{name} frame count mismatch: expected {frame_count}, "
            f"got {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def candidate_times(
    indices: np.ndarray,
    time: np.ndarray,
    name: str,
) -> np.ndarray:
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(
            f"{name} candidate indices must be a one-dimensional integer array"
        )
    if np.any(indices < 0) or np.any(indices >= time.size):
        raise ValueError(f"{name} candidate indices contain an out-of-range value")
    return time[indices]


def load_transcript(path: Path, utterance_id: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Transcript file does not exist: {path}")

    prefix = f"{utterance_id}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            text = line.split(":", 1)[1].strip()
            if not text:
                raise ValueError(f"Transcript is empty: {utterance_id}")
            return text

    raise KeyError(f"Transcript entry was not found: {utterance_id}")


def central_phone(full_context_label: str) -> str:
    match = re.search(r"-([^+]+)\+", full_context_label)
    if match is None:
        raise ValueError(
            f"Cannot extract central phone from full-context label: "
            f"{full_context_label}"
        )
    return normalize_phone(match.group(1))


def normalize_phone(phone: str) -> str:
    phone = phone.strip()
    aliases = {
        "sp": "pau",
        "silB": "sil",
        "silE": "sil",
    }
    return aliases.get(phone, phone)


def load_phone_alignment(path: Path) -> list[PhoneSegment]:
    if not path.is_file():
        raise FileNotFoundError(f"Alignment file does not exist: {path}")

    segments: list[PhoneSegment] = []
    previous_end = 0.0

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"Invalid lab line {line_number}: {line}")

        begin_raw, end_raw, label = parts
        begin = int(begin_raw) / 10_000_000.0
        end = int(end_raw) / 10_000_000.0

        if begin < 0.0 or end <= begin:
            raise ValueError(f"Invalid lab interval on line {line_number}")
        if begin < previous_end - 1e-7:
            raise ValueError(f"Overlapping lab interval on line {line_number}")

        segments.append(
            PhoneSegment(
                begin=begin,
                end=end,
                phone=central_phone(label),
            )
        )
        previous_end = end

    if not segments:
        raise ValueError(f"Alignment file is empty: {path}")

    return segments


def load_pyopenjtalk():
    try:
        import pyopenjtalk  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "pyopenjtalk is required for text-to-phone alignment. "
            "Install it in the active environment."
        ) from error
    return pyopenjtalk


def token_phones(pyopenjtalk, surface: str) -> tuple[str, ...]:
    result = pyopenjtalk.g2p(surface, kana=False, join=False)
    if isinstance(result, str):
        phones = result.split()
    else:
        phones = list(result)

    normalized = tuple(
        normalize_phone(str(phone))
        for phone in phones
        if normalize_phone(str(phone)) not in SILENCE_PHONES
    )
    return normalized


def text_tokens(text: str) -> list[TextToken]:
    pyopenjtalk = load_pyopenjtalk()
    features = pyopenjtalk.run_frontend(text)

    tokens: list[TextToken] = []
    for feature in features:
        surface = str(feature.get("string", "")).strip()
        if not surface:
            continue

        phones = token_phones(pyopenjtalk, surface)
        if not phones:
            continue

        tokens.append(TextToken(surface=surface, phones=phones))

    if not tokens:
        raise ValueError("OpenJTalk produced no alignable text tokens")

    return tokens


def global_phone_alignment(
    expected: Sequence[str],
    observed: Sequence[str],
) -> list[int | None]:
    """Map each expected phone index to an observed phone index."""

    n = len(expected)
    m = len(observed)

    costs = np.empty((n + 1, m + 1), dtype=np.int32)
    moves = np.empty((n + 1, m + 1), dtype=np.int8)

    costs[:, 0] = np.arange(n + 1, dtype=np.int32)
    costs[0, :] = np.arange(m + 1, dtype=np.int32)
    moves[:, 0] = 1
    moves[0, :] = 2
    moves[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = costs[i - 1, j - 1] + (
                0 if expected[i - 1] == observed[j - 1] else 1
            )
            delete_expected = costs[i - 1, j] + 1
            insert_observed = costs[i, j - 1] + 1

            best = min(diagonal, delete_expected, insert_observed)
            costs[i, j] = best

            if diagonal == best:
                moves[i, j] = 0
            elif delete_expected == best:
                moves[i, j] = 1
            else:
                moves[i, j] = 2

    mapping: list[int | None] = [None] * n
    i = n
    j = m

    while i > 0 or j > 0:
        move = moves[i, j]
        if i > 0 and j > 0 and move == 0:
            if expected[i - 1] == observed[j - 1]:
                mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            i -= 1
        else:
            j -= 1

    return mapping


def align_tokens_to_lab(
    tokens: Sequence[TextToken],
    lab_segments: Sequence[PhoneSegment],
) -> tuple[list[AlignedToken], float]:
    observed_segments = [
        segment
        for segment in lab_segments
        if segment.phone not in SILENCE_PHONES
    ]
    observed_phones = [segment.phone for segment in observed_segments]

    expected_phones: list[str] = []
    expected_token_indices: list[int] = []

    for token_index, token in enumerate(tokens):
        for phone in token.phones:
            expected_phones.append(phone)
            expected_token_indices.append(token_index)

    mapping = global_phone_alignment(expected_phones, observed_phones)

    mapped_by_token: list[list[int]] = [[] for _ in tokens]
    matched_total = 0

    for expected_index, observed_index in enumerate(mapping):
        if observed_index is None:
            continue
        token_index = expected_token_indices[expected_index]
        mapped_by_token[token_index].append(observed_index)
        matched_total += 1

    aligned: list[AlignedToken] = []
    for token, observed_indices in zip(tokens, mapped_by_token, strict=True):
        if not observed_indices:
            continue

        begin = observed_segments[min(observed_indices)].begin
        end = observed_segments[max(observed_indices)].end

        aligned.append(
            AlignedToken(
                surface=token.surface,
                begin=begin,
                end=end,
                expected_phone_count=len(token.phones),
                matched_phone_count=len(observed_indices),
            )
        )

    overall_confidence = (
        matched_total / len(expected_phones)
        if expected_phones
        else 0.0
    )

    return aligned, overall_confidence


def find_japanese_font(explicit_path: Path | None) -> FontProperties:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Japanese font file does not exist: {path}")
        return FontProperties(fname=str(path))

    available: list[tuple[str, Path]] = []
    for raw_path in font_manager.findSystemFonts(
        fontpaths=None,
        fontext="ttf",
    ):
        path = Path(raw_path)
        try:
            family = FontProperties(fname=str(path)).get_name()
        except Exception:
            continue
        available.append((family, path))

    for hint in JAPANESE_FONT_HINTS:
        for family, path in available:
            if hint.casefold() in family.casefold():
                return FontProperties(fname=str(path))

    for family, path in available:
        lowered = f"{family} {path.name}".casefold()
        if any(key in lowered for key in ("cjk", "japan", "ipa", "gothic", "mincho")):
            return FontProperties(fname=str(path))

    raise RuntimeError(
        "No Japanese-capable font was found. Install Noto Sans CJK JP "
        "or pass --font-file with a Japanese TrueType/OpenType font."
    )


def add_candidate_labels(
    axis,
    times: np.ndarray,
    strengths: np.ndarray,
    prefix: str,
    stride: int,
) -> None:
    for position, (time_value, strength_value) in enumerate(
        zip(times, strengths, strict=True),
        start=1,
    ):
        if (position - 1) % stride != 0:
            continue

        offset = 7 if position % 2 else 17
        axis.annotate(
            f"{prefix}{position:03d}",
            xy=(float(time_value), float(strength_value)),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=5,
            annotation_clip=True,
        )


def main() -> int:
    args = build_parser().parse_args()

    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1")
    if args.label_stride < 1:
        raise ValueError("--label-stride must be at least 1")
    if not 0.0 <= args.minimum_token_confidence <= 1.0:
        raise ValueError("--minimum-token-confidence must be in [0, 1]")

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    utterance_id = input_dir.name
    subset = input_dir.parent.name
    corpus_root = args.corpus_root.expanduser().resolve()

    transcript_path = (
        args.transcript_file.expanduser().resolve()
        if args.transcript_file is not None
        else corpus_root / subset / "transcript_utf8.txt"
    )
    transcript = load_transcript(transcript_path, utterance_id)

    lab_path = (
        args.lab_file.expanduser().resolve()
        if args.lab_file is not None
        else corpus_root / subset / "lab" / f"{utterance_id}.lab"
    )
    if not lab_path.is_file():
        raise FileNotFoundError(
            "Exact text-to-phone alignment requires a .lab file. "
            f"Expected: {lab_path}"
        )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_dir / f"{utterance_id}_aligned.png"
    )
    html_output_path = (
        args.html_output.expanduser().resolve()
        if args.html_output is not None
        else output_path.with_suffix(".html")
    )

    f0_dir = input_dir / "f0"
    scale_dir = input_dir / "scale_profile"
    boundary_dir = input_dir / "boundary"
    nucleus_dir = input_dir / "nucleus"

    observed_f0 = load_array(require_single(f0_dir, "*_f0.npy"))
    time = load_array(require_single(f0_dir, "*_f0_time.npy"))
    interpolated_f0 = load_array(require_single(f0_dir, "*_f0_interpolated.npy"))
    voicing = load_array(require_single(f0_dir, "*_voicing.npy"))
    interpolation_mask = load_array(require_single(f0_dir, "*_interpolation_mask.npy"))

    if time.ndim != 1 or time.size < 1 or not np.all(np.isfinite(time)):
        raise ValueError("Time axis must be a non-empty finite one-dimensional array")
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("Time axis must be strictly increasing")

    frame_count = time.size
    observed_f0 = validate_frame_vector("Observed F0", observed_f0, frame_count)
    interpolated_f0 = validate_frame_vector(
        "Interpolated F0",
        interpolated_f0,
        frame_count,
    )
    voicing = validate_frame_vector("Voicing mask", voicing, frame_count).astype(bool)
    interpolation_mask = validate_frame_vector(
        "Interpolation mask",
        interpolation_mask,
        frame_count,
    ).astype(bool)

    profile_path = require_single(scale_dir, "*_scale_profile.npz")
    with np.load(profile_path, allow_pickle=False) as archive:
        magnitude = np.array(archive["magnitude"], dtype=np.float64, copy=True)
        frequencies = np.array(archive["frequencies"], dtype=np.float64, copy=True)

    if magnitude.ndim != 2 or magnitude.shape[1] != frame_count:
        raise ValueError("ScaleProfile magnitude must have shape (scales, frames)")
    if frequencies.ndim != 1 or frequencies.size != magnitude.shape[0]:
        raise ValueError("ScaleProfile frequencies must align with scale dimension")
    if not np.all(np.isfinite(magnitude)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("ScaleProfile arrays must contain only finite values")

    boundary_path = require_single(boundary_dir, "*_boundary_candidates.npz")
    with np.load(boundary_path, allow_pickle=False) as archive:
        boundary_strength = np.array(
            archive["boundary_strength"],
            dtype=np.float64,
            copy=True,
        )
        boundary_indices = np.array(
            archive["candidate_frame_indices"],
            dtype=np.int64,
            copy=True,
        )
        boundary_candidate_strengths = np.array(
            archive["candidate_strengths"],
            dtype=np.float64,
            copy=True,
        )

    nucleus_path = require_single(nucleus_dir, "*_nucleus_candidates.npz")
    with np.load(nucleus_path, allow_pickle=False) as archive:
        nucleus_strength = np.array(
            archive["nucleus_strength"],
            dtype=np.float64,
            copy=True,
        )
        nucleus_indices = np.array(
            archive["candidate_frame_indices"],
            dtype=np.int64,
            copy=True,
        )
        nucleus_candidate_strengths = np.array(
            archive["candidate_strengths"],
            dtype=np.float64,
            copy=True,
        )

    boundary_strength = validate_frame_vector(
        "Boundary strength",
        boundary_strength,
        frame_count,
    )
    nucleus_strength = validate_frame_vector(
        "Nucleus strength",
        nucleus_strength,
        frame_count,
    )

    if boundary_indices.shape != boundary_candidate_strengths.shape:
        raise ValueError("Boundary candidate index and strength shapes must match")
    if nucleus_indices.shape != nucleus_candidate_strengths.shape:
        raise ValueError("Nucleus candidate index and strength shapes must match")

    boundary_time = candidate_times(boundary_indices, time, "Boundary")
    nucleus_time = candidate_times(nucleus_indices, time, "Nucleus")

    nucleus_metadata_path = require_single(
        nucleus_dir,
        "*_nucleus_candidates_metadata.json",
    )
    nucleus_metadata = json.loads(
        nucleus_metadata_path.read_text(encoding="utf-8")
    )
    excluded_count = nucleus_metadata["nucleus_summary"][
        "boundary_excluded_candidate_count"
    ]

    lab_segments = load_phone_alignment(lab_path)
    tokens = text_tokens(transcript)
    aligned_tokens, overall_confidence = align_tokens_to_lab(
        tokens,
        lab_segments,
    )

    if overall_confidence < args.minimum_token_confidence:
        raise ValueError(
            "Text-to-phone alignment confidence is below threshold: "
            f"{overall_confidence:.3f} < {args.minimum_token_confidence:.3f}"
        )

    japanese_font = find_japanese_font(args.font_file)

    figure, axes = plt.subplots(
        6,
        1,
        figsize=(10, 8.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={
            "height_ratios": [0.50, 0.50, 1.25, 1.85, 1.0, 1.0]
        },
    )

    ax_phone, ax_text, ax_f0, ax_profile, ax_boundary, ax_nucleus = axes

    duration_end = max(float(time[-1]), lab_segments[-1].end)
    ax_phone.set_xlim(0.0, duration_end)
    ax_phone.set_ylim(0.0, 1.0)
    ax_phone.set_yticks([])
    ax_phone.set_title(
        f"{utterance_id}: phone and text alignment "
        f"(confidence {overall_confidence:.1%})",
        fontsize=9,
    )

    for index, segment in enumerate(lab_segments):
        if segment.end < 0.0 or segment.begin > duration_end:
            continue
        ax_phone.axvspan(
            segment.begin,
            segment.end,
            alpha=0.11 if index % 2 == 0 else 0.20,
        )
        if segment.phone not in SILENCE_PHONES:
            ax_phone.text(
                (segment.begin + segment.end) / 2.0,
                0.50,
                segment.phone,
                ha="center",
                va="center",
                fontsize=5,
                rotation=90,
                clip_on=True,
            )
    ax_phone.set_ylabel("Phone", fontsize=7)

    ax_text.set_ylim(0.0, 1.0)
    ax_text.set_yticks([])
    ax_text.set_ylabel("Text", fontsize=7)

    for index, token in enumerate(aligned_tokens):
        if token.confidence < args.minimum_token_confidence:
            continue
        ax_text.axvspan(
            token.begin,
            token.end,
            alpha=0.08 if index % 2 == 0 else 0.15,
        )
        ax_text.text(
            (token.begin + token.end) / 2.0,
            0.50,
            token.surface,
            ha="center",
            va="center",
            fontsize=7,
            fontproperties=japanese_font,
            clip_on=True,
        )

    observed_plot = np.where(observed_f0 > 0.0, observed_f0, np.nan)
    ax_f0.plot(time, interpolated_f0, linewidth=1.2, label="Interpolated F0")
    ax_f0.scatter(
        time[voicing],
        observed_plot[voicing],
        s=6,
        label="Observed voiced F0",
    )
    if np.any(interpolation_mask):
        ax_f0.scatter(
            time[interpolation_mask],
            interpolated_f0[interpolation_mask],
            s=6,
            marker="x",
            label="Interpolated frames",
        )
    ax_f0.set_ylabel("F0 (Hz)")
    ax_f0.set_title("F0 contour")
    ax_f0.legend(loc="upper right", ncols=3, fontsize=7)
    ax_f0.grid(True, axis="x", alpha=0.25)

    time_edges = np.empty(frame_count + 1, dtype=np.float64)
    if frame_count == 1:
        time_edges[:] = [time[0] - 0.0025, time[0] + 0.0025]
    else:
        time_edges[1:-1] = (time[:-1] + time[1:]) / 2.0
        time_edges[0] = time[0] - (time[1] - time[0]) / 2.0
        time_edges[-1] = time[-1] + (time[-1] - time[-2]) / 2.0

    frequency_edges = np.empty(frequencies.size + 1, dtype=np.float64)
    if frequencies.size == 1:
        frequency_edges[:] = [frequencies[0] * 0.9, frequencies[0] * 1.1]
    else:
        frequency_edges[1:-1] = np.sqrt(
            frequencies[:-1] * frequencies[1:]
        )
        frequency_edges[0] = frequencies[0] ** 2 / frequency_edges[1]
        frequency_edges[-1] = frequencies[-1] ** 2 / frequency_edges[-2]

    mesh = ax_profile.pcolormesh(
        time_edges,
        frequency_edges,
        magnitude,
        shading="auto",
    )
    ax_profile.set_yscale("log")
    ax_profile.set_ylabel("Frequency (Hz)")
    ax_profile.set_title("ScaleProfile magnitude")
    figure.colorbar(mesh, ax=ax_profile, label="Magnitude")

    for value in boundary_time:
        ax_profile.axvline(value, linewidth=0.5, alpha=0.30)
    for value in nucleus_time:
        ax_profile.axvline(
            value,
            linewidth=0.5,
            alpha=0.30,
            linestyle="--",
        )

    ax_boundary.plot(
        time,
        boundary_strength,
        linewidth=1.0,
        label="Boundary strength",
    )
    if boundary_indices.size:
        ax_boundary.scatter(
            boundary_time,
            boundary_candidate_strengths,
            s=18,
            zorder=3,
            label=f"Boundary candidates ({boundary_indices.size})",
        )
    add_candidate_labels(
        ax_boundary,
        boundary_time,
        boundary_candidate_strengths,
        prefix="B",
        stride=args.label_stride,
    )
    ax_boundary.set_ylabel("Strength")
    ax_boundary.set_title("Boundary candidates")
    ax_boundary.legend(loc="upper right", fontsize=7)
    ax_boundary.grid(True, axis="x", alpha=0.25)

    ax_nucleus.plot(
        time,
        nucleus_strength,
        linewidth=1.0,
        label="Nucleus strength",
    )
    if nucleus_indices.size:
        ax_nucleus.scatter(
            nucleus_time,
            nucleus_candidate_strengths,
            s=18,
            zorder=3,
            label=f"Nucleus candidates ({nucleus_indices.size})",
        )
    add_candidate_labels(
        ax_nucleus,
        nucleus_time,
        nucleus_candidate_strengths,
        prefix="N",
        stride=args.label_stride,
    )
    ax_nucleus.set_ylabel("Strength")
    ax_nucleus.set_xlabel("Time (s)")
    ax_nucleus.set_title(
        f"Nucleus candidates; Boundary-excluded: {excluded_count}"
    )
    ax_nucleus.legend(loc="upper right", fontsize=7)
    ax_nucleus.grid(True, axis="x", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    encoded_png = base64.b64encode(output_path.read_bytes()).decode("ascii")
    safe_utterance_id = html.escape(utterance_id)
    safe_transcript = html.escape(transcript)

    token_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(token.surface)}</td>"
        f"<td>{token.begin:.3f}</td>"
        f"<td>{token.end:.3f}</td>"
        f"<td>{token.confidence:.1%}</td>"
        "</tr>"
        for token in aligned_tokens
    )

    html_document = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_utterance_id} MRPA-II alignment</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; padding: 1rem; background: Canvas; color: CanvasText; }}
    main {{ max-width: 980px; margin: 0 auto; }}
    h1 {{ font-size: 1.15rem; }}
    .transcript {{ font-size: 1.05rem; line-height: 1.7; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; }}
    img {{ display: block; width: min(100%, 900px); height: auto; margin: 1rem auto; background: white; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
    th, td {{ border: 1px solid #8888; padding: 0.25rem 0.4rem; text-align: left; }}
  </style>
</head>
<body>
  <main>
    <h1>{safe_utterance_id}: MRPA-II text and phone alignment</h1>
    <p class="transcript">{safe_transcript}</p>
    <div class="summary">
      <span>Alignment confidence: {overall_confidence:.1%}</span>
      <span>Frames: {frame_count}</span>
      <span>Scales: {frequencies.size}</span>
      <span>Boundary candidates: {boundary_indices.size}</span>
      <span>Nucleus candidates: {nucleus_indices.size}</span>
      <span>Boundary-excluded: {excluded_count}</span>
    </div>
    <img src="data:image/png;base64,{encoded_png}" alt="MRPA-II aligned visualization for {safe_utterance_id}">
    <h2>Aligned text tokens</h2>
    <table>
      <thead><tr><th>Text</th><th>Begin (s)</th><th>End (s)</th><th>Confidence</th></tr></thead>
      <tbody>{token_rows}</tbody>
    </table>
  </main>
</body>
</html>
"""

    html_output_path.write_text(html_document, encoding="utf-8")

    print(f"saved PNG: {output_path}")
    print(f"saved HTML: {html_output_path}")
    print(f"transcript: {transcript}")
    print(f"aligned tokens: {len(aligned_tokens)}")
    print(f"alignment confidence: {overall_confidence:.3f}")
    print(f"frames: {frame_count}")
    print(f"scales: {frequencies.size}")
    print(f"Boundary candidates: {boundary_indices.size}")
    print(f"Nucleus candidates: {nucleus_indices.size}")
    print(f"Boundary-excluded candidates: {excluded_count}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
