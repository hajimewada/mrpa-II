#!/usr/bin/env python3
"""Extract an F0 contour from a WAV file with PyWORLD.

The script:

1. loads a mono or multichannel WAV file;
2. converts multichannel audio to mono;
3. estimates F0 with DIO or Harvest;
4. refines the estimate with StoneMask;
5. saves the F0 contour as a NumPy array;
6. optionally saves time positions and JSON metadata.

Unvoiced frames are represented by 0.0 Hz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyworld as pw
import soundfile as sf


MIN_WORLD_SAMPLE_RATE = 16_000


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract an F0 contour from a WAV file using "
            "PyWORLD DIO or Harvest followed by StoneMask."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input WAV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path of the output F0 NumPy file (.npy).",
    )
    parser.add_argument(
        "--time-output",
        type=Path,
        default=None,
        help=(
            "Optional path for the frame-time NumPy file. "
            "Default: <output-stem>_time.npy"
        ),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help=(
            "Optional path for JSON metadata. "
            "Default: <output-stem>_metadata.json"
        ),
    )
    parser.add_argument(
        "--method",
        choices=("dio", "harvest"),
        default="dio",
        help="Initial F0 estimation method. Default: dio.",
    )
    parser.add_argument(
        "--frame-period",
        type=float,
        default=5.0,
        help="F0 frame period in milliseconds. Default: 5.0.",
    )
    parser.add_argument(
        "--f0-floor",
        type=float,
        default=50.0,
        help="Minimum F0 in Hz. Default: 50.0.",
    )
    parser.add_argument(
        "--f0-ceil",
        type=float,
        default=800.0,
        help="Maximum F0 in Hz. Default: 800.0.",
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate paths and numerical command-line arguments."""

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    if not args.input.is_file():
        raise ValueError(f"Input path is not a file: {args.input}")

    if args.output.suffix.lower() != ".npy":
        raise ValueError("--output must use the .npy extension")

    if args.time_output is not None and args.time_output.suffix.lower() != ".npy":
        raise ValueError("--time-output must use the .npy extension")

    if (
        args.metadata_output is not None
        and args.metadata_output.suffix.lower() != ".json"
    ):
        raise ValueError("--metadata-output must use the .json extension")

    if args.frame_period <= 0:
        raise ValueError("--frame-period must be greater than zero")

    if args.f0_floor <= 0:
        raise ValueError("--f0-floor must be greater than zero")

    if args.f0_ceil <= args.f0_floor:
        raise ValueError("--f0-ceil must be greater than --f0-floor")


def load_audio(input_path: Path) -> tuple[np.ndarray, int, int]:
    """Load an audio file and return mono float64 samples.

    Returns:
        A tuple containing:
        - mono audio samples;
        - sampling rate;
        - original number of channels.
    """

    audio, sample_rate = sf.read(
        input_path,
        dtype="float64",
        always_2d=True,
    )

    if audio.size == 0:
        raise ValueError(f"Input audio is empty: {input_path}")

    original_channels = int(audio.shape[1])

    if original_channels == 1:
        mono = audio[:, 0]
    else:
        mono = np.mean(audio, axis=1)

    mono = np.ascontiguousarray(mono, dtype=np.float64)

    if not np.all(np.isfinite(mono)):
        raise ValueError("Input audio contains NaN or infinite values")

    peak = float(np.max(np.abs(mono)))

    if peak == 0.0:
        raise ValueError("Input audio is silent")

    if sample_rate < MIN_WORLD_SAMPLE_RATE:
        raise ValueError(
            "PyWORLD requires speech sampled at 16 kHz or higher. "
            f"Input sampling rate: {sample_rate} Hz"
        )

    return mono, int(sample_rate), original_channels


def extract_f0(
    audio: np.ndarray,
    sample_rate: int,
    *,
    method: str,
    frame_period: float,
    f0_floor: float,
    f0_ceil: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and refine an F0 contour."""

    common_arguments: dict[str, Any] = {
        "f0_floor": float(f0_floor),
        "f0_ceil": float(f0_ceil),
        "frame_period": float(frame_period),
    }

    if method == "dio":
        raw_f0, time_axis = pw.dio(
            audio,
            sample_rate,
            **common_arguments,
        )
    elif method == "harvest":
        raw_f0, time_axis = pw.harvest(
            audio,
            sample_rate,
            **common_arguments,
        )
    else:
        raise ValueError(f"Unsupported F0 extraction method: {method}")

    refined_f0 = pw.stonemask(
        audio,
        raw_f0,
        time_axis,
        sample_rate,
    )

    refined_f0 = np.asarray(refined_f0, dtype=np.float64)
    time_axis = np.asarray(time_axis, dtype=np.float64)

    if refined_f0.ndim != 1:
        raise RuntimeError(
            f"Expected a one-dimensional F0 array, got {refined_f0.shape}"
        )

    if time_axis.shape != refined_f0.shape:
        raise RuntimeError(
            "Time-axis and F0 shapes do not match: "
            f"{time_axis.shape} != {refined_f0.shape}"
        )

    if not np.all(np.isfinite(refined_f0)):
        raise RuntimeError("Extracted F0 contains NaN or infinite values")

    if not np.all(np.isfinite(time_axis)):
        raise RuntimeError("Time axis contains NaN or infinite values")

    # WORLD represents unvoiced frames as zero.
    refined_f0[refined_f0 < 0.0] = 0.0

    return refined_f0, time_axis


def default_sibling_path(output_path: Path, suffix: str) -> Path:
    """Construct a related output path beside the primary output."""

    return output_path.with_name(f"{output_path.stem}{suffix}")


def build_metadata(
    *,
    input_path: Path,
    output_path: Path,
    time_output_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    original_channels: int,
    f0: np.ndarray,
    method: str,
    frame_period: float,
    f0_floor: float,
    f0_ceil: float,
) -> dict[str, Any]:
    """Build reproducibility metadata for one extraction run."""

    voiced_mask = f0 > 0.0
    voiced_f0 = f0[voiced_mask]

    metadata: dict[str, Any] = {
        "input_file": str(input_path.resolve()),
        "f0_output_file": str(output_path.resolve()),
        "time_output_file": str(time_output_path.resolve()),
        "method": method,
        "refinement": "stonemask",
        "sample_rate_hz": sample_rate,
        "original_channels": original_channels,
        "samples": int(audio.shape[0]),
        "duration_seconds": float(audio.shape[0] / sample_rate),
        "frame_period_ms": float(frame_period),
        "f0_floor_hz": float(f0_floor),
        "f0_ceil_hz": float(f0_ceil),
        "total_frames": int(f0.shape[0]),
        "voiced_frames": int(np.count_nonzero(voiced_mask)),
        "unvoiced_frames": int(np.count_nonzero(~voiced_mask)),
        "voiced_ratio": float(np.mean(voiced_mask)),
    }

    if voiced_f0.size:
        metadata["voiced_f0_hz"] = {
            "minimum": float(np.min(voiced_f0)),
            "maximum": float(np.max(voiced_f0)),
            "mean": float(np.mean(voiced_f0)),
            "median": float(np.median(voiced_f0)),
        }
    else:
        metadata["voiced_f0_hz"] = None

    return metadata


def save_outputs(
    *,
    f0: np.ndarray,
    time_axis: np.ndarray,
    metadata: dict[str, Any],
    output_path: Path,
    time_output_path: Path,
    metadata_output_path: Path,
) -> None:
    """Save numerical outputs and metadata."""

    for path in (output_path, time_output_path, metadata_output_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, f0)
    np.save(time_output_path, time_axis)

    with metadata_output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    """Run the command-line F0 extraction pipeline."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_arguments(args)

        time_output_path = args.time_output or default_sibling_path(
            args.output,
            "_time.npy",
        )
        metadata_output_path = (
            args.metadata_output
            or default_sibling_path(args.output, "_metadata.json")
        )

        audio, sample_rate, original_channels = load_audio(args.input)

        f0, time_axis = extract_f0(
            audio,
            sample_rate,
            method=args.method,
            frame_period=args.frame_period,
            f0_floor=args.f0_floor,
            f0_ceil=args.f0_ceil,
        )

        metadata = build_metadata(
            input_path=args.input,
            output_path=args.output,
            time_output_path=time_output_path,
            audio=audio,
            sample_rate=sample_rate,
            original_channels=original_channels,
            f0=f0,
            method=args.method,
            frame_period=args.frame_period,
            f0_floor=args.f0_floor,
            f0_ceil=args.f0_ceil,
        )

        save_outputs(
            f0=f0,
            time_axis=time_axis,
            metadata=metadata,
            output_path=args.output,
            time_output_path=time_output_path,
            metadata_output_path=metadata_output_path,
        )

        print(f"input          : {args.input}")
        print(f"sample rate    : {sample_rate} Hz")
        print(f"channels       : {original_channels}")
        print(f"duration       : {metadata['duration_seconds']:.3f} s")
        print(f"method         : {args.method} + StoneMask")
        print(f"frame period   : {args.frame_period:.3f} ms")
        print(f"saved F0       : {args.output}")
        print(f"saved time     : {time_output_path}")
        print(f"saved metadata : {metadata_output_path}")
        print(f"shape          : {f0.shape}")
        print(f"voiced frames  : {metadata['voiced_frames']}")
        print(f"voiced ratio   : {metadata['voiced_ratio']:.4f}")

        return 0

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
