#!/usr/bin/env python3
"""Interpolate an F0 contour while preserving observation masks.

This script converts a discontinuous F0 contour into a continuous latent
F0 contour for subsequent multiscale analysis.

Design principles:

1. Raw F0 observations are never overwritten.
2. Original voicing information is preserved in a separate mask.
3. Interpolated positions are recorded in a separate mask.
4. Interpolation and smoothing are treated as separate operations.
5. Unvoiced frames are represented by 0.0 Hz in the raw input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np


InterpolationDomain = Literal["log-hz", "hz"]
InterpolationMethod = Literal["linear"]
EdgePolicy = Literal["nearest", "error"]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a continuous latent F0 contour from PyWORLD F0 "
            "observations while preserving voicing and interpolation masks."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw F0 NumPy file (.npy).",
    )
    parser.add_argument(
        "--time-input",
        type=Path,
        required=True,
        help="Path to the corresponding frame-time NumPy file (.npy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the interpolated F0 NumPy file (.npy).",
    )
    parser.add_argument(
        "--voicing-output",
        type=Path,
        default=None,
        help=(
            "Optional path for the voicing mask. "
            "Default: <output-stem>_voicing.npy"
        ),
    )
    parser.add_argument(
        "--interpolation-mask-output",
        type=Path,
        default=None,
        help=(
            "Optional path for the interpolation mask. "
            "Default: <output-stem>_interp_mask.npy"
        ),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help=(
            "Optional path for interpolation metadata. "
            "Default: <output-stem>_metadata.json"
        ),
    )
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=None,
        help=(
            "Optional JSON metadata produced by extract_real_f0.py. "
            "Selected source information will be recorded in the output."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("linear",),
        default="linear",
        help="Interpolation method. Default: linear.",
    )
    parser.add_argument(
        "--domain",
        choices=("log-hz", "hz"),
        default="log-hz",
        help="Interpolation domain. Default: log-hz.",
    )
    parser.add_argument(
        "--edge-policy",
        choices=("nearest", "error"),
        default="nearest",
        help=(
            "How to handle missing frames before the first observation "
            "and after the final observation. Default: nearest."
        ),
    )

    return parser


def default_sibling_path(output_path: Path, suffix: str) -> Path:
    """Create a related output path beside the primary output."""

    return output_path.with_name(f"{output_path.stem}{suffix}")


def validate_npy_path(path: Path, argument_name: str) -> None:
    """Ensure that a path uses the .npy extension."""

    if path.suffix.lower() != ".npy":
        raise ValueError(f"{argument_name} must use the .npy extension")


def validate_json_path(path: Path, argument_name: str) -> None:
    """Ensure that a path uses the .json extension."""

    if path.suffix.lower() != ".json":
        raise ValueError(f"{argument_name} must use the .json extension")


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line paths and output extensions."""

    if not args.input.exists():
        raise FileNotFoundError(f"F0 input file does not exist: {args.input}")

    if not args.input.is_file():
        raise ValueError(f"F0 input path is not a file: {args.input}")

    if not args.time_input.exists():
        raise FileNotFoundError(
            f"Time input file does not exist: {args.time_input}"
        )

    if not args.time_input.is_file():
        raise ValueError(
            f"Time input path is not a file: {args.time_input}"
        )

    validate_npy_path(args.input, "--input")
    validate_npy_path(args.time_input, "--time-input")
    validate_npy_path(args.output, "--output")

    if args.voicing_output is not None:
        validate_npy_path(args.voicing_output, "--voicing-output")

    if args.interpolation_mask_output is not None:
        validate_npy_path(
            args.interpolation_mask_output,
            "--interpolation-mask-output",
        )

    if args.metadata_output is not None:
        validate_json_path(args.metadata_output, "--metadata-output")

    if args.source_metadata is not None:
        if not args.source_metadata.exists():
            raise FileNotFoundError(
                "Source metadata file does not exist: "
                f"{args.source_metadata}"
            )

        if not args.source_metadata.is_file():
            raise ValueError(
                "Source metadata path is not a file: "
                f"{args.source_metadata}"
            )

        validate_json_path(args.source_metadata, "--source-metadata")


def load_array(path: Path, name: str) -> np.ndarray:
    """Load a NumPy array without allowing pickled objects."""

    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load {name} from {path}: {error}") from error

    return np.asarray(array)


def load_inputs(
    f0_path: Path,
    time_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate raw F0 and frame-time arrays."""

    f0 = load_array(f0_path, "F0")
    time_axis = load_array(time_path, "time axis")

    if f0.ndim != 1:
        raise ValueError(
            f"F0 input must be one-dimensional, got shape {f0.shape}"
        )

    if time_axis.ndim != 1:
        raise ValueError(
            "Time input must be one-dimensional, "
            f"got shape {time_axis.shape}"
        )

    if f0.shape != time_axis.shape:
        raise ValueError(
            "F0 and time arrays must have identical shapes: "
            f"{f0.shape} != {time_axis.shape}"
        )

    if f0.size == 0:
        raise ValueError("F0 input is empty")

    f0 = np.asarray(f0, dtype=np.float64)
    time_axis = np.asarray(time_axis, dtype=np.float64)

    if not np.all(np.isfinite(f0)):
        raise ValueError("F0 input contains NaN or infinite values")

    if not np.all(np.isfinite(time_axis)):
        raise ValueError("Time input contains NaN or infinite values")

    if np.any(f0 < 0.0):
        raise ValueError("F0 input contains negative values")

    if f0.size > 1 and np.any(np.diff(time_axis) <= 0.0):
        raise ValueError("Time input must be strictly increasing")

    voiced_frames = int(np.count_nonzero(f0 > 0.0))

    if voiced_frames == 0:
        raise ValueError("F0 input contains no voiced frames")

    if voiced_frames == 1:
        raise ValueError(
            "At least two voiced frames are required for interpolation"
        )

    return f0, time_axis


def find_missing_region_counts(
    voicing_mask: np.ndarray,
) -> dict[str, int]:
    """Classify unvoiced frames as leading, internal, or trailing."""

    voiced_indices = np.flatnonzero(voicing_mask)

    if voiced_indices.size == 0:
        raise ValueError("Cannot classify gaps without voiced frames")

    first_voiced = int(voiced_indices[0])
    last_voiced = int(voiced_indices[-1])

    leading_missing = first_voiced
    trailing_missing = int(voicing_mask.size - last_voiced - 1)

    if last_voiced > first_voiced:
        internal_slice = voicing_mask[first_voiced : last_voiced + 1]
        internal_missing = int(np.count_nonzero(~internal_slice))
    else:
        internal_missing = 0

    return {
        "leading_missing_frames": leading_missing,
        "internal_missing_frames": internal_missing,
        "trailing_missing_frames": trailing_missing,
    }


def interpolate_f0(
    f0: np.ndarray,
    time_axis: np.ndarray,
    *,
    method: InterpolationMethod = "linear",
    domain: InterpolationDomain = "log-hz",
    edge_policy: EdgePolicy = "nearest",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a raw F0 contour.

    Args:
        f0:
            One-dimensional raw F0 array in Hz. Unvoiced frames must be
            represented by 0.0.
        time_axis:
            Strictly increasing frame timestamps in seconds.
        method:
            Interpolation method. Only linear interpolation is supported
            in the initial implementation.
        domain:
            Either log-hz or hz.
        edge_policy:
            nearest extends the nearest observed F0 to the signal edges.
            error rejects input containing leading or trailing gaps.

    Returns:
        A tuple of:
        - continuous F0 in Hz;
        - original voicing mask;
        - interpolation mask.
    """

    if method != "linear":
        raise ValueError(f"Unsupported interpolation method: {method}")

    if domain not in ("log-hz", "hz"):
        raise ValueError(f"Unsupported interpolation domain: {domain}")

    if edge_policy not in ("nearest", "error"):
        raise ValueError(f"Unsupported edge policy: {edge_policy}")

    f0 = np.asarray(f0, dtype=np.float64)
    time_axis = np.asarray(time_axis, dtype=np.float64)

    if f0.ndim != 1 or time_axis.ndim != 1:
        raise ValueError("F0 and time arrays must be one-dimensional")

    if f0.shape != time_axis.shape:
        raise ValueError("F0 and time arrays must have identical shapes")

    if f0.size == 0:
        raise ValueError("F0 input is empty")

    if not np.all(np.isfinite(f0)):
        raise ValueError("F0 input contains NaN or infinite values")

    if not np.all(np.isfinite(time_axis)):
        raise ValueError("Time input contains NaN or infinite values")

    if np.any(f0 < 0.0):
        raise ValueError("F0 input contains negative values")

    if f0.size > 1 and np.any(np.diff(time_axis) <= 0.0):
        raise ValueError("Time input must be strictly increasing")

    voicing_mask = f0 > 0.0
    voiced_indices = np.flatnonzero(voicing_mask)

    if voiced_indices.size == 0:
        raise ValueError("F0 input contains no voiced frames")

    if voiced_indices.size == 1:
        raise ValueError(
            "At least two voiced frames are required for interpolation"
        )

    interpolation_mask = ~voicing_mask

    if np.all(voicing_mask):
        return (
            f0.copy(),
            voicing_mask.astype(bool, copy=True),
            np.zeros_like(voicing_mask, dtype=bool),
        )

    first_voiced = int(voiced_indices[0])
    last_voiced = int(voiced_indices[-1])

    has_edge_gaps = first_voiced > 0 or last_voiced < f0.size - 1

    if edge_policy == "error" and has_edge_gaps:
        raise ValueError(
            "Leading or trailing missing frames are not allowed when "
            "--edge-policy error is selected"
        )

    observed_time = time_axis[voicing_mask]
    observed_f0 = f0[voicing_mask]

    if domain == "log-hz":
        observed_values = np.log(observed_f0)
    else:
        observed_values = observed_f0

    if edge_policy == "nearest":
        left_value = float(observed_values[0])
        right_value = float(observed_values[-1])
    else:
        # np.interp requires numerical edge values. Under edge-policy=error,
        # edge gaps have already been rejected, so these values are unused.
        left_value = float(observed_values[0])
        right_value = float(observed_values[-1])

    interpolated_values = np.interp(
        time_axis,
        observed_time,
        observed_values,
        left=left_value,
        right=right_value,
    )

    if domain == "log-hz":
        continuous_f0 = np.exp(interpolated_values)
    else:
        continuous_f0 = interpolated_values

    continuous_f0 = np.asarray(continuous_f0, dtype=np.float64)

    # Preserve all original observations exactly. This prevents numerical
    # round-off introduced by log and exponential conversion from changing
    # measured F0 values.
    continuous_f0[voicing_mask] = f0[voicing_mask]

    if not np.all(np.isfinite(continuous_f0)):
        raise RuntimeError(
            "Interpolated F0 contains NaN or infinite values"
        )

    if np.any(continuous_f0 <= 0.0):
        raise RuntimeError(
            "Interpolated F0 must contain only positive values"
        )

    if not np.array_equal(
        continuous_f0[voicing_mask],
        f0[voicing_mask],
    ):
        raise RuntimeError(
            "Observed F0 values changed during interpolation"
        )

    return (
        continuous_f0,
        voicing_mask.astype(bool, copy=False),
        interpolation_mask.astype(bool, copy=False),
    )


def load_source_metadata(path: Path | None) -> dict[str, Any] | None:
    """Load optional source-extraction metadata."""

    if path is None:
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not load source metadata from {path}: {error}"
        ) from error

    if not isinstance(metadata, dict):
        raise ValueError("Source metadata must contain a JSON object")

    return metadata


def build_metadata(
    *,
    input_f0_path: Path,
    input_time_path: Path,
    output_f0_path: Path,
    voicing_output_path: Path,
    interpolation_mask_output_path: Path,
    f0: np.ndarray,
    time_axis: np.ndarray,
    continuous_f0: np.ndarray,
    voicing_mask: np.ndarray,
    interpolation_mask: np.ndarray,
    method: str,
    domain: str,
    edge_policy: str,
    source_metadata_path: Path | None,
    source_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build interpolation metadata."""

    gap_counts = find_missing_region_counts(voicing_mask)

    observed_f0 = f0[voicing_mask]
    interpolated_f0 = continuous_f0[interpolation_mask]

    if time_axis.size > 1:
        frame_period_seconds = float(np.median(np.diff(time_axis)))
        frame_period_ms = frame_period_seconds * 1000.0
    else:
        frame_period_seconds = None
        frame_period_ms = None

    metadata: dict[str, Any] = {
        "input_f0_file": str(input_f0_path.resolve()),
        "input_time_file": str(input_time_path.resolve()),
        "output_f0_file": str(output_f0_path.resolve()),
        "voicing_output_file": str(voicing_output_path.resolve()),
        "interpolation_mask_output_file": str(
            interpolation_mask_output_path.resolve()
        ),
        "method": method,
        "domain": domain,
        "edge_policy": edge_policy,
        "gap_policy": "all",
        "total_frames": int(f0.size),
        "observed_frames": int(np.count_nonzero(voicing_mask)),
        "interpolated_frames": int(
            np.count_nonzero(interpolation_mask)
        ),
        "observed_ratio": float(np.mean(voicing_mask)),
        "interpolated_ratio": float(np.mean(interpolation_mask)),
        "frame_period_seconds": frame_period_seconds,
        "frame_period_ms": frame_period_ms,
        "observed_values_preserved": bool(
            np.array_equal(
                continuous_f0[voicing_mask],
                f0[voicing_mask],
            )
        ),
        "raw_f0_hz": {
            "minimum_observed": float(np.min(observed_f0)),
            "maximum_observed": float(np.max(observed_f0)),
            "mean_observed": float(np.mean(observed_f0)),
            "median_observed": float(np.median(observed_f0)),
        },
        "continuous_f0_hz": {
            "minimum": float(np.min(continuous_f0)),
            "maximum": float(np.max(continuous_f0)),
            "mean": float(np.mean(continuous_f0)),
            "median": float(np.median(continuous_f0)),
        },
        **gap_counts,
    }

    if interpolated_f0.size:
        metadata["interpolated_f0_hz"] = {
            "minimum": float(np.min(interpolated_f0)),
            "maximum": float(np.max(interpolated_f0)),
            "mean": float(np.mean(interpolated_f0)),
            "median": float(np.median(interpolated_f0)),
        }
    else:
        metadata["interpolated_f0_hz"] = None

    if source_metadata_path is not None:
        metadata["source_metadata_file"] = str(
            source_metadata_path.resolve()
        )

    if source_metadata is not None:
        # Retain the complete source object so each interpolated output can
        # be traced back to the extraction configuration.
        metadata["source_extraction_metadata"] = source_metadata

    return metadata


def save_outputs(
    *,
    continuous_f0: np.ndarray,
    voicing_mask: np.ndarray,
    interpolation_mask: np.ndarray,
    metadata: dict[str, Any],
    output_path: Path,
    voicing_output_path: Path,
    interpolation_mask_output_path: Path,
    metadata_output_path: Path,
) -> None:
    """Save interpolated F0, masks, and metadata."""

    output_paths = (
        output_path,
        voicing_output_path,
        interpolation_mask_output_path,
        metadata_output_path,
    )

    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("All output paths must be different")

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, continuous_f0)
    np.save(voicing_output_path, voicing_mask)
    np.save(interpolation_mask_output_path, interpolation_mask)

    with metadata_output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    """Run the command-line F0 interpolation pipeline."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_arguments(args)

        voicing_output_path = (
            args.voicing_output
            or default_sibling_path(args.output, "_voicing.npy")
        )
        interpolation_mask_output_path = (
            args.interpolation_mask_output
            or default_sibling_path(args.output, "_interp_mask.npy")
        )
        metadata_output_path = (
            args.metadata_output
            or default_sibling_path(args.output, "_metadata.json")
        )

        validate_npy_path(
            voicing_output_path,
            "--voicing-output",
        )
        validate_npy_path(
            interpolation_mask_output_path,
            "--interpolation-mask-output",
        )
        validate_json_path(
            metadata_output_path,
            "--metadata-output",
        )

        f0, time_axis = load_inputs(
            args.input,
            args.time_input,
        )

        continuous_f0, voicing_mask, interpolation_mask = interpolate_f0(
            f0,
            time_axis,
            method=args.method,
            domain=args.domain,
            edge_policy=args.edge_policy,
        )

        source_metadata = load_source_metadata(args.source_metadata)

        metadata = build_metadata(
            input_f0_path=args.input,
            input_time_path=args.time_input,
            output_f0_path=args.output,
            voicing_output_path=voicing_output_path,
            interpolation_mask_output_path=(
                interpolation_mask_output_path
            ),
            f0=f0,
            time_axis=time_axis,
            continuous_f0=continuous_f0,
            voicing_mask=voicing_mask,
            interpolation_mask=interpolation_mask,
            method=args.method,
            domain=args.domain,
            edge_policy=args.edge_policy,
            source_metadata_path=args.source_metadata,
            source_metadata=source_metadata,
        )

        save_outputs(
            continuous_f0=continuous_f0,
            voicing_mask=voicing_mask,
            interpolation_mask=interpolation_mask,
            metadata=metadata,
            output_path=args.output,
            voicing_output_path=voicing_output_path,
            interpolation_mask_output_path=(
                interpolation_mask_output_path
            ),
            metadata_output_path=metadata_output_path,
        )

        print(f"input F0        : {args.input}")
        print(f"input time       : {args.time_input}")
        print(f"method           : {args.method}")
        print(f"domain           : {args.domain}")
        print(f"edge policy      : {args.edge_policy}")
        print(f"saved F0         : {args.output}")
        print(f"saved voicing    : {voicing_output_path}")
        print(f"saved interp mask: {interpolation_mask_output_path}")
        print(f"saved metadata   : {metadata_output_path}")
        print(f"shape             : {continuous_f0.shape}")
        print(f"observed frames   : {metadata['observed_frames']}")
        print(f"interpolated      : {metadata['interpolated_frames']}")
        print(f"leading missing   : {metadata['leading_missing_frames']}")
        print(f"internal missing  : {metadata['internal_missing_frames']}")
        print(f"trailing missing  : {metadata['trailing_missing_frames']}")

        return 0

    except (
        FileNotFoundError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
