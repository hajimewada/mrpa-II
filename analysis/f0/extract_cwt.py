#!/usr/bin/env python3
"""Validate inputs for the MRPA-II Continuous Wavelet Transform pipeline.

Milestone 1 implements:

1. command-line argument parsing;
2. file-path and numerical-parameter validation;
3. continuous-F0 and time-axis loading;
4. uniform sampling-period inference;
5. optional mask loading and validation;
6. interpolation-metadata loading and consistency validation.

CWT signal preparation, physical-scale generation, transform computation,
and output generation are implemented in later milestones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


MINIMUM_FRAMES = 16
RECOMMENDED_MINIMUM_FRAMES = 128
MINIMUM_PRECISION = 12
DEFAULT_NUMBER_OF_SCALES = 64
DEFAULT_MINIMUM_PERIOD_SECONDS = 0.05
DEFAULT_MAX_INTERPOLATION_RATIO = 1.0
DEFAULT_TIME_RTOL = 1e-6
DEFAULT_TIME_ATOL = 1e-9
DEFAULT_MINIMUM_DYNAMIC_RANGE = 1e-12
DEFAULT_MAXIMUM_PERIOD_CAP_SECONDS = 2.0
PERIOD_RANGE_RTOL = 1e-12
PERIOD_RANGE_ATOL = 1e-12

SUPPORTED_CWT_DOMAINS = ("log-hz", "hz")
SUPPORTED_CENTERING_METHODS = ("median", "mean", "none")
SUPPORTED_SCALING_METHODS = ("none",)
SUPPORTED_PERIOD_SPACING = ("log", "linear")


def build_parser() -> argparse.ArgumentParser:
    """Create the CWT command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate continuous-F0 inputs and prepare an MRPA-II "
            "Continuous Wavelet Transform experiment."
        )
    )

    # Required inputs and outputs.
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Interpolated continuous-F0 NumPy file (.npy).",
    )
    parser.add_argument(
        "--time-input",
        type=Path,
        required=True,
        help="Frame-time NumPy file corresponding to the F0 input (.npy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CWT archive (.npz).",
    )

    # Optional provenance inputs.
    parser.add_argument(
        "--voicing-input",
        type=Path,
        default=None,
        help="Optional original voicing mask (.npy).",
    )
    parser.add_argument(
        "--interpolation-mask-input",
        type=Path,
        default=None,
        help="Optional interpolation mask (.npy).",
    )
    parser.add_argument(
        "--interpolation-metadata",
        type=Path,
        default=None,
        help="Optional interpolation metadata JSON file.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help=(
            "Output CWT metadata JSON file. "
            "Default: <output-stem>_metadata.json"
        ),
    )

    # Signal-space options.
    parser.add_argument(
        "--domain",
        choices=("log-hz", "hz"),
        default="log-hz",
        help="CWT input domain. Default: log-hz.",
    )
    parser.add_argument(
        "--centering",
        choices=("median", "mean", "none"),
        default="median",
        help="Centering method. Default: median.",
    )
    parser.add_argument(
        "--scaling",
        choices=("none",),
        default="none",
        help="Signal scaling method. Milestone baseline: none.",
    )

    # CWT and scale parameters.
    parser.add_argument(
        "--wavelet",
        type=str,
        default="morl",
        help="PyWavelets continuous-wavelet name. Default: morl.",
    )
    parser.add_argument(
        "--method",
        choices=("conv", "fft"),
        default="conv",
        help="CWT computation method. Default: conv.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=12,
        help="Wavelet precision. Must be at least 12. Default: 12.",
    )
    parser.add_argument(
        "--number-of-scales",
        type=int,
        default=DEFAULT_NUMBER_OF_SCALES,
        help="Number of CWT scales. Default: 64.",
    )
    parser.add_argument(
        "--minimum-period",
        type=float,
        default=DEFAULT_MINIMUM_PERIOD_SECONDS,
        help="Minimum physical period in seconds. Default: 0.05.",
    )
    parser.add_argument(
        "--maximum-period",
        type=float,
        default=None,
        help=(
            "Maximum physical period in seconds. "
            "Default: min(2.0, input duration / 2)."
        ),
    )
    parser.add_argument(
        "--scale-spacing",
        choices=("log", "linear"),
        default="log",
        help="Physical-period spacing. Default: log.",
    )

    # Quality controls.
    parser.add_argument(
        "--max-interpolation-ratio",
        type=float,
        default=DEFAULT_MAX_INTERPOLATION_RATIO,
        help=(
            "Maximum accepted interpolation ratio in [0, 1]. "
            "Default: 1.0."
        ),
    )
    parser.add_argument(
        "--time-rtol",
        type=float,
        default=DEFAULT_TIME_RTOL,
        help="Relative tolerance for uniform frame spacing.",
    )
    parser.add_argument(
        "--time-atol",
        type=float,
        default=DEFAULT_TIME_ATOL,
        help="Absolute tolerance in seconds for uniform frame spacing.",
    )
    parser.add_argument(
        "--minimum-dynamic-range",
        type=float,
        default=DEFAULT_MINIMUM_DYNAMIC_RANGE,
        help=(
            "Minimum required dynamic range after signal preparation. "
            "Default: 1e-12."
        ),
    )

    return parser


def default_metadata_path(output_path: Path) -> Path:
    """Construct the default metadata path beside the CWT archive."""

    return output_path.with_name(f"{output_path.stem}_metadata.json")


def validate_file_path(
    path: Path,
    *,
    argument_name: str,
    expected_suffix: str,
) -> None:
    """Validate one required input file."""

    if not path.exists():
        raise FileNotFoundError(
            f"{argument_name} file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{argument_name} path is not a file: {path}"
        )

    if path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"{argument_name} must use the {expected_suffix} extension"
        )


def validate_optional_file_path(
    path: Path | None,
    *,
    argument_name: str,
    expected_suffix: str,
) -> None:
    """Validate an optional input file when specified."""

    if path is None:
        return

    validate_file_path(
        path,
        argument_name=argument_name,
        expected_suffix=expected_suffix,
    )


def validate_output_path(
    path: Path,
    *,
    argument_name: str,
    expected_suffix: str,
) -> None:
    """Validate an output filename extension."""

    if path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"{argument_name} must use the {expected_suffix} extension"
        )


def validate_arguments(args: argparse.Namespace) -> Path:
    """Validate CLI paths and numerical parameters.

    Returns:
        The resolved metadata-output path.
    """

    validate_file_path(
        args.input,
        argument_name="--input",
        expected_suffix=".npy",
    )
    validate_file_path(
        args.time_input,
        argument_name="--time-input",
        expected_suffix=".npy",
    )
    validate_optional_file_path(
        args.voicing_input,
        argument_name="--voicing-input",
        expected_suffix=".npy",
    )
    validate_optional_file_path(
        args.interpolation_mask_input,
        argument_name="--interpolation-mask-input",
        expected_suffix=".npy",
    )
    validate_optional_file_path(
        args.interpolation_metadata,
        argument_name="--interpolation-metadata",
        expected_suffix=".json",
    )

    validate_output_path(
        args.output,
        argument_name="--output",
        expected_suffix=".npz",
    )

    metadata_output = (
        args.metadata_output
        if args.metadata_output is not None
        else default_metadata_path(args.output)
    )

    validate_output_path(
        metadata_output,
        argument_name="--metadata-output",
        expected_suffix=".json",
    )

    if args.output.resolve() == metadata_output.resolve():
        raise ValueError(
            "--output and --metadata-output must be different paths"
        )

    if not args.wavelet.strip():
        raise ValueError("--wavelet must not be empty")

    if args.precision < MINIMUM_PRECISION:
        raise ValueError(
            f"--precision must be at least {MINIMUM_PRECISION}"
        )

    if args.number_of_scales < 2:
        raise ValueError("--number-of-scales must be at least 2")

    if not np.isfinite(args.minimum_period):
        raise ValueError("--minimum-period must be finite")

    if args.minimum_period <= 0.0:
        raise ValueError("--minimum-period must be greater than zero")

    if args.maximum_period is not None:
        if not np.isfinite(args.maximum_period):
            raise ValueError("--maximum-period must be finite")

        if args.maximum_period <= args.minimum_period:
            raise ValueError(
                "--maximum-period must be greater than --minimum-period"
            )

    if not np.isfinite(args.max_interpolation_ratio):
        raise ValueError("--max-interpolation-ratio must be finite")

    if not 0.0 <= args.max_interpolation_ratio <= 1.0:
        raise ValueError(
            "--max-interpolation-ratio must be between 0 and 1"
        )

    if not np.isfinite(args.time_rtol) or args.time_rtol < 0.0:
        raise ValueError("--time-rtol must be finite and non-negative")

    if not np.isfinite(args.time_atol) or args.time_atol < 0.0:
        raise ValueError("--time-atol must be finite and non-negative")

    if not np.isfinite(args.minimum_dynamic_range):
        raise ValueError(
            "--minimum-dynamic-range must be finite"
        )

    if args.minimum_dynamic_range < 0.0:
        raise ValueError(
            "--minimum-dynamic-range must be non-negative"
        )

    return metadata_output


def load_numpy_array(path: Path, *, name: str) -> np.ndarray:
    """Load a NumPy array without permitting pickled objects."""

    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not load {name} from {path}: {error}"
        ) from error

    return np.asarray(value)


def validate_core_arrays(
    f0: np.ndarray,
    time_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate in-memory continuous-F0 and time arrays."""

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

    if f0.size < MINIMUM_FRAMES:
        raise ValueError(
            "CWT input must contain at least "
            f"{MINIMUM_FRAMES} frames; got {f0.size}"
        )

    continuous_f0 = np.asarray(f0, dtype=np.float64)
    frame_times = np.asarray(time_axis, dtype=np.float64)

    if not np.all(np.isfinite(continuous_f0)):
        raise ValueError("F0 input contains NaN or infinite values")

    if not np.all(np.isfinite(frame_times)):
        raise ValueError("Time input contains NaN or infinite values")

    if np.any(continuous_f0 <= 0.0):
        raise ValueError(
            "Continuous-F0 input must contain only positive values"
        )

    if np.any(np.diff(frame_times) <= 0.0):
        raise ValueError("Time input must be strictly increasing")

    return continuous_f0, frame_times


def load_core_inputs(
    f0_path: Path,
    time_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate continuous-F0 and frame-time arrays."""

    f0 = load_numpy_array(f0_path, name="continuous F0")
    time_axis = load_numpy_array(time_path, name="time axis")

    return validate_core_arrays(f0, time_axis)


def infer_sampling_period(
    time_axis: np.ndarray,
    *,
    rtol: float = DEFAULT_TIME_RTOL,
    atol: float = DEFAULT_TIME_ATOL,
) -> float:
    """Infer and validate the uniform frame sampling period."""

    time_axis = np.asarray(time_axis, dtype=np.float64)

    if time_axis.ndim != 1:
        raise ValueError("Time input must be one-dimensional")

    if time_axis.size < 2:
        raise ValueError(
            "At least two frame times are required "
            "to infer a sampling period"
        )

    if not np.all(np.isfinite(time_axis)):
        raise ValueError("Time input contains NaN or infinite values")

    differences = np.diff(time_axis)

    if np.any(differences <= 0.0):
        raise ValueError("Time input must be strictly increasing")

    sampling_period = float(np.median(differences))

    if not np.isfinite(sampling_period) or sampling_period <= 0.0:
        raise ValueError("Inferred sampling period is invalid")

    if not np.allclose(
        differences,
        sampling_period,
        rtol=rtol,
        atol=atol,
    ):
        maximum_deviation = float(
            np.max(np.abs(differences - sampling_period))
        )

        raise ValueError(
            "Time input is not uniformly sampled: "
            f"median period={sampling_period:.12g} seconds, "
            f"maximum deviation={maximum_deviation:.12g} seconds"
        )

    return sampling_period


def validate_mask_array(
    mask: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    """Validate an in-memory boolean mask without implicit conversion."""

    if mask.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got shape {mask.shape}"
        )

    if mask.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, "
            f"got {mask.shape}"
        )

    if mask.dtype != np.bool_:
        raise ValueError(
            f"{name} must use boolean dtype, got {mask.dtype}"
        )

    return np.asarray(mask, dtype=bool)


def load_optional_mask(
    path: Path | None,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray | None:
    """Load and validate an optional boolean mask."""

    if path is None:
        return None

    mask = load_numpy_array(path, name=name)

    return validate_mask_array(
        mask,
        expected_shape=expected_shape,
        name=name,
    )


def summarize_masks(
    *,
    voicing_mask: np.ndarray | None,
    interpolation_mask: np.ndarray | None,
) -> dict[str, Any]:
    """Summarize mask availability and mutual consistency."""

    summary: dict[str, Any] = {
        "voicing_mask_available": voicing_mask is not None,
        "interpolation_mask_available": interpolation_mask is not None,
        "mask_overlap_frames": None,
        "masks_are_complementary": None,
    }

    if voicing_mask is not None:
        summary["voiced_frames"] = int(
            np.count_nonzero(voicing_mask)
        )
        summary["voiced_ratio"] = float(np.mean(voicing_mask))
    else:
        summary["voiced_frames"] = None
        summary["voiced_ratio"] = None

    if interpolation_mask is not None:
        summary["interpolated_frames"] = int(
            np.count_nonzero(interpolation_mask)
        )
        summary["interpolation_ratio"] = float(
            np.mean(interpolation_mask)
        )
    else:
        summary["interpolated_frames"] = None
        summary["interpolation_ratio"] = None

    if voicing_mask is not None and interpolation_mask is not None:
        summary["mask_overlap_frames"] = int(
            np.count_nonzero(
                voicing_mask & interpolation_mask
            )
        )
        summary["masks_are_complementary"] = bool(
            np.array_equal(
                interpolation_mask,
                ~voicing_mask,
            )
        )

    return summary


def load_interpolation_metadata(
    path: Path | None,
) -> dict[str, Any] | None:
    """Load optional interpolation metadata."""

    if path is None:
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not load interpolation metadata from {path}: {error}"
        ) from error

    if not isinstance(metadata, dict):
        raise ValueError(
            "Interpolation metadata must contain a JSON object"
        )

    return metadata


def require_metadata_integer(
    metadata: dict[str, Any],
    key: str,
) -> int:
    """Read one required non-negative integer metadata field."""

    if key not in metadata:
        raise ValueError(
            f"Interpolation metadata is missing required field: {key}"
        )

    value = metadata[key]

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Interpolation metadata field {key!r} "
            "must be an integer"
        )

    if value < 0:
        raise ValueError(
            f"Interpolation metadata field {key!r} "
            "must not be negative"
        )

    return value


def validate_interpolation_metadata(
    metadata: dict[str, Any],
    *,
    number_of_frames: int,
    voicing_mask: np.ndarray | None,
    interpolation_mask: np.ndarray | None,
) -> None:
    """Validate metadata against the supplied CWT input arrays."""

    total_frames = require_metadata_integer(
        metadata,
        "total_frames",
    )
    observed_frames = require_metadata_integer(
        metadata,
        "observed_frames",
    )
    interpolated_frames = require_metadata_integer(
        metadata,
        "interpolated_frames",
    )

    if total_frames != number_of_frames:
        raise ValueError(
            "Interpolation metadata total_frames does not match "
            f"the CWT input: {total_frames} != {number_of_frames}"
        )

    if observed_frames + interpolated_frames != total_frames:
        raise ValueError(
            "Interpolation metadata frame counts are inconsistent: "
            "observed_frames + interpolated_frames "
            "must equal total_frames"
        )

    if metadata.get("observed_values_preserved") is not True:
        raise ValueError(
            "Interpolation metadata must confirm "
            "observed_values_preserved=true"
        )

    if voicing_mask is not None:
        actual_observed = int(np.count_nonzero(voicing_mask))

        if observed_frames != actual_observed:
            raise ValueError(
                "Interpolation metadata observed_frames does not "
                "match the voicing mask: "
                f"{observed_frames} != {actual_observed}"
            )

    if interpolation_mask is not None:
        actual_interpolated = int(
            np.count_nonzero(interpolation_mask)
        )

        if interpolated_frames != actual_interpolated:
            raise ValueError(
                "Interpolation metadata interpolated_frames does not "
                "match the interpolation mask: "
                f"{interpolated_frames} != {actual_interpolated}"
            )

    metadata_frame_period = metadata.get("frame_period_seconds")

    if metadata_frame_period is not None:
        if isinstance(metadata_frame_period, bool) or not isinstance(
            metadata_frame_period,
            (int, float),
        ):
            raise ValueError(
                "Interpolation metadata frame_period_seconds "
                "must be numerical or null"
            )

        if (
            not np.isfinite(metadata_frame_period)
            or metadata_frame_period <= 0.0
        ):
            raise ValueError(
                "Interpolation metadata frame_period_seconds "
                "must be finite and positive"
            )


def validate_metadata_sampling_period(
    metadata: dict[str, Any] | None,
    *,
    sampling_period: float,
    rtol: float,
    atol: float,
) -> None:
    """Compare optional metadata frame period with the time axis."""

    if metadata is None:
        return

    metadata_period = metadata.get("frame_period_seconds")

    if metadata_period is None:
        return

    if not np.isclose(
        float(metadata_period),
        sampling_period,
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError(
            "Interpolation metadata frame period does not match "
            "the time axis: "
            f"{metadata_period} != {sampling_period}"
        )


def validate_interpolation_ratio(
    interpolation_mask: np.ndarray | None,
    *,
    maximum_ratio: float,
) -> None:
    """Enforce an optional maximum interpolation ratio."""

    if interpolation_mask is None:
        return

    ratio = float(np.mean(interpolation_mask))

    if ratio > maximum_ratio:
        raise ValueError(
            "Interpolation ratio exceeds the configured maximum: "
            f"{ratio:.6f} > {maximum_ratio:.6f}"
        )


def validate_signal_array(
    signal: np.ndarray,
    *,
    name: str,
    require_positive: bool,
) -> np.ndarray:
    """Validate a one-dimensional numerical signal.

    Args:
        signal:
            Input array to validate.
        name:
            Human-readable name used in error messages.
        require_positive:
            If true, every value must be strictly greater than zero.

    Returns:
        An independent float64 copy of the validated signal.
    """

    array = np.asarray(signal)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got shape {array.shape}"
        )

    if array.size == 0:
        raise ValueError(f"{name} must not be empty")

    try:
        validated = np.array(
            array,
            dtype=np.float64,
            copy=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must contain numerical values"
        ) from error

    if not np.all(np.isfinite(validated)):
        raise ValueError(
            f"{name} contains NaN or infinite values"
        )

    if require_positive and np.any(validated <= 0.0):
        raise ValueError(
            f"{name} must contain only positive values"
        )

    return validated


def convert_f0_domain(
    continuous_f0: np.ndarray,
    *,
    domain: str,
) -> np.ndarray:
    """Convert continuous F0 into the selected CWT analysis domain.

    Supported domains:

    - log-hz: natural logarithm of F0 in Hz;
    - hz: unchanged F0 in Hz.

    The input array is never modified.
    """

    if domain not in SUPPORTED_CWT_DOMAINS:
        raise ValueError(
            f"Unsupported CWT signal domain: {domain}"
        )

    f0 = validate_signal_array(
        continuous_f0,
        name="Continuous-F0 input",
        require_positive=True,
    )

    if domain == "log-hz":
        converted = np.log(f0)
    else:
        converted = f0.copy()

    converted = np.asarray(converted, dtype=np.float64)

    if converted.shape != f0.shape:
        raise RuntimeError(
            "Domain conversion changed the signal shape"
        )

    if not np.all(np.isfinite(converted)):
        raise RuntimeError(
            "Domain-converted CWT signal contains "
            "NaN or infinite values"
        )

    return converted


def center_signal(
    signal: np.ndarray,
    *,
    centering: str,
) -> tuple[np.ndarray, float]:
    """Center a CWT input signal.

    Supported methods:

    - median: subtract the signal median;
    - mean: subtract the signal mean;
    - none: return an unchanged independent copy.
    """

    if centering not in SUPPORTED_CENTERING_METHODS:
        raise ValueError(
            f"Unsupported CWT signal centering: {centering}"
        )

    validated = validate_signal_array(
        signal,
        name="CWT domain signal",
        require_positive=False,
    )

    if centering == "median":
        center_value = float(np.median(validated))
        centered = validated - center_value
    elif centering == "mean":
        center_value = float(np.mean(validated))
        centered = validated - center_value
    else:
        center_value = 0.0
        centered = validated.copy()

    centered = np.asarray(centered, dtype=np.float64)

    if centered.shape != validated.shape:
        raise RuntimeError(
            "Centering changed the signal shape"
        )

    if not np.all(np.isfinite(centered)):
        raise RuntimeError(
            "Centered CWT signal contains NaN or infinite values"
        )

    return centered, center_value


def scale_signal(
    signal: np.ndarray,
    *,
    scaling: str,
) -> tuple[np.ndarray, float]:
    """Scale a prepared CWT signal.

    Milestone 2 supports only ``none`` scaling. The explicit scaling
    interface is retained so that standard or robust scaling can be
    evaluated in later experiments without changing the pipeline
    contract.

    Args:
        signal:
            One-dimensional finite numerical signal.
        scaling:
            Scaling method. Currently only ``none`` is supported.

    Returns:
        A tuple containing:
        - an independent float64 copy of the input signal;
        - the applied scale value, which is 1.0 for ``none``.

    Raises:
        ValueError:
            If the scaling method is unsupported or the signal is invalid.
        RuntimeError:
            If scaling unexpectedly changes the signal shape or produces
            non-finite values.
    """

    if scaling not in SUPPORTED_SCALING_METHODS:
        raise ValueError(
            f"Unsupported CWT signal scaling: {scaling}"
        )

    validated = validate_signal_array(
        signal,
        name="Centered CWT signal",
        require_positive=False,
    )

    if scaling == "none":
        scale_value = 1.0
        scaled = validated.copy()
    else:
        # This branch is unreachable while the supported methods contain
        # only "none", but it protects the function if the constant and
        # implementation become inconsistent.
        raise RuntimeError(
            f"CWT scaling method is declared but not implemented: {scaling}"
        )

    scaled = np.asarray(scaled, dtype=np.float64)

    if scaled.shape != validated.shape:
        raise RuntimeError(
            "Scaling changed the signal shape"
        )

    if not np.all(np.isfinite(scaled)):
        raise RuntimeError(
            "Scaled CWT signal contains NaN or infinite values"
        )

    return scaled, scale_value


def validate_signal_variation(
    signal: np.ndarray,
    *,
    minimum_dynamic_range: float = DEFAULT_MINIMUM_DYNAMIC_RANGE,
) -> float:
    """Validate that a prepared CWT signal has measurable variation.

    Dynamic range is defined as the difference between the maximum and
    minimum signal values:

        dynamic_range = max(signal) - min(signal)

    The signal is accepted only when its dynamic range is strictly
    greater than ``minimum_dynamic_range``.

    Args:
        signal:
            One-dimensional finite numerical signal. Positive and
            negative values are both permitted.
        minimum_dynamic_range:
            Finite, non-negative threshold for the required signal
            variation. Boolean values are not accepted.

    Returns:
        The measured dynamic range as a Python float.

    Raises:
        ValueError:
            If the signal is invalid, the threshold is invalid, or the
            measured dynamic range is not greater than the threshold.
    """

    if isinstance(
        minimum_dynamic_range,
        (bool, np.bool_),
    ):
        raise ValueError(
            "minimum_dynamic_range must be "
            "a finite non-negative number"
        )

    if not isinstance(
        minimum_dynamic_range,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(
            "minimum_dynamic_range must be "
            "a finite non-negative number"
        )

    threshold = float(minimum_dynamic_range)

    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "minimum_dynamic_range must be "
            "a finite non-negative number"
        )

    validated = validate_signal_array(
        signal,
        name="Prepared CWT signal",
        require_positive=False,
    )

    dynamic_range = float(np.ptp(validated))

    if not np.isfinite(dynamic_range):
        raise RuntimeError(
            "Prepared CWT signal produced a non-finite dynamic range"
        )

    if dynamic_range <= threshold:
        raise ValueError(
            "CWT input has no measurable variation after transformation: "
            f"dynamic range={dynamic_range:.12g}, "
            f"required>{threshold:.12g}"
        )

    return dynamic_range


def prepare_cwt_signal(
    continuous_f0: np.ndarray,
    *,
    domain: str = "log-hz",
    centering: str = "median",
    scaling: str = "none",
    minimum_dynamic_range: float = DEFAULT_MINIMUM_DYNAMIC_RANGE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare a validated one-dimensional signal for CWT analysis.

    The transformation order is:

    1. validate and copy continuous F0;
    2. convert the F0 domain;
    3. center the domain signal;
    4. apply the selected scaling policy;
    5. validate the final dynamic range;
    6. return the prepared signal and a reproducibility summary.

    The input F0 array is never modified.

    Args:
        continuous_f0:
            Positive, finite, one-dimensional F0 values in Hz.
        domain:
            Analysis domain. Supported values are ``log-hz`` and ``hz``.
        centering:
            Centering method. Supported values are ``median``, ``mean``,
            and ``none``.
        scaling:
            Scaling method. Milestone 2 supports only ``none``.
        minimum_dynamic_range:
            Final signal dynamic range must be strictly greater than this
            finite, non-negative threshold.

    Returns:
        A tuple containing:
        - the prepared float64 CWT input signal;
        - a JSON-compatible summary dictionary.

    Raises:
        ValueError:
            If an input option or signal value is invalid, or if the final
            signal does not contain sufficient variation.
        RuntimeError:
            If an internal transformation violates the expected shape,
            dtype, finiteness, or input-preservation contract.
    """

    original_f0 = validate_signal_array(
        continuous_f0,
        name="Continuous-F0 input",
        require_positive=True,
    )
    preservation_reference = original_f0.copy()

    domain_signal = convert_f0_domain(
        original_f0,
        domain=domain,
    )

    domain_dynamic_range = float(np.ptp(domain_signal))

    if not np.isfinite(domain_dynamic_range):
        raise RuntimeError(
            "Domain-converted CWT signal produced a non-finite "
            "dynamic range"
        )

    centered_signal, center_value = center_signal(
        domain_signal,
        centering=centering,
    )

    prepared_signal, scale_value = scale_signal(
        centered_signal,
        scaling=scaling,
    )

    prepared_dynamic_range = validate_signal_variation(
        prepared_signal,
        minimum_dynamic_range=minimum_dynamic_range,
    )

    if prepared_signal.shape != original_f0.shape:
        raise RuntimeError(
            "CWT signal preparation changed the signal shape"
        )

    if prepared_signal.dtype != np.float64:
        raise RuntimeError(
            "Prepared CWT signal must use float64 dtype"
        )

    if not np.all(np.isfinite(prepared_signal)):
        raise RuntimeError(
            "Prepared CWT signal contains NaN or infinite values"
        )

    if not np.array_equal(
        original_f0,
        preservation_reference,
    ):
        raise RuntimeError(
            "CWT signal preparation modified the validated F0 input"
        )

    # A centering transformation subtracts a constant, and the current
    # scaling policy is identity scaling. The dynamic range should
    # therefore remain unchanged except for floating-point round-off.
    if not np.isclose(
        domain_dynamic_range,
        prepared_dynamic_range,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "CWT signal preparation unexpectedly changed "
            "the dynamic range"
        )

    summary: dict[str, Any] = {
        "domain": domain,
        "centering": centering,
        "scaling": scaling,
        "center_value": float(center_value),
        "scale_value": float(scale_value),
        "number_of_frames": int(original_f0.size),
        "input_f0_min_hz": float(np.min(original_f0)),
        "input_f0_max_hz": float(np.max(original_f0)),
        "input_f0_mean_hz": float(np.mean(original_f0)),
        "input_f0_median_hz": float(np.median(original_f0)),
        "domain_signal_min": float(np.min(domain_signal)),
        "domain_signal_max": float(np.max(domain_signal)),
        "domain_dynamic_range": domain_dynamic_range,
        "prepared_signal_min": float(np.min(prepared_signal)),
        "prepared_signal_max": float(np.max(prepared_signal)),
        "prepared_signal_mean": float(np.mean(prepared_signal)),
        "prepared_signal_median": float(np.median(prepared_signal)),
        "prepared_dynamic_range": prepared_dynamic_range,
        "minimum_dynamic_range": float(minimum_dynamic_range),
    }

    return prepared_signal, summary


def validate_positive_real(
    value: object,
    *,
    name: str,
) -> float:
    """Validate and convert a strictly positive finite real number.

    Accepted values include Python and NumPy integer and floating-point
    scalar types. Boolean values, strings, null values, non-finite
    values, zero, and negative values are rejected.

    Args:
        value:
            Candidate numerical value.
        name:
            Human-readable parameter name used in error messages.

    Returns:
        The validated value as a Python float.

    Raises:
        ValueError:
            If the value is not a finite, strictly positive real number.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be a finite positive number"
        )

    if not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(
            f"{name} must be a finite positive number"
        )

    converted = float(value)

    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(
            f"{name} must be a finite positive number"
        )

    return converted


def resolve_period_range(
    *,
    minimum_period: float,
    maximum_period: float | None,
    sampling_period: float,
    duration: float,
    maximum_period_cap: float = (
        DEFAULT_MAXIMUM_PERIOD_CAP_SECONDS
    ),
) -> tuple[float, float, dict[str, Any]]:
    """Resolve a physically valid CWT period range.

    The minimum period must span at least four sampling frames. When
    ``maximum_period`` is omitted, the maximum is selected as the
    smaller of the configured cap and half of the input duration.

    An explicitly requested maximum period is never silently clamped.

    Args:
        minimum_period:
            Requested minimum physical period in seconds.
        maximum_period:
            Requested maximum period in seconds, or ``None`` for
            automatic resolution.
        sampling_period:
            Frame sampling period in seconds.
        duration:
            Total input-signal duration in seconds.
        maximum_period_cap:
            Upper cap used only when maximum-period selection is
            automatic.

    Returns:
        A tuple containing:
        - resolved minimum period in seconds;
        - resolved maximum period in seconds;
        - a JSON-compatible summary dictionary.

    Raises:
        ValueError:
            If an input is invalid or no valid period interval can be
            constructed.
    """

    minimum = validate_positive_real(
        minimum_period,
        name="minimum_period",
    )
    sampling = validate_positive_real(
        sampling_period,
        name="sampling_period",
    )
    signal_duration = validate_positive_real(
        duration,
        name="duration",
    )
    period_cap = validate_positive_real(
        maximum_period_cap,
        name="maximum_period_cap",
    )

    sampling_minimum = 4.0 * sampling
    duration_maximum = signal_duration / 2.0

    minimum_matches_limit = np.isclose(
        minimum,
        sampling_minimum,
        rtol=PERIOD_RANGE_RTOL,
        atol=PERIOD_RANGE_ATOL,
    )

    if minimum < sampling_minimum and not minimum_matches_limit:
        raise ValueError(
            "minimum period must span at least four frames: "
            f"requested={minimum:.12g} s, "
            f"required>={sampling_minimum:.12g} s"
        )

    # Normalize a boundary value that differs only by floating-point
    # representation to the exact operational minimum.
    if minimum_matches_limit:
        minimum = sampling_minimum

    if maximum_period is None:
        maximum = min(
            period_cap,
            duration_maximum,
        )
        maximum_was_automatic = True
        requested_maximum: float | None = None
    else:
        maximum = validate_positive_real(
            maximum_period,
            name="maximum_period",
        )
        maximum_was_automatic = False
        requested_maximum = maximum

        maximum_matches_duration_limit = np.isclose(
            maximum,
            duration_maximum,
            rtol=PERIOD_RANGE_RTOL,
            atol=PERIOD_RANGE_ATOL,
        )

        if (
            maximum > duration_maximum
            and not maximum_matches_duration_limit
        ):
            raise ValueError(
                "maximum period exceeds half of the input duration: "
                f"requested={maximum:.12g} s, "
                f"allowed<={duration_maximum:.12g} s"
            )

        if maximum_matches_duration_limit:
            maximum = duration_maximum

    if maximum <= minimum or np.isclose(
        maximum,
        minimum,
        rtol=PERIOD_RANGE_RTOL,
        atol=PERIOD_RANGE_ATOL,
    ):
        raise ValueError(
            "input duration is too short for the requested period range: "
            f"minimum={minimum:.12g} s, "
            f"maximum={maximum:.12g} s"
        )

    minimum_period_frames = minimum / sampling
    maximum_period_frames = maximum / sampling
    minimum_cycles_in_signal = signal_duration / minimum
    maximum_cycles_in_signal = signal_duration / maximum

    summary: dict[str, Any] = {
        "requested_minimum_period_seconds": float(minimum_period),
        "requested_maximum_period_seconds": requested_maximum,
        "sampling_minimum_period_seconds": sampling_minimum,
        "duration_maximum_period_seconds": duration_maximum,
        "maximum_period_cap_seconds": period_cap,
        "resolved_minimum_period_seconds": minimum,
        "resolved_maximum_period_seconds": maximum,
        "maximum_period_was_automatic": maximum_was_automatic,
        "minimum_period_frames": minimum_period_frames,
        "maximum_period_frames": maximum_period_frames,
        "minimum_cycles_in_signal": minimum_cycles_in_signal,
        "maximum_cycles_in_signal": maximum_cycles_in_signal,
    }

    return minimum, maximum, summary


def generate_period_grid(
    *,
    minimum_period: float,
    maximum_period: float,
    number_of_scales: int,
    spacing: str,
) -> np.ndarray:
    """Generate an ordered physical-period grid for CWT analysis.

    The grid includes both the minimum and maximum period.

    Supported spacing policies:

    - ``log``: geometrically spaced periods using ``numpy.geomspace``;
    - ``linear``: arithmetically spaced periods using
      ``numpy.linspace``.

    Args:
        minimum_period:
            Strictly positive lower endpoint in seconds.
        maximum_period:
            Strictly positive upper endpoint in seconds. It must be
            strictly greater than ``minimum_period``.
        number_of_scales:
            Number of periods to generate. It must be an integer of at
            least two. Boolean values are rejected.
        spacing:
            Either ``log`` or ``linear``.

    Returns:
        A one-dimensional float64 array of strictly increasing periods
        in seconds.

    Raises:
        ValueError:
            If an input is invalid or the generated grid violates the
            period-grid contract.
        RuntimeError:
            If NumPy unexpectedly returns an array with an invalid
            shape, dtype, or numerical value.
    """

    minimum = validate_positive_real(
        minimum_period,
        name="minimum_period",
    )
    maximum = validate_positive_real(
        maximum_period,
        name="maximum_period",
    )

    if isinstance(number_of_scales, (bool, np.bool_)):
        raise ValueError(
            "number_of_scales must be an integer of at least 2"
        )

    if not isinstance(
        number_of_scales,
        (int, np.integer),
    ):
        raise ValueError(
            "number_of_scales must be an integer of at least 2"
        )

    scale_count = int(number_of_scales)

    if scale_count < 2:
        raise ValueError(
            "number_of_scales must be an integer of at least 2"
        )

    if not isinstance(spacing, str):
        raise ValueError(
            "spacing must be one of: log, linear"
        )

    if spacing not in SUPPORTED_PERIOD_SPACING:
        raise ValueError(
            f"Unsupported period spacing: {spacing}"
        )

    if maximum <= minimum or np.isclose(
        maximum,
        minimum,
        rtol=PERIOD_RANGE_RTOL,
        atol=PERIOD_RANGE_ATOL,
    ):
        raise ValueError(
            "maximum_period must be strictly greater than "
            "minimum_period"
        )

    if spacing == "log":
        periods = np.geomspace(
            minimum,
            maximum,
            num=scale_count,
            endpoint=True,
            dtype=np.float64,
        )
    elif spacing == "linear":
        periods = np.linspace(
            minimum,
            maximum,
            num=scale_count,
            endpoint=True,
            dtype=np.float64,
        )
    else:
        # Protected by the supported-spacing check above.
        raise RuntimeError(
            f"Period spacing is declared but not implemented: {spacing}"
        )

    periods = np.asarray(periods, dtype=np.float64)

    expected_shape = (scale_count,)

    if periods.shape != expected_shape:
        raise RuntimeError(
            "Generated period grid has an unexpected shape: "
            f"{periods.shape} != {expected_shape}"
        )

    if periods.dtype != np.float64:
        raise RuntimeError(
            "Generated period grid must use float64 dtype"
        )

    if not np.all(np.isfinite(periods)):
        raise RuntimeError(
            "Generated period grid contains NaN or infinite values"
        )

    if np.any(periods <= 0.0):
        raise RuntimeError(
            "Generated period grid must contain only positive values"
        )

    if np.any(np.diff(periods) <= 0.0):
        raise RuntimeError(
            "Generated period grid must be strictly increasing"
        )

    # Set endpoints explicitly to remove any endpoint representation
    # drift and preserve the requested interval exactly.
    periods[0] = minimum
    periods[-1] = maximum

    return periods


def main() -> int:
    """Run Milestone 1 validation and Milestone 2 signal preparation."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        metadata_output = validate_arguments(args)

        continuous_f0, time_axis = load_core_inputs(
            args.input,
            args.time_input,
        )

        sampling_period = infer_sampling_period(
            time_axis,
            rtol=args.time_rtol,
            atol=args.time_atol,
        )
        sampling_frequency = 1.0 / sampling_period

        voicing_mask = load_optional_mask(
            args.voicing_input,
            expected_shape=continuous_f0.shape,
            name="voicing mask",
        )
        interpolation_mask = load_optional_mask(
            args.interpolation_mask_input,
            expected_shape=continuous_f0.shape,
            name="interpolation mask",
        )

        mask_summary = summarize_masks(
            voicing_mask=voicing_mask,
            interpolation_mask=interpolation_mask,
        )

        if mask_summary["mask_overlap_frames"] not in (None, 0):
            raise ValueError(
                "Voicing and interpolation masks overlap at "
                f"{mask_summary['mask_overlap_frames']} frames"
            )

        validate_interpolation_ratio(
            interpolation_mask,
            maximum_ratio=args.max_interpolation_ratio,
        )

        interpolation_metadata = load_interpolation_metadata(
            args.interpolation_metadata
        )

        if interpolation_metadata is not None:
            validate_interpolation_metadata(
                interpolation_metadata,
                number_of_frames=int(continuous_f0.size),
                voicing_mask=voicing_mask,
                interpolation_mask=interpolation_mask,
            )
            validate_metadata_sampling_period(
                interpolation_metadata,
                sampling_period=sampling_period,
                rtol=args.time_rtol,
                atol=args.time_atol,
            )

        prepared_signal, signal_summary = prepare_cwt_signal(
            continuous_f0,
            domain=args.domain,
            centering=args.centering,
            scaling=args.scaling,
            minimum_dynamic_range=args.minimum_dynamic_range,
        )

        duration = float(
            time_axis[-1] - time_axis[0] + sampling_period
        )
        short_input_warning = (
            continuous_f0.size < RECOMMENDED_MINIMUM_FRAMES
        )

        print("Milestone 1 input validation: OK")
        print(f"input F0          : {args.input}")
        print(f"input time         : {args.time_input}")
        print(f"output CWT         : {args.output}")
        print(f"output metadata    : {metadata_output}")
        print(f"frames             : {continuous_f0.size}")
        print(f"duration           : {duration:.6f} s")
        print(f"sampling period    : {sampling_period:.9f} s")
        print(f"sampling frequency : {sampling_frequency:.6f} Hz")
        print(f"voicing mask       : {args.voicing_input}")
        print(f"interpolation mask : {args.interpolation_mask_input}")
        print(
            "interpolation meta : "
            f"{args.interpolation_metadata}"
        )
        print(
            "interpolation ratio: "
            f"{mask_summary['interpolation_ratio']}"
        )
        print(
            "masks complementary: "
            f"{mask_summary['masks_are_complementary']}"
        )
        print(f"short input warning: {short_input_warning}")
        print()
        print("Milestone 2 signal preparation: OK")
        print(
            f"domain              : "
            f"{signal_summary['domain']}"
        )
        print(
            f"centering           : "
            f"{signal_summary['centering']}"
        )
        print(
            f"scaling             : "
            f"{signal_summary['scaling']}"
        )
        print(
            f"center value        : "
            f"{signal_summary['center_value']:.12g}"
        )
        print(
            f"scale value         : "
            f"{signal_summary['scale_value']:.12g}"
        )
        print(
            f"domain range        : "
            f"{signal_summary['domain_dynamic_range']:.12g}"
        )
        print(
            f"prepared range      : "
            f"{signal_summary['prepared_dynamic_range']:.12g}"
        )
        print(
            f"prepared mean       : "
            f"{signal_summary['prepared_signal_mean']:.12g}"
        )
        print(
            f"prepared median     : "
            f"{signal_summary['prepared_signal_median']:.12g}"
        )
        print(
            f"prepared shape      : "
            f"{prepared_signal.shape}"
        )
        print("CWT computation     : not run in Milestone 2")

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
