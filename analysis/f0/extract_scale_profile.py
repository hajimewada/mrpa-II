#!/usr/bin/env python3
"""Build statistical ScaleProfile features from persisted MRPA-II CWT data.

Milestone 6 implements:

1. persisted CWT loading and validation;
2. CWT magnitude generation;
3. scale-level statistical feature generation;
4. frame-level statistical feature generation;
5. ScaleProfile construction and validation;
6. ScaleProfile metadata generation;
7. output persistence;
8. CLI integration and acceptance testing.

Boundary discovery, Nucleus discovery, ProsodicNode construction, and
language-specific interpretation are implemented in later milestones.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import json
import sys

from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from analysis.f0.extract_cwt import (
    CWTMetadata,
    MINIMUM_FRAMES,
    MINIMUM_PRECISION,
    validate_cwt_metadata,
)

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]

class ScaleStatistics(TypedDict):
    """Statistics computed across time for every CWT scale."""

    scale_mean: Float64Array
    scale_std: Float64Array
    scale_median: Float64Array
    scale_max: Float64Array
    scale_rms: Float64Array


class FrameStatistics(TypedDict):
    """Statistics computed across scales for every time frame."""

    frame_total: Float64Array
    frame_mean: Float64Array
    frame_max: Float64Array
    dominant_scale_index: Int64Array
    dominant_frequency_hz: Float64Array
    frequency_centroid_hz: Float64Array
    frequency_dispersion_hz: Float64Array

class ScaleProfile(TypedDict):
    """Complete numerical ScaleProfile representation."""

    frequencies: Float64Array
    magnitude: Float64Array

    scale_mean: Float64Array
    scale_std: Float64Array
    scale_median: Float64Array
    scale_max: Float64Array
    scale_rms: Float64Array

    frame_total: Float64Array
    frame_mean: Float64Array
    frame_max: Float64Array

    dominant_scale_index: Int64Array
    dominant_frequency_hz: Float64Array
    frequency_centroid_hz: Float64Array
    frequency_dispersion_hz: Float64Array

class SourceCWTMetadata(TypedDict):
    """Reproducibility information inherited from the source CWT."""

    metadata_format_version: int

    wavelet: str
    sampling_period: float
    sampling_frequency: float
    method: str
    precision: int

    coefficient_dtype: str

    number_of_scales: int
    number_of_frames: int

    minimum_frequency_hz: float
    maximum_frequency_hz: float

class FeatureConfigurationMetadata(TypedDict):
    """Fixed conventions used to construct a ScaleProfile."""

    magnitude_transform: str
    standard_deviation_convention: str
    zero_energy_policy: str
    dominant_scale_tie_policy: str
    centroid_coordinate: str
    dispersion_coordinate: str

class ScaleProfileShapeMetadata(TypedDict):
    """Shape information for persisted ScaleProfile arrays."""

    number_of_scales: int
    number_of_frames: int

    frequencies_shape: list[int]
    magnitude_shape: list[int]
    scale_statistic_shape: list[int]
    frame_statistic_shape: list[int]

    number_of_scale_statistics: int
    number_of_frame_statistics: int


class ScaleStatisticsSummaryMetadata(TypedDict):
    """Aggregate summary of scale-level statistical features."""

    minimum_scale_mean: float
    maximum_scale_mean: float

    minimum_scale_rms: float
    maximum_scale_rms: float

class FrameStatisticsSummaryMetadata(TypedDict):
    """Aggregate summary of frame-level statistical features."""

    minimum_frame_total: float
    maximum_frame_total: float

    minimum_centroid_frequency_hz: float
    maximum_centroid_frequency_hz: float

    zero_energy_frame_count: int
    zero_energy_frame_ratio: float

class ScaleProfileMetadata(TypedDict):
    """Top-level persisted ScaleProfile metadata."""

    format_version: int

    source_cwt: SourceCWTMetadata

    feature_configuration: FeatureConfigurationMetadata

    shape_summary: ScaleProfileShapeMetadata

    scale_statistics_summary: ScaleStatisticsSummaryMetadata

    frame_statistics_summary: FrameStatisticsSummaryMetadata

SCALE_PROFILE_FORMAT_VERSION = 1

MAGNITUDE_TRANSFORM = "absolute"
STANDARD_DEVIATION_CONVENTION = "population"
ZERO_ENERGY_POLICY = "finite-zero-statistics"
DOMINANT_SCALE_TIE_POLICY = "first-index"
CENTROID_COORDINATE = "frequency_hz"
DISPERSION_COORDINATE = "frequency_hz"

SCALE_PROFILE_RTOL = 1e-12
SCALE_PROFILE_ATOL = 1e-12

SOURCE_CWT_REQUIRED_KEYS = (
    "coefficients",
    "frequencies",
)


SCALE_PROFILE_REQUIRED_KEYS = (
    "frequencies",
    "magnitude",
    "scale_mean",
    "scale_std",
    "scale_median",
    "scale_max",
    "scale_rms",
    "frame_total",
    "frame_mean",
    "frame_max",
    "dominant_scale_index",
    "dominant_frequency_hz",
    "frequency_centroid_hz",
    "frequency_dispersion_hz",
)

SCALE_PROFILE_METADATA_REQUIRED_KEYS = (
    "format_version",
    "source_cwt",
    "feature_configuration",
    "shape_summary",
    "scale_statistics_summary",
    "frame_statistics_summary",
)

SOURCE_CWT_METADATA_REQUIRED_KEYS = (
    "metadata_format_version",
    "wavelet",
    "sampling_period",
    "sampling_frequency",
    "method",
    "precision",
    "coefficient_dtype",
    "number_of_scales",
    "number_of_frames",
    "minimum_frequency_hz",
    "maximum_frequency_hz",
)

FEATURE_CONFIGURATION_REQUIRED_KEYS = (
    "magnitude_transform",
    "standard_deviation_convention",
    "zero_energy_policy",
    "dominant_scale_tie_policy",
    "centroid_coordinate",
    "dispersion_coordinate",
)

SHAPE_SUMMARY_REQUIRED_KEYS = (
    "number_of_scales",
    "number_of_frames",
    "frequencies_shape",
    "magnitude_shape",
    "scale_statistic_shape",
    "frame_statistic_shape",
    "number_of_scale_statistics",
    "number_of_frame_statistics",
)

SCALE_STATISTICS_SUMMARY_REQUIRED_KEYS = (
    "minimum_scale_mean",
    "maximum_scale_mean",
    "minimum_scale_rms",
    "maximum_scale_rms",
)

FRAME_STATISTICS_SUMMARY_REQUIRED_KEYS = (
    "minimum_frame_total",
    "maximum_frame_total",
    "minimum_centroid_frequency_hz",
    "maximum_centroid_frequency_hz",
    "zero_energy_frame_count",
    "zero_energy_frame_ratio",
)

NUMBER_OF_SCALE_STATISTICS = 5
NUMBER_OF_FRAME_STATISTICS = 7


def load_cwt_outputs(
    input_path: Path,
    metadata_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    """Load persisted CWT arrays and metadata.

    Args:
        input_path:
            Path to the compressed NumPy archive produced by the
            Milestone 5 CWT pipeline.

        metadata_path:
            Path to the JSON metadata file associated with the CWT
            archive.

    Returns:
        A tuple containing:

        - an independent copy of the CWT coefficient matrix;
        - an independent float64 copy of the frequency array;
        - the decoded CWT metadata dictionary.

    Raises:
        FileNotFoundError:
            If either input file does not exist.

        KeyError:
            If the CWT archive does not contain one or more required
            arrays.

        OSError:
            If an input file cannot be read.

        json.JSONDecodeError:
            If the metadata file does not contain valid JSON.

    Notes:
        This function performs file loading and required-key checking
        only. Array shapes, numerical values, frequency ordering, and
        metadata consistency are validated separately by
        ``validate_loaded_cwt``.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"CWT archive does not exist: {input_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"CWT metadata file does not exist: {metadata_path}"
        )

    with np.load(
        input_path,
        allow_pickle=False,
    ) as archive:
        available_keys = set(archive.files)

        missing_keys = tuple(
            key
            for key in SOURCE_CWT_REQUIRED_KEYS
            if key not in available_keys
        )

        if missing_keys:
            missing_names = ", ".join(missing_keys)

            raise KeyError(
                "CWT archive is missing required arrays: "
                f"{missing_names}"
            )

        coefficients = np.array(
            archive["coefficients"],
            copy=True,
        )

        frequencies = np.array(
            archive["frequencies"],
            dtype=np.float64,
            copy=True,
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    return (
        coefficients,
        frequencies,
        metadata,
    )


def validate_loaded_cwt(
    coefficients: np.ndarray,
    frequencies: np.ndarray,
    metadata: dict[str, object],
) -> tuple[
    np.ndarray,
    np.ndarray,
    CWTMetadata,
]:
    """Validate loaded CWT arrays and their persisted metadata.

    Args:
        coefficients:
            CWT coefficient matrix loaded from the persisted NPZ
            archive.

        frequencies:
            Frequency coordinates loaded from the persisted NPZ
            archive.

        metadata:
            Decoded CWT metadata loaded from the associated JSON file.

    Returns:
        A tuple containing:

        - an independent validated copy of the CWT coefficients;
        - an independent float64 copy of the frequencies;
        - an independent validated copy of the CWT metadata.

    Raises:
        TypeError:
            If the metadata root is not a dictionary, or if the
            coefficient array does not use a supported numerical dtype.

        ValueError:
            If the arrays or metadata violate the persisted CWT
            contract.

        KeyError:
            If required metadata fields are missing.
    """

    if not isinstance(metadata, dict):
        raise TypeError(
            "CWT metadata root must be a dictionary"
        )

    validated_coefficients = np.array(
        coefficients,
        copy=True,
    )

    if validated_coefficients.ndim != 2:
        raise ValueError(
            "CWT coefficients must be two-dimensional"
        )

    if (
        np.issubdtype(
            validated_coefficients.dtype,
            np.bool_,
        )
        or not np.issubdtype(
            validated_coefficients.dtype,
            np.number,
        )
    ):
        raise TypeError(
            "CWT coefficients must use a non-boolean "
            "numerical dtype"
        )

    number_of_scales, number_of_frames = (
        validated_coefficients.shape
    )

    if number_of_scales < 2:
        raise ValueError(
            "CWT coefficients must contain at least two scales"
        )

    if number_of_frames < MINIMUM_FRAMES:
        raise ValueError(
            "CWT coefficients must contain at least "
            f"{MINIMUM_FRAMES} frames; "
            f"got {number_of_frames}"
        )

    if not np.all(
        np.isfinite(validated_coefficients)
    ):
        raise ValueError(
            "CWT coefficients must contain only finite values"
        )

    validated_frequencies = np.array(
        frequencies,
        dtype=np.float64,
        copy=True,
    )

    if validated_frequencies.ndim != 1:
        raise ValueError(
            "CWT frequencies must be one-dimensional"
        )

    if validated_frequencies.size != number_of_scales:
        raise ValueError(
            "CWT frequency count does not match "
            "coefficient scale count; "
            f"expected {number_of_scales}, "
            f"got {validated_frequencies.size}"
        )

    if not np.all(
        np.isfinite(validated_frequencies)
    ):
        raise ValueError(
            "CWT frequencies must contain only finite values"
        )

    if np.any(validated_frequencies <= 0.0):
        raise ValueError(
            "CWT frequencies must be strictly positive"
        )

    if np.any(
        np.diff(validated_frequencies) >= 0.0
    ):
        raise ValueError(
            "CWT frequencies must be strictly decreasing"
        )

    validated_metadata = validate_cwt_metadata(
        cast(
            CWTMetadata,
            deepcopy(metadata),
        )
    )

    metadata_scale_count = validated_metadata[
        "cwt_summary"
    ]["n_scales"]

    scale_summary_count = validated_metadata[
        "scale_summary"
    ]["number_of_scales"]

    if (
        metadata_scale_count != number_of_scales
        or scale_summary_count != number_of_scales
    ):
        raise ValueError(
            "CWT metadata scale count does not match "
            "coefficient scale count"
        )

    metadata_frame_count = validated_metadata[
        "cwt_summary"
    ]["n_frames"]

    if metadata_frame_count != number_of_frames:
        raise ValueError(
            "CWT metadata frame count does not match "
            "coefficient frame count"
        )

    actual_minimum_frequency = float(
        np.min(validated_frequencies)
    )
    actual_maximum_frequency = float(
        np.max(validated_frequencies)
    )

    cwt_minimum_frequency = float(
        validated_metadata[
            "cwt_summary"
        ]["frequency_min_hz"]
    )
    cwt_maximum_frequency = float(
        validated_metadata[
            "cwt_summary"
        ]["frequency_max_hz"]
    )

    scale_minimum_frequency = float(
        validated_metadata[
            "scale_summary"
        ]["minimum_frequency_hz"]
    )
    scale_maximum_frequency = float(
        validated_metadata[
            "scale_summary"
        ]["maximum_frequency_hz"]
    )

    if not np.isclose(
        cwt_minimum_frequency,
        actual_minimum_frequency,
        rtol=SCALE_PROFILE_RTOL,
        atol=SCALE_PROFILE_ATOL,
    ):
        raise ValueError(
            "CWT metadata minimum frequency does not match "
            "the loaded frequency array"
        )

    if not np.isclose(
        cwt_maximum_frequency,
        actual_maximum_frequency,
        rtol=SCALE_PROFILE_RTOL,
        atol=SCALE_PROFILE_ATOL,
    ):
        raise ValueError(
            "CWT metadata maximum frequency does not match "
            "the loaded frequency array"
        )

    if not np.isclose(
        scale_minimum_frequency,
        actual_minimum_frequency,
        rtol=SCALE_PROFILE_RTOL,
        atol=SCALE_PROFILE_ATOL,
    ):
        raise ValueError(
            "CWT scale metadata minimum frequency does not "
            "match the loaded frequency array"
        )

    if not np.isclose(
        scale_maximum_frequency,
        actual_maximum_frequency,
        rtol=SCALE_PROFILE_RTOL,
        atol=SCALE_PROFILE_ATOL,
    ):
        raise ValueError(
            "CWT scale metadata maximum frequency does not "
            "match the loaded frequency array"
        )

    return (
        validated_coefficients,
        validated_frequencies,
        validated_metadata,
    )


def compute_cwt_magnitude(
    coefficients: np.ndarray,
) -> Float64Array:
    """Compute a non-negative float64 CWT magnitude representation.

    Args:
        coefficients:
            Validated real-valued or complex-valued CWT coefficient
            matrix with shape ``(n_scales, n_frames)``.

    Returns:
        An independent float64 array containing the absolute CWT
        coefficient values with the same shape as the input.

    Raises:
        ValueError:
            If the computed magnitude has an unexpected shape or
            contains non-finite or negative values.
    """

    magnitude = np.array(
        np.abs(coefficients),
        dtype=np.float64,
        copy=True,
    )

    if magnitude.shape != coefficients.shape:
        raise ValueError(
            "CWT magnitude shape does not match "
            "the coefficient shape"
        )

    if not np.all(np.isfinite(magnitude)):
        raise ValueError(
            "CWT magnitude must contain only finite values"
        )

    if np.any(magnitude < 0.0):
        raise ValueError(
            "CWT magnitude must be non-negative"
        )

    return magnitude


def compute_scale_statistics(
    magnitude: Float64Array,
) -> ScaleStatistics:
    """Compute time-aggregated statistics for every CWT scale.

    Args:
        magnitude:
            Validated non-negative finite CWT magnitude matrix with
            shape ``(n_scales, n_frames)``.

    Returns:
        Independent float64 arrays containing:

        - mean magnitude for each scale;
        - population standard deviation for each scale;
        - median magnitude for each scale;
        - maximum magnitude for each scale;
        - root-mean-square magnitude for each scale.

        Every returned array has shape ``(n_scales,)``.

    Raises:
        ValueError:
            If the magnitude array is not two-dimensional, is empty,
            contains non-finite values, contains negative values, or
            produces non-finite statistics.
    """

    validated_magnitude = np.array(
        magnitude,
        dtype=np.float64,
        copy=True,
    )

    if validated_magnitude.ndim != 2:
        raise ValueError(
            "CWT magnitude must be two-dimensional"
        )

    number_of_scales, number_of_frames = (
        validated_magnitude.shape
    )

    if number_of_scales < 1:
        raise ValueError(
            "CWT magnitude must contain at least one scale"
        )

    if number_of_frames < 1:
        raise ValueError(
            "CWT magnitude must contain at least one frame"
        )

    if not np.all(np.isfinite(validated_magnitude)):
        raise ValueError(
            "CWT magnitude must contain only finite values"
        )

    if np.any(validated_magnitude < 0.0):
        raise ValueError(
            "CWT magnitude must be non-negative"
        )

    scale_mean = np.asarray(
        np.mean(
            validated_magnitude,
            axis=1,
        ),
        dtype=np.float64,
    )

    scale_std = np.asarray(
        np.std(
            validated_magnitude,
            axis=1,
            ddof=0,
        ),
        dtype=np.float64,
    )

    scale_median = np.asarray(
        np.median(
            validated_magnitude,
            axis=1,
        ),
        dtype=np.float64,
    )

    scale_max = np.asarray(
        np.max(
            validated_magnitude,
            axis=1,
        ),
        dtype=np.float64,
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        scale_rms = np.asarray(
            np.sqrt(
                np.mean(
                    np.square(
                        validated_magnitude,
                    ),
                    axis=1,
                )
            ),
            dtype=np.float64,
        )

    expected_shape = (
        number_of_scales,
    )

    statistics = {
        "scale_mean": scale_mean,
        "scale_std": scale_std,
        "scale_median": scale_median,
        "scale_max": scale_max,
        "scale_rms": scale_rms,
    }

    for name, values in statistics.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} has an unexpected shape; "
                f"expected {expected_shape}, "
                f"got {values.shape}"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} must contain only finite values"
            )

        if np.any(values < 0.0):
            raise ValueError(
                f"{name} must be non-negative"
            )

    return statistics


def compute_frame_statistics(
    magnitude: Float64Array,
    frequencies: Float64Array,
) -> FrameStatistics:
    """Compute scale-aggregated statistics for every time frame.

    Args:
        magnitude:
            Validated non-negative finite CWT magnitude matrix with
            shape ``(n_scales, n_frames)``.

        frequencies:
            Validated positive, strictly decreasing float64 frequency
            array with shape ``(n_scales,)``.

    Returns:
        Independent arrays containing:

        - total magnitude for each frame;
        - mean magnitude for each frame;
        - maximum magnitude for each frame;
        - dominant scale index for each frame;
        - dominant frequency for each frame;
        - magnitude-weighted frequency centroid for each frame;
        - magnitude-weighted frequency dispersion for each frame.

        Every returned array has shape ``(n_frames,)``.

        For a zero-energy frame:

        - total, mean, maximum, centroid, and dispersion are zero;
        - dominant scale index is zero;
        - dominant frequency is ``frequencies[0]``.

    Raises:
        ValueError:
            If the magnitude or frequency arrays violate the frame
            statistics contract, or if a computed statistic contains
            an invalid value.
    """

    validated_magnitude = np.array(
        magnitude,
        dtype=np.float64,
        copy=True,
    )

    validated_frequencies = np.array(
        frequencies,
        dtype=np.float64,
        copy=True,
    )

    if validated_magnitude.ndim != 2:
        raise ValueError(
            "CWT magnitude must be two-dimensional"
        )

    number_of_scales, number_of_frames = (
        validated_magnitude.shape
    )

    if number_of_scales < 1:
        raise ValueError(
            "CWT magnitude must contain at least one scale"
        )

    if number_of_frames < 1:
        raise ValueError(
            "CWT magnitude must contain at least one frame"
        )

    if not np.all(np.isfinite(validated_magnitude)):
        raise ValueError(
            "CWT magnitude must contain only finite values"
        )

    if np.any(validated_magnitude < 0.0):
        raise ValueError(
            "CWT magnitude must be non-negative"
        )

    if validated_frequencies.ndim != 1:
        raise ValueError(
            "CWT frequencies must be one-dimensional"
        )

    if validated_frequencies.size != number_of_scales:
        raise ValueError(
            "CWT frequency count does not match "
            "the magnitude scale count; "
            f"expected {number_of_scales}, "
            f"got {validated_frequencies.size}"
        )

    if not np.all(np.isfinite(validated_frequencies)):
        raise ValueError(
            "CWT frequencies must contain only finite values"
        )

    if np.any(validated_frequencies <= 0.0):
        raise ValueError(
            "CWT frequencies must be strictly positive"
        )

    if np.any(
        np.diff(validated_frequencies) >= 0.0
    ):
        raise ValueError(
            "CWT frequencies must be strictly decreasing"
        )

    with np.errstate(
        over="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        frame_total = np.asarray(
            np.sum(
                validated_magnitude,
                axis=0,
            ),
            dtype=np.float64,
        )

        frame_mean = np.asarray(
            np.mean(
                validated_magnitude,
                axis=0,
            ),
            dtype=np.float64,
        )

        frame_max = np.asarray(
            np.max(
                validated_magnitude,
                axis=0,
            ),
            dtype=np.float64,
        )

        dominant_scale_index = np.asarray(
            np.argmax(
                validated_magnitude,
                axis=0,
            ),
            dtype=np.int64,
        )

        dominant_frequency_hz = np.asarray(
            validated_frequencies[
                dominant_scale_index
            ],
            dtype=np.float64,
        )

        zero_energy_mask = frame_total == 0.0
        nonzero_energy_mask = ~zero_energy_mask

        weighted_frequency_sum = np.sum(
            validated_magnitude
            * validated_frequencies[:, np.newaxis],
            axis=0,
        )

        frequency_centroid_hz = np.zeros(
            number_of_frames,
            dtype=np.float64,
        )

        np.divide(
            weighted_frequency_sum,
            frame_total,
            out=frequency_centroid_hz,
            where=nonzero_energy_mask,
        )

        frequency_offsets = (
            validated_frequencies[:, np.newaxis]
            - frequency_centroid_hz[np.newaxis, :]
        )

        weighted_squared_offsets = np.sum(
            validated_magnitude
            * np.square(frequency_offsets),
            axis=0,
        )

        frequency_variance = np.zeros(
            number_of_frames,
            dtype=np.float64,
        )

        np.divide(
            weighted_squared_offsets,
            frame_total,
            out=frequency_variance,
            where=nonzero_energy_mask,
        )

        frequency_variance = np.maximum(
            frequency_variance,
            0.0,
        )

        frequency_dispersion_hz = np.sqrt(
            frequency_variance
        )

    expected_shape = (
        number_of_frames,
    )

    float_statistics = {
        "frame_total": frame_total,
        "frame_mean": frame_mean,
        "frame_max": frame_max,
        "dominant_frequency_hz": (
            dominant_frequency_hz
        ),
        "frequency_centroid_hz": (
            frequency_centroid_hz
        ),
        "frequency_dispersion_hz": (
            frequency_dispersion_hz
        ),
    }

    for name, values in float_statistics.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} has an unexpected shape; "
                f"expected {expected_shape}, "
                f"got {values.shape}"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} must contain only finite values"
            )

        if np.any(values < 0.0):
            raise ValueError(
                f"{name} must be non-negative"
            )

    if dominant_scale_index.shape != expected_shape:
        raise ValueError(
            "dominant_scale_index has an unexpected shape; "
            f"expected {expected_shape}, "
            f"got {dominant_scale_index.shape}"
        )

    if (
        np.any(dominant_scale_index < 0)
        or np.any(
            dominant_scale_index
            >= number_of_scales
        )
    ):
        raise ValueError(
            "dominant_scale_index contains an "
            "out-of-range value"
        )

    return {
        "frame_total": frame_total,
        "frame_mean": frame_mean,
        "frame_max": frame_max,
        "dominant_scale_index": dominant_scale_index,
        "dominant_frequency_hz": dominant_frequency_hz,
        "frequency_centroid_hz": frequency_centroid_hz,
        "frequency_dispersion_hz": (
            frequency_dispersion_hz
        ),
    }


def build_scale_profile(
    *,
    frequencies: Float64Array,
    magnitude: Float64Array,
    scale_statistics: ScaleStatistics,
    frame_statistics: FrameStatistics,
) -> ScaleProfile:
    """Build a unified ScaleProfile from computed CWT features.

    Args:
        frequencies:
            Positive physical frequency coordinates with shape
            ``(n_scales,)``.

        magnitude:
            Non-negative CWT magnitude matrix with shape
            ``(n_scales, n_frames)``.

        scale_statistics:
            Statistics computed across time for each scale.

        frame_statistics:
            Statistics computed across scales for each frame.

    Returns:
        A ScaleProfile containing independent copies of all required
        numerical arrays.

    Raises:
        KeyError:
            If a required scale-level or frame-level statistic is
            missing.

        ValueError:
            If an input array has an unexpected rank or shape.
    """

    validated_frequencies = np.array(
        frequencies,
        dtype=np.float64,
        copy=True,
    )

    validated_magnitude = np.array(
        magnitude,
        dtype=np.float64,
        copy=True,
    )

    if validated_frequencies.ndim != 1:
        raise ValueError(
            "ScaleProfile frequencies must be one-dimensional"
        )

    if validated_magnitude.ndim != 2:
        raise ValueError(
            "ScaleProfile magnitude must be two-dimensional"
        )

    number_of_scales, number_of_frames = (
        validated_magnitude.shape
    )

    if validated_frequencies.shape != (
        number_of_scales,
    ):
        raise ValueError(
            "ScaleProfile frequency shape does not match "
            "the magnitude scale dimension; "
            f"expected {(number_of_scales,)}, "
            f"got {validated_frequencies.shape}"
        )

    required_scale_keys = (
        "scale_mean",
        "scale_std",
        "scale_median",
        "scale_max",
        "scale_rms",
    )

    required_frame_keys = (
        "frame_total",
        "frame_mean",
        "frame_max",
        "dominant_scale_index",
        "dominant_frequency_hz",
        "frequency_centroid_hz",
        "frequency_dispersion_hz",
    )

    for key in required_scale_keys:
        if key not in scale_statistics:
            raise KeyError(
                "Scale statistics are missing required array: "
                f"{key}"
            )

    for key in required_frame_keys:
        if key not in frame_statistics:
            raise KeyError(
                "Frame statistics are missing required array: "
                f"{key}"
            )

    validated_scale_statistics: dict[
        str,
        Float64Array,
    ] = {}

    for key in required_scale_keys:
        values = np.array(
            scale_statistics[key],
            dtype=np.float64,
            copy=True,
        )

        expected_shape = (
            number_of_scales,
        )

        if values.shape != expected_shape:
            raise ValueError(
                f"{key} has an unexpected shape; "
                f"expected {expected_shape}, "
                f"got {values.shape}"
            )

        validated_scale_statistics[key] = values

    validated_frame_statistics: dict[
        str,
        np.ndarray,
    ] = {}

    for key in required_frame_keys:
        dtype = (
            np.int64
            if key == "dominant_scale_index"
            else np.float64
        )

        values = np.array(
            frame_statistics[key],
            dtype=dtype,
            copy=True,
        )

        expected_shape = (
            number_of_frames,
        )

        if values.shape != expected_shape:
            raise ValueError(
                f"{key} has an unexpected shape; "
                f"expected {expected_shape}, "
                f"got {values.shape}"
            )

        validated_frame_statistics[key] = values

    scale_profile: ScaleProfile = {
        "frequencies": validated_frequencies,
        "magnitude": validated_magnitude,
        "scale_mean": validated_scale_statistics[
            "scale_mean"
        ],
        "scale_std": validated_scale_statistics[
            "scale_std"
        ],
        "scale_median": validated_scale_statistics[
            "scale_median"
        ],
        "scale_max": validated_scale_statistics[
            "scale_max"
        ],
        "scale_rms": validated_scale_statistics[
            "scale_rms"
        ],
        "frame_total": validated_frame_statistics[
            "frame_total"
        ],
        "frame_mean": validated_frame_statistics[
            "frame_mean"
        ],
        "frame_max": validated_frame_statistics[
            "frame_max"
        ],
        "dominant_scale_index": (
            validated_frame_statistics[
                "dominant_scale_index"
            ]
        ),
        "dominant_frequency_hz": (
            validated_frame_statistics[
                "dominant_frequency_hz"
            ]
        ),
        "frequency_centroid_hz": (
            validated_frame_statistics[
                "frequency_centroid_hz"
            ]
        ),
        "frequency_dispersion_hz": (
            validated_frame_statistics[
                "frequency_dispersion_hz"
            ]
        ),
    }

    if set(scale_profile) != set(
        SCALE_PROFILE_REQUIRED_KEYS
    ):
        raise ValueError(
            "ScaleProfile keys do not match "
            "the required archive schema"
        )

    return scale_profile


def validate_scale_profile_arrays(
    scale_profile: ScaleProfile,
) -> ScaleProfile:
    """Validate all numerical arrays in a ScaleProfile.

    Args:
        scale_profile:
            ScaleProfile containing frequency coordinates, CWT
            magnitude, scale-level statistics, and frame-level
            statistics.

    Returns:
        A ScaleProfile containing independent validated copies of all
        required arrays.

    Raises:
        KeyError:
            If a required ScaleProfile array is missing.

        TypeError:
            If ``dominant_scale_index`` does not use an integer dtype.

        ValueError:
            If an array violates the ScaleProfile shape or numerical
            contract.
    """

    missing_keys = tuple(
        key
        for key in SCALE_PROFILE_REQUIRED_KEYS
        if key not in scale_profile
    )

    if missing_keys:
        raise KeyError(
            "ScaleProfile is missing required arrays: "
            f"{', '.join(missing_keys)}"
        )

    unexpected_keys = tuple(
        key
        for key in scale_profile
        if key not in SCALE_PROFILE_REQUIRED_KEYS
    )

    if unexpected_keys:
        raise ValueError(
            "ScaleProfile contains unexpected arrays: "
            f"{', '.join(unexpected_keys)}"
        )

    frequencies = np.array(
        scale_profile["frequencies"],
        dtype=np.float64,
        copy=True,
    )

    magnitude = np.array(
        scale_profile["magnitude"],
        dtype=np.float64,
        copy=True,
    )

    if frequencies.ndim != 1:
        raise ValueError(
            "ScaleProfile frequencies must be one-dimensional"
        )

    if magnitude.ndim != 2:
        raise ValueError(
            "ScaleProfile magnitude must be two-dimensional"
        )

    number_of_scales, number_of_frames = magnitude.shape

    if number_of_scales < 1:
        raise ValueError(
            "ScaleProfile must contain at least one scale"
        )

    if number_of_frames < 1:
        raise ValueError(
            "ScaleProfile must contain at least one frame"
        )

    if frequencies.shape != (number_of_scales,):
        raise ValueError(
            "ScaleProfile frequency shape does not match "
            "the magnitude scale dimension; "
            f"expected {(number_of_scales,)}, "
            f"got {frequencies.shape}"
        )

    if not np.all(np.isfinite(frequencies)):
        raise ValueError(
            "ScaleProfile frequencies must contain only finite values"
        )

    if np.any(frequencies <= 0.0):
        raise ValueError(
            "ScaleProfile frequencies must be strictly positive"
        )

    if np.any(np.diff(frequencies) >= 0.0):
        raise ValueError(
            "ScaleProfile frequencies must be strictly decreasing"
        )

    if not np.all(np.isfinite(magnitude)):
        raise ValueError(
            "ScaleProfile magnitude must contain only finite values"
        )

    if np.any(magnitude < 0.0):
        raise ValueError(
            "ScaleProfile magnitude must be non-negative"
        )

    scale_keys = (
        "scale_mean",
        "scale_std",
        "scale_median",
        "scale_max",
        "scale_rms",
    )

    frame_float_keys = (
        "frame_total",
        "frame_mean",
        "frame_max",
        "dominant_frequency_hz",
        "frequency_centroid_hz",
        "frequency_dispersion_hz",
    )

    validated_scale_statistics: dict[
        str,
        Float64Array,
    ] = {}

    for key in scale_keys:
        values = np.array(
            scale_profile[key],
            dtype=np.float64,
            copy=True,
        )

        expected_shape = (number_of_scales,)

        if values.shape != expected_shape:
            raise ValueError(
                f"{key} has an unexpected shape; "
                f"expected {expected_shape}, "
                f"got {values.shape}"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{key} must contain only finite values"
            )

        if np.any(values < 0.0):
            raise ValueError(
                f"{key} must be non-negative"
            )

        validated_scale_statistics[key] = values

    validated_frame_statistics: dict[
        str,
        Float64Array,
    ] = {}

    for key in frame_float_keys:
        values = np.array(
            scale_profile[key],
            dtype=np.float64,
            copy=True,
        )

        expected_shape = (number_of_frames,)

        if values.shape != expected_shape:
            raise ValueError(
                f"{key} has an unexpected shape; "
                f"expected {expected_shape}, "
                f"got {values.shape}"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{key} must contain only finite values"
            )

        if np.any(values < 0.0):
            raise ValueError(
                f"{key} must be non-negative"
            )

        validated_frame_statistics[key] = values

    source_dominant_index = np.asarray(
        scale_profile["dominant_scale_index"]
    )

    if (
        np.issubdtype(
            source_dominant_index.dtype,
            np.bool_,
        )
        or not np.issubdtype(
            source_dominant_index.dtype,
            np.integer,
        )
    ):
        raise TypeError(
            "dominant_scale_index must use a "
            "non-boolean integer dtype"
        )

    dominant_scale_index = np.array(
        source_dominant_index,
        dtype=np.int64,
        copy=True,
    )

    expected_frame_shape = (number_of_frames,)

    if dominant_scale_index.shape != expected_frame_shape:
        raise ValueError(
            "dominant_scale_index has an unexpected shape; "
            f"expected {expected_frame_shape}, "
            f"got {dominant_scale_index.shape}"
        )

    if (
        np.any(dominant_scale_index < 0)
        or np.any(
            dominant_scale_index >= number_of_scales
        )
    ):
        raise ValueError(
            "dominant_scale_index contains an "
            "out-of-range value"
        )

    dominant_frequency_hz = (
        validated_frame_statistics[
            "dominant_frequency_hz"
        ]
    )

    expected_dominant_frequency = frequencies[
        dominant_scale_index
    ]

    if not np.allclose(
        dominant_frequency_hz,
        expected_dominant_frequency,
        rtol=SCALE_PROFILE_RTOL,
        atol=SCALE_PROFILE_ATOL,
    ):
        raise ValueError(
            "dominant_frequency_hz is inconsistent with "
            "dominant_scale_index"
        )

    frame_total = validated_frame_statistics[
        "frame_total"
    ]

    zero_energy_mask = frame_total == 0.0
    nonzero_energy_mask = ~zero_energy_mask

    frequency_centroid_hz = (
        validated_frame_statistics[
            "frequency_centroid_hz"
        ]
    )

    frequency_dispersion_hz = (
        validated_frame_statistics[
            "frequency_dispersion_hz"
        ]
    )

    if np.any(
        frequency_centroid_hz[zero_energy_mask]
        != 0.0
    ):
        raise ValueError(
            "Zero-energy frames must have zero "
            "frequency centroid"
        )

    if np.any(
        frequency_dispersion_hz[zero_energy_mask]
        != 0.0
    ):
        raise ValueError(
            "Zero-energy frames must have zero "
            "frequency dispersion"
        )

    minimum_frequency = float(np.min(frequencies))
    maximum_frequency = float(np.max(frequencies))

    if np.any(
        frequency_centroid_hz[nonzero_energy_mask]
        < minimum_frequency - SCALE_PROFILE_ATOL
    ):
        raise ValueError(
            "frequency_centroid_hz is below the "
            "available frequency range"
        )

    if np.any(
        frequency_centroid_hz[nonzero_energy_mask]
        > maximum_frequency + SCALE_PROFILE_ATOL
    ):
        raise ValueError(
            "frequency_centroid_hz is above the "
            "available frequency range"
        )

    validated_profile: ScaleProfile = {
        "frequencies": frequencies,
        "magnitude": magnitude,
        "scale_mean": validated_scale_statistics[
            "scale_mean"
        ],
        "scale_std": validated_scale_statistics[
            "scale_std"
        ],
        "scale_median": validated_scale_statistics[
            "scale_median"
        ],
        "scale_max": validated_scale_statistics[
            "scale_max"
        ],
        "scale_rms": validated_scale_statistics[
            "scale_rms"
        ],
        "frame_total": validated_frame_statistics[
            "frame_total"
        ],
        "frame_mean": validated_frame_statistics[
            "frame_mean"
        ],
        "frame_max": validated_frame_statistics[
            "frame_max"
        ],
        "dominant_scale_index": dominant_scale_index,
        "dominant_frequency_hz": dominant_frequency_hz,
        "frequency_centroid_hz": frequency_centroid_hz,
        "frequency_dispersion_hz": (
            frequency_dispersion_hz
        ),
    }

    return validated_profile


def build_scale_profile_metadata(
    *,
    source_cwt_metadata: CWTMetadata,
    scale_profile: ScaleProfile,
) -> ScaleProfileMetadata:
    """Build JSON-compatible metadata for a ScaleProfile.

    Args:
        source_cwt_metadata:
            Validated metadata associated with the source CWT.

        scale_profile:
            Validated ScaleProfile numerical arrays.

    Returns:
        Versioned, JSON-compatible ScaleProfile metadata.

    Raises:
        ValueError:
            If an array required for summary generation is empty.
    """

    number_of_scales, number_of_frames = (
        scale_profile["magnitude"].shape
    )

    if number_of_scales < 1:
        raise ValueError(
            "ScaleProfile metadata requires at least one scale"
        )

    if number_of_frames < 1:
        raise ValueError(
            "ScaleProfile metadata requires at least one frame"
        )

    scale_mean = scale_profile["scale_mean"]
    scale_rms = scale_profile["scale_rms"]
    frame_total = scale_profile["frame_total"]
    frequency_centroid_hz = scale_profile[
        "frequency_centroid_hz"
    ]

    if scale_mean.size == 0:
        raise ValueError(
            "scale_mean must not be empty"
        )

    if scale_rms.size == 0:
        raise ValueError(
            "scale_rms must not be empty"
        )

    if frame_total.size == 0:
        raise ValueError(
            "frame_total must not be empty"
        )

    if frequency_centroid_hz.size == 0:
        raise ValueError(
            "frequency_centroid_hz must not be empty"
        )

    zero_energy_frame_count = int(
        np.count_nonzero(
            frame_total == 0.0
        )
    )

    zero_energy_frame_ratio = float(
        zero_energy_frame_count
        / number_of_frames
    )

    metadata: ScaleProfileMetadata = {
        "format_version": (
            SCALE_PROFILE_FORMAT_VERSION
        ),
        "source_cwt": {
            "metadata_format_version": int(
                source_cwt_metadata[
                    "format_version"
                ]
            ),
            "wavelet": str(
                source_cwt_metadata[
                    "cwt_configuration"
                ]["wavelet"]
            ),
            "sampling_period": float(
                source_cwt_metadata[
                    "cwt_configuration"
                ]["sampling_period"]
            ),
            "sampling_frequency": float(
                source_cwt_metadata[
                    "cwt_configuration"
                ]["sampling_frequency"]
            ),
            "method": str(
                source_cwt_metadata[
                    "cwt_configuration"
                ]["method"]
            ),
            "precision": int(
                source_cwt_metadata[
                    "cwt_configuration"
                ]["precision"]
            ),
            "coefficient_dtype": str(
                source_cwt_metadata[
                    "cwt_summary"
                ]["coefficient_dtype"]
            ),
            "number_of_scales": int(
                source_cwt_metadata[
                    "cwt_summary"
                ]["n_scales"]
            ),
            "number_of_frames": int(
                source_cwt_metadata[
                    "cwt_summary"
                ]["n_frames"]
            ),
            "minimum_frequency_hz": float(
                source_cwt_metadata[
                    "cwt_summary"
                ]["frequency_min_hz"]
            ),
            "maximum_frequency_hz": float(
                source_cwt_metadata[
                    "cwt_summary"
                ]["frequency_max_hz"]
            ),
        },
        "feature_configuration": {
            "magnitude_transform": (
                MAGNITUDE_TRANSFORM
            ),
            "standard_deviation_convention": (
                STANDARD_DEVIATION_CONVENTION
            ),
            "zero_energy_policy": (
                ZERO_ENERGY_POLICY
            ),
            "dominant_scale_tie_policy": (
                DOMINANT_SCALE_TIE_POLICY
            ),
            "centroid_coordinate": (
                CENTROID_COORDINATE
            ),
            "dispersion_coordinate": (
                DISPERSION_COORDINATE
            ),
        },
        "shape_summary": {
            "number_of_scales": int(
                number_of_scales
            ),
            "number_of_frames": int(
                number_of_frames
            ),
            "frequencies_shape": [
                int(value)
                for value in scale_profile[
                    "frequencies"
                ].shape
            ],
            "magnitude_shape": [
                int(value)
                for value in scale_profile[
                    "magnitude"
                ].shape
            ],
            "scale_statistic_shape": [
                int(value)
                for value in scale_mean.shape
            ],
            "frame_statistic_shape": [
                int(value)
                for value in frame_total.shape
            ],
            "number_of_scale_statistics": int(
                NUMBER_OF_SCALE_STATISTICS
            ),
            "number_of_frame_statistics": int(
                NUMBER_OF_FRAME_STATISTICS
            ),
        },
        "scale_statistics_summary": {
            "minimum_scale_mean": float(
                np.min(scale_mean)
            ),
            "maximum_scale_mean": float(
                np.max(scale_mean)
            ),
            "minimum_scale_rms": float(
                np.min(scale_rms)
            ),
            "maximum_scale_rms": float(
                np.max(scale_rms)
            ),
        },
        "frame_statistics_summary": {
            "minimum_frame_total": float(
                np.min(frame_total)
            ),
            "maximum_frame_total": float(
                np.max(frame_total)
            ),
            "minimum_centroid_frequency_hz": float(
                np.min(
                    frequency_centroid_hz
                )
            ),
            "maximum_centroid_frequency_hz": float(
                np.max(
                    frequency_centroid_hz
                )
            ),
            "zero_energy_frame_count": (
                zero_energy_frame_count
            ),
            "zero_energy_frame_ratio": (
                zero_energy_frame_ratio
            ),
        },
    }

    return metadata


def _require_exact_metadata_keys(
    mapping: Mapping[str, object],
    *,
    required_keys: tuple[str, ...],
    name: str,
) -> None:
    """Require an exact set of metadata keys."""

    missing_keys = tuple(
        key
        for key in required_keys
        if key not in mapping
    )

    if missing_keys:
        raise KeyError(
            f"{name} is missing required fields: "
            f"{', '.join(missing_keys)}"
        )

    unexpected_keys = tuple(
        key
        for key in mapping
        if key not in required_keys
    )

    if unexpected_keys:
        raise ValueError(
            f"{name} contains unexpected fields: "
            f"{', '.join(unexpected_keys)}"
        )


def _require_metadata_mapping(
    value: object,
    *,
    name: str,
) -> Mapping[str, object]:
    """Validate and return a metadata mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(
            f"{name} must be a mapping"
        )

    return value


def _require_metadata_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    """Validate one non-boolean metadata integer."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer and must not be boolean"
        )

    validated_value = int(value)

    if (
        minimum is not None
        and validated_value < minimum
    ):
        raise ValueError(
            f"{name} must be at least {minimum}; "
            f"got {validated_value}"
        )

    return validated_value


def _require_metadata_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    """Validate one finite non-boolean metadata number."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            f"{name} must be a real number and must not be boolean"
        )

    validated_value = float(value)

    if not np.isfinite(validated_value):
        raise ValueError(
            f"{name} must be finite"
        )

    if (
        minimum is not None
        and validated_value < minimum
    ):
        raise ValueError(
            f"{name} must be at least {minimum}; "
            f"got {validated_value}"
        )

    return validated_value


def _require_metadata_string(
    value: object,
    *,
    name: str,
) -> str:
    """Validate one non-empty metadata string."""

    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string"
        )

    validated_value = value.strip()

    if not validated_value:
        raise ValueError(
            f"{name} must not be empty"
        )

    return validated_value


def validate_scale_profile_metadata(
    metadata: object,
    *,
    scale_profile: ScaleProfile,
) -> ScaleProfileMetadata:
    """Validate ScaleProfile metadata against numerical profile arrays.

    Args:
        metadata:
            ScaleProfile metadata to validate.

        scale_profile:
            Validated ScaleProfile numerical arrays associated with the
            metadata.

    Returns:
        An independent validated copy of the metadata.

    Raises:
        TypeError:
            If a metadata section or value has an invalid type.

        KeyError:
            If a required metadata field is absent.

        ValueError:
            If metadata values violate the ScaleProfile contract or do
            not match the associated ScaleProfile arrays.
    """

    root = _require_metadata_mapping(
        metadata,
        name="ScaleProfile metadata root",
    )

    _require_exact_metadata_keys(
        root,
        required_keys=(
            SCALE_PROFILE_METADATA_REQUIRED_KEYS
        ),
        name="ScaleProfile metadata",
    )

    validated_metadata = cast(
        ScaleProfileMetadata,
        deepcopy(dict(root)),
    )

    format_version = _require_metadata_integer(
        root["format_version"],
        name="format_version",
        minimum=1,
    )

    if format_version != SCALE_PROFILE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported ScaleProfile metadata format version; "
            f"expected {SCALE_PROFILE_FORMAT_VERSION}, "
            f"got {format_version}"
        )

    source_cwt = _require_metadata_mapping(
        root["source_cwt"],
        name="source_cwt",
    )

    feature_configuration = _require_metadata_mapping(
        root["feature_configuration"],
        name="feature_configuration",
    )

    shape_summary = _require_metadata_mapping(
        root["shape_summary"],
        name="shape_summary",
    )

    scale_summary = _require_metadata_mapping(
        root["scale_statistics_summary"],
        name="scale_statistics_summary",
    )

    frame_summary = _require_metadata_mapping(
        root["frame_statistics_summary"],
        name="frame_statistics_summary",
    )

    _require_exact_metadata_keys(
        source_cwt,
        required_keys=SOURCE_CWT_METADATA_REQUIRED_KEYS,
        name="source_cwt",
    )

    _require_exact_metadata_keys(
        feature_configuration,
        required_keys=(
            FEATURE_CONFIGURATION_REQUIRED_KEYS
        ),
        name="feature_configuration",
    )

    _require_exact_metadata_keys(
        shape_summary,
        required_keys=SHAPE_SUMMARY_REQUIRED_KEYS,
        name="shape_summary",
    )

    _require_exact_metadata_keys(
        scale_summary,
        required_keys=(
            SCALE_STATISTICS_SUMMARY_REQUIRED_KEYS
        ),
        name="scale_statistics_summary",
    )

    _require_exact_metadata_keys(
        frame_summary,
        required_keys=(
            FRAME_STATISTICS_SUMMARY_REQUIRED_KEYS
        ),
        name="frame_statistics_summary",
    )

    source_metadata_version = _require_metadata_integer(
        source_cwt["metadata_format_version"],
        name="source_cwt.metadata_format_version",
        minimum=1,
    )

    if source_metadata_version != 1:
        raise ValueError(
            "Unsupported source CWT metadata format version; "
            f"expected 1, got {source_metadata_version}"
        )

    _require_metadata_string(
        source_cwt["wavelet"],
        name="source_cwt.wavelet",
    )

    sampling_period = _require_metadata_float(
        source_cwt["sampling_period"],
        name="source_cwt.sampling_period",
        minimum=0.0,
    )

    sampling_frequency = _require_metadata_float(
        source_cwt["sampling_frequency"],
        name="source_cwt.sampling_frequency",
        minimum=0.0,
    )

    if sampling_period <= 0.0:
        raise ValueError(
            "source_cwt.sampling_period must be positive"
        )

    if sampling_frequency <= 0.0:
        raise ValueError(
            "source_cwt.sampling_frequency must be positive"
        )

    if not np.isclose(
        sampling_frequency,
        1.0 / sampling_period,
        rtol=SCALE_PROFILE_RTOL,
        atol=SCALE_PROFILE_ATOL,
    ):
        raise ValueError(
            "source_cwt.sampling_frequency is inconsistent "
            "with source_cwt.sampling_period"
        )

    method = _require_metadata_string(
        source_cwt["method"],
        name="source_cwt.method",
    )

    if method not in ("conv", "fft"):
        raise ValueError(
            "source_cwt.method must be one of: conv, fft"
        )

    _require_metadata_integer(
        source_cwt["precision"],
        name="source_cwt.precision",
        minimum=MINIMUM_PRECISION,
    )

    _require_metadata_string(
        source_cwt["coefficient_dtype"],
        name="source_cwt.coefficient_dtype",
    )

    source_scale_count = _require_metadata_integer(
        source_cwt["number_of_scales"],
        name="source_cwt.number_of_scales",
        minimum=1,
    )

    source_frame_count = _require_metadata_integer(
        source_cwt["number_of_frames"],
        name="source_cwt.number_of_frames",
        minimum=1,
    )

    source_minimum_frequency = _require_metadata_float(
        source_cwt["minimum_frequency_hz"],
        name="source_cwt.minimum_frequency_hz",
        minimum=0.0,
    )

    source_maximum_frequency = _require_metadata_float(
        source_cwt["maximum_frequency_hz"],
        name="source_cwt.maximum_frequency_hz",
        minimum=0.0,
    )

    if source_minimum_frequency <= 0.0:
        raise ValueError(
            "source_cwt.minimum_frequency_hz must be positive"
        )

    if source_maximum_frequency <= 0.0:
        raise ValueError(
            "source_cwt.maximum_frequency_hz must be positive"
        )

    if (
        source_minimum_frequency
        >= source_maximum_frequency
    ):
        raise ValueError(
            "source_cwt frequency range must be increasing"
        )

    expected_configuration = {
        "magnitude_transform": MAGNITUDE_TRANSFORM,
        "standard_deviation_convention": (
            STANDARD_DEVIATION_CONVENTION
        ),
        "zero_energy_policy": ZERO_ENERGY_POLICY,
        "dominant_scale_tie_policy": (
            DOMINANT_SCALE_TIE_POLICY
        ),
        "centroid_coordinate": CENTROID_COORDINATE,
        "dispersion_coordinate": DISPERSION_COORDINATE,
    }

    if dict(feature_configuration) != expected_configuration:
        raise ValueError(
            "Unsupported ScaleProfile feature configuration"
        )

    number_of_scales = _require_metadata_integer(
        shape_summary["number_of_scales"],
        name="shape_summary.number_of_scales",
        minimum=1,
    )

    number_of_frames = _require_metadata_integer(
        shape_summary["number_of_frames"],
        name="shape_summary.number_of_frames",
        minimum=1,
    )

    expected_shapes = {
        "frequencies_shape": [
            number_of_scales
        ],
        "magnitude_shape": [
            number_of_scales,
            number_of_frames,
        ],
        "scale_statistic_shape": [
            number_of_scales
        ],
        "frame_statistic_shape": [
            number_of_frames
        ],
    }

    for name, expected_shape in expected_shapes.items():
        actual_shape = shape_summary[name]

        if not isinstance(actual_shape, list):
            raise TypeError(
                f"shape_summary.{name} must be a list"
            )

        if actual_shape != expected_shape:
            raise ValueError(
                f"shape_summary.{name} is inconsistent; "
                f"expected {expected_shape}, "
                f"got {actual_shape}"
            )

    scale_statistic_count = _require_metadata_integer(
        shape_summary["number_of_scale_statistics"],
        name="shape_summary.number_of_scale_statistics",
        minimum=1,
    )

    frame_statistic_count = _require_metadata_integer(
        shape_summary["number_of_frame_statistics"],
        name="shape_summary.number_of_frame_statistics",
        minimum=1,
    )

    if scale_statistic_count != NUMBER_OF_SCALE_STATISTICS:
        raise ValueError(
            "Unexpected number of scale statistics"
        )

    if frame_statistic_count != NUMBER_OF_FRAME_STATISTICS:
        raise ValueError(
            "Unexpected number of frame statistics"
        )

    actual_scale_count = int(
        scale_profile["frequencies"].size
    )

    actual_frame_count = int(
        scale_profile["magnitude"].shape[1]
    )

    if (
        source_scale_count != actual_scale_count
        or number_of_scales != actual_scale_count
    ):
        raise ValueError(
            "ScaleProfile metadata scale count does not "
            "match the ScaleProfile arrays"
        )

    if (
        source_frame_count != actual_frame_count
        or number_of_frames != actual_frame_count
    ):
        raise ValueError(
            "ScaleProfile metadata frame count does not "
            "match the ScaleProfile arrays"
        )

    actual_minimum_frequency = float(
        np.min(scale_profile["frequencies"])
    )

    actual_maximum_frequency = float(
        np.max(scale_profile["frequencies"])
    )

    if not np.isclose(
        source_minimum_frequency,
        actual_minimum_frequency,
        rtol=SCALE_PROFILE_RTOL,
        atol=SCALE_PROFILE_ATOL,
    ):
        raise ValueError(
            "Source CWT minimum frequency does not match "
            "the ScaleProfile arrays"
        )

    if not np.isclose(
        source_maximum_frequency,
        actual_maximum_frequency,
        rtol=SCALE_PROFILE_RTOL,
        atol=SCALE_PROFILE_ATOL,
    ):
        raise ValueError(
            "Source CWT maximum frequency does not match "
            "the ScaleProfile arrays"
        )

    minimum_scale_mean = _require_metadata_float(
        scale_summary["minimum_scale_mean"],
        name=(
            "scale_statistics_summary."
            "minimum_scale_mean"
        ),
        minimum=0.0,
    )

    maximum_scale_mean = _require_metadata_float(
        scale_summary["maximum_scale_mean"],
        name=(
            "scale_statistics_summary."
            "maximum_scale_mean"
        ),
        minimum=0.0,
    )

    minimum_scale_rms = _require_metadata_float(
        scale_summary["minimum_scale_rms"],
        name=(
            "scale_statistics_summary."
            "minimum_scale_rms"
        ),
        minimum=0.0,
    )

    maximum_scale_rms = _require_metadata_float(
        scale_summary["maximum_scale_rms"],
        name=(
            "scale_statistics_summary."
            "maximum_scale_rms"
        ),
        minimum=0.0,
    )

    if minimum_scale_mean > maximum_scale_mean:
        raise ValueError(
            "minimum_scale_mean must not exceed "
            "maximum_scale_mean"
        )

    if minimum_scale_rms > maximum_scale_rms:
        raise ValueError(
            "minimum_scale_rms must not exceed "
            "maximum_scale_rms"
        )

    actual_scale_summary = {
        "minimum_scale_mean": minimum_scale_mean,
        "maximum_scale_mean": maximum_scale_mean,
        "minimum_scale_rms": minimum_scale_rms,
        "maximum_scale_rms": maximum_scale_rms,
    }

    expected_scale_summary = {
        "minimum_scale_mean": float(
            np.min(scale_profile["scale_mean"])
        ),
        "maximum_scale_mean": float(
            np.max(scale_profile["scale_mean"])
        ),
        "minimum_scale_rms": float(
            np.min(scale_profile["scale_rms"])
        ),
        "maximum_scale_rms": float(
            np.max(scale_profile["scale_rms"])
        ),
    }

    for name, expected_value in (
        expected_scale_summary.items()
    ):
        actual_value = actual_scale_summary[name]

        if not np.isclose(
            actual_value,
            expected_value,
            rtol=SCALE_PROFILE_RTOL,
            atol=SCALE_PROFILE_ATOL,
        ):
            raise ValueError(
                f"scale_statistics_summary.{name} "
                "does not match the ScaleProfile arrays"
            )

    frame_total = scale_profile["frame_total"]
    centroid = scale_profile[
        "frequency_centroid_hz"
    ]

    zero_energy_count = int(
        np.count_nonzero(
            frame_total == 0.0
        )
    )

    zero_energy_ratio = float(
        zero_energy_count
        / actual_frame_count
    )

    minimum_frame_total = _require_metadata_float(
        frame_summary["minimum_frame_total"],
        name=(
            "frame_statistics_summary."
            "minimum_frame_total"
        ),
        minimum=0.0,
    )

    maximum_frame_total = _require_metadata_float(
        frame_summary["maximum_frame_total"],
        name=(
            "frame_statistics_summary."
            "maximum_frame_total"
        ),
        minimum=0.0,
    )

    minimum_centroid_frequency_hz = (
        _require_metadata_float(
            frame_summary[
                "minimum_centroid_frequency_hz"
            ],
            name=(
                "frame_statistics_summary."
                "minimum_centroid_frequency_hz"
            ),
            minimum=0.0,
        )
    )

    maximum_centroid_frequency_hz = (
        _require_metadata_float(
            frame_summary[
                "maximum_centroid_frequency_hz"
            ],
            name=(
                "frame_statistics_summary."
                "maximum_centroid_frequency_hz"
            ),
            minimum=0.0,
        )
    )

    if minimum_frame_total > maximum_frame_total:
        raise ValueError(
            "minimum_frame_total must not exceed "
            "maximum_frame_total"
        )

    if (
        minimum_centroid_frequency_hz
        > maximum_centroid_frequency_hz
    ):
        raise ValueError(
            "minimum_centroid_frequency_hz must not exceed "
            "maximum_centroid_frequency_hz"
        )

    minimum_frame_total = _require_metadata_float(
        frame_summary["minimum_frame_total"],
        name=(
            "frame_statistics_summary."
            "minimum_frame_total"
        ),
        minimum=0.0,
    )

    maximum_frame_total = _require_metadata_float(
        frame_summary["maximum_frame_total"],
        name=(
            "frame_statistics_summary."
            "maximum_frame_total"
        ),
        minimum=0.0,
    )

    minimum_centroid_frequency_hz = (
        _require_metadata_float(
            frame_summary[
                "minimum_centroid_frequency_hz"
            ],
            name=(
                "frame_statistics_summary."
                "minimum_centroid_frequency_hz"
            ),
            minimum=0.0,
        )
    )

    maximum_centroid_frequency_hz = (
        _require_metadata_float(
            frame_summary[
                "maximum_centroid_frequency_hz"
            ],
            name=(
                "frame_statistics_summary."
                "maximum_centroid_frequency_hz"
            ),
            minimum=0.0,
        )
    )

    if minimum_frame_total > maximum_frame_total:
        raise ValueError(
            "minimum_frame_total must not exceed "
            "maximum_frame_total"
        )

    if (
        minimum_centroid_frequency_hz
        > maximum_centroid_frequency_hz
    ):
        raise ValueError(
            "minimum_centroid_frequency_hz must not exceed "
            "maximum_centroid_frequency_hz"
        )

    actual_frame_summary = {
        "minimum_frame_total": minimum_frame_total,
        "maximum_frame_total": maximum_frame_total,
        "minimum_centroid_frequency_hz": (
            minimum_centroid_frequency_hz
        ),
        "maximum_centroid_frequency_hz": (
            maximum_centroid_frequency_hz
        ),
    }

    expected_frame_summary = {
        "minimum_frame_total": float(
            np.min(frame_total)
        ),
        "maximum_frame_total": float(
            np.max(frame_total)
        ),
        "minimum_centroid_frequency_hz": float(
            np.min(centroid)
        ),
        "maximum_centroid_frequency_hz": float(
            np.max(centroid)
        ),
    }

    for name, expected_value in (
        expected_frame_summary.items()
    ):
        actual_value = actual_frame_summary[name]

        if not np.isclose(
            actual_value,
            expected_value,
            rtol=SCALE_PROFILE_RTOL,
            atol=SCALE_PROFILE_ATOL,
        ):
            raise ValueError(
                f"frame_statistics_summary.{name} "
                "does not match the ScaleProfile arrays"
            )

    metadata_zero_energy_count = (
        _require_metadata_integer(
            frame_summary[
                "zero_energy_frame_count"
            ],
            name=(
                "frame_statistics_summary."
                "zero_energy_frame_count"
            ),
            minimum=0,
        )
    )

    metadata_zero_energy_ratio = (
        _require_metadata_float(
            frame_summary[
                "zero_energy_frame_ratio"
            ],
            name=(
                "frame_statistics_summary."
                "zero_energy_frame_ratio"
            ),
            minimum=0.0,
        )
    )

    if metadata_zero_energy_count > actual_frame_count:
        raise ValueError(
            "zero_energy_frame_count exceeds "
            "the number of frames"
        )

    if metadata_zero_energy_ratio > 1.0:
        raise ValueError(
            "zero_energy_frame_ratio must not exceed 1.0"
        )

    if metadata_zero_energy_count != zero_energy_count:
        raise ValueError(
            "zero_energy_frame_count does not match "
            "the ScaleProfile arrays"
        )

    if not np.isclose(
        metadata_zero_energy_ratio,
        zero_energy_ratio,
        rtol=SCALE_PROFILE_RTOL,
        atol=SCALE_PROFILE_ATOL,
    ):
        raise ValueError(
            "zero_energy_frame_ratio does not match "
            "the ScaleProfile arrays"
        )

    try:
        json.dumps(
            validated_metadata
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "ScaleProfile metadata must be "
            "JSON serializable"
        ) from error

    return validated_metadata


#6.12
def extract_scale_profile(
    *,
    coefficients: np.ndarray,
    frequencies: np.ndarray,
    source_cwt_metadata: dict[str, object],
) -> tuple[
    ScaleProfile,
    ScaleProfileMetadata,
]:
    """Build a validated ScaleProfile from persisted CWT data.

    Args:
        coefficients:
            Real-valued or complex-valued CWT coefficient matrix.

        frequencies:
            Physical frequency coordinates associated with the CWT
            scale dimension.

        source_cwt_metadata:
            Decoded metadata associated with the source CWT.

    Returns:
        A tuple containing a validated ScaleProfile and its validated
        JSON-compatible metadata.

    Raises:
        TypeError:
            If an input or generated metadata value has an invalid
            type.

        KeyError:
            If a required input or metadata field is absent.

        ValueError:
            If an input, generated feature, or metadata value violates
            the ScaleProfile contract.
    """

    (
        validated_coefficients,
        validated_frequencies,
        validated_source_metadata,
    ) = validate_loaded_cwt(
        coefficients,
        frequencies,
        source_cwt_metadata,
    )

    magnitude = compute_cwt_magnitude(
        validated_coefficients,
    )

    scale_statistics = compute_scale_statistics(
        magnitude,
    )

    frame_statistics = compute_frame_statistics(
        magnitude,
        validated_frequencies,
    )

    scale_profile = build_scale_profile(
        frequencies=validated_frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    validated_scale_profile = (
        validate_scale_profile_arrays(
            scale_profile,
        )
    )

    scale_profile_metadata = (
        build_scale_profile_metadata(
            source_cwt_metadata=(
                validated_source_metadata
            ),
            scale_profile=(
                validated_scale_profile
            ),
        )
    )

    validated_scale_profile_metadata = (
        validate_scale_profile_metadata(
            scale_profile_metadata,
            scale_profile=(
                validated_scale_profile
            ),
        )
    )

    return (
        validated_scale_profile,
        validated_scale_profile_metadata,
    )


#6.13
def save_scale_profile_outputs(
    output_path: Path,
    metadata_path: Path,
    scale_profile: ScaleProfile,
    metadata: ScaleProfileMetadata,
) -> None:
    """Validate and save ScaleProfile arrays and metadata."""

    profile = validate_scale_profile_arrays(
        scale_profile
    )

    validated_metadata = validate_scale_profile_metadata(
        metadata,
        scale_profile=profile,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays = {
        key: profile[key]
        for key in SCALE_PROFILE_REQUIRED_KEYS
    }

    np.savez_compressed(
        output_path,
        **arrays,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            validated_metadata,
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the ScaleProfile command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build ScaleProfile features from persisted CWT data."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metadata-input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
    )

    return parser


def default_scale_profile_metadata_path(
    output_path: Path,
) -> Path:
    """Construct the default ScaleProfile metadata path."""

    return output_path.with_name(
        f"{output_path.stem}_metadata.json"
    )


def validate_arguments(
    args: argparse.Namespace,
) -> Path:
    """Validate CLI paths and return the metadata output path."""

    if not args.input.is_file():
        raise FileNotFoundError(
            f"CWT archive does not exist: {args.input}"
        )

    if not args.metadata_input.is_file():
        raise FileNotFoundError(
            "CWT metadata file does not exist: "
            f"{args.metadata_input}"
        )

    if args.input.suffix.lower() != ".npz":
        raise ValueError(
            "--input must use the .npz extension"
        )

    if args.metadata_input.suffix.lower() != ".json":
        raise ValueError(
            "--metadata-input must use the .json extension"
        )

    if args.output.suffix.lower() != ".npz":
        raise ValueError(
            "--output must use the .npz extension"
        )

    metadata_output = (
        args.metadata_output
        if args.metadata_output is not None
        else default_scale_profile_metadata_path(
            args.output
        )
    )

    if metadata_output.suffix.lower() != ".json":
        raise ValueError(
            "--metadata-output must use the .json extension"
        )

    paths = [
        args.input.resolve(),
        args.metadata_input.resolve(),
        args.output.resolve(),
        metadata_output.resolve(),
    ]

    if len(set(paths)) != len(paths):
        raise ValueError(
            "CLI input and output paths must be different"
        )

    return metadata_output


def main() -> int:
    """Run ScaleProfile extraction and persistence."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        metadata_output = validate_arguments(args)

        (
            coefficients,
            frequencies,
            source_metadata,
        ) = load_cwt_outputs(
            args.input,
            args.metadata_input,
        )

        profile, metadata = extract_scale_profile(
            coefficients=coefficients,
            frequencies=frequencies,
            source_cwt_metadata=source_metadata,
        )

        save_scale_profile_outputs(
            args.output,
            metadata_output,
            profile,
            metadata,
        )

        print("ScaleProfile extraction: OK")
        print(
            f"scales: "
            f"{profile['frequencies'].size}"
        )
        print(
            f"frames: "
            f"{profile['magnitude'].shape[1]}"
        )
        print(f"output: {args.output}")
        print(f"metadata: {metadata_output}")

        return 0

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"error: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())