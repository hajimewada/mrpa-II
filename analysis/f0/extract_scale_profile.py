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

from copy import deepcopy
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from analysis.f0.extract_cwt import (
    CWTMetadata,
    MINIMUM_FRAMES,
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
