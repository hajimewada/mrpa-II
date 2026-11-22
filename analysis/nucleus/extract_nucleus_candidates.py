#!/usr/bin/env python3
"""Discover Nucleus candidates from validated prosodic features."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from analysis.boundary.extract_boundary_candidates import (
    BOUNDARY_CANDIDATE_REQUIRED_KEYS,
    BoundaryCandidates,
    BoundaryMetadata,
    compute_robust_normalized_series,
    load_scale_profile_outputs,
    validate_boundary_candidates,
    validate_boundary_metadata,
    validate_loaded_scale_profile,
)

from analysis.f0.extract_scale_profile import (
    SCALE_PROFILE_REQUIRED_KEYS,
    ScaleProfile,
    ScaleProfileMetadata,
    validate_scale_profile_arrays,
)

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]

class NucleusEvidence(TypedDict):
    """Feature-specific evidence for prosodic prominence."""

    frame_total_level: Float64Array
    frame_mean_level: Float64Array
    frame_max_level: Float64Array
    dominant_frequency_level_hz: Float64Array
    frequency_centroid_level_hz: Float64Array
    frequency_dispersion_concentration: Float64Array


class NucleusCandidates(TypedDict):
    """Nucleus strength and extracted candidate locations."""

    nucleus_strength: Float64Array
    candidate_frame_indices: Int64Array
    candidate_strengths: Float64Array


class SourceScaleProfileMetadata(TypedDict):
    """Summary of the source ScaleProfile representation."""

    metadata_format_version: int
    number_of_scales: int
    number_of_frames: int
    minimum_frequency_hz: float
    maximum_frequency_hz: float


class SourceBoundaryMetadata(TypedDict):
    """Summary of the source Boundary representation."""

    metadata_format_version: int
    number_of_frames: int
    number_of_candidates: int
    threshold: float
    minimum_candidate_distance: int


class NucleusConfigurationMetadata(TypedDict):
    """Configuration used for Nucleus candidate discovery."""

    normalization: str
    local_contrast_radius: int
    threshold: float
    minimum_candidate_distance: int
    boundary_exclusion_distance: int
    plateau_policy: str
    feature_names: list[str]
    feature_weights: dict[str, float]


class NucleusShapeMetadata(TypedDict):
    """Shapes of persisted Nucleus candidate arrays."""

    number_of_frames: int
    number_of_candidates: int
    nucleus_strength_shape: list[int]
    candidate_index_shape: list[int]
    candidate_strength_shape: list[int]


class NucleusSummaryMetadata(TypedDict):
    """Aggregate Nucleus strength and candidate statistics."""

    minimum_nucleus_strength: float
    maximum_nucleus_strength: float
    minimum_candidate_strength: float
    maximum_candidate_strength: float
    boundary_excluded_candidate_count: int


class NucleusMetadata(TypedDict):
    """Top-level persisted Nucleus candidate metadata."""

    format_version: int
    source_scale_profile: SourceScaleProfileMetadata
    source_boundary: SourceBoundaryMetadata
    nucleus_configuration: NucleusConfigurationMetadata
    shape_summary: NucleusShapeMetadata
    nucleus_summary: NucleusSummaryMetadata


NUCLEUS_FORMAT_VERSION = 1

NUCLEUS_NORMALIZATION = "median-mad"
NUCLEUS_PLATEAU_POLICY = "first-index"

NUCLEUS_RTOL = 1e-12
NUCLEUS_ATOL = 1e-12


NUCLEUS_FEATURE_NAMES = (
    "frame_total",
    "frame_mean",
    "frame_max",
    "dominant_frequency_hz",
    "frequency_centroid_hz",
    "frequency_dispersion_hz",
)


NUCLEUS_EVIDENCE_REQUIRED_KEYS = (
    "frame_total_level",
    "frame_mean_level",
    "frame_max_level",
    "dominant_frequency_level_hz",
    "frequency_centroid_level_hz",
    "frequency_dispersion_concentration",
)


NUCLEUS_CANDIDATE_REQUIRED_KEYS = (
    "nucleus_strength",
    "candidate_frame_indices",
    "candidate_strengths",
)


NUCLEUS_METADATA_REQUIRED_KEYS = (
    "format_version",
    "source_scale_profile",
    "source_boundary",
    "nucleus_configuration",
    "shape_summary",
    "nucleus_summary",
)


SOURCE_SCALE_PROFILE_METADATA_REQUIRED_KEYS = (
    "metadata_format_version",
    "number_of_scales",
    "number_of_frames",
    "minimum_frequency_hz",
    "maximum_frequency_hz",
)


SOURCE_BOUNDARY_METADATA_REQUIRED_KEYS = (
    "metadata_format_version",
    "number_of_frames",
    "number_of_candidates",
    "threshold",
    "minimum_candidate_distance",
)


NUCLEUS_CONFIGURATION_METADATA_REQUIRED_KEYS = (
    "normalization",
    "local_contrast_radius",
    "threshold",
    "minimum_candidate_distance",
    "boundary_exclusion_distance",
    "plateau_policy",
    "feature_names",
    "feature_weights",
)


NUCLEUS_SHAPE_METADATA_REQUIRED_KEYS = (
    "number_of_frames",
    "number_of_candidates",
    "nucleus_strength_shape",
    "candidate_index_shape",
    "candidate_strength_shape",
)


NUCLEUS_SUMMARY_METADATA_REQUIRED_KEYS = (
    "minimum_nucleus_strength",
    "maximum_nucleus_strength",
    "minimum_candidate_strength",
    "maximum_candidate_strength",
    "boundary_excluded_candidate_count",
)


class NucleusConfiguration(TypedDict):
    """Runtime configuration for Nucleus candidate discovery."""

    normalization: str
    local_contrast_radius: int
    threshold: float
    minimum_candidate_distance: int
    boundary_exclusion_distance: int
    plateau_policy: str
    feature_weights: dict[str, float]


DEFAULT_LOCAL_CONTRAST_RADIUS = 2
DEFAULT_NUCLEUS_THRESHOLD = 1.0
DEFAULT_MINIMUM_NUCLEUS_CANDIDATE_DISTANCE = 1
DEFAULT_BOUNDARY_EXCLUSION_DISTANCE = 1

DEFAULT_NUCLEUS_FEATURE_WEIGHTS = {
    name: 1.0
    for name in NUCLEUS_FEATURE_NAMES
}


def build_default_nucleus_configuration(
) -> NucleusConfiguration:
    """Build an independent default Nucleus configuration."""

    return {
        "normalization": NUCLEUS_NORMALIZATION,
        "local_contrast_radius": (
            DEFAULT_LOCAL_CONTRAST_RADIUS
        ),
        "threshold": DEFAULT_NUCLEUS_THRESHOLD,
        "minimum_candidate_distance": (
            DEFAULT_MINIMUM_NUCLEUS_CANDIDATE_DISTANCE
        ),
        "boundary_exclusion_distance": (
            DEFAULT_BOUNDARY_EXCLUSION_DISTANCE
        ),
        "plateau_policy": NUCLEUS_PLATEAU_POLICY,
        "feature_weights": dict(
            DEFAULT_NUCLEUS_FEATURE_WEIGHTS
        ),
    }


def validate_nucleus_configuration(
    configuration: NucleusConfiguration,
) -> NucleusConfiguration:
    """Validate and copy a Nucleus configuration."""

    normalization = configuration["normalization"]

    if not isinstance(normalization, str):
        raise TypeError(
            "Nucleus normalization must be a string"
        )

    if normalization != NUCLEUS_NORMALIZATION:
        raise ValueError(
            "Unsupported Nucleus normalization"
        )

    plateau_policy = configuration[
        "plateau_policy"
    ]

    if not isinstance(plateau_policy, str):
        raise TypeError(
            "Nucleus plateau policy must be a string"
        )

    if plateau_policy != NUCLEUS_PLATEAU_POLICY:
        raise ValueError(
            "Unsupported Nucleus plateau policy"
        )

    local_contrast_radius = configuration[
        "local_contrast_radius"
    ]

    if (
        isinstance(local_contrast_radius, bool)
        or not isinstance(
            local_contrast_radius,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Local contrast radius must be an integer"
        )

    local_contrast_radius = int(
        local_contrast_radius
    )

    if local_contrast_radius < 1:
        raise ValueError(
            "Local contrast radius must be at least 1"
        )

    threshold = configuration["threshold"]

    if (
        isinstance(threshold, bool)
        or not isinstance(
            threshold,
            (Real, np.integer, np.floating),
        )
    ):
        raise TypeError(
            "Nucleus threshold must be a real number"
        )

    threshold = float(threshold)

    if not np.isfinite(threshold):
        raise ValueError(
            "Nucleus threshold must be finite"
        )

    if threshold < 0.0:
        raise ValueError(
            "Nucleus threshold must be non-negative"
        )

    minimum_distance = configuration[
        "minimum_candidate_distance"
    ]

    if (
        isinstance(minimum_distance, bool)
        or not isinstance(
            minimum_distance,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Minimum candidate distance must be an integer"
        )

    minimum_distance = int(
        minimum_distance
    )

    if minimum_distance < 1:
        raise ValueError(
            "Minimum candidate distance must be at least 1"
        )

    exclusion_distance = configuration[
        "boundary_exclusion_distance"
    ]

    if (
        isinstance(exclusion_distance, bool)
        or not isinstance(
            exclusion_distance,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Boundary exclusion distance must be an integer"
        )

    exclusion_distance = int(
        exclusion_distance
    )

    if exclusion_distance < 0:
        raise ValueError(
            "Boundary exclusion distance must be non-negative"
        )

    weights = configuration["feature_weights"]

    if not isinstance(weights, dict):
        raise TypeError(
            "Nucleus feature weights must be a dictionary"
        )

    if set(weights) != set(NUCLEUS_FEATURE_NAMES):
        raise ValueError(
            "Nucleus feature weights must match "
            "the required feature names"
        )

    validated_weights: dict[str, float] = {}

    for name in NUCLEUS_FEATURE_NAMES:
        value = weights[name]

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (Real, np.integer, np.floating),
            )
        ):
            raise TypeError(
                f"Nucleus weight for {name} "
                "must be a real number"
            )

        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                f"Nucleus weight for {name} must be finite"
            )

        if value < 0.0:
            raise ValueError(
                f"Nucleus weight for {name} "
                "must be non-negative"
            )

        validated_weights[name] = value

    if not any(
        value > 0.0
        for value in validated_weights.values()
    ):
        raise ValueError(
            "At least one Nucleus feature weight "
            "must be positive"
        )

    return {
        "normalization": normalization,
        "local_contrast_radius": (
            local_contrast_radius
        ),
        "threshold": threshold,
        "minimum_candidate_distance": (
            minimum_distance
        ),
        "boundary_exclusion_distance": (
            exclusion_distance
        ),
        "plateau_policy": plateau_policy,
        "feature_weights": validated_weights,
    }


#8.5
def load_boundary_outputs(
    input_path: Path,
    metadata_path: Path,
) -> tuple[
    BoundaryCandidates,
    dict[str, object],
]:
    """Load persisted Boundary candidate arrays and metadata."""

    if not input_path.is_file():
        raise FileNotFoundError(
            "Boundary archive does not exist: "
            f"{input_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Boundary metadata does not exist: "
            f"{metadata_path}"
        )

    with np.load(
        input_path,
        allow_pickle=False,
    ) as archive:
        missing_keys = [
            key
            for key in BOUNDARY_CANDIDATE_REQUIRED_KEYS
            if key not in archive.files
        ]

        if missing_keys:
            raise KeyError(
                "Boundary archive is missing required arrays: "
                + ", ".join(missing_keys)
            )

        candidates: BoundaryCandidates = {
            "boundary_strength": np.array(
                archive["boundary_strength"],
                copy=True,
            ),
            "candidate_frame_indices": np.array(
                archive["candidate_frame_indices"],
                copy=True,
            ),
            "candidate_strengths": np.array(
                archive["candidate_strengths"],
                copy=True,
            ),
        }

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    if not isinstance(metadata, dict):
        raise TypeError(
            "Boundary metadata root must be a dictionary"
        )

    return candidates, metadata


#8.6
def validate_loaded_boundary_outputs(
    candidates: BoundaryCandidates,
    metadata: object,
    source_metadata: ScaleProfileMetadata,
) -> tuple[
    BoundaryCandidates,
    BoundaryMetadata,
]:
    """Validate loaded Boundary arrays and metadata."""

    if not isinstance(metadata, dict):
        raise TypeError(
            "Boundary metadata root must be a dictionary"
        )

    configuration = metadata.get(
        "boundary_configuration"
    )

    if not isinstance(configuration, dict):
        raise TypeError(
            "Boundary configuration metadata "
            "must be a dictionary"
        )

    if "minimum_candidate_distance" not in configuration:
        raise KeyError(
            "minimum_candidate_distance"
        )

    minimum_distance = configuration[
        "minimum_candidate_distance"
    ]

    validated_candidates = validate_boundary_candidates(
        candidates,
        minimum_distance=minimum_distance,
    )

    validated_metadata = validate_boundary_metadata(
        metadata,
        source_metadata,
        validated_candidates,
    )

    return (
        validated_candidates,
        validated_metadata,
    )


#8.8
def compute_local_contrast(
    values: np.ndarray,
    radius: int,
) -> Float64Array:
    """Compute non-negative contrast against local neighbors."""

    series = np.array(
        values,
        dtype=np.float64,
        copy=True,
    )

    if series.ndim != 1:
        raise ValueError(
            "Frame series must be one-dimensional"
        )

    if series.size < 1:
        raise ValueError(
            "Frame series must contain at least one value"
        )

    if not np.all(np.isfinite(series)):
        raise ValueError(
            "Frame series must contain only finite values"
        )

    if (
        isinstance(radius, bool)
        or not isinstance(
            radius,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Local contrast radius must be an integer"
        )

    radius = int(radius)

    if radius < 1:
        raise ValueError(
            "Local contrast radius must be at least 1"
        )

    contrast = np.zeros(
        series.shape,
        dtype=np.float64,
    )

    if series.size == 1:
        return contrast

    for index in range(series.size):
        start = max(
            0,
            index - radius,
        )
        stop = min(
            series.size,
            index + radius + 1,
        )

        left = series[
            start:index
        ]
        right = series[
            index + 1:stop
        ]

        neighbors = np.concatenate(
            (
                left,
                right,
            )
        )

        if neighbors.size == 0:
            continue

        baseline = float(
            np.median(neighbors)
        )

        contrast[index] = max(
            series[index] - baseline,
            0.0,
        )

    if not np.all(np.isfinite(contrast)):
        raise ValueError(
            "Local contrast must contain only finite values"
        )

    return contrast


#8.8
def compute_normalized_local_contrast(
    values: np.ndarray,
    radius: int,
) -> Float64Array:
    """Compute robustly normalized non-negative local contrast."""

    contrast = compute_local_contrast(
        values,
        radius=radius,
    )

    normalized = compute_robust_normalized_series(
        contrast
    )

    if normalized.shape != contrast.shape:
        raise ValueError(
            "Normalized local contrast has an unexpected shape"
        )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "Normalized local contrast must contain "
            "only finite values"
        )

    if np.any(normalized < 0.0):
        raise ValueError(
            "Normalized local contrast must be non-negative"
        )

    return np.array(
        normalized,
        dtype=np.float64,
        copy=True,
    )


#8.9
def build_nucleus_evidence(
    scale_profile: ScaleProfile,
    configuration: NucleusConfiguration,
) -> NucleusEvidence:
    """Build normalized Nucleus evidence from frame features."""

    validated_profile = validate_scale_profile_arrays(
        scale_profile
    )

    validated_configuration = (
        validate_nucleus_configuration(
            configuration
        )
    )

    feature_mapping = {
        "frame_total_level": "frame_total",
        "frame_mean_level": "frame_mean",
        "frame_max_level": "frame_max",
        "dominant_frequency_level_hz": (
            "dominant_frequency_hz"
        ),
        "frequency_centroid_level_hz": (
            "frequency_centroid_hz"
        ),
        "frequency_dispersion_concentration": (
            "frequency_dispersion_hz"
        ),
    }

    frame_count = int(
        validated_profile["magnitude"].shape[1]
    )

    radius = validated_configuration[
        "local_contrast_radius"
    ]

    evidence: NucleusEvidence = {}

    for evidence_name, profile_name in (
        feature_mapping.items()
    ):
        values = compute_normalized_local_contrast(
            validated_profile[profile_name],
            radius=radius,
        )

        if values.shape != (frame_count,):
            raise ValueError(
                f"{evidence_name} has an unexpected shape"
            )

        if values.dtype != np.float64:
            raise TypeError(
                f"{evidence_name} must use float64"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{evidence_name} must contain only "
                "finite values"
            )

        if np.any(values < 0.0):
            raise ValueError(
                f"{evidence_name} must be non-negative"
            )

        evidence[evidence_name] = np.array(
            values,
            dtype=np.float64,
            copy=True,
        )

    if set(evidence) != set(
        NUCLEUS_EVIDENCE_REQUIRED_KEYS
    ):
        raise ValueError(
            "Nucleus evidence keys do not match "
            "the required schema"
        )

    return evidence


#8.10
def compute_nucleus_strength(
    evidence: NucleusEvidence,
    configuration: NucleusConfiguration,
) -> Float64Array:
    """Combine feature evidence into one Nucleus strength series."""

    validated_configuration = (
        validate_nucleus_configuration(
            configuration
        )
    )

    if set(evidence) != set(
        NUCLEUS_EVIDENCE_REQUIRED_KEYS
    ):
        raise ValueError(
            "Nucleus evidence keys do not match "
            "the required schema"
        )

    evidence_mapping = {
        "frame_total": "frame_total_level",
        "frame_mean": "frame_mean_level",
        "frame_max": "frame_max_level",
        "dominant_frequency_hz": (
            "dominant_frequency_level_hz"
        ),
        "frequency_centroid_hz": (
            "frequency_centroid_level_hz"
        ),
        "frequency_dispersion_hz": (
            "frequency_dispersion_concentration"
        ),
    }

    first_values = np.asarray(
        evidence["frame_total_level"]
    )

    if first_values.ndim != 1:
        raise ValueError(
            "Nucleus evidence must be one-dimensional"
        )

    frame_count = first_values.size

    if frame_count < 1:
        raise ValueError(
            "Nucleus evidence must contain at least one frame"
        )

    strength = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    total_weight = 0.0

    for feature_name in NUCLEUS_FEATURE_NAMES:
        evidence_name = evidence_mapping[
            feature_name
        ]

        values = np.array(
            evidence[evidence_name],
            dtype=np.float64,
            copy=True,
        )

        if values.ndim != 1:
            raise ValueError(
                f"{evidence_name} must be one-dimensional"
            )

        if values.shape != (frame_count,):
            raise ValueError(
                f"{evidence_name} has an unexpected shape"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{evidence_name} must contain only "
                "finite values"
            )

        if np.any(values < 0.0):
            raise ValueError(
                f"{evidence_name} must be non-negative"
            )

        weight = validated_configuration[
            "feature_weights"
        ][feature_name]

        strength += np.multiply(
            weight,
            values,
        )
        total_weight += weight

    strength = np.divide(
        strength,
        total_weight,
    )

    if not np.all(np.isfinite(strength)):
        raise ValueError(
            "Nucleus strength must contain only finite values"
        )

    if np.any(strength < 0.0):
        raise ValueError(
            "Nucleus strength must be non-negative"
        )

    return np.array(
        strength,
        dtype=np.float64,
        copy=True,
    )


#8.11
def validate_nucleus_strength(
    nucleus_strength: np.ndarray,
    expected_frame_count: int,
) -> Float64Array:
    """Validate a Nucleus strength series."""

    if (
        isinstance(expected_frame_count, bool)
        or not isinstance(
            expected_frame_count,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Expected frame count must be an integer"
        )

    expected_frame_count = int(
        expected_frame_count
    )

    if expected_frame_count < 1:
        raise ValueError(
            "Expected frame count must be at least 1"
        )

    validated = np.array(
        nucleus_strength,
        dtype=np.float64,
        copy=True,
    )

    if validated.ndim != 1:
        raise ValueError(
            "Nucleus strength must be one-dimensional"
        )

    if validated.shape != (
        expected_frame_count,
    ):
        raise ValueError(
            "Nucleus strength shape does not match "
            "the expected frame count"
        )

    if not np.all(np.isfinite(validated)):
        raise ValueError(
            "Nucleus strength must contain only finite values"
        )

    if np.any(validated < 0.0):
        raise ValueError(
            "Nucleus strength must be non-negative"
        )

    return validated


#8.12
def find_nucleus_local_maxima(
    nucleus_strength: np.ndarray,
    plateau_policy: str = NUCLEUS_PLATEAU_POLICY,
) -> Int64Array:
    """Find Nucleus local maxima, including endpoint peaks."""

    strength = validate_nucleus_strength(
        nucleus_strength,
        expected_frame_count=np.asarray(
            nucleus_strength
        ).size,
    )

    if not isinstance(
        plateau_policy,
        str,
    ):
        raise TypeError(
            "Nucleus plateau policy must be a string"
        )

    if plateau_policy != NUCLEUS_PLATEAU_POLICY:
        raise ValueError(
            "Unsupported Nucleus plateau policy"
        )

    if strength.size == 1:
        if strength[0] > 0.0:
            return np.array(
                [0],
                dtype=np.int64,
            )

        return np.array(
            [],
            dtype=np.int64,
        )

    candidate_indices: list[int] = []
    index = 0

    while index < strength.size:
        plateau_start = index
        plateau_end = index

        while (
            plateau_end + 1 < strength.size
            and strength[plateau_end + 1]
            == strength[plateau_start]
        ):
            plateau_end += 1

        plateau_value = strength[plateau_start]

        has_left_neighbor = plateau_start > 0
        has_right_neighbor = (
            plateau_end + 1 < strength.size
        )

        left_is_lower = (
            not has_left_neighbor
            or plateau_value
            > strength[plateau_start - 1]
        )

        right_is_lower = (
            not has_right_neighbor
            or plateau_value
            > strength[plateau_end + 1]
        )

        if (
            plateau_value > 0.0
            and left_is_lower
            and right_is_lower
        ):
            candidate_indices.append(
                plateau_start
            )

        index = plateau_end + 1

    return np.array(
        candidate_indices,
        dtype=np.int64,
    )


#8.13
def apply_nucleus_threshold(
    nucleus_strength: np.ndarray,
    candidate_indices: np.ndarray,
    threshold: float,
) -> Int64Array:
    """Filter Nucleus candidates using an inclusive threshold."""

    strength = validate_nucleus_strength(
        nucleus_strength,
        expected_frame_count=np.asarray(
            nucleus_strength
        ).size,
    )

    source_indices = np.asarray(
        candidate_indices
    )

    if (
        np.issubdtype(
            source_indices.dtype,
            np.bool_,
        )
        or not np.issubdtype(
            source_indices.dtype,
            np.integer,
        )
    ):
        raise TypeError(
            "Candidate indices must use an integer dtype"
        )

    indices = np.array(
        source_indices,
        dtype=np.int64,
        copy=True,
    )

    if indices.ndim != 1:
        raise ValueError(
            "Candidate indices must be one-dimensional"
        )

    if (
        np.any(indices < 0)
        or np.any(indices >= strength.size)
    ):
        raise ValueError(
            "Candidate indices contain an out-of-range value"
        )

    if np.unique(indices).size != indices.size:
        raise ValueError(
            "Candidate indices must not contain duplicates"
        )

    if (
        indices.size > 1
        and np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(
            "Candidate indices must be strictly increasing"
        )

    if (
        isinstance(threshold, bool)
        or not isinstance(
            threshold,
            (Real, np.integer, np.floating),
        )
    ):
        raise TypeError(
            "Nucleus threshold must be a real number"
        )

    threshold = float(threshold)

    if not np.isfinite(threshold):
        raise ValueError(
            "Nucleus threshold must be finite"
        )

    if threshold < 0.0:
        raise ValueError(
            "Nucleus threshold must be non-negative"
        )

    selected = indices[
        strength[indices] >= threshold
    ]

    return np.array(
        selected,
        dtype=np.int64,
        copy=True,
    )


#8.14
def enforce_minimum_nucleus_candidate_distance(
    nucleus_strength: np.ndarray,
    candidate_indices: np.ndarray,
    minimum_distance: int,
) -> Int64Array:
    """Suppress nearby Nucleus candidates, keeping stronger candidates."""

    strength = validate_nucleus_strength(
        nucleus_strength,
        expected_frame_count=np.asarray(
            nucleus_strength
        ).size,
    )

    source_indices = np.asarray(
        candidate_indices
    )

    if (
        np.issubdtype(
            source_indices.dtype,
            np.bool_,
        )
        or not np.issubdtype(
            source_indices.dtype,
            np.integer,
        )
    ):
        raise TypeError(
            "Candidate indices must use an integer dtype"
        )

    indices = np.array(
        source_indices,
        dtype=np.int64,
        copy=True,
    )

    if indices.ndim != 1:
        raise ValueError(
            "Candidate indices must be one-dimensional"
        )

    if (
        np.any(indices < 0)
        or np.any(indices >= strength.size)
    ):
        raise ValueError(
            "Candidate indices contain an out-of-range value"
        )

    if np.unique(indices).size != indices.size:
        raise ValueError(
            "Candidate indices must not contain duplicates"
        )

    if (
        indices.size > 1
        and np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(
            "Candidate indices must be strictly increasing"
        )

    if (
        isinstance(minimum_distance, bool)
        or not isinstance(
            minimum_distance,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Minimum Nucleus candidate distance "
            "must be an integer"
        )

    minimum_distance = int(
        minimum_distance
    )

    if minimum_distance < 1:
        raise ValueError(
            "Minimum Nucleus candidate distance "
            "must be at least 1"
        )

    ranked_indices = sorted(
        indices.tolist(),
        key=lambda index: (
            -strength[index],
            index,
        ),
    )

    selected_indices: list[int] = []

    for index in ranked_indices:
        if all(
            abs(index - selected) >= minimum_distance
            for selected in selected_indices
        ):
            selected_indices.append(
                index
            )

    selected_indices.sort()

    return np.array(
        selected_indices,
        dtype=np.int64,
    )


#8.15
def exclude_boundary_adjacent_nucleus_candidates(
    candidate_indices: np.ndarray,
    boundary_indices: np.ndarray,
    exclusion_distance: int,
    frame_count: int,
) -> tuple[
    Int64Array,
    int,
]:
    """Exclude Nucleus candidates near Boundary candidates."""

    if (
        isinstance(frame_count, bool)
        or not isinstance(
            frame_count,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Frame count must be an integer"
        )

    frame_count = int(frame_count)

    if frame_count < 1:
        raise ValueError(
            "Frame count must be at least 1"
        )

    source_candidate_indices = np.asarray(
        candidate_indices
    )

    if (
        np.issubdtype(
            source_candidate_indices.dtype,
            np.bool_,
        )
        or not np.issubdtype(
            source_candidate_indices.dtype,
            np.integer,
        )
    ):
        raise TypeError(
            "Nucleus candidate indices must use "
            "an integer dtype"
        )

    candidates = np.array(
        source_candidate_indices,
        dtype=np.int64,
        copy=True,
    )

    if candidates.ndim != 1:
        raise ValueError(
            "Nucleus candidate indices must be "
            "one-dimensional"
        )

    if (
        np.any(candidates < 0)
        or np.any(candidates >= frame_count)
    ):
        raise ValueError(
            "Nucleus candidate indices contain "
            "an out-of-range value"
        )

    if np.unique(candidates).size != candidates.size:
        raise ValueError(
            "Nucleus candidate indices must not "
            "contain duplicates"
        )

    if (
        candidates.size > 1
        and np.any(np.diff(candidates) <= 0)
    ):
        raise ValueError(
            "Nucleus candidate indices must be "
            "strictly increasing"
        )

    source_boundary_indices = np.asarray(
        boundary_indices
    )

    if (
        np.issubdtype(
            source_boundary_indices.dtype,
            np.bool_,
        )
        or not np.issubdtype(
            source_boundary_indices.dtype,
            np.integer,
        )
    ):
        raise TypeError(
            "Boundary candidate indices must use "
            "an integer dtype"
        )

    boundaries = np.array(
        source_boundary_indices,
        dtype=np.int64,
        copy=True,
    )

    if boundaries.ndim != 1:
        raise ValueError(
            "Boundary candidate indices must be "
            "one-dimensional"
        )

    if (
        np.any(boundaries < 0)
        or np.any(boundaries >= frame_count)
    ):
        raise ValueError(
            "Boundary candidate indices contain "
            "an out-of-range value"
        )

    if np.unique(boundaries).size != boundaries.size:
        raise ValueError(
            "Boundary candidate indices must not "
            "contain duplicates"
        )

    if (
        boundaries.size > 1
        and np.any(np.diff(boundaries) <= 0)
    ):
        raise ValueError(
            "Boundary candidate indices must be "
            "strictly increasing"
        )

    if (
        isinstance(exclusion_distance, bool)
        or not isinstance(
            exclusion_distance,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Boundary exclusion distance must be "
            "an integer"
        )

    exclusion_distance = int(
        exclusion_distance
    )

    if exclusion_distance < 0:
        raise ValueError(
            "Boundary exclusion distance must be "
            "non-negative"
        )

    if (
        exclusion_distance == 0
        or candidates.size == 0
        or boundaries.size == 0
    ):
        return (
            np.array(
                candidates,
                dtype=np.int64,
                copy=True,
            ),
            0,
        )

    kept_indices: list[int] = []

    for candidate_index in candidates:
        is_excluded = bool(
            np.any(
                np.abs(
                    boundaries - candidate_index
                )
                <= exclusion_distance
            )
        )

        if not is_excluded:
            kept_indices.append(
                int(candidate_index)
            )

    kept = np.array(
        kept_indices,
        dtype=np.int64,
    )

    excluded_count = int(
        candidates.size - kept.size
    )

    return kept, excluded_count


#8.16
def build_nucleus_candidates(
    nucleus_strength: np.ndarray,
    candidate_indices: np.ndarray,
) -> NucleusCandidates:
    """Build persistent Nucleus candidate arrays."""

    strength = validate_nucleus_strength(
        nucleus_strength,
        expected_frame_count=np.asarray(
            nucleus_strength
        ).size,
    )

    source_indices = np.asarray(
        candidate_indices
    )

    if (
        np.issubdtype(
            source_indices.dtype,
            np.bool_,
        )
        or not np.issubdtype(
            source_indices.dtype,
            np.integer,
        )
    ):
        raise TypeError(
            "Candidate indices must use an integer dtype"
        )

    indices = np.array(
        source_indices,
        dtype=np.int64,
        copy=True,
    )

    if indices.ndim != 1:
        raise ValueError(
            "Candidate indices must be one-dimensional"
        )

    if (
        np.any(indices < 0)
        or np.any(indices >= strength.size)
    ):
        raise ValueError(
            "Candidate indices contain an out-of-range value"
        )

    if np.unique(indices).size != indices.size:
        raise ValueError(
            "Candidate indices must not contain duplicates"
        )

    if (
        indices.size > 1
        and np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(
            "Candidate indices must be strictly increasing"
        )

    candidate_strengths = np.array(
        strength[indices],
        dtype=np.float64,
        copy=True,
    )

    candidates: NucleusCandidates = {
        "nucleus_strength": np.array(
            strength,
            dtype=np.float64,
            copy=True,
        ),
        "candidate_frame_indices": indices,
        "candidate_strengths": candidate_strengths,
    }

    if set(candidates) != set(
        NUCLEUS_CANDIDATE_REQUIRED_KEYS
    ):
        raise ValueError(
            "Nucleus candidate keys do not match "
            "the required schema"
        )

    return candidates


#8.17
def validate_nucleus_candidates(
    candidates: NucleusCandidates,
    minimum_distance: int,
) -> NucleusCandidates:
    """Validate Nucleus candidate arrays."""

    if set(candidates) != set(
        NUCLEUS_CANDIDATE_REQUIRED_KEYS
    ):
        raise ValueError(
            "Nucleus candidate keys do not match "
            "the required schema"
        )

    strength = validate_nucleus_strength(
        candidates["nucleus_strength"],
        expected_frame_count=np.asarray(
            candidates["nucleus_strength"]
        ).size,
    )

    source_indices = np.asarray(
        candidates["candidate_frame_indices"]
    )

    if (
        np.issubdtype(
            source_indices.dtype,
            np.bool_,
        )
        or not np.issubdtype(
            source_indices.dtype,
            np.integer,
        )
    ):
        raise TypeError(
            "Candidate indices must use an integer dtype"
        )

    indices = np.array(
        source_indices,
        dtype=np.int64,
        copy=True,
    )

    if indices.ndim != 1:
        raise ValueError(
            "Candidate indices must be one-dimensional"
        )

    if (
        np.any(indices < 0)
        or np.any(indices >= strength.size)
    ):
        raise ValueError(
            "Candidate indices contain an out-of-range value"
        )

    if np.unique(indices).size != indices.size:
        raise ValueError(
            "Candidate indices must not contain duplicates"
        )

    if (
        indices.size > 1
        and np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(
            "Candidate indices must be strictly increasing"
        )

    candidate_strengths = np.array(
        candidates["candidate_strengths"],
        dtype=np.float64,
        copy=True,
    )

    if candidate_strengths.ndim != 1:
        raise ValueError(
            "Candidate strengths must be one-dimensional"
        )

    if indices.shape != candidate_strengths.shape:
        raise ValueError(
            "Candidate index and strength shapes must match"
        )

    if not np.all(np.isfinite(candidate_strengths)):
        raise ValueError(
            "Candidate strengths must contain only "
            "finite values"
        )

    if np.any(candidate_strengths < 0.0):
        raise ValueError(
            "Candidate strengths must be non-negative"
        )

    if not np.allclose(
        candidate_strengths,
        strength[indices],
        rtol=NUCLEUS_RTOL,
        atol=NUCLEUS_ATOL,
    ):
        raise ValueError(
            "Candidate strengths do not match "
            "the Nucleus strength series"
        )

    if (
        isinstance(minimum_distance, bool)
        or not isinstance(
            minimum_distance,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Minimum Nucleus candidate distance "
            "must be an integer"
        )

    minimum_distance = int(
        minimum_distance
    )

    if minimum_distance < 1:
        raise ValueError(
            "Minimum Nucleus candidate distance "
            "must be at least 1"
        )

    if (
        indices.size > 1
        and np.any(
            np.diff(indices) < minimum_distance
        )
    ):
        raise ValueError(
            "Nucleus candidate indices violate "
            "the minimum distance"
        )

    return {
        "nucleus_strength": strength,
        "candidate_frame_indices": indices,
        "candidate_strengths": candidate_strengths,
    }


#8.18
def build_nucleus_metadata(
    source_metadata: ScaleProfileMetadata,
    boundary_metadata: BoundaryMetadata,
    configuration: NucleusConfiguration,
    candidates: NucleusCandidates,
    boundary_excluded_candidate_count: int,
) -> NucleusMetadata:
    """Build versioned metadata for Nucleus candidate outputs."""

    if (
        isinstance(
            boundary_excluded_candidate_count,
            bool,
        )
        or not isinstance(
            boundary_excluded_candidate_count,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "Boundary excluded candidate count "
            "must be an integer"
        )

    boundary_excluded_candidate_count = int(
        boundary_excluded_candidate_count
    )

    if boundary_excluded_candidate_count < 0:
        raise ValueError(
            "Boundary excluded candidate count "
            "must be non-negative"
        )

    validated_configuration = (
        validate_nucleus_configuration(
            configuration
        )
    )

    validated_candidates = (
        validate_nucleus_candidates(
            candidates,
            minimum_distance=validated_configuration[
                "minimum_candidate_distance"
            ],
        )
    )

    strength = validated_candidates[
        "nucleus_strength"
    ]
    candidate_indices = validated_candidates[
        "candidate_frame_indices"
    ]
    candidate_strengths = validated_candidates[
        "candidate_strengths"
    ]

    source_shape_summary = source_metadata[
        "shape_summary"
    ]
    source_cwt = source_metadata[
        "source_cwt"
    ]

    boundary_shape_summary = boundary_metadata[
        "shape_summary"
    ]
    boundary_configuration = boundary_metadata[
        "boundary_configuration"
    ]

    if candidate_strengths.size > 0:
        minimum_candidate_strength = float(
            np.min(candidate_strengths)
        )
        maximum_candidate_strength = float(
            np.max(candidate_strengths)
        )
    else:
        minimum_candidate_strength = 0.0
        maximum_candidate_strength = 0.0

    metadata: NucleusMetadata = {
        "format_version": NUCLEUS_FORMAT_VERSION,
        "source_scale_profile": {
            "metadata_format_version": int(
                source_metadata["format_version"]
            ),
            "number_of_scales": int(
                source_shape_summary[
                    "number_of_scales"
                ]
            ),
            "number_of_frames": int(
                source_shape_summary[
                    "number_of_frames"
                ]
            ),
            "minimum_frequency_hz": float(
                source_cwt[
                    "minimum_frequency_hz"
                ]
            ),
            "maximum_frequency_hz": float(
                source_cwt[
                    "maximum_frequency_hz"
                ]
            ),
        },
        "source_boundary": {
            "metadata_format_version": int(
                boundary_metadata[
                    "format_version"
                ]
            ),
            "number_of_frames": int(
                boundary_shape_summary[
                    "number_of_frames"
                ]
            ),
            "number_of_candidates": int(
                boundary_shape_summary[
                    "number_of_candidates"
                ]
            ),
            "threshold": float(
                boundary_configuration[
                    "threshold"
                ]
            ),
            "minimum_candidate_distance": int(
                boundary_configuration[
                    "minimum_candidate_distance"
                ]
            ),
        },
        "nucleus_configuration": {
            "normalization": (
                validated_configuration[
                    "normalization"
                ]
            ),
            "local_contrast_radius": int(
                validated_configuration[
                    "local_contrast_radius"
                ]
            ),
            "threshold": float(
                validated_configuration[
                    "threshold"
                ]
            ),
            "minimum_candidate_distance": int(
                validated_configuration[
                    "minimum_candidate_distance"
                ]
            ),
            "boundary_exclusion_distance": int(
                validated_configuration[
                    "boundary_exclusion_distance"
                ]
            ),
            "plateau_policy": (
                validated_configuration[
                    "plateau_policy"
                ]
            ),
            "feature_names": list(
                NUCLEUS_FEATURE_NAMES
            ),
            "feature_weights": dict(
                validated_configuration[
                    "feature_weights"
                ]
            ),
        },
        "shape_summary": {
            "number_of_frames": int(
                strength.size
            ),
            "number_of_candidates": int(
                candidate_indices.size
            ),
            "nucleus_strength_shape": list(
                strength.shape
            ),
            "candidate_index_shape": list(
                candidate_indices.shape
            ),
            "candidate_strength_shape": list(
                candidate_strengths.shape
            ),
        },
        "nucleus_summary": {
            "minimum_nucleus_strength": float(
                np.min(strength)
            ),
            "maximum_nucleus_strength": float(
                np.max(strength)
            ),
            "minimum_candidate_strength": (
                minimum_candidate_strength
            ),
            "maximum_candidate_strength": (
                maximum_candidate_strength
            ),
            "boundary_excluded_candidate_count": (
                boundary_excluded_candidate_count
            ),
        },
    }

    if set(metadata) != set(
        NUCLEUS_METADATA_REQUIRED_KEYS
    ):
        raise ValueError(
            "Nucleus metadata keys do not match "
            "the required schema"
        )

    try:
        json.dumps(
            metadata,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Nucleus metadata must be JSON serializable"
        ) from error

    return metadata


#8.19
def validate_nucleus_metadata(
    metadata: object,
    source_metadata: ScaleProfileMetadata,
    boundary_metadata: BoundaryMetadata,
    candidates: NucleusCandidates,
) -> NucleusMetadata:
    """Validate Nucleus metadata against source data and arrays."""

    if not isinstance(metadata, dict):
        raise TypeError(
            "Nucleus metadata root must be a dictionary"
        )

    if set(metadata) != set(
        NUCLEUS_METADATA_REQUIRED_KEYS
    ):
        raise ValueError(
            "Nucleus metadata keys do not match "
            "the required schema"
        )

    configuration_section = metadata[
        "nucleus_configuration"
    ]

    if not isinstance(
        configuration_section,
        dict,
    ):
        raise TypeError(
            "Nucleus configuration metadata "
            "must be a dictionary"
        )

    configuration: NucleusConfiguration = {
        "normalization": configuration_section[
            "normalization"
        ],
        "local_contrast_radius": (
            configuration_section[
                "local_contrast_radius"
            ]
        ),
        "threshold": configuration_section[
            "threshold"
        ],
        "minimum_candidate_distance": (
            configuration_section[
                "minimum_candidate_distance"
            ]
        ),
        "boundary_exclusion_distance": (
            configuration_section[
                "boundary_exclusion_distance"
            ]
        ),
        "plateau_policy": configuration_section[
            "plateau_policy"
        ],
        "feature_weights": configuration_section[
            "feature_weights"
        ],
    }

    validated_configuration = (
        validate_nucleus_configuration(
            configuration
        )
    )

    validated_candidates = (
        validate_nucleus_candidates(
            candidates,
            minimum_distance=validated_configuration[
                "minimum_candidate_distance"
            ],
        )
    )

    summary_section = metadata[
        "nucleus_summary"
    ]

    if not isinstance(
        summary_section,
        dict,
    ):
        raise TypeError(
            "Nucleus summary metadata "
            "must be a dictionary"
        )

    if (
        "boundary_excluded_candidate_count"
        not in summary_section
    ):
        raise KeyError(
            "boundary_excluded_candidate_count"
        )

    boundary_excluded_candidate_count = (
        summary_section[
            "boundary_excluded_candidate_count"
        ]
    )

    expected_metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        validated_configuration,
        validated_candidates,
        boundary_excluded_candidate_count=(
            boundary_excluded_candidate_count
        ),
    )

    if metadata != expected_metadata:
        raise ValueError(
            "Nucleus metadata does not match "
            "the source data and candidate arrays"
        )

    validated_metadata = deepcopy(
        expected_metadata
    )

    try:
        json.dumps(
            validated_metadata,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Nucleus metadata must be JSON serializable"
        ) from error

    return validated_metadata


#8.20
def extract_nucleus_candidates(
    scale_profile: ScaleProfile,
    source_metadata: ScaleProfileMetadata,
    boundary_candidates: BoundaryCandidates,
    boundary_metadata: BoundaryMetadata,
    configuration: NucleusConfiguration,
) -> tuple[
    NucleusCandidates,
    NucleusMetadata,
]:
    """Extract validated Nucleus candidates from persisted inputs."""

    (
        validated_profile,
        validated_source_metadata,
    ) = validate_loaded_scale_profile(
        scale_profile,
        source_metadata,
    )

    (
        validated_boundary_candidates,
        validated_boundary_metadata,
    ) = validate_loaded_boundary_outputs(
        boundary_candidates,
        boundary_metadata,
        validated_source_metadata,
    )

    validated_configuration = (
        validate_nucleus_configuration(
            configuration
        )
    )

    evidence = build_nucleus_evidence(
        validated_profile,
        validated_configuration,
    )

    nucleus_strength = compute_nucleus_strength(
        evidence,
        validated_configuration,
    )

    nucleus_strength = validate_nucleus_strength(
        nucleus_strength,
        expected_frame_count=validated_profile[
            "magnitude"
        ].shape[1],
    )

    local_maxima = find_nucleus_local_maxima(
        nucleus_strength,
        plateau_policy=validated_configuration[
            "plateau_policy"
        ],
    )

    thresholded_indices = apply_nucleus_threshold(
        nucleus_strength,
        local_maxima,
        threshold=validated_configuration[
            "threshold"
        ],
    )

    separated_indices = (
        enforce_minimum_nucleus_candidate_distance(
            nucleus_strength,
            thresholded_indices,
            minimum_distance=validated_configuration[
                "minimum_candidate_distance"
            ],
        )
    )

    (
        final_indices,
        boundary_excluded_candidate_count,
    ) = exclude_boundary_adjacent_nucleus_candidates(
        separated_indices,
        validated_boundary_candidates[
            "candidate_frame_indices"
        ],
        exclusion_distance=validated_configuration[
            "boundary_exclusion_distance"
        ],
        frame_count=nucleus_strength.size,
    )

    candidates = build_nucleus_candidates(
        nucleus_strength,
        final_indices,
    )

    validated_candidates = validate_nucleus_candidates(
        candidates,
        minimum_distance=validated_configuration[
            "minimum_candidate_distance"
        ],
    )

    metadata = build_nucleus_metadata(
        validated_source_metadata,
        validated_boundary_metadata,
        validated_configuration,
        validated_candidates,
        boundary_excluded_candidate_count=(
            boundary_excluded_candidate_count
        ),
    )

    validated_metadata = validate_nucleus_metadata(
        metadata,
        validated_source_metadata,
        validated_boundary_metadata,
        validated_candidates,
    )

    return (
        validated_candidates,
        validated_metadata,
    )


#8.21
def save_nucleus_outputs(
    output_path: Path,
    metadata_path: Path,
    candidates: NucleusCandidates,
    metadata: NucleusMetadata,
    source_metadata: ScaleProfileMetadata,
    boundary_metadata: BoundaryMetadata,
) -> None:
    """Persist validated Nucleus candidate arrays and metadata."""

    if not isinstance(
        output_path,
        Path,
    ):
        raise TypeError(
            "Nucleus output path must be a Path"
        )

    if not isinstance(
        metadata_path,
        Path,
    ):
        raise TypeError(
            "Nucleus metadata path must be a Path"
        )

    configuration_section = metadata.get(
        "nucleus_configuration"
    )

    if not isinstance(
        configuration_section,
        dict,
    ):
        raise TypeError(
            "Nucleus configuration metadata "
            "must be a dictionary"
        )

    if (
        "minimum_candidate_distance"
        not in configuration_section
    ):
        raise KeyError(
            "minimum_candidate_distance"
        )

    minimum_distance = configuration_section[
        "minimum_candidate_distance"
    ]

    validated_candidates = validate_nucleus_candidates(
        candidates,
        minimum_distance=minimum_distance,
    )

    validated_metadata = validate_nucleus_metadata(
        metadata,
        source_metadata,
        boundary_metadata,
        validated_candidates,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        nucleus_strength=validated_candidates[
            "nucleus_strength"
        ],
        candidate_frame_indices=validated_candidates[
            "candidate_frame_indices"
        ],
        candidate_strengths=validated_candidates[
            "candidate_strengths"
        ],
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            validated_metadata,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )

        stream.write("\n")


#8.22
def build_parser() -> argparse.ArgumentParser:
    """Build the Nucleus candidate CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Discover Nucleus candidates from persisted "
            "ScaleProfile and Boundary candidate features."
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
        "--boundary-input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--boundary-metadata-input",
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
    parser.add_argument(
        "--local-contrast-radius",
        type=int,
        default=DEFAULT_LOCAL_CONTRAST_RADIUS,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_NUCLEUS_THRESHOLD,
    )
    parser.add_argument(
        "--minimum-distance",
        type=int,
        default=(
            DEFAULT_MINIMUM_NUCLEUS_CANDIDATE_DISTANCE
        ),
    )
    parser.add_argument(
        "--boundary-exclusion-distance",
        type=int,
        default=DEFAULT_BOUNDARY_EXCLUSION_DISTANCE,
    )
    parser.add_argument(
        "--frame-total-weight",
        type=float,
        default=DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
            "frame_total"
        ],
    )
    parser.add_argument(
        "--frame-mean-weight",
        type=float,
        default=DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
            "frame_mean"
        ],
    )
    parser.add_argument(
        "--frame-max-weight",
        type=float,
        default=DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
            "frame_max"
        ],
    )
    parser.add_argument(
        "--dominant-frequency-weight",
        type=float,
        default=DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
            "dominant_frequency_hz"
        ],
    )
    parser.add_argument(
        "--frequency-centroid-weight",
        type=float,
        default=DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
            "frequency_centroid_hz"
        ],
    )
    parser.add_argument(
        "--frequency-dispersion-weight",
        type=float,
        default=DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
            "frequency_dispersion_hz"
        ],
    )

    return parser


def default_nucleus_metadata_path(
    output_path: Path,
) -> Path:
    """Construct the default Nucleus metadata path."""

    return output_path.with_name(
        f"{output_path.stem}_metadata.json"
    )


#8.23
def validate_arguments(
    args: argparse.Namespace,
) -> tuple[
    Path,
    NucleusConfiguration,
]:
    """Validate Nucleus CLI arguments."""

    if not args.input.is_file():
        raise FileNotFoundError(
            "ScaleProfile archive does not exist: "
            f"{args.input}"
        )

    if not args.metadata_input.is_file():
        raise FileNotFoundError(
            "ScaleProfile metadata does not exist: "
            f"{args.metadata_input}"
        )

    if not args.boundary_input.is_file():
        raise FileNotFoundError(
            "Boundary archive does not exist: "
            f"{args.boundary_input}"
        )

    if not args.boundary_metadata_input.is_file():
        raise FileNotFoundError(
            "Boundary metadata does not exist: "
            f"{args.boundary_metadata_input}"
        )

    if args.input.suffix.lower() != ".npz":
        raise ValueError(
            "--input must use the .npz extension"
        )

    if args.metadata_input.suffix.lower() != ".json":
        raise ValueError(
            "--metadata-input must use the .json extension"
        )

    if args.boundary_input.suffix.lower() != ".npz":
        raise ValueError(
            "--boundary-input must use the .npz extension"
        )

    if (
        args.boundary_metadata_input.suffix.lower()
        != ".json"
    ):
        raise ValueError(
            "--boundary-metadata-input must use "
            "the .json extension"
        )

    if args.output.suffix.lower() != ".npz":
        raise ValueError(
            "--output must use the .npz extension"
        )

    metadata_output = args.metadata_output

    if metadata_output is None:
        metadata_output = (
            default_nucleus_metadata_path(
                args.output
            )
        )

    if metadata_output.suffix.lower() != ".json":
        raise ValueError(
            "--metadata-output must use "
            "the .json extension"
        )

    resolved_paths = [
        args.input.resolve(),
        args.metadata_input.resolve(),
        args.boundary_input.resolve(),
        args.boundary_metadata_input.resolve(),
        args.output.resolve(),
        metadata_output.resolve(),
    ]

    if len(set(resolved_paths)) != len(
        resolved_paths
    ):
        raise ValueError(
            "CLI input and output paths must be different"
        )

    configuration = (
        build_default_nucleus_configuration()
    )

    configuration["local_contrast_radius"] = (
        args.local_contrast_radius
    )
    configuration["threshold"] = args.threshold
    configuration[
        "minimum_candidate_distance"
    ] = args.minimum_distance
    configuration[
        "boundary_exclusion_distance"
    ] = args.boundary_exclusion_distance

    configuration["feature_weights"] = {
        "frame_total": args.frame_total_weight,
        "frame_mean": args.frame_mean_weight,
        "frame_max": args.frame_max_weight,
        "dominant_frequency_hz": (
            args.dominant_frequency_weight
        ),
        "frequency_centroid_hz": (
            args.frequency_centroid_weight
        ),
        "frequency_dispersion_hz": (
            args.frequency_dispersion_weight
        ),
    }

    validated_configuration = (
        validate_nucleus_configuration(
            configuration
        )
    )

    return (
        metadata_output,
        validated_configuration,
    )



def main() -> int:
    """Run Nucleus candidate discovery and persistence."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        metadata_output, configuration = (
            validate_arguments(args)
        )

        scale_profile, source_metadata = (
            load_scale_profile_outputs(
                args.input,
                args.metadata_input,
            )
        )

        (
            boundary_candidates,
            boundary_metadata,
        ) = load_boundary_outputs(
            args.boundary_input,
            args.boundary_metadata_input,
        )

        candidates, metadata = (
            extract_nucleus_candidates(
                scale_profile,
                source_metadata,
                boundary_candidates,
                boundary_metadata,
                configuration,
            )
        )

        save_nucleus_outputs(
            args.output,
            metadata_output,
            candidates,
            metadata,
            source_metadata,
            boundary_metadata,
        )

        print("Nucleus candidate extraction: OK")
        print(
            "frames:",
            candidates["nucleus_strength"].size,
        )
        print(
            "candidates:",
            candidates[
                "candidate_frame_indices"
            ].size,
        )
        print(
            "boundary excluded:",
            metadata["nucleus_summary"][
                "boundary_excluded_candidate_count"
            ],
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
    ) as error:
        print(
            f"error: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())