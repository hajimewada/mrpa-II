#!/usr/bin/env python3
"""Discover boundary candidates from validated ScaleProfile features."""

from __future__ import annotations

import argparse
import sys

import json
from pathlib import Path
from copy import deepcopy

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from analysis.f0.extract_scale_profile import (
    SCALE_PROFILE_REQUIRED_KEYS,
    ScaleProfile,
    ScaleProfileMetadata,
    validate_scale_profile_arrays,
    validate_scale_profile_metadata,
)

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]


class BoundaryEvidence(TypedDict):
    """Feature-specific evidence for prosodic boundaries."""

    frame_total_delta: Float64Array
    frame_mean_delta: Float64Array
    frame_max_delta: Float64Array
    dominant_frequency_delta_hz: Float64Array
    frequency_centroid_delta_hz: Float64Array
    frequency_dispersion_delta_hz: Float64Array


class BoundaryCandidates(TypedDict):
    """Boundary strength and extracted candidate locations."""

    boundary_strength: Float64Array
    candidate_frame_indices: Int64Array
    candidate_strengths: Float64Array


class SourceScaleProfileMetadata(TypedDict):
    """Summary of the source ScaleProfile representation."""

    metadata_format_version: int
    number_of_scales: int
    number_of_frames: int
    minimum_frequency_hz: float
    maximum_frequency_hz: float


class BoundaryConfigurationMetadata(TypedDict):
    """Configuration used for boundary candidate discovery."""

    normalization: str
    threshold: float
    minimum_candidate_distance: int
    plateau_policy: str
    feature_names: list[str]
    feature_weights: dict[str, float]


class BoundaryShapeMetadata(TypedDict):
    """Shapes of persisted Boundary candidate arrays."""

    number_of_frames: int
    number_of_candidates: int
    boundary_strength_shape: list[int]
    candidate_index_shape: list[int]
    candidate_strength_shape: list[int]


class BoundarySummaryMetadata(TypedDict):
    """Aggregate Boundary strength and candidate statistics."""

    minimum_boundary_strength: float
    maximum_boundary_strength: float
    minimum_candidate_strength: float
    maximum_candidate_strength: float


class BoundaryMetadata(TypedDict):
    """Top-level persisted Boundary candidate metadata."""

    format_version: int
    source_scale_profile: SourceScaleProfileMetadata
    boundary_configuration: BoundaryConfigurationMetadata
    shape_summary: BoundaryShapeMetadata
    boundary_summary: BoundarySummaryMetadata


BOUNDARY_FORMAT_VERSION = 1

BOUNDARY_NORMALIZATION = "median-mad"
BOUNDARY_PLATEAU_POLICY = "first-index"

BOUNDARY_RTOL = 1e-12
BOUNDARY_ATOL = 1e-12

BOUNDARY_FEATURE_NAMES = (
    "frame_total",
    "frame_mean",
    "frame_max",
    "dominant_frequency_hz",
    "frequency_centroid_hz",
    "frequency_dispersion_hz",
)

BOUNDARY_EVIDENCE_REQUIRED_KEYS = (
    "frame_total_delta",
    "frame_mean_delta",
    "frame_max_delta",
    "dominant_frequency_delta_hz",
    "frequency_centroid_delta_hz",
    "frequency_dispersion_delta_hz",
)

BOUNDARY_CANDIDATE_REQUIRED_KEYS = (
    "boundary_strength",
    "candidate_frame_indices",
    "candidate_strengths",
)

BOUNDARY_METADATA_REQUIRED_KEYS = (
    "format_version",
    "source_scale_profile",
    "boundary_configuration",
    "shape_summary",
    "boundary_summary",
)


class BoundaryConfiguration(TypedDict):
    """Runtime configuration for boundary candidate discovery."""

    normalization: str
    threshold: float
    minimum_candidate_distance: int
    plateau_policy: str
    feature_weights: dict[str, float]


DEFAULT_BOUNDARY_THRESHOLD = 1.0
DEFAULT_MINIMUM_CANDIDATE_DISTANCE = 1

DEFAULT_BOUNDARY_FEATURE_WEIGHTS = {
    name: 1.0
    for name in BOUNDARY_FEATURE_NAMES
}


def build_default_boundary_configuration(
) -> BoundaryConfiguration:
    """Build an independent default Boundary configuration."""

    return {
        "normalization": BOUNDARY_NORMALIZATION,
        "threshold": DEFAULT_BOUNDARY_THRESHOLD,
        "minimum_candidate_distance": (
            DEFAULT_MINIMUM_CANDIDATE_DISTANCE
        ),
        "plateau_policy": BOUNDARY_PLATEAU_POLICY,
        "feature_weights": dict(
            DEFAULT_BOUNDARY_FEATURE_WEIGHTS
        ),
    }


def validate_boundary_configuration(
    configuration: BoundaryConfiguration,
) -> BoundaryConfiguration:
    """Validate and copy a Boundary configuration."""

    if configuration["normalization"] != BOUNDARY_NORMALIZATION:
        raise ValueError(
            "Unsupported Boundary normalization"
        )

    if configuration["plateau_policy"] != BOUNDARY_PLATEAU_POLICY:
        raise ValueError(
            "Unsupported Boundary plateau policy"
        )

    threshold = configuration["threshold"]

    if isinstance(threshold, bool):
        raise TypeError(
            "Boundary threshold must be a real number"
        )

    threshold = float(threshold)

    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "Boundary threshold must be finite and non-negative"
        )

    minimum_distance = configuration[
        "minimum_candidate_distance"
    ]

    if isinstance(minimum_distance, bool) or not isinstance(
        minimum_distance,
        (int, np.integer),
    ):
        raise TypeError(
            "Minimum candidate distance must be an integer"
        )

    minimum_distance = int(minimum_distance)

    if minimum_distance < 1:
        raise ValueError(
            "Minimum candidate distance must be at least 1"
        )

    weights = configuration["feature_weights"]

    if set(weights) != set(BOUNDARY_FEATURE_NAMES):
        raise ValueError(
            "Boundary feature weights must match "
            "the required feature names"
        )

    validated_weights: dict[str, float] = {}

    for name in BOUNDARY_FEATURE_NAMES:
        value = weights[name]

        if isinstance(value, bool):
            raise TypeError(
                f"Boundary weight for {name} must be a real number"
            )

        value = float(value)

        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Boundary weight for {name} must be "
                "finite and non-negative"
            )

        validated_weights[name] = value

    if not any(
        value > 0.0
        for value in validated_weights.values()
    ):
        raise ValueError(
            "At least one Boundary feature weight must be positive"
        )

    return {
        "normalization": BOUNDARY_NORMALIZATION,
        "threshold": threshold,
        "minimum_candidate_distance": minimum_distance,
        "plateau_policy": BOUNDARY_PLATEAU_POLICY,
        "feature_weights": validated_weights,
    }


def load_scale_profile_outputs(
    input_path: Path,
    metadata_path: Path,
) -> tuple[
    ScaleProfile,
    dict[str, object],
]:
    """Load persisted ScaleProfile arrays and metadata."""

    if not input_path.is_file():
        raise FileNotFoundError(
            f"ScaleProfile archive does not exist: {input_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"ScaleProfile metadata does not exist: {metadata_path}"
        )

    with np.load(
        input_path,
        allow_pickle=False,
    ) as archive:
        missing_keys = [
            key
            for key in SCALE_PROFILE_REQUIRED_KEYS
            if key not in archive.files
        ]

        if missing_keys:
            raise KeyError(
                "ScaleProfile archive is missing required arrays: "
                + ", ".join(missing_keys)
            )

        profile = {
            key: np.array(
                archive[key],
                copy=True,
            )
            for key in SCALE_PROFILE_REQUIRED_KEYS
        }

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    return profile, metadata


def validate_loaded_scale_profile(
    profile: ScaleProfile,
    metadata: object,
) -> tuple[
    ScaleProfile,
    ScaleProfileMetadata,
]:
    """Validate loaded ScaleProfile arrays and metadata."""

    validated_profile = validate_scale_profile_arrays(
        profile
    )

    validated_metadata = validate_scale_profile_metadata(
        metadata,
        scale_profile=validated_profile,
    )

    return validated_profile, validated_metadata


def compute_absolute_frame_difference(
    values: np.ndarray,
) -> Float64Array:
    """Compute absolute differences between adjacent frames."""

    validated_values = np.array(
        values,
        dtype=np.float64,
        copy=True,
    )

    if validated_values.ndim != 1:
        raise ValueError(
            "Frame series must be one-dimensional"
        )

    if validated_values.size < 1:
        raise ValueError(
            "Frame series must contain at least one value"
        )

    if not np.all(np.isfinite(validated_values)):
        raise ValueError(
            "Frame series must contain only finite values"
        )

    difference = np.zeros(
        validated_values.shape,
        dtype=np.float64,
    )

    difference[1:] = np.abs(
        np.diff(validated_values)
    )

    if not np.all(np.isfinite(difference)):
        raise ValueError(
            "Frame difference must contain only finite values"
        )

    return difference


def compute_robust_normalized_series(
    values: np.ndarray,
) -> Float64Array:
    """Normalize a frame series using its median and MAD."""

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

    median = float(np.median(series))

    mad = float(
        np.median(
            np.abs(series - median)
        )
    )

    if mad == 0.0:
        return np.zeros(
            series.shape,
            dtype=np.float64,
        )

    normalized = np.array(
        (series - median) / mad,
        dtype=np.float64,
        copy=True,
    )

    normalized = np.maximum(
        normalized,
        0.0,
    )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "Normalized frame series must contain "
            "only finite values"
        )

    return normalized


def build_boundary_evidence(
    scale_profile: ScaleProfile,
) -> BoundaryEvidence:
    """Build normalized Boundary evidence from frame features."""

    feature_mapping = {
        "frame_total_delta": "frame_total",
        "frame_mean_delta": "frame_mean",
        "frame_max_delta": "frame_max",
        "dominant_frequency_delta_hz": (
            "dominant_frequency_hz"
        ),
        "frequency_centroid_delta_hz": (
            "frequency_centroid_hz"
        ),
        "frequency_dispersion_delta_hz": (
            "frequency_dispersion_hz"
        ),
    }

    evidence: BoundaryEvidence = {}

    expected_frame_count = int(
        scale_profile["magnitude"].shape[1]
    )

    for evidence_name, profile_name in (
        feature_mapping.items()
    ):
        difference = compute_absolute_frame_difference(
            scale_profile[profile_name]
        )

        normalized = compute_robust_normalized_series(
            difference
        )

        if normalized.shape != (
            expected_frame_count,
        ):
            raise ValueError(
                f"{evidence_name} has an unexpected shape"
            )

        evidence[evidence_name] = normalized

    if set(evidence) != set(
        BOUNDARY_EVIDENCE_REQUIRED_KEYS
    ):
        raise ValueError(
            "Boundary evidence keys do not match "
            "the required schema"
        )

    return evidence


def compute_boundary_strength(
    evidence: BoundaryEvidence,
    configuration: BoundaryConfiguration,
) -> Float64Array:
    """Combine feature evidence into one Boundary strength series."""

    validated_configuration = (
        validate_boundary_configuration(
            configuration
        )
    )

    evidence_mapping = {
        "frame_total": "frame_total_delta",
        "frame_mean": "frame_mean_delta",
        "frame_max": "frame_max_delta",
        "dominant_frequency_hz": (
            "dominant_frequency_delta_hz"
        ),
        "frequency_centroid_hz": (
            "frequency_centroid_delta_hz"
        ),
        "frequency_dispersion_hz": (
            "frequency_dispersion_delta_hz"
        ),
    }

    if set(evidence) != set(
        BOUNDARY_EVIDENCE_REQUIRED_KEYS
    ):
        raise ValueError(
            "Boundary evidence keys do not match "
            "the required schema"
        )

    first_values = np.asarray(
        evidence["frame_total_delta"]
    )

    if first_values.ndim != 1:
        raise ValueError(
            "Boundary evidence must be one-dimensional"
        )

    frame_count = first_values.size

    if frame_count < 1:
        raise ValueError(
            "Boundary evidence must contain at least one frame"
        )

    strength = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    total_weight = 0.0

    for feature_name in BOUNDARY_FEATURE_NAMES:
        evidence_name = evidence_mapping[
            feature_name
        ]

        values = np.array(
            evidence[evidence_name],
            dtype=np.float64,
            copy=True,
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

        strength += weight * values
        total_weight += weight

    strength = strength / total_weight

    if not np.all(np.isfinite(strength)):
        raise ValueError(
            "Boundary strength must contain only finite values"
        )

    if np.any(strength < 0.0):
        raise ValueError(
            "Boundary strength must be non-negative"
        )

    return strength

#7.9
def validate_boundary_strength(
    boundary_strength: np.ndarray,
    expected_frame_count: int,
) -> Float64Array:
    """Validate a Boundary strength series."""

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
        boundary_strength,
        dtype=np.float64,
        copy=True,
    )

    if validated.ndim != 1:
        raise ValueError(
            "Boundary strength must be one-dimensional"
        )

    if validated.shape != (
        expected_frame_count,
    ):
        raise ValueError(
            "Boundary strength shape does not match "
            "the expected frame count"
        )

    if not np.all(np.isfinite(validated)):
        raise ValueError(
            "Boundary strength must contain only finite values"
        )

    if np.any(validated < 0.0):
        raise ValueError(
            "Boundary strength must be non-negative"
        )

    if validated[0] != 0.0:
        raise ValueError(
            "Boundary strength must start with zero"
        )

    return validated


#7.10
def find_local_maxima(
    boundary_strength: np.ndarray,
    plateau_policy: str = BOUNDARY_PLATEAU_POLICY,
) -> Int64Array:
    """Find local maxima using the configured plateau policy."""

    strength = validate_boundary_strength(
        boundary_strength,
        expected_frame_count=np.asarray(
            boundary_strength
        ).size,
    )

    if plateau_policy != BOUNDARY_PLATEAU_POLICY:
        raise ValueError(
            "Unsupported Boundary plateau policy"
        )

    if strength.size < 3:
        return np.array(
            [],
            dtype=np.int64,
        )

    candidate_indices: list[int] = []
    index = 1
    final_index = strength.size - 1

    while index < final_index:
        if strength[index] <= strength[index - 1]:
            index += 1
            continue

        plateau_end = index

        while (
            plateau_end + 1 < strength.size
            and strength[plateau_end + 1]
            == strength[index]
        ):
            plateau_end += 1

        if (
            plateau_end < final_index
            and strength[index]
            > strength[plateau_end + 1]
        ):
            candidate_indices.append(index)

        index = plateau_end + 1

    return np.array(
        candidate_indices,
        dtype=np.int64,
    )


#7.11
def apply_boundary_threshold(
    boundary_strength: np.ndarray,
    candidate_indices: np.ndarray,
    threshold: float,
) -> Int64Array:
    """Filter Boundary candidates using an inclusive threshold."""

    strength = validate_boundary_strength(
        boundary_strength,
        expected_frame_count=np.asarray(
            boundary_strength
        ).size,
    )

    indices = np.array(
        candidate_indices,
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

    if isinstance(threshold, bool):
        raise TypeError(
            "Boundary threshold must be a real number"
        )

    threshold = float(threshold)

    if not np.isfinite(threshold):
        raise ValueError(
            "Boundary threshold must be finite"
        )

    if threshold < 0.0:
        raise ValueError(
            "Boundary threshold must be non-negative"
        )

    selected = indices[
        strength[indices] >= threshold
    ]

    return np.array(
        selected,
        dtype=np.int64,
        copy=True,
    )


#7.12
def enforce_minimum_candidate_distance(
    boundary_strength: np.ndarray,
    candidate_indices: np.ndarray,
    minimum_distance: int,
) -> Int64Array:
    """Suppress nearby candidates, keeping stronger candidates."""

    strength = validate_boundary_strength(
        boundary_strength,
        expected_frame_count=np.asarray(
            boundary_strength
        ).size,
    )

    indices = np.array(
        candidate_indices,
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
            "Minimum candidate distance must be an integer"
        )

    minimum_distance = int(minimum_distance)

    if minimum_distance < 1:
        raise ValueError(
            "Minimum candidate distance must be at least 1"
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
            selected_indices.append(index)

    selected_indices.sort()

    return np.array(
        selected_indices,
        dtype=np.int64,
    )


#7.13
def build_boundary_candidates(
    boundary_strength: np.ndarray,
    candidate_indices: np.ndarray,
) -> BoundaryCandidates:
    """Build persistent Boundary candidate arrays."""

    strength = validate_boundary_strength(
        boundary_strength,
        expected_frame_count=np.asarray(
            boundary_strength
        ).size,
    )

    indices = np.array(
        candidate_indices,
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

    candidates: BoundaryCandidates = {
        "boundary_strength": np.array(
            strength,
            dtype=np.float64,
            copy=True,
        ),
        "candidate_frame_indices": indices,
        "candidate_strengths": candidate_strengths,
    }

    if set(candidates) != set(
        BOUNDARY_CANDIDATE_REQUIRED_KEYS
    ):
        raise ValueError(
            "Boundary candidate keys do not match "
            "the required schema"
        )

    return candidates


#7.14
def validate_boundary_candidates(
    candidates: BoundaryCandidates,
    minimum_distance: int,
) -> BoundaryCandidates:
    """Validate Boundary candidate arrays."""

    if set(candidates) != set(
        BOUNDARY_CANDIDATE_REQUIRED_KEYS
    ):
        raise ValueError(
            "Boundary candidate keys do not match "
            "the required schema"
        )

    strength = validate_boundary_strength(
        candidates["boundary_strength"],
        expected_frame_count=np.asarray(
            candidates["boundary_strength"]
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

    candidate_strengths = np.array(
        candidates["candidate_strengths"],
        dtype=np.float64,
        copy=True,
    )

    if indices.ndim != 1:
        raise ValueError(
            "Candidate indices must be one-dimensional"
        )

    if candidate_strengths.ndim != 1:
        raise ValueError(
            "Candidate strengths must be one-dimensional"
        )

    if indices.shape != candidate_strengths.shape:
        raise ValueError(
            "Candidate index and strength shapes must match"
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

    if not np.all(np.isfinite(candidate_strengths)):
        raise ValueError(
            "Candidate strengths must contain only finite values"
        )

    if np.any(candidate_strengths < 0.0):
        raise ValueError(
            "Candidate strengths must be non-negative"
        )

    if not np.allclose(
        candidate_strengths,
        strength[indices],
        rtol=BOUNDARY_RTOL,
        atol=BOUNDARY_ATOL,
    ):
        raise ValueError(
            "Candidate strengths do not match "
            "the Boundary strength series"
        )

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

    minimum_distance = int(minimum_distance)

    if minimum_distance < 1:
        raise ValueError(
            "Minimum candidate distance must be at least 1"
        )

    if (
        indices.size > 1
        and np.any(
            np.diff(indices) < minimum_distance
        )
    ):
        raise ValueError(
            "Candidate indices violate "
            "the minimum distance"
        )

    return {
        "boundary_strength": strength,
        "candidate_frame_indices": indices,
        "candidate_strengths": candidate_strengths,
    }


#7.15
def build_boundary_metadata(
    source_metadata: ScaleProfileMetadata,
    configuration: BoundaryConfiguration,
    candidates: BoundaryCandidates,
) -> BoundaryMetadata:
    """Build JSON-compatible Boundary candidate metadata."""

    validated_configuration = (
        validate_boundary_configuration(
            configuration
        )
    )

    minimum_distance = validated_configuration[
        "minimum_candidate_distance"
    ]

    validated_candidates = validate_boundary_candidates(
        candidates,
        minimum_distance=minimum_distance,
    )

    boundary_strength = validated_candidates[
        "boundary_strength"
    ]
    candidate_strengths = validated_candidates[
        "candidate_strengths"
    ]
    candidate_indices = validated_candidates[
        "candidate_frame_indices"
    ]

    source_summary = source_metadata[
        "shape_summary"
    ]
    source_cwt = source_metadata[
        "source_cwt"
    ]

    if candidate_strengths.size == 0:
        minimum_candidate_strength = 0.0
        maximum_candidate_strength = 0.0
    else:
        minimum_candidate_strength = float(
            np.min(candidate_strengths)
        )
        maximum_candidate_strength = float(
            np.max(candidate_strengths)
        )

    return {
        "format_version": BOUNDARY_FORMAT_VERSION,
        "source_scale_profile": {
            "metadata_format_version": int(
                source_metadata["format_version"]
            ),
            "number_of_scales": int(
                source_summary["number_of_scales"]
            ),
            "number_of_frames": int(
                source_summary["number_of_frames"]
            ),
            "minimum_frequency_hz": float(
                source_cwt["minimum_frequency_hz"]
            ),
            "maximum_frequency_hz": float(
                source_cwt["maximum_frequency_hz"]
            ),
        },
        "boundary_configuration": {
            "normalization": validated_configuration[
                "normalization"
            ],
            "threshold": float(
                validated_configuration["threshold"]
            ),
            "minimum_candidate_distance": int(
                minimum_distance
            ),
            "plateau_policy": validated_configuration[
                "plateau_policy"
            ],
            "feature_names": list(
                BOUNDARY_FEATURE_NAMES
            ),
            "feature_weights": dict(
                validated_configuration[
                    "feature_weights"
                ]
            ),
        },
        "shape_summary": {
            "number_of_frames": int(
                boundary_strength.size
            ),
            "number_of_candidates": int(
                candidate_indices.size
            ),
            "boundary_strength_shape": list(
                boundary_strength.shape
            ),
            "candidate_index_shape": list(
                candidate_indices.shape
            ),
            "candidate_strength_shape": list(
                candidate_strengths.shape
            ),
        },
        "boundary_summary": {
            "minimum_boundary_strength": float(
                np.min(boundary_strength)
            ),
            "maximum_boundary_strength": float(
                np.max(boundary_strength)
            ),
            "minimum_candidate_strength": (
                minimum_candidate_strength
            ),
            "maximum_candidate_strength": (
                maximum_candidate_strength
            ),
        },
    }


#7.16
def validate_boundary_metadata(
    metadata: object,
    source_metadata: ScaleProfileMetadata,
    candidates: BoundaryCandidates,
) -> BoundaryMetadata:
    """Validate Boundary metadata against candidate arrays."""

    if not isinstance(metadata, dict):
        raise TypeError(
            "Boundary metadata root must be a dictionary"
        )

    if set(metadata) != set(
        BOUNDARY_METADATA_REQUIRED_KEYS
    ):
        raise ValueError(
            "Boundary metadata keys do not match "
            "the required schema"
        )

    configuration_section = metadata[
        "boundary_configuration"
    ]

    if not isinstance(
        configuration_section,
        dict,
    ):
        raise TypeError(
            "Boundary configuration metadata "
            "must be a dictionary"
        )

    configuration: BoundaryConfiguration = {
        "normalization": configuration_section[
            "normalization"
        ],
        "threshold": configuration_section[
            "threshold"
        ],
        "minimum_candidate_distance": (
            configuration_section[
                "minimum_candidate_distance"
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
        validate_boundary_configuration(
            configuration
        )
    )

    validated_candidates = validate_boundary_candidates(
        candidates,
        minimum_distance=validated_configuration[
            "minimum_candidate_distance"
        ],
    )

    expected_metadata = build_boundary_metadata(
        source_metadata,
        validated_configuration,
        validated_candidates,
    )

    if metadata != expected_metadata:
        raise ValueError(
            "Boundary metadata does not match "
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
            "Boundary metadata must be JSON serializable"
        ) from error

    return validated_metadata


#7.17
def extract_boundary_candidates(
    scale_profile: ScaleProfile,
    source_metadata: ScaleProfileMetadata,
    configuration: BoundaryConfiguration,
) -> tuple[
    BoundaryCandidates,
    BoundaryMetadata,
]:
    """Extract validated Boundary candidates from a ScaleProfile."""

    validated_profile, validated_source_metadata = (
        validate_loaded_scale_profile(
            scale_profile,
            source_metadata,
        )
    )

    validated_configuration = (
        validate_boundary_configuration(
            configuration
        )
    )

    evidence = build_boundary_evidence(
        validated_profile
    )

    boundary_strength = compute_boundary_strength(
        evidence,
        validated_configuration,
    )

    boundary_strength = validate_boundary_strength(
        boundary_strength,
        expected_frame_count=validated_profile[
            "magnitude"
        ].shape[1],
    )

    candidate_indices = find_local_maxima(
        boundary_strength,
        plateau_policy=validated_configuration[
            "plateau_policy"
        ],
    )

    candidate_indices = apply_boundary_threshold(
        boundary_strength,
        candidate_indices,
        threshold=validated_configuration[
            "threshold"
        ],
    )

    candidate_indices = (
        enforce_minimum_candidate_distance(
            boundary_strength,
            candidate_indices,
            minimum_distance=validated_configuration[
                "minimum_candidate_distance"
            ],
        )
    )

    candidates = build_boundary_candidates(
        boundary_strength,
        candidate_indices,
    )

    validated_candidates = validate_boundary_candidates(
        candidates,
        minimum_distance=validated_configuration[
            "minimum_candidate_distance"
        ],
    )

    metadata = build_boundary_metadata(
        validated_source_metadata,
        validated_configuration,
        validated_candidates,
    )

    validated_metadata = validate_boundary_metadata(
        metadata,
        validated_source_metadata,
        validated_candidates,
    )

    return validated_candidates, validated_metadata


#7.18
def save_boundary_outputs(
    output_path: Path,
    metadata_path: Path,
    candidates: BoundaryCandidates,
    metadata: BoundaryMetadata,
    source_metadata: ScaleProfileMetadata,
) -> None:
    """Validate and save Boundary candidate outputs."""

    minimum_distance = metadata[
        "boundary_configuration"
    ]["minimum_candidate_distance"]

    validated_candidates = validate_boundary_candidates(
        candidates,
        minimum_distance=minimum_distance,
    )

    validated_metadata = validate_boundary_metadata(
        metadata,
        source_metadata,
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
        boundary_strength=validated_candidates[
            "boundary_strength"
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
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


#7.19
def build_parser() -> argparse.ArgumentParser:
    """Build the Boundary candidate CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Discover Boundary candidates from "
            "persisted ScaleProfile features."
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_BOUNDARY_THRESHOLD,
    )
    parser.add_argument(
        "--minimum-distance",
        type=int,
        default=DEFAULT_MINIMUM_CANDIDATE_DISTANCE,
    )

    return parser


def default_boundary_metadata_path(
    output_path: Path,
) -> Path:
    """Construct the default Boundary metadata path."""

    return output_path.with_name(
        f"{output_path.stem}_metadata.json"
    )


def validate_arguments(
    args: argparse.Namespace,
) -> tuple[
    Path,
    BoundaryConfiguration,
]:
    """Validate CLI arguments."""

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

    metadata_output = args.metadata_output

    if metadata_output is None:
        metadata_output = default_boundary_metadata_path(
            args.output
        )

    if metadata_output.suffix.lower() != ".json":
        raise ValueError(
            "--metadata-output must use the .json extension"
        )

    resolved_paths = [
        args.input.resolve(),
        args.metadata_input.resolve(),
        args.output.resolve(),
        metadata_output.resolve(),
    ]

    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError(
            "CLI input and output paths must be different"
        )

    configuration = (
        build_default_boundary_configuration()
    )
    configuration["threshold"] = args.threshold
    configuration[
        "minimum_candidate_distance"
    ] = args.minimum_distance

    validated_configuration = (
        validate_boundary_configuration(
            configuration
        )
    )

    return (
        metadata_output,
        validated_configuration,
    )


def main() -> int:
    """Run Boundary candidate discovery and persistence."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        metadata_output, configuration = (
            validate_arguments(args)
        )

        profile, source_metadata = (
            load_scale_profile_outputs(
                args.input,
                args.metadata_input,
            )
        )

        candidates, metadata = (
            extract_boundary_candidates(
                profile,
                source_metadata,
                configuration,
            )
        )

        save_boundary_outputs(
            args.output,
            metadata_output,
            candidates,
            metadata,
            source_metadata,
        )

        print("Boundary candidate extraction: OK")
        print(
            "frames:",
            candidates["boundary_strength"].size,
        )
        print(
            "candidates:",
            candidates[
                "candidate_frame_indices"
            ].size,
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