from __future__ import annotations

from argparse import Namespace
import subprocess
import sys

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.f0.extract_cwt import MINIMUM_FRAMES

from analysis.boundary.extract_boundary_candidates import (
    BOUNDARY_EVIDENCE_REQUIRED_KEYS,
    BOUNDARY_CANDIDATE_REQUIRED_KEYS,
    BOUNDARY_FEATURE_NAMES,
    BOUNDARY_METADATA_REQUIRED_KEYS,
    DEFAULT_BOUNDARY_THRESHOLD,
    DEFAULT_MINIMUM_CANDIDATE_DISTANCE,
    build_parser,
    apply_boundary_threshold,
    build_boundary_candidates,
    build_boundary_evidence,
    build_boundary_metadata,
    build_default_boundary_configuration,
    compute_absolute_frame_difference,
    compute_boundary_strength,
    compute_robust_normalized_series,
    default_boundary_metadata_path,
    enforce_minimum_candidate_distance,
    extract_boundary_candidates,
    find_local_maxima,
    load_scale_profile_outputs,
    save_boundary_outputs,
    validate_arguments,
    validate_loaded_scale_profile,
    validate_boundary_candidates,
    validate_boundary_metadata,
    validate_boundary_strength,
)

from analysis.f0.extract_scale_profile import (
    SCALE_PROFILE_REQUIRED_KEYS,
    build_scale_profile_metadata,
    extract_scale_profile,
    save_scale_profile_outputs,
)


def test_load_scale_profile_outputs_loads_persisted_data(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        expected_profile,
        expected_metadata,
    ) = create_persisted_scale_profile(
        tmp_path,
    )

    loaded_profile, loaded_metadata = (
        load_scale_profile_outputs(
            input_path,
            metadata_path,
        )
    )

    for key in SCALE_PROFILE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            loaded_profile[key],
            expected_profile[key],
        )

    assert loaded_metadata == expected_metadata


def test_load_scale_profile_outputs_rejects_missing_archive(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="archive does not exist",
    ):
        load_scale_profile_outputs(
            tmp_path / "missing.npz",
            metadata_path,
        )


def test_load_scale_profile_outputs_rejects_missing_metadata(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "profile.npz"
    np.savez_compressed(
        input_path,
        frequencies=np.array([1.0]),
    )

    with pytest.raises(
        FileNotFoundError,
        match="metadata does not exist",
    ):
        load_scale_profile_outputs(
            input_path,
            tmp_path / "missing.json",
        )


def create_persisted_scale_profile(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, np.ndarray], dict[str, object]]:
    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    coefficients = np.arange(
        3 * MINIMUM_FRAMES,
        dtype=np.float64,
    ).reshape(
        3,
        MINIMUM_FRAMES,
    )

    source_metadata = {
        "format_version": 1,
        "cwt_configuration": {
            "wavelet": "mexh",
            "sampling_period": 0.005,
            "sampling_frequency": 200.0,
            "method": "conv",
            "precision": 12,
        },
        "signal_summary": {},
        "scale_summary": {
            "wavelet": "mexh",
            "number_of_scales": 3,
            "minimum_frequency_hz": 5.0,
            "maximum_frequency_hz": 20.0,
        },
        "cwt_summary": {
            "n_scales": 3,
            "n_frames": MINIMUM_FRAMES,
            "frequency_min_hz": 5.0,
            "frequency_max_hz": 20.0,
            "coefficient_dtype": "float64",
        },
    }

    profile, metadata = extract_scale_profile(
        coefficients=coefficients,
        frequencies=frequencies,
        source_cwt_metadata=source_metadata,
    )

    input_path = tmp_path / "scale_profile.npz"
    metadata_path = tmp_path / "scale_profile_metadata.json"

    save_scale_profile_outputs(
        input_path,
        metadata_path,
        profile,
        metadata,
    )

    return input_path, metadata_path, profile, metadata


def test_validate_loaded_scale_profile_accepts_valid_data(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        expected_profile,
        expected_metadata,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    profile, metadata = load_scale_profile_outputs(
        input_path,
        metadata_path,
    )

    validated_profile, validated_metadata = (
        validate_loaded_scale_profile(
            profile,
            metadata,
        )
    )

    for key in SCALE_PROFILE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            validated_profile[key],
            expected_profile[key],
        )

    assert validated_metadata == expected_metadata


def test_validate_loaded_scale_profile_returns_independent_copies(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        loaded_profile,
        loaded_metadata,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    validated_profile, validated_metadata = (
        validate_loaded_scale_profile(
            loaded_profile,
            loaded_metadata,
        )
    )

    validated_profile["magnitude"][0, 0] = 999.0
    validated_metadata["shape_summary"][
        "number_of_frames"
    ] = 999

    assert loaded_profile["magnitude"][0, 0] != 999.0
    assert loaded_metadata["shape_summary"][
        "number_of_frames"
    ] != 999


def test_validate_loaded_scale_profile_rejects_invalid_profile(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        profile,
        metadata,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    profile["magnitude"][0, 0] = -1.0

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        validate_loaded_scale_profile(
            profile,
            metadata,
        )


def test_compute_absolute_frame_difference_returns_expected_values():
    values = np.array(
        [1.0, 4.0, 2.0, 2.0],
        dtype=np.float64,
    )

    result = compute_absolute_frame_difference(
        values
    )

    expected = np.array(
        [0.0, 3.0, 2.0, 0.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_compute_absolute_frame_difference_accepts_single_value():
    result = compute_absolute_frame_difference(
        np.array(
            [5.0],
            dtype=np.float64,
        )
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [0.0],
            dtype=np.float64,
        ),
    )


def test_compute_absolute_frame_difference_returns_independent_float64():
    values = np.array(
        [1, 3, 6],
        dtype=np.int64,
    )

    result = compute_absolute_frame_difference(
        values
    )

    result[0] = 999.0

    assert values[0] == 1
    assert result.dtype == np.float64
    assert not np.shares_memory(
        result,
        values,
    )


def test_compute_absolute_frame_difference_rejects_non_1d_input():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        compute_absolute_frame_difference(
            np.zeros(
                (2, 3),
                dtype=np.float64,
            )
        )


def test_compute_absolute_frame_difference_rejects_empty_input():
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        compute_absolute_frame_difference(
            np.array(
                [],
                dtype=np.float64,
            )
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_compute_absolute_frame_difference_rejects_nonfinite_input(
    invalid_value: float,
):
    values = np.array(
        [1.0, invalid_value, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        compute_absolute_frame_difference(
            values
        )


def test_compute_robust_normalized_series_returns_expected_values():
    values = np.array(
        [0.0, 1.0, 2.0, 5.0],
        dtype=np.float64,
    )

    result = compute_robust_normalized_series(
        values
    )

    expected = np.array(
        [0.0, 0.0, 0.5, 3.5],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_compute_robust_normalized_series_handles_zero_mad():
    values = np.full(
        5,
        3.0,
        dtype=np.float64,
    )

    result = compute_robust_normalized_series(
        values
    )

    np.testing.assert_array_equal(
        result,
        np.zeros(5),
    )


def test_compute_robust_normalized_series_returns_independent_float64():
    values = np.array(
        [0, 1, 2, 5],
        dtype=np.int64,
    )

    result = compute_robust_normalized_series(
        values
    )

    result[0] = 999.0

    assert values[0] == 0
    assert result.dtype == np.float64
    assert not np.shares_memory(
        result,
        values,
    )


def test_compute_robust_normalized_series_rejects_non_1d_input():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        compute_robust_normalized_series(
            np.zeros(
                (2, 3),
                dtype=np.float64,
            )
        )


def test_compute_robust_normalized_series_rejects_empty_input():
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        compute_robust_normalized_series(
            np.array(
                [],
                dtype=np.float64,
            )
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_compute_robust_normalized_series_rejects_nonfinite_input(
    invalid_value: float,
):
    values = np.array(
        [0.0, invalid_value, 1.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        compute_robust_normalized_series(
            values
        )


def test_build_boundary_evidence_returns_required_arrays(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        profile,
        metadata,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    validated_profile, _ = (
        validate_loaded_scale_profile(
            profile,
            metadata,
        )
    )

    evidence = build_boundary_evidence(
        validated_profile
    )

    assert set(evidence) == set(
        BOUNDARY_EVIDENCE_REQUIRED_KEYS
    )

    frame_count = validated_profile[
        "magnitude"
    ].shape[1]

    for values in evidence.values():
        assert values.shape == (frame_count,)
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
        assert values[0] == 0.0


def test_build_boundary_evidence_matches_component_functions(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        profile,
        metadata,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    evidence = build_boundary_evidence(
        profile
    )

    expected = compute_robust_normalized_series(
        compute_absolute_frame_difference(
            profile["frame_total"]
        )
    )

    np.testing.assert_array_equal(
        evidence["frame_total_delta"],
        expected,
    )


def test_build_boundary_evidence_does_not_modify_profile(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        profile,
        metadata,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    original = profile["frame_total"].copy()

    evidence = build_boundary_evidence(
        profile
    )

    evidence["frame_total_delta"][0] = 999.0

    np.testing.assert_array_equal(
        profile["frame_total"],
        original,
    )

#7.8
def test_compute_boundary_strength_returns_weighted_mean():
    evidence = {
        key: np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        )
        for key in BOUNDARY_EVIDENCE_REQUIRED_KEYS
    }

    configuration = (
        build_default_boundary_configuration()
    )

    result = compute_boundary_strength(
        evidence,
        configuration,
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        ),
    )


def test_compute_boundary_strength_applies_feature_weights():
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in BOUNDARY_EVIDENCE_REQUIRED_KEYS
    }

    evidence["frame_total_delta"] = np.array(
        [0.0, 2.0, 4.0],
        dtype=np.float64,
    )

    configuration = (
        build_default_boundary_configuration()
    )

    for name in configuration["feature_weights"]:
        configuration["feature_weights"][name] = 0.0

    configuration["feature_weights"][
        "frame_total"
    ] = 2.0

    result = compute_boundary_strength(
        evidence,
        configuration,
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [0.0, 2.0, 4.0],
            dtype=np.float64,
        ),
    )


def test_compute_boundary_strength_rejects_shape_mismatch():
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in BOUNDARY_EVIDENCE_REQUIRED_KEYS
    }

    evidence["frame_mean_delta"] = np.zeros(
        2,
        dtype=np.float64,
    )

    configuration = (
        build_default_boundary_configuration()
    )

    with pytest.raises(
        ValueError,
        match="unexpected shape",
    ):
        compute_boundary_strength(
            evidence,
            configuration,
        )


#7.9
def test_validate_boundary_strength_accepts_valid_series():
    strength = np.array(
        [0.0, 1.0, 2.0, 0.5],
        dtype=np.float64,
    )

    validated = validate_boundary_strength(
        strength,
        expected_frame_count=4,
    )

    np.testing.assert_array_equal(
        validated,
        strength,
    )


def test_validate_boundary_strength_returns_independent_float64():
    strength = np.array(
        [0, 1, 2],
        dtype=np.int64,
    )

    validated = validate_boundary_strength(
        strength,
        expected_frame_count=3,
    )

    validated[1] = 999.0

    assert strength[1] == 1
    assert validated.dtype == np.float64
    assert not np.shares_memory(
        validated,
        strength,
    )


def test_validate_boundary_strength_rejects_non_1d_series():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_boundary_strength(
            np.zeros(
                (2, 3),
                dtype=np.float64,
            ),
            expected_frame_count=3,
        )


def test_validate_boundary_strength_rejects_frame_count_mismatch():
    with pytest.raises(
        ValueError,
        match="expected frame count",
    ):
        validate_boundary_strength(
            np.zeros(
                3,
                dtype=np.float64,
            ),
            expected_frame_count=4,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_boundary_strength_rejects_nonfinite_values(
    invalid_value: float,
):
    strength = np.array(
        [0.0, invalid_value, 1.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        validate_boundary_strength(
            strength,
            expected_frame_count=3,
        )


def test_validate_boundary_strength_rejects_negative_values():
    strength = np.array(
        [0.0, -1.0, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        validate_boundary_strength(
            strength,
            expected_frame_count=3,
        )


def test_validate_boundary_strength_rejects_nonzero_first_frame():
    strength = np.array(
        [1.0, 0.5, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="start with zero",
    ):
        validate_boundary_strength(
            strength,
            expected_frame_count=3,
        )


@pytest.mark.parametrize(
    "invalid_count",
    [
        True,
        3.5,
        "3",
    ],
)
def test_validate_boundary_strength_rejects_invalid_frame_count_type(
    invalid_count: object,
):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        validate_boundary_strength(
            np.zeros(
                3,
                dtype=np.float64,
            ),
            expected_frame_count=invalid_count,
        )


@pytest.mark.parametrize(
    "invalid_count",
    [
        0,
        -1,
    ],
)
def test_validate_boundary_strength_rejects_nonpositive_frame_count(
    invalid_count: int,
):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        validate_boundary_strength(
            np.zeros(
                1,
                dtype=np.float64,
            ),
            expected_frame_count=invalid_count,
        )


#7.10
def test_find_local_maxima_finds_isolated_peaks():
    strength = np.array(
        [0.0, 1.0, 3.0, 1.0, 2.0, 0.0],
        dtype=np.float64,
    )

    result = find_local_maxima(strength)

    np.testing.assert_array_equal(
        result,
        np.array(
            [2, 4],
            dtype=np.int64,
        ),
    )


def test_find_local_maxima_uses_first_plateau_index():
    strength = np.array(
        [0.0, 1.0, 3.0, 3.0, 3.0, 1.0],
        dtype=np.float64,
    )

    result = find_local_maxima(strength)

    np.testing.assert_array_equal(
        result,
        np.array(
            [2],
            dtype=np.int64,
        ),
    )


@pytest.mark.parametrize(
    "strength",
    [
        np.array(
            [0.0],
            dtype=np.float64,
        ),
        np.array(
            [0.0, 1.0],
            dtype=np.float64,
        ),
        np.array(
            [0.0, 1.0, 2.0, 3.0],
            dtype=np.float64,
        ),
        np.zeros(
            5,
            dtype=np.float64,
        ),
    ],
)
def test_find_local_maxima_returns_empty_array_without_internal_peak(
    strength: np.ndarray,
):
    result = find_local_maxima(strength)

    assert result.dtype == np.int64
    assert result.size == 0


def test_find_local_maxima_rejects_unsupported_plateau_policy():
    strength = np.array(
        [0.0, 2.0, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported Boundary plateau policy",
    ):
        find_local_maxima(
            strength,
            plateau_policy="last-index",
        )


#7.11
def test_apply_boundary_threshold_keeps_qualifying_candidates():
    strength = np.array(
        [0.0, 1.0, 3.0, 1.0, 2.0, 0.0],
        dtype=np.float64,
    )

    indices = np.array(
        [1, 2, 4],
        dtype=np.int64,
    )

    result = apply_boundary_threshold(
        strength,
        indices,
        threshold=2.0,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [2, 4],
            dtype=np.int64,
        ),
    )


def test_apply_boundary_threshold_is_inclusive():
    strength = np.array(
        [0.0, 2.0, 0.0],
        dtype=np.float64,
    )

    result = apply_boundary_threshold(
        strength,
        np.array(
            [1],
            dtype=np.int64,
        ),
        threshold=2.0,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [1],
            dtype=np.int64,
        ),
    )


def test_apply_boundary_threshold_accepts_empty_candidates():
    result = apply_boundary_threshold(
        np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [],
            dtype=np.int64,
        ),
        threshold=1.0,
    )

    assert result.dtype == np.int64
    assert result.size == 0


def test_apply_boundary_threshold_rejects_out_of_range_index():
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        apply_boundary_threshold(
            np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [3],
                dtype=np.int64,
            ),
            threshold=1.0,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        np.nan,
        np.inf,
        -np.inf,
        -1.0,
    ],
)
def test_apply_boundary_threshold_rejects_invalid_threshold(
    invalid_threshold: float,
):
    with pytest.raises(
        ValueError,
        match="finite|non-negative",
    ):
        apply_boundary_threshold(
            np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [1],
                dtype=np.int64,
            ),
            threshold=invalid_threshold,
        )


def test_apply_boundary_threshold_rejects_boolean_threshold():
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        apply_boundary_threshold(
            np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [1],
                dtype=np.int64,
            ),
            threshold=True,
        )


#7.12
def test_enforce_minimum_candidate_distance_keeps_stronger_candidate():
    strength = np.array(
        [0.0, 2.0, 5.0, 3.0, 0.0],
        dtype=np.float64,
    )

    result = enforce_minimum_candidate_distance(
        strength,
        np.array(
            [1, 2, 3],
            dtype=np.int64,
        ),
        minimum_distance=2,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [2],
            dtype=np.int64,
        ),
    )


def test_enforce_minimum_candidate_distance_keeps_separated_candidates():
    strength = np.array(
        [0.0, 3.0, 0.0, 0.0, 2.0, 0.0],
        dtype=np.float64,
    )

    result = enforce_minimum_candidate_distance(
        strength,
        np.array(
            [1, 4],
            dtype=np.int64,
        ),
        minimum_distance=2,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [1, 4],
            dtype=np.int64,
        ),
    )


def test_enforce_minimum_candidate_distance_prefers_lower_index_on_tie():
    strength = np.array(
        [0.0, 4.0, 4.0, 0.0],
        dtype=np.float64,
    )

    result = enforce_minimum_candidate_distance(
        strength,
        np.array(
            [1, 2],
            dtype=np.int64,
        ),
        minimum_distance=2,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [1],
            dtype=np.int64,
        ),
    )


def test_enforce_minimum_candidate_distance_accepts_empty_candidates():
    result = enforce_minimum_candidate_distance(
        np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [],
            dtype=np.int64,
        ),
        minimum_distance=2,
    )

    assert result.dtype == np.int64
    assert result.size == 0


@pytest.mark.parametrize(
    "invalid_distance",
    [
        True,
        1.5,
        "2",
    ],
)
def test_enforce_minimum_candidate_distance_rejects_invalid_type(
    invalid_distance: object,
):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        enforce_minimum_candidate_distance(
            np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [1],
                dtype=np.int64,
            ),
            minimum_distance=invalid_distance,
        )


@pytest.mark.parametrize(
    "invalid_distance",
    [
        0,
        -1,
    ],
)
def test_enforce_minimum_candidate_distance_rejects_nonpositive_value(
    invalid_distance: int,
):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        enforce_minimum_candidate_distance(
            np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [1],
                dtype=np.int64,
            ),
            minimum_distance=invalid_distance,
        )


#7.13
def test_build_boundary_candidates_returns_expected_arrays():
    strength = np.array(
        [0.0, 2.0, 0.5, 4.0, 0.0],
        dtype=np.float64,
    )

    candidates = build_boundary_candidates(
        strength,
        np.array(
            [1, 3],
            dtype=np.int64,
        ),
    )

    assert set(candidates) == set(
        BOUNDARY_CANDIDATE_REQUIRED_KEYS
    )

    np.testing.assert_array_equal(
        candidates["boundary_strength"],
        strength,
    )

    np.testing.assert_array_equal(
        candidates["candidate_frame_indices"],
        np.array(
            [1, 3],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        candidates["candidate_strengths"],
        np.array(
            [2.0, 4.0],
            dtype=np.float64,
        ),
    )


def test_build_boundary_candidates_accepts_empty_candidates():
    candidates = build_boundary_candidates(
        np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [],
            dtype=np.int64,
        ),
    )

    assert candidates[
        "candidate_frame_indices"
    ].shape == (0,)

    assert candidates[
        "candidate_strengths"
    ].shape == (0,)


def test_build_boundary_candidates_returns_independent_arrays():
    strength = np.array(
        [0.0, 2.0, 0.0],
        dtype=np.float64,
    )

    indices = np.array(
        [1],
        dtype=np.int64,
    )

    candidates = build_boundary_candidates(
        strength,
        indices,
    )

    candidates["boundary_strength"][1] = 999.0
    candidates["candidate_frame_indices"][0] = 0

    assert strength[1] == 2.0
    assert indices[0] == 1


def test_build_boundary_candidates_rejects_out_of_range_index():
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        build_boundary_candidates(
            np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [3],
                dtype=np.int64,
            ),
        )


def test_build_boundary_candidates_rejects_duplicate_indices():
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        build_boundary_candidates(
            np.array(
                [0.0, 1.0, 2.0, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [1, 1],
                dtype=np.int64,
            ),
        )


def test_build_boundary_candidates_rejects_unsorted_indices():
    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        build_boundary_candidates(
            np.array(
                [0.0, 1.0, 2.0, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [2, 1],
                dtype=np.int64,
            ),
        )


#7.14
def test_validate_boundary_candidates_accepts_valid_arrays():
    candidates = build_boundary_candidates(
        np.array(
            [0.0, 2.0, 0.0, 4.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [1, 3],
            dtype=np.int64,
        ),
    )

    validated = validate_boundary_candidates(
        candidates,
        minimum_distance=2,
    )

    for key in BOUNDARY_CANDIDATE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            validated[key],
            candidates[key],
        )


def test_validate_boundary_candidates_accepts_empty_candidates():
    candidates = build_boundary_candidates(
        np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [],
            dtype=np.int64,
        ),
    )

    validated = validate_boundary_candidates(
        candidates,
        minimum_distance=2,
    )

    assert validated[
        "candidate_frame_indices"
    ].size == 0


def test_validate_boundary_candidates_rejects_strength_mismatch():
    candidates = build_boundary_candidates(
        np.array(
            [0.0, 2.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [1],
            dtype=np.int64,
        ),
    )

    candidates["candidate_strengths"][0] = 3.0

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        validate_boundary_candidates(
            candidates,
            minimum_distance=1,
        )


def test_validate_boundary_candidates_rejects_minimum_distance_violation():
    candidates = build_boundary_candidates(
        np.array(
            [0.0, 3.0, 2.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [1, 2],
            dtype=np.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="minimum distance",
    ):
        validate_boundary_candidates(
            candidates,
            minimum_distance=2,
        )


#7.15
def test_build_boundary_metadata_returns_expected_schema(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        profile,
        source_metadata,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    configuration = (
        build_default_boundary_configuration()
    )

    candidates = build_boundary_candidates(
        np.array(
            [0.0, 2.0, 0.0, 4.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [1, 3],
            dtype=np.int64,
        ),
    )

    metadata = build_boundary_metadata(
        source_metadata,
        configuration,
        candidates,
    )

    assert set(metadata) == set(
        BOUNDARY_METADATA_REQUIRED_KEYS
    )

    assert metadata["format_version"] == 1

    assert metadata["shape_summary"][
        "number_of_frames"
    ] == 5

    assert metadata["shape_summary"][
        "number_of_candidates"
    ] == 2

    assert metadata["boundary_summary"][
        "maximum_boundary_strength"
    ] == pytest.approx(4.0)


def test_build_boundary_metadata_handles_empty_candidates(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        profile,
        source_metadata,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    candidates = build_boundary_candidates(
        np.zeros(
            5,
            dtype=np.float64,
        ),
        np.array(
            [],
            dtype=np.int64,
        ),
    )

    metadata = build_boundary_metadata(
        source_metadata,
        build_default_boundary_configuration(),
        candidates,
    )

    summary = metadata["boundary_summary"]

    assert summary[
        "minimum_candidate_strength"
    ] == 0.0

    assert summary[
        "maximum_candidate_strength"
    ] == 0.0


def test_build_boundary_metadata_is_json_serializable(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        profile,
        source_metadata,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    candidates = build_boundary_candidates(
        np.array(
            [0.0, 2.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [1],
            dtype=np.int64,
        ),
    )

    metadata = build_boundary_metadata(
        source_metadata,
        build_default_boundary_configuration(),
        candidates,
    )

    encoded = json.dumps(metadata)
    assert json.loads(encoded) == metadata


#7.16
def test_validate_boundary_metadata_accepts_valid_metadata(
    tmp_path: Path,
) -> None:
    _, _, _, source_metadata = (
        create_persisted_scale_profile(
            tmp_path
        )
    )

    configuration = (
        build_default_boundary_configuration()
    )

    candidates = build_boundary_candidates(
        np.array(
            [0.0, 2.0, 0.0, 4.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [1, 3],
            dtype=np.int64,
        ),
    )

    metadata = build_boundary_metadata(
        source_metadata,
        configuration,
        candidates,
    )

    validated = validate_boundary_metadata(
        metadata,
        source_metadata,
        candidates,
    )

    assert validated == metadata
    assert validated is not metadata


def test_validate_boundary_metadata_rejects_summary_mismatch(
    tmp_path: Path,
) -> None:
    _, _, _, source_metadata = (
        create_persisted_scale_profile(
            tmp_path
        )
    )

    configuration = (
        build_default_boundary_configuration()
    )

    candidates = build_boundary_candidates(
        np.array(
            [0.0, 2.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [1],
            dtype=np.int64,
        ),
    )

    metadata = build_boundary_metadata(
        source_metadata,
        configuration,
        candidates,
    )

    metadata["boundary_summary"][
        "maximum_boundary_strength"
    ] = 3.0

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_boundary_metadata(
            metadata,
            source_metadata,
            candidates,
        )


#7.17
def test_extract_boundary_candidates_returns_valid_outputs(
    tmp_path: Path,
) -> None:
    _, _, profile, source_metadata = (
        create_persisted_scale_profile(
            tmp_path
        )
    )

    configuration = (
        build_default_boundary_configuration()
    )

    candidates, metadata = (
        extract_boundary_candidates(
            profile,
            source_metadata,
            configuration,
        )
    )

    assert set(candidates) == set(
        BOUNDARY_CANDIDATE_REQUIRED_KEYS
    )

    assert set(metadata) == set(
        BOUNDARY_METADATA_REQUIRED_KEYS
    )

    assert candidates["boundary_strength"].shape == (
        profile["magnitude"].shape[1],
    )

    assert metadata["shape_summary"][
        "number_of_frames"
    ] == profile["magnitude"].shape[1]


def test_extract_boundary_candidates_handles_zero_candidates(
    tmp_path: Path,
) -> None:
    _, _, profile, source_metadata = (
        create_persisted_scale_profile(
            tmp_path
        )
    )

    configuration = (
        build_default_boundary_configuration()
    )

    configuration["threshold"] = 1.0e12

    candidates, metadata = (
        extract_boundary_candidates(
            profile,
            source_metadata,
            configuration,
        )
    )

    assert candidates[
        "candidate_frame_indices"
    ].size == 0

    assert candidates[
        "candidate_strengths"
    ].size == 0

    assert metadata["shape_summary"][
        "number_of_candidates"
    ] == 0

    assert metadata["boundary_summary"][
        "minimum_candidate_strength"
    ] == 0.0

    assert metadata["boundary_summary"][
        "maximum_candidate_strength"
    ] == 0.0



#7.18
def test_save_boundary_outputs_persists_valid_outputs(
    tmp_path: Path,
) -> None:
    _, _, profile, source_metadata = (
        create_persisted_scale_profile(
            tmp_path
        )
    )

    configuration = (
        build_default_boundary_configuration()
    )

    candidates, metadata = (
        extract_boundary_candidates(
            profile,
            source_metadata,
            configuration,
        )
    )

    output_path = tmp_path / "boundary_candidates.npz"
    metadata_path = tmp_path / "boundary_metadata.json"

    save_boundary_outputs(
        output_path,
        metadata_path,
        candidates,
        metadata,
        source_metadata,
    )

    assert output_path.is_file()
    assert metadata_path.is_file()

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        assert set(archive.files) == set(
            BOUNDARY_CANDIDATE_REQUIRED_KEYS
        )

        for key in BOUNDARY_CANDIDATE_REQUIRED_KEYS:
            np.testing.assert_array_equal(
                archive[key],
                candidates[key],
            )

    loaded_metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8",
        )
    )

    assert loaded_metadata == metadata
    assert metadata_path.read_bytes().endswith(
        b"\n"
    )


def test_save_boundary_outputs_creates_parent_directories(
    tmp_path: Path,
) -> None:
    _, _, profile, source_metadata = (
        create_persisted_scale_profile(
            tmp_path
        )
    )

    candidates, metadata = (
        extract_boundary_candidates(
            profile,
            source_metadata,
            build_default_boundary_configuration(),
        )
    )

    output_path = (
        tmp_path
        / "arrays"
        / "boundary_candidates.npz"
    )
    metadata_path = (
        tmp_path
        / "metadata"
        / "boundary_metadata.json"
    )

    save_boundary_outputs(
        output_path,
        metadata_path,
        candidates,
        metadata,
        source_metadata,
    )

    assert output_path.is_file()
    assert metadata_path.is_file()


#7.19
def test_build_parser_accepts_required_arguments():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--input",
            "scale_profile.npz",
            "--metadata-input",
            "scale_profile_metadata.json",
            "--output",
            "boundary_candidates.npz",
        ]
    )

    assert args.input == Path(
        "scale_profile.npz"
    )
    assert args.metadata_input == Path(
        "scale_profile_metadata.json"
    )
    assert args.output == Path(
        "boundary_candidates.npz"
    )
    assert args.metadata_output is None
    assert (
        args.threshold
        == DEFAULT_BOUNDARY_THRESHOLD
    )
    assert (
        args.minimum_distance
        == DEFAULT_MINIMUM_CANDIDATE_DISTANCE
    )


def test_build_parser_accepts_optional_arguments():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--input",
            "scale_profile.npz",
            "--metadata-input",
            "scale_profile_metadata.json",
            "--output",
            "boundary_candidates.npz",
            "--metadata-output",
            "boundary_metadata.json",
            "--threshold",
            "2.5",
            "--minimum-distance",
            "4",
        ]
    )

    assert args.metadata_output == Path(
        "boundary_metadata.json"
    )
    assert args.threshold == pytest.approx(2.5)
    assert args.minimum_distance == 4


@pytest.mark.parametrize(
    "missing_option",
    [
        "--input",
        "--metadata-input",
        "--output",
    ],
)
def test_build_parser_requires_input_and_output_arguments(
    missing_option: str,
):
    parser = build_parser()

    arguments = [
        "--input",
        "scale_profile.npz",
        "--metadata-input",
        "scale_profile_metadata.json",
        "--output",
        "boundary_candidates.npz",
    ]

    option_index = arguments.index(
        missing_option
    )

    del arguments[
        option_index : option_index + 2
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)


#7.20
def create_boundary_cli_arguments(
    tmp_path: Path,
    metadata_output: Path | None = None,
) -> Namespace:
    input_path = tmp_path / "scale_profile.npz"
    metadata_input = (
        tmp_path / "scale_profile_metadata.json"
    )

    input_path.write_bytes(b"placeholder")
    metadata_input.write_text(
        "{}",
        encoding="utf-8",
    )

    return Namespace(
        input=input_path,
        metadata_input=metadata_input,
        output=tmp_path / "boundary_candidates.npz",
        metadata_output=metadata_output,
        threshold=DEFAULT_BOUNDARY_THRESHOLD,
        minimum_distance=(
            DEFAULT_MINIMUM_CANDIDATE_DISTANCE
        ),
    )


def test_default_boundary_metadata_path():
    result = default_boundary_metadata_path(
        Path("results/boundary_candidates.npz")
    )

    assert result == Path(
        "results/boundary_candidates_metadata.json"
    )


def test_validate_arguments_returns_default_outputs(
    tmp_path: Path,
):
    args = create_boundary_cli_arguments(
        tmp_path
    )

    metadata_output, configuration = (
        validate_arguments(args)
    )

    assert metadata_output == (
        tmp_path
        / "boundary_candidates_metadata.json"
    )

    assert configuration["threshold"] == (
        DEFAULT_BOUNDARY_THRESHOLD
    )

    assert configuration[
        "minimum_candidate_distance"
    ] == DEFAULT_MINIMUM_CANDIDATE_DISTANCE


def test_validate_arguments_preserves_metadata_output(
    tmp_path: Path,
):
    expected_path = (
        tmp_path / "custom_metadata.json"
    )

    args = create_boundary_cli_arguments(
        tmp_path,
        metadata_output=expected_path,
    )

    metadata_output, _ = validate_arguments(
        args
    )

    assert metadata_output == expected_path


@pytest.mark.parametrize(
    "field_name",
    [
        "input",
        "metadata_input",
    ],
)
def test_validate_arguments_rejects_missing_input(
    tmp_path: Path,
    field_name: str,
):
    args = create_boundary_cli_arguments(
        tmp_path
    )

    setattr(
        args,
        field_name,
        tmp_path / "missing",
    )

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        validate_arguments(args)


@pytest.mark.parametrize(
    (
        "field_name",
        "filename",
        "expected_message",
    ),
    [
        (
            "input",
            "profile.npy",
            "--input must use the .npz extension",
        ),
        (
            "metadata_input",
            "metadata.txt",
            "--metadata-input must use the .json extension",
        ),
        (
            "output",
            "boundary.npy",
            "--output must use the .npz extension",
        ),
        (
            "metadata_output",
            "boundary.txt",
            "--metadata-output must use the .json extension",
        ),
    ],
)
def test_validate_arguments_rejects_invalid_extension(
    tmp_path: Path,
    field_name: str,
    filename: str,
    expected_message: str,
):
    args = create_boundary_cli_arguments(
        tmp_path
    )

    invalid_path = tmp_path / filename

    if field_name == "input":
        args.input.rename(invalid_path)
    elif field_name == "metadata_input":
        args.metadata_input.rename(
            invalid_path
        )

    setattr(
        args,
        field_name,
        invalid_path,
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        validate_arguments(args)


def test_validate_arguments_rejects_path_collision(
    tmp_path: Path,
):
    args = create_boundary_cli_arguments(
        tmp_path
    )

    args.output = args.input

    with pytest.raises(
        ValueError,
        match="paths must be different",
    ):
        validate_arguments(args)


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_message",
    ),
    [
        (
            "threshold",
            -1.0,
            "threshold must be finite and non-negative",
        ),
        (
            "minimum_distance",
            0,
            "distance must be at least 1",
        ),
    ],
)
def test_validate_arguments_rejects_invalid_configuration(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
    expected_message: str,
):
    args = create_boundary_cli_arguments(
        tmp_path
    )

    setattr(
        args,
        field_name,
        invalid_value,
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        validate_arguments(args)


#
def test_boundary_cli_creates_outputs(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_input,
        _,
        _,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    output_path = (
        tmp_path / "boundary_candidates.npz"
    )
    metadata_output = (
        tmp_path / "boundary_metadata.json"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.boundary.extract_boundary_candidates",
            "--input",
            str(input_path),
            "--metadata-input",
            str(metadata_input),
            "--output",
            str(output_path),
            "--metadata-output",
            str(metadata_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert (
        "Boundary candidate extraction: OK"
        in result.stdout
    )
    assert output_path.is_file()
    assert metadata_output.is_file()


def test_boundary_cli_persists_valid_outputs(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_input,
        _,
        source_metadata,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    output_path = (
        tmp_path / "boundary_candidates.npz"
    )
    metadata_output = (
        tmp_path / "boundary_metadata.json"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.boundary.extract_boundary_candidates",
            "--input",
            str(input_path),
            "--metadata-input",
            str(metadata_input),
            "--output",
            str(output_path),
            "--metadata-output",
            str(metadata_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        assert set(archive.files) == set(
            BOUNDARY_CANDIDATE_REQUIRED_KEYS
        )

        candidates = {
            key: np.array(
                archive[key],
                copy=True,
            )
            for key in (
                BOUNDARY_CANDIDATE_REQUIRED_KEYS
            )
        }

    metadata = json.loads(
        metadata_output.read_text(
            encoding="utf-8",
        )
    )

    minimum_distance = metadata[
        "boundary_configuration"
    ]["minimum_candidate_distance"]

    validated_candidates = (
        validate_boundary_candidates(
            candidates,
            minimum_distance=minimum_distance,
        )
    )

    validated_metadata = validate_boundary_metadata(
        metadata,
        source_metadata,
        validated_candidates,
    )

    assert validated_metadata == metadata


def test_boundary_cli_uses_default_metadata_path(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_input,
        _,
        _,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    output_path = (
        tmp_path / "boundary_candidates.npz"
    )
    expected_metadata_path = (
        tmp_path
        / "boundary_candidates_metadata.json"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.boundary.extract_boundary_candidates",
            "--input",
            str(input_path),
            "--metadata-input",
            str(metadata_input),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert output_path.is_file()
    assert expected_metadata_path.is_file()


def test_boundary_cli_handles_zero_candidates(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_input,
        _,
        _,
    ) = create_persisted_scale_profile(
        tmp_path
    )

    output_path = (
        tmp_path / "boundary_candidates.npz"
    )
    metadata_output = (
        tmp_path / "boundary_metadata.json"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.boundary.extract_boundary_candidates",
            "--input",
            str(input_path),
            "--metadata-input",
            str(metadata_input),
            "--output",
            str(output_path),
            "--metadata-output",
            str(metadata_output),
            "--threshold",
            "1000000000000",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "candidates: 0" in result.stdout

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        assert archive[
            "candidate_frame_indices"
        ].size == 0

        assert archive[
            "candidate_strengths"
        ].size == 0


def test_boundary_cli_rejects_missing_input(
    tmp_path: Path,
) -> None:
    metadata_input = (
        tmp_path / "scale_profile_metadata.json"
    )

    metadata_input.write_text(
        "{}",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.boundary.extract_boundary_candidates",
            "--input",
            str(tmp_path / "missing.npz"),
            "--metadata-input",
            str(metadata_input),
            "--output",
            str(tmp_path / "boundary.npz"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert (
        "ScaleProfile archive does not exist"
        in result.stderr
    )