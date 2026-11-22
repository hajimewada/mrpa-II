from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from matplotlib.path import Path
from pathlib import Path
import numpy as np
import pytest

from analysis.boundary.extract_boundary_candidates import (
    BOUNDARY_CANDIDATE_REQUIRED_KEYS,
    build_boundary_candidates,
    build_boundary_metadata,
    build_default_boundary_configuration,
    compute_robust_normalized_series,
    extract_boundary_candidates,
    load_scale_profile_outputs,
    validate_loaded_scale_profile,
)

from analysis.f0.extract_scale_profile import (
    SCALE_PROFILE_REQUIRED_KEYS,
    extract_scale_profile,
)

from analysis.nucleus.extract_nucleus_candidates import (
    DEFAULT_BOUNDARY_EXCLUSION_DISTANCE,
    DEFAULT_LOCAL_CONTRAST_RADIUS,
    DEFAULT_MINIMUM_NUCLEUS_CANDIDATE_DISTANCE,
    DEFAULT_NUCLEUS_FEATURE_WEIGHTS,
    DEFAULT_NUCLEUS_THRESHOLD,
    NUCLEUS_CANDIDATE_REQUIRED_KEYS,
    NUCLEUS_CONFIGURATION_METADATA_REQUIRED_KEYS,
    NUCLEUS_EVIDENCE_REQUIRED_KEYS,
    NUCLEUS_FEATURE_NAMES,
    NUCLEUS_FORMAT_VERSION,
    NUCLEUS_METADATA_REQUIRED_KEYS,
    NUCLEUS_SHAPE_METADATA_REQUIRED_KEYS,
    NUCLEUS_SUMMARY_METADATA_REQUIRED_KEYS,
    SOURCE_BOUNDARY_METADATA_REQUIRED_KEYS,
    SOURCE_SCALE_PROFILE_METADATA_REQUIRED_KEYS,
    build_nucleus_candidates,
    apply_nucleus_threshold,
    build_nucleus_evidence,
    build_nucleus_metadata,
    build_default_nucleus_configuration,
    build_parser,
    default_nucleus_metadata_path,
    compute_local_contrast,
    compute_normalized_local_contrast,
    compute_nucleus_strength,
    enforce_minimum_nucleus_candidate_distance,
    exclude_boundary_adjacent_nucleus_candidates,
    extract_nucleus_candidates,
    find_nucleus_local_maxima,
    load_boundary_outputs,
    main,
    save_nucleus_outputs,
    validate_arguments,
    validate_nucleus_configuration,
    validate_loaded_boundary_outputs,
    validate_nucleus_candidates,
    validate_nucleus_metadata,
    validate_nucleus_strength,
)



def test_nucleus_contract_constants():
    assert NUCLEUS_FORMAT_VERSION == 1

    assert len(NUCLEUS_FEATURE_NAMES) == 6
    assert len(NUCLEUS_EVIDENCE_REQUIRED_KEYS) == 6
    assert len(NUCLEUS_CANDIDATE_REQUIRED_KEYS) == 3
    assert len(NUCLEUS_METADATA_REQUIRED_KEYS) == 6

    assert len(
        SOURCE_SCALE_PROFILE_METADATA_REQUIRED_KEYS
    ) == 5

    assert len(
        SOURCE_BOUNDARY_METADATA_REQUIRED_KEYS
    ) == 5

    assert len(
        NUCLEUS_CONFIGURATION_METADATA_REQUIRED_KEYS
    ) == 8

    assert len(
        NUCLEUS_SHAPE_METADATA_REQUIRED_KEYS
    ) == 5

    assert len(
        NUCLEUS_SUMMARY_METADATA_REQUIRED_KEYS
    ) == 5


#step 8.2
def test_validate_nucleus_configuration_accepts_default_configuration():
    configuration = (
        build_default_nucleus_configuration()
    )

    validated = validate_nucleus_configuration(
        configuration
    )

    assert validated == configuration


def test_validate_nucleus_configuration_returns_independent_copy():
    configuration = (
        build_default_nucleus_configuration()
    )

    validated = validate_nucleus_configuration(
        configuration
    )

    assert validated is not configuration
    assert (
        validated["feature_weights"]
        is not configuration["feature_weights"]
    )

    validated["feature_weights"]["frame_total"] = 9.0

    assert (
        configuration["feature_weights"]["frame_total"]
        == 1.0
    )


def test_build_default_nucleus_configuration_returns_independent_weights():
    first = build_default_nucleus_configuration()
    second = build_default_nucleus_configuration()

    first["feature_weights"]["frame_total"] = 9.0

    assert (
        second["feature_weights"]["frame_total"]
        == 1.0
    )


def test_validate_nucleus_configuration_accepts_explicit_valid_values():
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration["local_contrast_radius"] = 4
    configuration["threshold"] = 2.5
    configuration["minimum_candidate_distance"] = 3
    configuration["boundary_exclusion_distance"] = 2
    configuration["feature_weights"]["frame_total"] = 2.0

    validated = validate_nucleus_configuration(
        configuration
    )

    assert validated["local_contrast_radius"] == 4
    assert validated["threshold"] == pytest.approx(2.5)
    assert validated["minimum_candidate_distance"] == 3
    assert validated["boundary_exclusion_distance"] == 2
    assert (
        validated["feature_weights"]["frame_total"]
        == pytest.approx(2.0)
    )


def test_validate_nucleus_configuration_accepts_zero_boundary_exclusion():
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration["boundary_exclusion_distance"] = 0

    validated = validate_nucleus_configuration(
        configuration
    )

    assert validated["boundary_exclusion_distance"] == 0


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_message",
    ),
    [
        (
            "normalization",
            "z-score",
            "Unsupported Nucleus normalization",
        ),
        (
            "plateau_policy",
            "last-index",
            "Unsupported Nucleus plateau policy",
        ),
    ],
)
def test_validate_nucleus_configuration_rejects_unsupported_policy(
    field_name: str,
    invalid_value: str,
    expected_message: str,
):
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        validate_nucleus_configuration(
            configuration
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "normalization",
        "plateau_policy",
    ],
)
def test_validate_nucleus_configuration_rejects_nonstring_policy(
    field_name: str,
):
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration[field_name] = 1

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        validate_nucleus_configuration(
            configuration
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_message",
    ),
    [
        (
            "local_contrast_radius",
            True,
            "Local contrast radius must be an integer",
        ),
        (
            "local_contrast_radius",
            1.5,
            "Local contrast radius must be an integer",
        ),
        (
            "minimum_candidate_distance",
            True,
            "Minimum candidate distance must be an integer",
        ),
        (
            "minimum_candidate_distance",
            1.5,
            "Minimum candidate distance must be an integer",
        ),
        (
            "boundary_exclusion_distance",
            True,
            "Boundary exclusion distance must be an integer",
        ),
        (
            "boundary_exclusion_distance",
            1.5,
            "Boundary exclusion distance must be an integer",
        ),
    ],
)
def test_validate_nucleus_configuration_rejects_invalid_integer_type(
    field_name: str,
    invalid_value: object,
    expected_message: str,
):
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration[field_name] = invalid_value

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        validate_nucleus_configuration(
            configuration
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_message",
    ),
    [
        (
            "local_contrast_radius",
            0,
            "Local contrast radius must be at least 1",
        ),
        (
            "minimum_candidate_distance",
            0,
            "Minimum candidate distance must be at least 1",
        ),
        (
            "boundary_exclusion_distance",
            -1,
            "Boundary exclusion distance must be non-negative",
        ),
    ],
)
def test_validate_nucleus_configuration_rejects_invalid_integer_range(
    field_name: str,
    invalid_value: int,
    expected_message: str,
):
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        validate_nucleus_configuration(
            configuration
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        True,
        "1.0",
        None,
    ],
)
def test_validate_nucleus_configuration_rejects_invalid_threshold_type(
    invalid_threshold: object,
):
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration["threshold"] = invalid_threshold

    with pytest.raises(
        TypeError,
        match="Nucleus threshold must be a real number",
    ):
        validate_nucleus_configuration(
            configuration
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_nucleus_configuration_rejects_invalid_threshold_value(
    invalid_threshold: float,
):
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration["threshold"] = invalid_threshold

    with pytest.raises(
        ValueError,
        match="finite|non-negative",
    ):
        validate_nucleus_configuration(
            configuration
        )


def test_validate_nucleus_configuration_rejects_missing_feature_weight():
    configuration = (
        build_default_nucleus_configuration()
    )

    del configuration["feature_weights"]["frame_total"]

    with pytest.raises(
        ValueError,
        match="must match the required feature names",
    ):
        validate_nucleus_configuration(
            configuration
        )


def test_validate_nucleus_configuration_rejects_unexpected_feature_weight():
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration["feature_weights"]["unexpected"] = 1.0

    with pytest.raises(
        ValueError,
        match="must match the required feature names",
    ):
        validate_nucleus_configuration(
            configuration
        )


def test_validate_nucleus_configuration_rejects_non_dictionary_weights():
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration["feature_weights"] = []

    with pytest.raises(
        TypeError,
        match="must be a dictionary",
    ):
        validate_nucleus_configuration(
            configuration
        )


@pytest.mark.parametrize(
    "invalid_weight",
    [
        True,
        "1.0",
        None,
    ],
)
def test_validate_nucleus_configuration_rejects_invalid_weight_type(
    invalid_weight: object,
):
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration["feature_weights"][
        "frame_total"
    ] = invalid_weight

    with pytest.raises(
        TypeError,
        match="must be a real number",
    ):
        validate_nucleus_configuration(
            configuration
        )


@pytest.mark.parametrize(
    "invalid_weight",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_nucleus_configuration_rejects_invalid_weight_value(
    invalid_weight: float,
):
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration["feature_weights"][
        "frame_total"
    ] = invalid_weight

    with pytest.raises(
        ValueError,
        match="finite|non-negative",
    ):
        validate_nucleus_configuration(
            configuration
        )


def test_validate_nucleus_configuration_rejects_all_zero_weights():
    configuration = (
        build_default_nucleus_configuration()
    )

    configuration["feature_weights"] = {
        name: 0.0
        for name in NUCLEUS_FEATURE_NAMES
    }

    with pytest.raises(
        ValueError,
        match="At least one Nucleus feature weight must be positive",
    ):
        validate_nucleus_configuration(
            configuration
        )


#8.3
def create_scale_profile_files(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    dict[str, np.ndarray],
    dict[str, object],
]:
    frame_count = 5

    profile = {
        "frequencies": np.array(
            [20.0, 10.0],
            dtype=np.float64,
        ),
        "magnitude": np.array(
            [
                [1.0, 2.0, 3.0, 2.0, 1.0],
                [2.0, 3.0, 4.0, 3.0, 2.0],
            ],
            dtype=np.float64,
        ),
        "scale_mean": np.array(
            [1.8, 2.8],
            dtype=np.float64,
        ),
        "scale_std": np.array(
            [0.7483314773547883, 0.7483314773547883],
            dtype=np.float64,
        ),
        "scale_median": np.array(
            [2.0, 3.0],
            dtype=np.float64,
        ),
        "scale_max": np.array(
            [3.0, 4.0],
            dtype=np.float64,
        ),
        "scale_rms": np.array(
            [
                np.sqrt(19.0 / 5.0),
                np.sqrt(42.0 / 5.0),
            ],
            dtype=np.float64,
        ),
        "frame_total": np.array(
            [3.0, 5.0, 7.0, 5.0, 3.0],
            dtype=np.float64,
        ),
        "frame_mean": np.array(
            [1.5, 2.5, 3.5, 2.5, 1.5],
            dtype=np.float64,
        ),
        "frame_max": np.array(
            [2.0, 3.0, 4.0, 3.0, 2.0],
            dtype=np.float64,
        ),
        "dominant_scale_index": np.array(
            [1, 1, 1, 1, 1],
            dtype=np.int64,
        ),
        "dominant_frequency_hz": np.full(
            frame_count,
            10.0,
            dtype=np.float64,
        ),
        "frequency_centroid_hz": np.array(
            [40.0 / 3.0, 14.0, 100.0 / 7.0, 14.0, 40.0 / 3.0],
            dtype=np.float64,
        ),
        "frequency_dispersion_hz": np.array(
            [4.714045207910317, 4.898979485566356, 4.948716593053935, 4.898979485566356, 4.714045207910317],
            dtype=np.float64,
        ),
    }

    metadata = {
        "format_version": 1,
        "source_cwt": {},
        "feature_configuration": {},
        "shape_summary": {},
        "scale_statistics_summary": {},
        "frame_statistics_summary": {},
    }

    input_path = tmp_path / "scale_profile.npz"
    metadata_path = tmp_path / "scale_profile_metadata.json"

    np.savez_compressed(
        input_path,
        **profile,
    )

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    return (
        input_path,
        metadata_path,
        profile,
        metadata,
    )

def test_load_scale_profile_outputs_loads_persisted_data(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        expected_profile,
        expected_metadata,
    ) = create_scale_profile_files(
        tmp_path
    )

    loaded_profile, loaded_metadata = (
        load_scale_profile_outputs(
            input_path,
            metadata_path,
        )
    )

    assert set(loaded_profile) == set(
        SCALE_PROFILE_REQUIRED_KEYS
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
    input_path = tmp_path / "scale_profile.npz"

    np.savez_compressed(
        input_path,
        frequencies=np.array(
            [20.0, 10.0],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        FileNotFoundError,
        match="metadata does not exist",
    ):
        load_scale_profile_outputs(
            input_path,
            tmp_path / "missing.json",
        )


#8.4
def test_validate_loaded_scale_profile_rejects_invalid_metadata(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        _,
        _,
    ) = create_scale_profile_files(
        tmp_path
    )

    profile, metadata = load_scale_profile_outputs(
        input_path,
        metadata_path,
    )

    with pytest.raises(
        KeyError,
    ):
        validate_loaded_scale_profile(
            profile,
            metadata,
        )


def test_validate_loaded_scale_profile_rejects_invalid_profile(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        _,
        _,
    ) = create_scale_profile_files(
        tmp_path
    )

    profile, metadata = load_scale_profile_outputs(
        input_path,
        metadata_path,
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


#8.5
def create_boundary_files(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    dict[str, np.ndarray],
    dict[str, object],
]:
    candidates = {
        "boundary_strength": np.array(
            [0.0, 2.0, 0.5, 4.0, 0.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": np.array(
            [1, 3],
            dtype=np.int64,
        ),
        "candidate_strengths": np.array(
            [2.0, 4.0],
            dtype=np.float64,
        ),
    }

    metadata = {
        "format_version": 1,
        "source_scale_profile": {},
        "boundary_configuration": {},
        "shape_summary": {},
        "boundary_summary": {},
    }

    input_path = (
        tmp_path / "boundary_candidates.npz"
    )
    metadata_path = (
        tmp_path / "boundary_metadata.json"
    )

    np.savez_compressed(
        input_path,
        boundary_strength=candidates[
            "boundary_strength"
        ],
        candidate_frame_indices=candidates[
            "candidate_frame_indices"
        ],
        candidate_strengths=candidates[
            "candidate_strengths"
        ],
    )

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    return (
        input_path,
        metadata_path,
        candidates,
        metadata,
    )

def test_load_boundary_outputs_loads_persisted_data(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        expected_candidates,
        expected_metadata,
    ) = create_boundary_files(
        tmp_path
    )

    loaded_candidates, loaded_metadata = (
        load_boundary_outputs(
            input_path,
            metadata_path,
        )
    )

    assert set(loaded_candidates) == set(
        BOUNDARY_CANDIDATE_REQUIRED_KEYS
    )

    for key in BOUNDARY_CANDIDATE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            loaded_candidates[key],
            expected_candidates[key],
        )

        assert not np.shares_memory(
            loaded_candidates[key],
            expected_candidates[key],
        )

    assert loaded_metadata == expected_metadata


def test_load_boundary_outputs_rejects_missing_archive(
    tmp_path: Path,
) -> None:
    metadata_path = (
        tmp_path / "boundary_metadata.json"
    )

    metadata_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Boundary archive does not exist",
    ):
        load_boundary_outputs(
            tmp_path / "missing.npz",
            metadata_path,
        )


def test_load_boundary_outputs_rejects_missing_metadata(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "boundary_candidates.npz"
    )

    np.savez_compressed(
        input_path,
        boundary_strength=np.zeros(
            3,
            dtype=np.float64,
        ),
        candidate_frame_indices=np.array(
            [],
            dtype=np.int64,
        ),
        candidate_strengths=np.array(
            [],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Boundary metadata does not exist",
    ):
        load_boundary_outputs(
            input_path,
            tmp_path / "missing.json",
        )


def test_load_boundary_outputs_rejects_missing_array(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "boundary_candidates.npz"
    )
    metadata_path = (
        tmp_path / "boundary_metadata.json"
    )

    np.savez_compressed(
        input_path,
        boundary_strength=np.zeros(
            3,
            dtype=np.float64,
        ),
        candidate_frame_indices=np.array(
            [],
            dtype=np.int64,
        ),
    )

    metadata_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        KeyError,
        match="candidate_strengths",
    ):
        load_boundary_outputs(
            input_path,
            metadata_path,
        )


#8.6
def create_valid_boundary_validation_inputs(
) -> tuple[
    dict[str, np.ndarray],
    dict[str, object],
    dict[str, object],
]:
    source_metadata = {
        "format_version": 1,
        "shape_summary": {
            "number_of_scales": 2,
            "number_of_frames": 5,
        },
        "source_cwt": {
            "minimum_frequency_hz": 10.0,
            "maximum_frequency_hz": 20.0,
        },
    }

    candidates = build_boundary_candidates(
        np.array(
            [0.0, 2.0, 0.5, 4.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [1, 3],
            dtype=np.int64,
        ),
    )

    configuration = (
        build_default_boundary_configuration()
    )

    metadata = build_boundary_metadata(
        source_metadata,
        configuration,
        candidates,
    )

    return (
        candidates,
        metadata,
        source_metadata,
    )


def test_validate_loaded_boundary_outputs_accepts_valid_data():
    (
        candidates,
        metadata,
        source_metadata,
    ) = create_valid_boundary_validation_inputs()

    validated_candidates, validated_metadata = (
        validate_loaded_boundary_outputs(
            candidates,
            metadata,
            source_metadata,
        )
    )

    for key in BOUNDARY_CANDIDATE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            validated_candidates[key],
            candidates[key],
        )

    assert validated_metadata == metadata


def test_validate_loaded_boundary_outputs_returns_independent_copies():
    (
        candidates,
        metadata,
        source_metadata,
    ) = create_valid_boundary_validation_inputs()

    validated_candidates, validated_metadata = (
        validate_loaded_boundary_outputs(
            candidates,
            metadata,
            source_metadata,
        )
    )

    validated_candidates[
        "boundary_strength"
    ][1] = 999.0

    validated_metadata[
        "boundary_configuration"
    ]["threshold"] = 999.0

    assert candidates[
        "boundary_strength"
    ][1] == 2.0

    assert metadata[
        "boundary_configuration"
    ]["threshold"] != 999.0


def test_validate_loaded_boundary_outputs_rejects_invalid_candidates():
    (
        candidates,
        metadata,
        source_metadata,
    ) = create_valid_boundary_validation_inputs()

    candidates[
        "candidate_strengths"
    ][0] = 3.0

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        validate_loaded_boundary_outputs(
            candidates,
            metadata,
            source_metadata,
        )


def test_validate_loaded_boundary_outputs_rejects_invalid_metadata():
    (
        candidates,
        metadata,
        source_metadata,
    ) = create_valid_boundary_validation_inputs()

    metadata["boundary_summary"][
        "maximum_boundary_strength"
    ] = 999.0

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_loaded_boundary_outputs(
            candidates,
            metadata,
            source_metadata,
        )


def test_validate_loaded_boundary_outputs_rejects_source_frame_count_mismatch():
    (
        candidates,
        metadata,
        source_metadata,
    ) = create_valid_boundary_validation_inputs()

    source_metadata["shape_summary"][
        "number_of_frames"
    ] = 6

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_loaded_boundary_outputs(
            candidates,
            metadata,
            source_metadata,
        )


#8.7
def test_compute_local_contrast_returns_expected_values():
    values = np.array(
        [1.0, 2.0, 5.0, 2.0, 1.0],
        dtype=np.float64,
    )

    result = compute_local_contrast(
        values,
        radius=1,
    )

    expected = np.array(
        [0.0, 0.0, 3.0, 0.0, 0.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_compute_local_contrast_handles_multiple_peaks():
    values = np.array(
        [1.0, 4.0, 1.0, 5.0, 1.0],
        dtype=np.float64,
    )

    result = compute_local_contrast(
        values,
        radius=1,
    )

    expected = np.array(
        [0.0, 3.0, 0.0, 4.0, 0.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_compute_local_contrast_returns_zero_for_constant_series():
    result = compute_local_contrast(
        np.full(
            5,
            3.0,
            dtype=np.float64,
        ),
        radius=2,
    )

    np.testing.assert_array_equal(
        result,
        np.zeros(
            5,
            dtype=np.float64,
        ),
    )


def test_compute_local_contrast_handles_endpoint_peaks():
    values = np.array(
        [5.0, 1.0, 1.0, 1.0, 4.0],
        dtype=np.float64,
    )

    result = compute_local_contrast(
        values,
        radius=1,
    )

    expected = np.array(
        [4.0, 0.0, 0.0, 0.0, 3.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_compute_local_contrast_accepts_single_frame():
    result = compute_local_contrast(
        np.array(
            [5.0],
            dtype=np.float64,
        ),
        radius=1,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [0.0],
            dtype=np.float64,
        ),
    )


def test_compute_local_contrast_accepts_large_radius():
    values = np.array(
        [1.0, 5.0, 1.0],
        dtype=np.float64,
    )

    result = compute_local_contrast(
        values,
        radius=10,
    )

    expected = np.array(
        [0.0, 4.0, 0.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_compute_local_contrast_returns_independent_float64():
    values = np.array(
        [1, 2, 5, 2, 1],
        dtype=np.int64,
    )

    result = compute_local_contrast(
        values,
        radius=1,
    )

    result[0] = 999.0

    assert values[0] == 1
    assert result.dtype == np.float64
    assert not np.shares_memory(
        result,
        values,
    )
    assert np.all(result >= 0.0)


def test_compute_local_contrast_rejects_non_1d_input():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        compute_local_contrast(
            np.zeros(
                (2, 3),
                dtype=np.float64,
            ),
            radius=1,
        )


def test_compute_local_contrast_rejects_empty_input():
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        compute_local_contrast(
            np.array(
                [],
                dtype=np.float64,
            ),
            radius=1,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_compute_local_contrast_rejects_nonfinite_input(
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
        compute_local_contrast(
            values,
            radius=1,
        )


@pytest.mark.parametrize(
    "invalid_radius",
    [
        True,
        1.5,
        "1",
        None,
    ],
    ids=[
        "boolean",
        "float",
        "string",
        "none",
    ],
)
def test_compute_local_contrast_rejects_invalid_radius_type(
    invalid_radius: object,
):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        compute_local_contrast(
            np.array(
                [1.0, 2.0, 1.0],
                dtype=np.float64,
            ),
            radius=invalid_radius,
        )


@pytest.mark.parametrize(
    "invalid_radius",
    [
        0,
        -1,
    ],
    ids=[
        "zero",
        "negative",
    ],
)
def test_compute_local_contrast_rejects_nonpositive_radius(
    invalid_radius: int,
):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        compute_local_contrast(
            np.array(
                [1.0, 2.0, 1.0],
                dtype=np.float64,
            ),
            radius=invalid_radius,
        )


#8.8
def test_compute_normalized_local_contrast_matches_component_functions():
    values = np.array(
        [1.0, 2.0, 5.0, 2.0, 1.0],
        dtype=np.float64,
    )

    result = compute_normalized_local_contrast(
        values,
        radius=1,
    )

    expected = compute_robust_normalized_series(
        compute_local_contrast(
            values,
            radius=1,
        )
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_compute_normalized_local_contrast_handles_multiple_peaks():
    values = np.array(
        [1.0, 4.0, 1.0, 7.0, 1.0],
        dtype=np.float64,
    )

    result = compute_normalized_local_contrast(
        values,
        radius=1,
    )

    expected = compute_robust_normalized_series(
        np.array(
            [0.0, 3.0, 0.0, 6.0, 0.0],
            dtype=np.float64,
        )
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_compute_normalized_local_contrast_returns_zero_for_constant_series():
    result = compute_normalized_local_contrast(
        np.full(
            5,
            3.0,
            dtype=np.float64,
        ),
        radius=2,
    )

    np.testing.assert_array_equal(
        result,
        np.zeros(
            5,
            dtype=np.float64,
        ),
    )


def test_compute_normalized_local_contrast_handles_zero_mad():
    values = np.array(
        [1.0, 1.0, 5.0, 1.0, 1.0],
        dtype=np.float64,
    )

    contrast = compute_local_contrast(
        values,
        radius=1,
    )

    assert np.median(
        np.abs(
            contrast - np.median(contrast)
        )
    ) == 0.0

    result = compute_normalized_local_contrast(
        values,
        radius=1,
    )

    np.testing.assert_array_equal(
        result,
        np.zeros(
            5,
            dtype=np.float64,
        ),
    )


def test_compute_normalized_local_contrast_accepts_single_frame():
    result = compute_normalized_local_contrast(
        np.array(
            [5.0],
            dtype=np.float64,
        ),
        radius=1,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [0.0],
            dtype=np.float64,
        ),
    )


def test_compute_normalized_local_contrast_accepts_large_radius():
    values = np.array(
        [1.0, 5.0, 1.0],
        dtype=np.float64,
    )

    result = compute_normalized_local_contrast(
        values,
        radius=10,
    )

    expected = compute_robust_normalized_series(
        compute_local_contrast(
            values,
            radius=10,
        )
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_compute_normalized_local_contrast_returns_independent_float64():
    values = np.array(
        [1, 2, 5, 2, 1],
        dtype=np.int64,
    )

    original = values.copy()

    result = compute_normalized_local_contrast(
        values,
        radius=1,
    )

    result[0] = 999.0

    np.testing.assert_array_equal(
        values,
        original,
    )

    assert result.dtype == np.float64
    assert not np.shares_memory(
        result,
        values,
    )


def test_compute_normalized_local_contrast_returns_finite_nonnegative_series():
    result = compute_normalized_local_contrast(
        np.array(
            [2.0, -1.0, 6.0, 0.0, 3.0],
            dtype=np.float64,
        ),
        radius=2,
    )

    assert result.shape == (5,)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_compute_normalized_local_contrast_rejects_non_1d_input():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        compute_normalized_local_contrast(
            np.zeros(
                (2, 3),
                dtype=np.float64,
            ),
            radius=1,
        )


def test_compute_normalized_local_contrast_rejects_empty_input():
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        compute_normalized_local_contrast(
            np.array(
                [],
                dtype=np.float64,
            ),
            radius=1,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_compute_normalized_local_contrast_rejects_nonfinite_input(
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
        compute_normalized_local_contrast(
            values,
            radius=1,
        )


@pytest.mark.parametrize(
    "invalid_radius",
    [
        True,
        1.5,
        "1",
        None,
    ],
    ids=[
        "boolean",
        "float",
        "string",
        "none",
    ],
)
def test_compute_normalized_local_contrast_rejects_invalid_radius_type(
    invalid_radius: object,
):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        compute_normalized_local_contrast(
            np.array(
                [1.0, 2.0, 1.0],
                dtype=np.float64,
            ),
            radius=invalid_radius,
        )


@pytest.mark.parametrize(
    "invalid_radius",
    [
        0,
        -1,
    ],
    ids=[
        "zero",
        "negative",
    ],
)
def test_compute_normalized_local_contrast_rejects_nonpositive_radius(
    invalid_radius: int,
):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        compute_normalized_local_contrast(
            np.array(
                [1.0, 2.0, 1.0],
                dtype=np.float64,
            ),
            radius=invalid_radius,
        )


#8.9
def test_build_nucleus_evidence_returns_required_arrays(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        profile,
        _,
    ) = create_scale_profile_files(
        tmp_path
    )

    configuration = (
        build_default_nucleus_configuration()
    )

    evidence = build_nucleus_evidence(
        profile,
        configuration,
    )

    assert set(evidence) == set(
        NUCLEUS_EVIDENCE_REQUIRED_KEYS
    )

    frame_count = profile[
        "magnitude"
    ].shape[1]

    for values in evidence.values():
        assert values.shape == (frame_count,)
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)


@pytest.mark.parametrize(
    (
        "evidence_name",
        "profile_name",
    ),
    [
        (
            "frame_total_level",
            "frame_total",
        ),
        (
            "frame_mean_level",
            "frame_mean",
        ),
        (
            "frame_max_level",
            "frame_max",
        ),
        (
            "dominant_frequency_level_hz",
            "dominant_frequency_hz",
        ),
        (
            "frequency_centroid_level_hz",
            "frequency_centroid_hz",
        ),
        (
            "frequency_dispersion_concentration",
            "frequency_dispersion_hz",
        ),
    ],
)
def test_build_nucleus_evidence_matches_component_function(
    tmp_path: Path,
    evidence_name: str,
    profile_name: str,
) -> None:
    (
        _,
        _,
        profile,
        _,
    ) = create_scale_profile_files(
        tmp_path
    )

    configuration = (
        build_default_nucleus_configuration()
    )

    evidence = build_nucleus_evidence(
        profile,
        configuration,
    )

    expected = compute_normalized_local_contrast(
        profile[profile_name],
        radius=configuration[
            "local_contrast_radius"
        ],
    )

    np.testing.assert_array_equal(
        evidence[evidence_name],
        expected,
    )


def test_build_nucleus_evidence_applies_local_contrast_radius(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        profile,
        _,
    ) = create_scale_profile_files(
        tmp_path
    )

    configuration = (
        build_default_nucleus_configuration()
    )
    configuration["local_contrast_radius"] = 1

    evidence = build_nucleus_evidence(
        profile,
        configuration,
    )

    expected = compute_normalized_local_contrast(
        profile["frame_total"],
        radius=1,
    )

    np.testing.assert_array_equal(
        evidence["frame_total_level"],
        expected,
    )


def test_build_nucleus_evidence_does_not_modify_profile(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        profile,
        _,
    ) = create_scale_profile_files(
        tmp_path
    )

    original_frame_total = profile[
        "frame_total"
    ].copy()

    evidence = build_nucleus_evidence(
        profile,
        build_default_nucleus_configuration(),
    )

    evidence["frame_total_level"][0] = 999.0

    np.testing.assert_array_equal(
        profile["frame_total"],
        original_frame_total,
    )

    assert not np.shares_memory(
        evidence["frame_total_level"],
        profile["frame_total"],
    )


def test_build_nucleus_evidence_rejects_invalid_profile(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        profile,
        _,
    ) = create_scale_profile_files(
        tmp_path
    )

    profile["magnitude"][0, 0] = -1.0

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        build_nucleus_evidence(
            profile,
            build_default_nucleus_configuration(),
        )


def test_build_nucleus_evidence_rejects_invalid_configuration(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        profile,
        _,
    ) = create_scale_profile_files(
        tmp_path
    )

    configuration = (
        build_default_nucleus_configuration()
    )
    configuration["local_contrast_radius"] = 0

    with pytest.raises(
        ValueError,
        match="Local contrast radius must be at least 1",
    ):
        build_nucleus_evidence(
            profile,
            configuration,
        )


#8.10
def test_compute_nucleus_strength_returns_feature_mean():
    evidence = {
        key: np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    result = compute_nucleus_strength(
        evidence,
        build_default_nucleus_configuration(),
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        ),
    )


def test_compute_nucleus_strength_applies_single_feature_weight():
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    evidence["frame_total_level"] = np.array(
        [1.0, 3.0, 5.0],
        dtype=np.float64,
    )

    configuration = (
        build_default_nucleus_configuration()
    )

    for name in configuration["feature_weights"]:
        configuration["feature_weights"][name] = 0.0

    configuration["feature_weights"][
        "frame_total"
    ] = 2.0

    result = compute_nucleus_strength(
        evidence,
        configuration,
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [1.0, 3.0, 5.0],
            dtype=np.float64,
        ),
    )


def test_compute_nucleus_strength_returns_weighted_mean():
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    evidence["frame_total_level"] = np.array(
        [0.0, 2.0, 4.0],
        dtype=np.float64,
    )
    evidence["frame_mean_level"] = np.array(
        [0.0, 4.0, 8.0],
        dtype=np.float64,
    )

    configuration = (
        build_default_nucleus_configuration()
    )

    for name in configuration["feature_weights"]:
        configuration["feature_weights"][name] = 0.0

    configuration["feature_weights"][
        "frame_total"
    ] = 1.0
    configuration["feature_weights"][
        "frame_mean"
    ] = 3.0

    result = compute_nucleus_strength(
        evidence,
        configuration,
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [0.0, 3.5, 7.0],
            dtype=np.float64,
        ),
    )


def test_compute_nucleus_strength_returns_independent_float64():
    evidence = {
        key: np.array(
            [0, 1, 2],
            dtype=np.int64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    original = {
        key: values.copy()
        for key, values in evidence.items()
    }

    result = compute_nucleus_strength(
        evidence,
        build_default_nucleus_configuration(),
    )

    result[0] = 999.0

    for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            evidence[key],
            original[key],
        )

    assert result.dtype == np.float64

    for values in evidence.values():
        assert not np.shares_memory(
            result,
            values,
        )


def test_compute_nucleus_strength_returns_finite_nonnegative_series():
    evidence = {
        key: np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    result = compute_nucleus_strength(
        evidence,
        build_default_nucleus_configuration(),
    )

    assert result.shape == (3,)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_compute_nucleus_strength_rejects_missing_key():
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    del evidence["frame_total_level"]

    with pytest.raises(
        ValueError,
        match="keys do not match",
    ):
        compute_nucleus_strength(
            evidence,
            build_default_nucleus_configuration(),
        )


def test_compute_nucleus_strength_rejects_unexpected_key():
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    evidence["unexpected"] = np.zeros(
        3,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="keys do not match",
    ):
        compute_nucleus_strength(
            evidence,
            build_default_nucleus_configuration(),
        )


def test_compute_nucleus_strength_rejects_shape_mismatch():
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    evidence["frame_mean_level"] = np.zeros(
        2,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="unexpected shape",
    ):
        compute_nucleus_strength(
            evidence,
            build_default_nucleus_configuration(),
        )


def test_compute_nucleus_strength_rejects_non_1d_evidence():
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    evidence["frame_total_level"] = np.zeros(
        (1, 3),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        compute_nucleus_strength(
            evidence,
            build_default_nucleus_configuration(),
        )


def test_compute_nucleus_strength_rejects_empty_evidence():
    evidence = {
        key: np.array(
            [],
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    with pytest.raises(
        ValueError,
        match="at least one frame",
    ):
        compute_nucleus_strength(
            evidence,
            build_default_nucleus_configuration(),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_compute_nucleus_strength_rejects_nonfinite_evidence(
    invalid_value: float,
):
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    evidence["frame_total_level"][1] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        compute_nucleus_strength(
            evidence,
            build_default_nucleus_configuration(),
        )


def test_compute_nucleus_strength_rejects_negative_evidence():
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    evidence["frame_total_level"][1] = -1.0

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        compute_nucleus_strength(
            evidence,
            build_default_nucleus_configuration(),
        )


def test_compute_nucleus_strength_rejects_invalid_configuration():
    evidence = {
        key: np.zeros(
            3,
            dtype=np.float64,
        )
        for key in NUCLEUS_EVIDENCE_REQUIRED_KEYS
    }

    configuration = (
        build_default_nucleus_configuration()
    )
    configuration["threshold"] = -1.0

    with pytest.raises(
        ValueError,
        match="threshold must be non-negative",
    ):
        compute_nucleus_strength(
            evidence,
            configuration,
        )


#8.11
def test_validate_nucleus_strength_accepts_valid_series():
    strength = np.array(
        [1.0, 0.5, 2.0, 0.0],
        dtype=np.float64,
    )

    validated = validate_nucleus_strength(
        strength,
        expected_frame_count=4,
    )

    np.testing.assert_array_equal(
        validated,
        strength,
    )


def test_validate_nucleus_strength_accepts_positive_first_frame():
    strength = np.array(
        [3.0, 1.0, 0.0],
        dtype=np.float64,
    )

    validated = validate_nucleus_strength(
        strength,
        expected_frame_count=3,
    )

    assert validated[0] == pytest.approx(3.0)


def test_validate_nucleus_strength_returns_independent_float64():
    strength = np.array(
        [1, 2, 0],
        dtype=np.int64,
    )

    validated = validate_nucleus_strength(
        strength,
        expected_frame_count=3,
    )

    validated[1] = 999.0

    assert strength[1] == 2
    assert validated.dtype == np.float64
    assert not np.shares_memory(
        validated,
        strength,
    )


def test_validate_nucleus_strength_rejects_non_1d_series():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_nucleus_strength(
            np.zeros(
                (2, 3),
                dtype=np.float64,
            ),
            expected_frame_count=6,
        )


def test_validate_nucleus_strength_rejects_frame_count_mismatch():
    with pytest.raises(
        ValueError,
        match="expected frame count",
    ):
        validate_nucleus_strength(
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
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_validate_nucleus_strength_rejects_nonfinite_values(
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
        validate_nucleus_strength(
            strength,
            expected_frame_count=3,
        )


def test_validate_nucleus_strength_rejects_negative_values():
    strength = np.array(
        [0.0, -1.0, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        validate_nucleus_strength(
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
    ids=[
        "boolean",
        "float",
        "string",
    ],
)
def test_validate_nucleus_strength_rejects_invalid_frame_count_type(
    invalid_count: object,
):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        validate_nucleus_strength(
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
    ids=[
        "zero",
        "negative",
    ],
)
def test_validate_nucleus_strength_rejects_nonpositive_frame_count(
    invalid_count: int,
):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        validate_nucleus_strength(
            np.zeros(
                1,
                dtype=np.float64,
            ),
            expected_frame_count=invalid_count,
        )


#8.12
def test_find_nucleus_local_maxima_finds_isolated_peaks():
    strength = np.array(
        [0.0, 1.0, 3.0, 1.0, 2.0, 0.0],
        dtype=np.float64,
    )

    result = find_nucleus_local_maxima(
        strength
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [2, 4],
            dtype=np.int64,
        ),
    )


def test_find_nucleus_local_maxima_uses_first_plateau_index():
    strength = np.array(
        [0.0, 1.0, 3.0, 3.0, 3.0, 1.0],
        dtype=np.float64,
    )

    result = find_nucleus_local_maxima(
        strength
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [2],
            dtype=np.int64,
        ),
    )


@pytest.mark.parametrize(
    (
        "strength",
        "expected",
    ),
    [
        (
            np.array(
                [3.0, 1.0, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [0],
                dtype=np.int64,
            ),
        ),
        (
            np.array(
                [0.0, 1.0, 3.0],
                dtype=np.float64,
            ),
            np.array(
                [2],
                dtype=np.int64,
            ),
        ),
    ],
    ids=[
        "first-frame",
        "last-frame",
    ],
)
def test_find_nucleus_local_maxima_accepts_endpoint_peaks(
    strength: np.ndarray,
    expected: np.ndarray,
):
    result = find_nucleus_local_maxima(
        strength
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


@pytest.mark.parametrize(
    (
        "strength",
        "expected",
    ),
    [
        (
            np.array(
                [3.0, 3.0, 1.0],
                dtype=np.float64,
            ),
            np.array(
                [0],
                dtype=np.int64,
            ),
        ),
        (
            np.array(
                [1.0, 3.0, 3.0],
                dtype=np.float64,
            ),
            np.array(
                [1],
                dtype=np.int64,
            ),
        ),
    ],
    ids=[
        "first-plateau",
        "last-plateau",
    ],
)
def test_find_nucleus_local_maxima_accepts_endpoint_plateaus(
    strength: np.ndarray,
    expected: np.ndarray,
):
    result = find_nucleus_local_maxima(
        strength
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


@pytest.mark.parametrize(
    (
        "strength",
        "expected",
    ),
    [
        (
            np.array(
                [0.0, 1.0, 2.0, 3.0],
                dtype=np.float64,
            ),
            np.array(
                [3],
                dtype=np.int64,
            ),
        ),
        (
            np.array(
                [3.0, 2.0, 1.0, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [0],
                dtype=np.int64,
            ),
        ),
    ],
    ids=[
        "increasing",
        "decreasing",
    ],
)
def test_find_nucleus_local_maxima_handles_monotonic_series(
    strength: np.ndarray,
    expected: np.ndarray,
):
    result = find_nucleus_local_maxima(
        strength
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_find_nucleus_local_maxima_returns_empty_for_zero_series():
    result = find_nucleus_local_maxima(
        np.zeros(
            5,
            dtype=np.float64,
        )
    )

    assert result.dtype == np.int64
    assert result.size == 0


@pytest.mark.parametrize(
    (
        "strength",
        "expected",
    ),
    [
        (
            np.array(
                [2.0],
                dtype=np.float64,
            ),
            np.array(
                [0],
                dtype=np.int64,
            ),
        ),
        (
            np.array(
                [0.0],
                dtype=np.float64,
            ),
            np.array(
                [],
                dtype=np.int64,
            ),
        ),
    ],
    ids=[
        "positive",
        "zero",
    ],
)
def test_find_nucleus_local_maxima_handles_single_frame(
    strength: np.ndarray,
    expected: np.ndarray,
):
    result = find_nucleus_local_maxima(
        strength
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


@pytest.mark.parametrize(
    (
        "strength",
        "expected",
    ),
    [
        (
            np.array(
                [2.0, 1.0],
                dtype=np.float64,
            ),
            np.array(
                [0],
                dtype=np.int64,
            ),
        ),
        (
            np.array(
                [1.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [1],
                dtype=np.int64,
            ),
        ),
        (
            np.array(
                [2.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [0],
                dtype=np.int64,
            ),
        ),
    ],
    ids=[
        "first-higher",
        "last-higher",
        "plateau",
    ],
)
def test_find_nucleus_local_maxima_handles_two_frames(
    strength: np.ndarray,
    expected: np.ndarray,
):
    result = find_nucleus_local_maxima(
        strength
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_find_nucleus_local_maxima_rejects_unsupported_policy():
    strength = np.array(
        [0.0, 2.0, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported Nucleus plateau policy",
    ):
        find_nucleus_local_maxima(
            strength,
            plateau_policy="last-index",
        )


def test_find_nucleus_local_maxima_rejects_nonstring_policy():
    strength = np.array(
        [0.0, 2.0, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        find_nucleus_local_maxima(
            strength,
            plateau_policy=1,
        )


def test_find_nucleus_local_maxima_rejects_negative_strength():
    strength = np.array(
        [0.0, -1.0, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        find_nucleus_local_maxima(
            strength
        )


#8.13
def test_apply_nucleus_threshold_keeps_qualifying_candidates():
    strength = np.array(
        [0.5, 2.0, 1.0, 3.0, 0.0],
        dtype=np.float64,
    )

    indices = np.array(
        [0, 1, 3],
        dtype=np.int64,
    )

    result = apply_nucleus_threshold(
        strength,
        indices,
        threshold=2.0,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [1, 3],
            dtype=np.int64,
        ),
    )


def test_apply_nucleus_threshold_is_inclusive():
    strength = np.array(
        [0.0, 2.0, 0.0],
        dtype=np.float64,
    )

    result = apply_nucleus_threshold(
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


def test_apply_nucleus_threshold_can_remove_all_candidates():
    strength = np.array(
        [1.0, 2.0, 1.0],
        dtype=np.float64,
    )

    result = apply_nucleus_threshold(
        strength,
        np.array(
            [0, 1, 2],
            dtype=np.int64,
        ),
        threshold=3.0,
    )

    assert result.dtype == np.int64
    assert result.size == 0


def test_apply_nucleus_threshold_accepts_empty_candidates():
    result = apply_nucleus_threshold(
        np.array(
            [1.0, 2.0, 1.0],
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


def test_apply_nucleus_threshold_keeps_endpoint_candidates():
    strength = np.array(
        [3.0, 1.0, 2.0],
        dtype=np.float64,
    )

    result = apply_nucleus_threshold(
        strength,
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
        threshold=2.0,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )


def test_apply_nucleus_threshold_returns_independent_int64():
    indices = np.array(
        [0, 2],
        dtype=np.int32,
    )

    result = apply_nucleus_threshold(
        np.array(
            [3.0, 0.0, 2.0],
            dtype=np.float64,
        ),
        indices,
        threshold=1.0,
    )

    result[0] = 1

    assert indices[0] == 0
    assert result.dtype == np.int64
    assert not np.shares_memory(
        result,
        indices,
    )


@pytest.mark.parametrize(
    "indices",
    [
        np.array(
            [0.0, 2.0],
            dtype=np.float64,
        ),
        np.array(
            [True, False],
            dtype=np.bool_,
        ),
    ],
    ids=[
        "float",
        "boolean",
    ],
)
def test_apply_nucleus_threshold_rejects_noninteger_indices(
    indices: np.ndarray,
):
    with pytest.raises(
        TypeError,
        match="integer dtype",
    ):
        apply_nucleus_threshold(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            indices,
            threshold=1.0,
        )


def test_apply_nucleus_threshold_rejects_non_1d_indices():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        apply_nucleus_threshold(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [[0, 2]],
                dtype=np.int64,
            ),
            threshold=1.0,
        )


@pytest.mark.parametrize(
    "invalid_index",
    [
        -1,
        3,
    ],
    ids=[
        "negative",
        "past-end",
    ],
)
def test_apply_nucleus_threshold_rejects_out_of_range_index(
    invalid_index: int,
):
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        apply_nucleus_threshold(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [invalid_index],
                dtype=np.int64,
            ),
            threshold=1.0,
        )


def test_apply_nucleus_threshold_rejects_duplicate_indices():
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        apply_nucleus_threshold(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [0, 0],
                dtype=np.int64,
            ),
            threshold=1.0,
        )


def test_apply_nucleus_threshold_rejects_unsorted_indices():
    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        apply_nucleus_threshold(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [2, 0],
                dtype=np.int64,
            ),
            threshold=1.0,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        True,
        "1.0",
        None,
    ],
    ids=[
        "boolean",
        "string",
        "none",
    ],
)
def test_apply_nucleus_threshold_rejects_invalid_threshold_type(
    invalid_threshold: object,
):
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        apply_nucleus_threshold(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [0, 2],
                dtype=np.int64,
            ),
            threshold=invalid_threshold,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
    ids=[
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_apply_nucleus_threshold_rejects_invalid_threshold_value(
    invalid_threshold: float,
):
    with pytest.raises(
        ValueError,
        match="finite|non-negative",
    ):
        apply_nucleus_threshold(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [0, 2],
                dtype=np.int64,
            ),
            threshold=invalid_threshold,
        )


def test_apply_nucleus_threshold_rejects_invalid_strength():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        apply_nucleus_threshold(
            np.array(
                [1.0, -1.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [0, 2],
                dtype=np.int64,
            ),
            threshold=1.0,
        )


#8.14
def test_enforce_minimum_nucleus_candidate_distance_keeps_strongest():
    strength = np.array(
        [0.0, 3.0, 5.0, 4.0, 0.0],
        dtype=np.float64,
    )

    result = enforce_minimum_nucleus_candidate_distance(
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


def test_enforce_minimum_nucleus_candidate_distance_keeps_separated():
    strength = np.array(
        [0.0, 3.0, 0.0, 0.0, 2.0, 0.0],
        dtype=np.float64,
    )

    result = enforce_minimum_nucleus_candidate_distance(
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


def test_enforce_minimum_nucleus_candidate_distance_prefers_lower_index():
    strength = np.array(
        [0.0, 4.0, 4.0, 0.0],
        dtype=np.float64,
    )

    result = enforce_minimum_nucleus_candidate_distance(
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


def test_enforce_minimum_nucleus_candidate_distance_one_keeps_all():
    strength = np.array(
        [1.0, 3.0, 2.0, 4.0],
        dtype=np.float64,
    )

    result = enforce_minimum_nucleus_candidate_distance(
        strength,
        np.array(
            [0, 1, 2, 3],
            dtype=np.int64,
        ),
        minimum_distance=1,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [0, 1, 2, 3],
            dtype=np.int64,
        ),
    )


def test_enforce_minimum_nucleus_candidate_distance_handles_endpoints():
    strength = np.array(
        [4.0, 0.0, 0.0, 3.0],
        dtype=np.float64,
    )

    result = enforce_minimum_nucleus_candidate_distance(
        strength,
        np.array(
            [0, 3],
            dtype=np.int64,
        ),
        minimum_distance=2,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [0, 3],
            dtype=np.int64,
        ),
    )


def test_enforce_minimum_nucleus_candidate_distance_accepts_empty():
    result = enforce_minimum_nucleus_candidate_distance(
        np.array(
            [1.0, 2.0, 1.0],
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


def test_enforce_minimum_nucleus_candidate_distance_accepts_single():
    result = enforce_minimum_nucleus_candidate_distance(
        np.array(
            [1.0, 3.0, 1.0],
            dtype=np.float64,
        ),
        np.array(
            [1],
            dtype=np.int64,
        ),
        minimum_distance=10,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [1],
            dtype=np.int64,
        ),
    )


def test_enforce_minimum_nucleus_candidate_distance_returns_independent_int64():
    indices = np.array(
        [0, 3],
        dtype=np.int32,
    )

    result = enforce_minimum_nucleus_candidate_distance(
        np.array(
            [4.0, 0.0, 0.0, 3.0],
            dtype=np.float64,
        ),
        indices,
        minimum_distance=2,
    )

    result[0] = 1

    assert indices[0] == 0
    assert result.dtype == np.int64
    assert not np.shares_memory(
        result,
        indices,
    )


@pytest.mark.parametrize(
    "indices",
    [
        np.array(
            [0.0, 2.0],
            dtype=np.float64,
        ),
        np.array(
            [True, False],
            dtype=np.bool_,
        ),
    ],
    ids=[
        "float",
        "boolean",
    ],
)
def test_enforce_minimum_nucleus_candidate_distance_rejects_noninteger_indices(
    indices: np.ndarray,
):
    with pytest.raises(
        TypeError,
        match="integer dtype",
    ):
        enforce_minimum_nucleus_candidate_distance(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            indices,
            minimum_distance=2,
        )


def test_enforce_minimum_nucleus_candidate_distance_rejects_non_1d_indices():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        enforce_minimum_nucleus_candidate_distance(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [[0, 2]],
                dtype=np.int64,
            ),
            minimum_distance=2,
        )


@pytest.mark.parametrize(
    "invalid_index",
    [
        -1,
        3,
    ],
    ids=[
        "negative",
        "past-end",
    ],
)
def test_enforce_minimum_nucleus_candidate_distance_rejects_out_of_range(
    invalid_index: int,
):
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        enforce_minimum_nucleus_candidate_distance(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [invalid_index],
                dtype=np.int64,
            ),
            minimum_distance=2,
        )


def test_enforce_minimum_nucleus_candidate_distance_rejects_duplicates():
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        enforce_minimum_nucleus_candidate_distance(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [0, 0],
                dtype=np.int64,
            ),
            minimum_distance=2,
        )


def test_enforce_minimum_nucleus_candidate_distance_rejects_unsorted():
    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        enforce_minimum_nucleus_candidate_distance(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [2, 0],
                dtype=np.int64,
            ),
            minimum_distance=2,
        )


@pytest.mark.parametrize(
    "invalid_distance",
    [
        True,
        1.5,
        "2",
    ],
    ids=[
        "boolean",
        "float",
        "string",
    ],
)
def test_enforce_minimum_nucleus_candidate_distance_rejects_invalid_type(
    invalid_distance: object,
):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        enforce_minimum_nucleus_candidate_distance(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [0, 2],
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
    ids=[
        "zero",
        "negative",
    ],
)
def test_enforce_minimum_nucleus_candidate_distance_rejects_nonpositive(
    invalid_distance: int,
):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        enforce_minimum_nucleus_candidate_distance(
            np.array(
                [3.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [0, 2],
                dtype=np.int64,
            ),
            minimum_distance=invalid_distance,
        )


def test_enforce_minimum_nucleus_candidate_distance_rejects_invalid_strength():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        enforce_minimum_nucleus_candidate_distance(
            np.array(
                [3.0, -1.0, 2.0],
                dtype=np.float64,
            ),
            np.array(
                [0, 2],
                dtype=np.int64,
            ),
            minimum_distance=2,
        )


#8.15
def test_exclude_boundary_adjacent_nucleus_candidates_excludes_nearby():
    kept, excluded_count = (
        exclude_boundary_adjacent_nucleus_candidates(
            np.array(
                [2, 5, 8],
                dtype=np.int64,
            ),
            np.array(
                [4, 9],
                dtype=np.int64,
            ),
            exclusion_distance=1,
            frame_count=10,
        )
    )

    np.testing.assert_array_equal(
        kept,
        np.array(
            [2],
            dtype=np.int64,
        ),
    )
    assert excluded_count == 2


def test_exclude_boundary_adjacent_nucleus_candidates_excludes_same_frame():
    kept, excluded_count = (
        exclude_boundary_adjacent_nucleus_candidates(
            np.array(
                [1, 3, 5],
                dtype=np.int64,
            ),
            np.array(
                [3],
                dtype=np.int64,
            ),
            exclusion_distance=1,
            frame_count=6,
        )
    )

    np.testing.assert_array_equal(
        kept,
        np.array(
            [1, 5],
            dtype=np.int64,
        ),
    )
    assert excluded_count == 1


def test_exclude_boundary_adjacent_nucleus_candidates_keeps_distant_candidates():
    kept, excluded_count = (
        exclude_boundary_adjacent_nucleus_candidates(
            np.array(
                [1, 5, 9],
                dtype=np.int64,
            ),
            np.array(
                [3, 7],
                dtype=np.int64,
            ),
            exclusion_distance=1,
            frame_count=10,
        )
    )

    np.testing.assert_array_equal(
        kept,
        np.array(
            [1, 5, 9],
            dtype=np.int64,
        ),
    )
    assert excluded_count == 0


def test_exclude_boundary_adjacent_nucleus_candidates_zero_disables_exclusion():
    nucleus_indices = np.array(
        [1, 3, 5],
        dtype=np.int64,
    )

    kept, excluded_count = (
        exclude_boundary_adjacent_nucleus_candidates(
            nucleus_indices,
            np.array(
                [1, 3, 5],
                dtype=np.int64,
            ),
            exclusion_distance=0,
            frame_count=6,
        )
    )

    np.testing.assert_array_equal(
        kept,
        nucleus_indices,
    )
    assert excluded_count == 0
    assert kept is not nucleus_indices
    assert not np.shares_memory(
        kept,
        nucleus_indices,
    )


def test_exclude_boundary_adjacent_nucleus_candidates_accepts_empty_boundary():
    nucleus_indices = np.array(
        [1, 3],
        dtype=np.int64,
    )

    kept, excluded_count = (
        exclude_boundary_adjacent_nucleus_candidates(
            nucleus_indices,
            np.array(
                [],
                dtype=np.int64,
            ),
            exclusion_distance=1,
            frame_count=5,
        )
    )

    np.testing.assert_array_equal(
        kept,
        nucleus_indices,
    )
    assert excluded_count == 0


def test_exclude_boundary_adjacent_nucleus_candidates_accepts_empty_nucleus():
    kept, excluded_count = (
        exclude_boundary_adjacent_nucleus_candidates(
            np.array(
                [],
                dtype=np.int64,
            ),
            np.array(
                [1, 3],
                dtype=np.int64,
            ),
            exclusion_distance=1,
            frame_count=5,
        )
    )

    assert kept.dtype == np.int64
    assert kept.size == 0
    assert excluded_count == 0


def test_exclude_boundary_adjacent_nucleus_candidates_accepts_both_empty():
    kept, excluded_count = (
        exclude_boundary_adjacent_nucleus_candidates(
            np.array(
                [],
                dtype=np.int64,
            ),
            np.array(
                [],
                dtype=np.int64,
            ),
            exclusion_distance=1,
            frame_count=5,
        )
    )

    assert kept.dtype == np.int64
    assert kept.size == 0
    assert excluded_count == 0


def test_exclude_boundary_adjacent_nucleus_candidates_handles_endpoints():
    kept, excluded_count = (
        exclude_boundary_adjacent_nucleus_candidates(
            np.array(
                [0, 4],
                dtype=np.int64,
            ),
            np.array(
                [1, 3],
                dtype=np.int64,
            ),
            exclusion_distance=1,
            frame_count=5,
        )
    )

    assert kept.dtype == np.int64
    assert kept.size == 0
    assert excluded_count == 2


def test_exclude_boundary_adjacent_nucleus_candidates_returns_independent_int64():
    nucleus_indices = np.array(
        [1, 5],
        dtype=np.int32,
    )

    kept, excluded_count = (
        exclude_boundary_adjacent_nucleus_candidates(
            nucleus_indices,
            np.array(
                [3],
                dtype=np.int32,
            ),
            exclusion_distance=1,
            frame_count=7,
        )
    )

    kept[0] = 2

    assert nucleus_indices[0] == 1
    assert kept.dtype == np.int64
    assert excluded_count == 0
    assert not np.shares_memory(
        kept,
        nucleus_indices,
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "nucleus",
        "boundary",
    ],
)
@pytest.mark.parametrize(
    "invalid_indices",
    [
        np.array(
            [1.0, 3.0],
            dtype=np.float64,
        ),
        np.array(
            [True, False],
            dtype=np.bool_,
        ),
    ],
    ids=[
        "float",
        "boolean",
    ],
)
def test_exclude_boundary_adjacent_nucleus_candidates_rejects_noninteger_indices(
    field_name: str,
    invalid_indices: np.ndarray,
):
    nucleus_indices = np.array(
        [1, 3],
        dtype=np.int64,
    )
    boundary_indices = np.array(
        [2],
        dtype=np.int64,
    )

    if field_name == "nucleus":
        nucleus_indices = invalid_indices
    else:
        boundary_indices = invalid_indices

    with pytest.raises(
        TypeError,
        match="integer dtype",
    ):
        exclude_boundary_adjacent_nucleus_candidates(
            nucleus_indices,
            boundary_indices,
            exclusion_distance=1,
            frame_count=5,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "nucleus",
        "boundary",
    ],
)
def test_exclude_boundary_adjacent_nucleus_candidates_rejects_non_1d_indices(
    field_name: str,
):
    nucleus_indices = np.array(
        [1, 3],
        dtype=np.int64,
    )
    boundary_indices = np.array(
        [2],
        dtype=np.int64,
    )

    if field_name == "nucleus":
        nucleus_indices = np.array(
            [[1, 3]],
            dtype=np.int64,
        )
    else:
        boundary_indices = np.array(
            [[2]],
            dtype=np.int64,
        )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        exclude_boundary_adjacent_nucleus_candidates(
            nucleus_indices,
            boundary_indices,
            exclusion_distance=1,
            frame_count=5,
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_index",
    ),
    [
        ("nucleus", -1),
        ("nucleus", 5),
        ("boundary", -1),
        ("boundary", 5),
    ],
)
def test_exclude_boundary_adjacent_nucleus_candidates_rejects_out_of_range(
    field_name: str,
    invalid_index: int,
):
    nucleus_indices = np.array(
        [1],
        dtype=np.int64,
    )
    boundary_indices = np.array(
        [2],
        dtype=np.int64,
    )

    if field_name == "nucleus":
        nucleus_indices = np.array(
            [invalid_index],
            dtype=np.int64,
        )
    else:
        boundary_indices = np.array(
            [invalid_index],
            dtype=np.int64,
        )

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        exclude_boundary_adjacent_nucleus_candidates(
            nucleus_indices,
            boundary_indices,
            exclusion_distance=1,
            frame_count=5,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "nucleus",
        "boundary",
    ],
)
def test_exclude_boundary_adjacent_nucleus_candidates_rejects_duplicates(
    field_name: str,
):
    nucleus_indices = np.array(
        [1, 3],
        dtype=np.int64,
    )
    boundary_indices = np.array(
        [2, 4],
        dtype=np.int64,
    )

    if field_name == "nucleus":
        nucleus_indices = np.array(
            [1, 1],
            dtype=np.int64,
        )
    else:
        boundary_indices = np.array(
            [2, 2],
            dtype=np.int64,
        )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        exclude_boundary_adjacent_nucleus_candidates(
            nucleus_indices,
            boundary_indices,
            exclusion_distance=1,
            frame_count=5,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "nucleus",
        "boundary",
    ],
)
def test_exclude_boundary_adjacent_nucleus_candidates_rejects_unsorted(
    field_name: str,
):
    nucleus_indices = np.array(
        [1, 3],
        dtype=np.int64,
    )
    boundary_indices = np.array(
        [2, 4],
        dtype=np.int64,
    )

    if field_name == "nucleus":
        nucleus_indices = np.array(
            [3, 1],
            dtype=np.int64,
        )
    else:
        boundary_indices = np.array(
            [4, 2],
            dtype=np.int64,
        )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        exclude_boundary_adjacent_nucleus_candidates(
            nucleus_indices,
            boundary_indices,
            exclusion_distance=1,
            frame_count=5,
        )


@pytest.mark.parametrize(
    "invalid_distance",
    [
        True,
        1.5,
        "1",
    ],
    ids=[
        "boolean",
        "float",
        "string",
    ],
)
def test_exclude_boundary_adjacent_nucleus_candidates_rejects_invalid_distance_type(
    invalid_distance: object,
):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        exclude_boundary_adjacent_nucleus_candidates(
            np.array(
                [1],
                dtype=np.int64,
            ),
            np.array(
                [2],
                dtype=np.int64,
            ),
            exclusion_distance=invalid_distance,
            frame_count=5,
        )


def test_exclude_boundary_adjacent_nucleus_candidates_rejects_negative_distance():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        exclude_boundary_adjacent_nucleus_candidates(
            np.array(
                [1],
                dtype=np.int64,
            ),
            np.array(
                [2],
                dtype=np.int64,
            ),
            exclusion_distance=-1,
            frame_count=5,
        )


@pytest.mark.parametrize(
    "invalid_count",
    [
        True,
        5.0,
        "5",
    ],
    ids=[
        "boolean",
        "float",
        "string",
    ],
)
def test_exclude_boundary_adjacent_nucleus_candidates_rejects_invalid_frame_count_type(
    invalid_count: object,
):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        exclude_boundary_adjacent_nucleus_candidates(
            np.array(
                [1],
                dtype=np.int64,
            ),
            np.array(
                [2],
                dtype=np.int64,
            ),
            exclusion_distance=1,
            frame_count=invalid_count,
        )


@pytest.mark.parametrize(
    "invalid_count",
    [
        0,
        -1,
    ],
    ids=[
        "zero",
        "negative",
    ],
)
def test_exclude_boundary_adjacent_nucleus_candidates_rejects_nonpositive_frame_count(
    invalid_count: int,
):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        exclude_boundary_adjacent_nucleus_candidates(
            np.array(
                [],
                dtype=np.int64,
            ),
            np.array(
                [],
                dtype=np.int64,
            ),
            exclusion_distance=1,
            frame_count=invalid_count,
        )


#8.16
def test_build_nucleus_candidates_returns_expected_arrays():
    strength = np.array(
        [2.0, 0.5, 4.0, 1.0],
        dtype=np.float64,
    )

    candidates = build_nucleus_candidates(
        strength,
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )

    assert set(candidates) == set(
        NUCLEUS_CANDIDATE_REQUIRED_KEYS
    )

    np.testing.assert_array_equal(
        candidates["nucleus_strength"],
        strength,
    )

    np.testing.assert_array_equal(
        candidates["candidate_frame_indices"],
        np.array(
            [0, 2],
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


def test_build_nucleus_candidates_accepts_endpoint_candidates():
    candidates = build_nucleus_candidates(
        np.array(
            [3.0, 1.0, 2.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        candidates["candidate_frame_indices"],
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        candidates["candidate_strengths"],
        np.array(
            [3.0, 2.0],
            dtype=np.float64,
        ),
    )


def test_build_nucleus_candidates_accepts_empty_candidates():
    strength = np.array(
        [1.0, 2.0, 1.0],
        dtype=np.float64,
    )

    candidates = build_nucleus_candidates(
        strength,
        np.array(
            [],
            dtype=np.int64,
        ),
    )

    assert candidates[
        "candidate_frame_indices"
    ].shape == (0,)

    assert candidates[
        "candidate_frame_indices"
    ].dtype == np.int64

    assert candidates[
        "candidate_strengths"
    ].shape == (0,)

    assert candidates[
        "candidate_strengths"
    ].dtype == np.float64


def test_build_nucleus_candidates_accepts_single_candidate():
    candidates = build_nucleus_candidates(
        np.array(
            [1.0, 3.0, 1.0],
            dtype=np.float64,
        ),
        np.array(
            [1],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        candidates["candidate_frame_indices"],
        np.array(
            [1],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        candidates["candidate_strengths"],
        np.array(
            [3.0],
            dtype=np.float64,
        ),
    )


def test_build_nucleus_candidates_returns_independent_arrays():
    strength = np.array(
        [2, 1, 4],
        dtype=np.int64,
    )

    indices = np.array(
        [0, 2],
        dtype=np.int32,
    )

    candidates = build_nucleus_candidates(
        strength,
        indices,
    )

    assert candidates[
        "nucleus_strength"
    ].dtype == np.float64

    assert candidates[
        "candidate_frame_indices"
    ].dtype == np.int64

    assert candidates[
        "candidate_strengths"
    ].dtype == np.float64

    candidates["nucleus_strength"][0] = 999.0
    candidates["candidate_frame_indices"][0] = 1
    candidates["candidate_strengths"][0] = 999.0

    assert strength[0] == 2
    assert indices[0] == 0

    assert not np.shares_memory(
        candidates["nucleus_strength"],
        strength,
    )

    assert not np.shares_memory(
        candidates["candidate_frame_indices"],
        indices,
    )

    assert not np.shares_memory(
        candidates["candidate_strengths"],
        strength,
    )


@pytest.mark.parametrize(
    "indices",
    [
        np.array(
            [0.0, 2.0],
            dtype=np.float64,
        ),
        np.array(
            [True, False],
            dtype=np.bool_,
        ),
    ],
    ids=[
        "float",
        "boolean",
    ],
)
def test_build_nucleus_candidates_rejects_noninteger_indices(
    indices: np.ndarray,
):
    with pytest.raises(
        TypeError,
        match="integer dtype",
    ):
        build_nucleus_candidates(
            np.array(
                [2.0, 1.0, 4.0],
                dtype=np.float64,
            ),
            indices,
        )


def test_build_nucleus_candidates_rejects_non_1d_indices():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        build_nucleus_candidates(
            np.array(
                [2.0, 1.0, 4.0],
                dtype=np.float64,
            ),
            np.array(
                [[0, 2]],
                dtype=np.int64,
            ),
        )


@pytest.mark.parametrize(
    "invalid_index",
    [
        -1,
        3,
    ],
    ids=[
        "negative",
        "past-end",
    ],
)
def test_build_nucleus_candidates_rejects_out_of_range_index(
    invalid_index: int,
):
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        build_nucleus_candidates(
            np.array(
                [2.0, 1.0, 4.0],
                dtype=np.float64,
            ),
            np.array(
                [invalid_index],
                dtype=np.int64,
            ),
        )


def test_build_nucleus_candidates_rejects_duplicate_indices():
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        build_nucleus_candidates(
            np.array(
                [2.0, 1.0, 4.0],
                dtype=np.float64,
            ),
            np.array(
                [0, 0],
                dtype=np.int64,
            ),
        )


def test_build_nucleus_candidates_rejects_unsorted_indices():
    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        build_nucleus_candidates(
            np.array(
                [2.0, 1.0, 4.0],
                dtype=np.float64,
            ),
            np.array(
                [2, 0],
                dtype=np.int64,
            ),
        )


def test_build_nucleus_candidates_rejects_invalid_strength():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        build_nucleus_candidates(
            np.array(
                [2.0, -1.0, 4.0],
                dtype=np.float64,
            ),
            np.array(
                [0, 2],
                dtype=np.int64,
            ),
        )


#8.17
def test_validate_nucleus_candidates_accepts_valid_arrays():
    candidates = build_nucleus_candidates(
        np.array(
            [2.0, 0.0, 4.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )

    validated = validate_nucleus_candidates(
        candidates,
        minimum_distance=2,
    )

    for key in NUCLEUS_CANDIDATE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            validated[key],
            candidates[key],
        )


def test_validate_nucleus_candidates_accepts_endpoint_candidates():
    candidates = build_nucleus_candidates(
        np.array(
            [3.0, 0.0, 2.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )

    validated = validate_nucleus_candidates(
        candidates,
        minimum_distance=2,
    )

    np.testing.assert_array_equal(
        validated["candidate_frame_indices"],
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )


def test_validate_nucleus_candidates_accepts_empty_candidates():
    candidates = build_nucleus_candidates(
        np.array(
            [1.0, 2.0, 1.0],
            dtype=np.float64,
        ),
        np.array(
            [],
            dtype=np.int64,
        ),
    )

    validated = validate_nucleus_candidates(
        candidates,
        minimum_distance=2,
    )

    assert validated[
        "candidate_frame_indices"
    ].shape == (0,)

    assert validated[
        "candidate_strengths"
    ].shape == (0,)


def test_validate_nucleus_candidates_accepts_single_candidate():
    candidates = build_nucleus_candidates(
        np.array(
            [1.0, 3.0, 1.0],
            dtype=np.float64,
        ),
        np.array(
            [1],
            dtype=np.int64,
        ),
    )

    validated = validate_nucleus_candidates(
        candidates,
        minimum_distance=10,
    )

    np.testing.assert_array_equal(
        validated["candidate_frame_indices"],
        np.array(
            [1],
            dtype=np.int64,
        ),
    )


def test_validate_nucleus_candidates_accepts_minimum_distance_one():
    candidates = build_nucleus_candidates(
        np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 1, 2],
            dtype=np.int64,
        ),
    )

    validated = validate_nucleus_candidates(
        candidates,
        minimum_distance=1,
    )

    np.testing.assert_array_equal(
        validated["candidate_frame_indices"],
        np.array(
            [0, 1, 2],
            dtype=np.int64,
        ),
    )


def test_validate_nucleus_candidates_accepts_numpy_integer_distance():
    candidates = build_nucleus_candidates(
        np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )

    validated = validate_nucleus_candidates(
        candidates,
        minimum_distance=np.int64(2),
    )

    np.testing.assert_array_equal(
        validated["candidate_frame_indices"],
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )


def test_validate_nucleus_candidates_returns_independent_arrays():
    candidates = build_nucleus_candidates(
        np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )

    validated = validate_nucleus_candidates(
        candidates,
        minimum_distance=2,
    )

    for key in NUCLEUS_CANDIDATE_REQUIRED_KEYS:
        assert validated[key] is not candidates[key]
        assert not np.shares_memory(
            validated[key],
            candidates[key],
        )

    validated["nucleus_strength"][0] = 999.0
    validated["candidate_frame_indices"][0] = 1
    validated["candidate_strengths"][0] = 999.0

    assert candidates["nucleus_strength"][0] == 2.0
    assert candidates["candidate_frame_indices"][0] == 0
    assert candidates["candidate_strengths"][0] == 2.0


def test_validate_nucleus_candidates_rejects_missing_key():
    candidates = build_nucleus_candidates(
        np.array(
            [1.0, 3.0, 1.0],
            dtype=np.float64,
        ),
        np.array(
            [1],
            dtype=np.int64,
        ),
    )

    del candidates["candidate_strengths"]

    with pytest.raises(
        ValueError,
        match="keys do not match",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


def test_validate_nucleus_candidates_rejects_unexpected_key():
    candidates = build_nucleus_candidates(
        np.array(
            [1.0, 3.0, 1.0],
            dtype=np.float64,
        ),
        np.array(
            [1],
            dtype=np.int64,
        ),
    )

    candidates["unexpected"] = np.zeros(
        1,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="keys do not match",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


@pytest.mark.parametrize(
    "indices",
    [
        np.array(
            [0.0, 2.0],
            dtype=np.float64,
        ),
        np.array(
            [True, False],
            dtype=np.bool_,
        ),
    ],
    ids=[
        "float",
        "boolean",
    ],
)
def test_validate_nucleus_candidates_rejects_noninteger_indices(
    indices: np.ndarray,
):
    candidates = {
        "nucleus_strength": np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": indices,
        "candidate_strengths": np.array(
            [2.0, 4.0],
            dtype=np.float64,
        ),
    }

    with pytest.raises(
        TypeError,
        match="integer dtype",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


def test_validate_nucleus_candidates_rejects_non_1d_indices():
    candidates = {
        "nucleus_strength": np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": np.array(
            [[0, 2]],
            dtype=np.int64,
        ),
        "candidate_strengths": np.array(
            [2.0, 4.0],
            dtype=np.float64,
        ),
    }

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


@pytest.mark.parametrize(
    "invalid_index",
    [
        -1,
        3,
    ],
    ids=[
        "negative",
        "past-end",
    ],
)
def test_validate_nucleus_candidates_rejects_out_of_range_index(
    invalid_index: int,
):
    candidates = {
        "nucleus_strength": np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": np.array(
            [invalid_index],
            dtype=np.int64,
        ),
        "candidate_strengths": np.array(
            [2.0],
            dtype=np.float64,
        ),
    }

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


def test_validate_nucleus_candidates_rejects_duplicate_indices():
    candidates = {
        "nucleus_strength": np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": np.array(
            [0, 0],
            dtype=np.int64,
        ),
        "candidate_strengths": np.array(
            [2.0, 2.0],
            dtype=np.float64,
        ),
    }

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


def test_validate_nucleus_candidates_rejects_unsorted_indices():
    candidates = {
        "nucleus_strength": np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": np.array(
            [2, 0],
            dtype=np.int64,
        ),
        "candidate_strengths": np.array(
            [4.0, 2.0],
            dtype=np.float64,
        ),
    }

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


def test_validate_nucleus_candidates_rejects_non_1d_strengths():
    candidates = {
        "nucleus_strength": np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": np.array(
            [0, 2],
            dtype=np.int64,
        ),
        "candidate_strengths": np.array(
            [[2.0, 4.0]],
            dtype=np.float64,
        ),
    }

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


def test_validate_nucleus_candidates_rejects_strength_shape_mismatch():
    candidates = {
        "nucleus_strength": np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": np.array(
            [0, 2],
            dtype=np.int64,
        ),
        "candidate_strengths": np.array(
            [2.0],
            dtype=np.float64,
        ),
    }

    with pytest.raises(
        ValueError,
        match="shapes must match",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_validate_nucleus_candidates_rejects_nonfinite_strengths(
    invalid_value: float,
):
    candidates = {
        "nucleus_strength": np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": np.array(
            [0, 2],
            dtype=np.int64,
        ),
        "candidate_strengths": np.array(
            [invalid_value, 4.0],
            dtype=np.float64,
        ),
    }

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


def test_validate_nucleus_candidates_rejects_negative_strengths():
    candidates = {
        "nucleus_strength": np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": np.array(
            [0, 2],
            dtype=np.int64,
        ),
        "candidate_strengths": np.array(
            [-1.0, 4.0],
            dtype=np.float64,
        ),
    }

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


def test_validate_nucleus_candidates_rejects_strength_mismatch():
    candidates = build_nucleus_candidates(
        np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )

    candidates["candidate_strengths"][0] = 3.0

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


@pytest.mark.parametrize(
    "invalid_distance",
    [
        True,
        2.5,
        "2",
    ],
    ids=[
        "boolean",
        "float",
        "string",
    ],
)
def test_validate_nucleus_candidates_rejects_invalid_distance_type(
    invalid_distance: object,
):
    candidates = build_nucleus_candidates(
        np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=invalid_distance,
        )


@pytest.mark.parametrize(
    "invalid_distance",
    [
        0,
        -1,
    ],
    ids=[
        "zero",
        "negative",
    ],
)
def test_validate_nucleus_candidates_rejects_nonpositive_distance(
    invalid_distance: int,
):
    candidates = build_nucleus_candidates(
        np.array(
            [2.0, 0.0, 4.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 2],
            dtype=np.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=invalid_distance,
        )


def test_validate_nucleus_candidates_rejects_distance_violation():
    candidates = build_nucleus_candidates(
        np.array(
            [2.0, 3.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 1],
            dtype=np.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="violate the minimum distance",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=2,
        )


def test_validate_nucleus_candidates_rejects_invalid_nucleus_strength():
    candidates = {
        "nucleus_strength": np.array(
            [2.0, -1.0, 4.0],
            dtype=np.float64,
        ),
        "candidate_frame_indices": np.array(
            [0, 2],
            dtype=np.int64,
        ),
        "candidate_strengths": np.array(
            [2.0, 4.0],
            dtype=np.float64,
        ),
    }

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        validate_nucleus_candidates(
            candidates,
            minimum_distance=1,
        )


#8.18
def create_valid_nucleus_metadata_inputs():
    source_metadata = {
        "format_version": 1,
        "shape_summary": {
            "number_of_scales": 2,
            "number_of_frames": 5,
        },
        "source_cwt": {
            "minimum_frequency_hz": 10.0,
            "maximum_frequency_hz": 20.0,
        },
    }

    boundary_candidates = build_boundary_candidates(
        np.array(
            [0.0, 2.0, 0.0, 4.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [1, 3],
            dtype=np.int64,
        ),
    )

    boundary_configuration = (
        build_default_boundary_configuration()
    )

    boundary_metadata = build_boundary_metadata(
        source_metadata,
        boundary_configuration,
        boundary_candidates,
    )

    nucleus_configuration = (
        build_default_nucleus_configuration()
    )

    nucleus_candidates = build_nucleus_candidates(
        np.array(
            [1.0, 0.0, 3.0, 0.0, 2.0],
            dtype=np.float64,
        ),
        np.array(
            [0, 2, 4],
            dtype=np.int64,
        ),
    )

    return (
        source_metadata,
        boundary_metadata,
        nucleus_configuration,
        nucleus_candidates,
    )


def test_build_nucleus_metadata_returns_required_sections():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    assert set(metadata) == set(
        NUCLEUS_METADATA_REQUIRED_KEYS
    )

    assert set(
        metadata["source_scale_profile"]
    ) == set(
        SOURCE_SCALE_PROFILE_METADATA_REQUIRED_KEYS
    )

    assert set(
        metadata["source_boundary"]
    ) == set(
        SOURCE_BOUNDARY_METADATA_REQUIRED_KEYS
    )

    assert set(
        metadata["nucleus_configuration"]
    ) == set(
        NUCLEUS_CONFIGURATION_METADATA_REQUIRED_KEYS
    )

    assert set(
        metadata["shape_summary"]
    ) == set(
        NUCLEUS_SHAPE_METADATA_REQUIRED_KEYS
    )

    assert set(
        metadata["nucleus_summary"]
    ) == set(
        NUCLEUS_SUMMARY_METADATA_REQUIRED_KEYS
    )


def test_build_nucleus_metadata_records_sources_and_configuration():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    assert metadata["format_version"] == (
        NUCLEUS_FORMAT_VERSION
    )

    source_section = metadata[
        "source_scale_profile"
    ]

    assert source_section[
        "metadata_format_version"
    ] == 1

    assert source_section[
        "number_of_scales"
    ] == 2

    assert source_section[
        "number_of_frames"
    ] == 5

    assert source_section[
        "minimum_frequency_hz"
    ] == pytest.approx(10.0)

    assert source_section[
        "maximum_frequency_hz"
    ] == pytest.approx(20.0)

    boundary_section = metadata[
        "source_boundary"
    ]

    assert boundary_section[
        "metadata_format_version"
    ] == boundary_metadata["format_version"]

    assert boundary_section[
        "number_of_frames"
    ] == 5

    assert boundary_section[
        "number_of_candidates"
    ] == 2

    assert boundary_section[
        "threshold"
    ] == pytest.approx(
        boundary_metadata[
            "boundary_configuration"
        ]["threshold"]
    )

    assert boundary_section[
        "minimum_candidate_distance"
    ] == boundary_metadata[
        "boundary_configuration"
    ]["minimum_candidate_distance"]

    configuration_section = metadata[
        "nucleus_configuration"
    ]

    assert configuration_section[
        "normalization"
    ] == configuration["normalization"]

    assert configuration_section[
        "local_contrast_radius"
    ] == configuration["local_contrast_radius"]

    assert configuration_section[
        "threshold"
    ] == pytest.approx(
        configuration["threshold"]
    )

    assert configuration_section[
        "minimum_candidate_distance"
    ] == configuration[
        "minimum_candidate_distance"
    ]

    assert configuration_section[
        "boundary_exclusion_distance"
    ] == configuration[
        "boundary_exclusion_distance"
    ]

    assert configuration_section[
        "plateau_policy"
    ] == configuration["plateau_policy"]

    assert configuration_section[
        "feature_names"
    ] == list(NUCLEUS_FEATURE_NAMES)

    assert configuration_section[
        "feature_weights"
    ] == configuration["feature_weights"]


def test_build_nucleus_metadata_records_array_shapes():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    shape_summary = metadata[
        "shape_summary"
    ]

    assert shape_summary[
        "number_of_frames"
    ] == 5

    assert shape_summary[
        "number_of_candidates"
    ] == 3

    assert shape_summary[
        "nucleus_strength_shape"
    ] == [5]

    assert shape_summary[
        "candidate_index_shape"
    ] == [3]

    assert shape_summary[
        "candidate_strength_shape"
    ] == [3]


def test_build_nucleus_metadata_records_strength_summary():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    summary = metadata[
        "nucleus_summary"
    ]

    assert summary[
        "minimum_nucleus_strength"
    ] == pytest.approx(0.0)

    assert summary[
        "maximum_nucleus_strength"
    ] == pytest.approx(3.0)

    assert summary[
        "minimum_candidate_strength"
    ] == pytest.approx(1.0)

    assert summary[
        "maximum_candidate_strength"
    ] == pytest.approx(3.0)

    assert summary[
        "boundary_excluded_candidate_count"
    ] == 2


def test_build_nucleus_metadata_handles_zero_candidates():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        _,
    ) = create_valid_nucleus_metadata_inputs()

    candidates = build_nucleus_candidates(
        np.array(
            [1.0, 0.0, 3.0, 0.0, 2.0],
            dtype=np.float64,
        ),
        np.array(
            [],
            dtype=np.int64,
        ),
    )

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=3,
    )

    assert metadata["shape_summary"][
        "number_of_candidates"
    ] == 0

    assert metadata["nucleus_summary"][
        "minimum_candidate_strength"
    ] == 0.0

    assert metadata["nucleus_summary"][
        "maximum_candidate_strength"
    ] == 0.0

    assert metadata["nucleus_summary"][
        "boundary_excluded_candidate_count"
    ] == 3


def test_build_nucleus_metadata_is_strictly_json_serializable():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    encoded = json.dumps(
        metadata,
        allow_nan=False,
    )

    decoded = json.loads(encoded)

    assert decoded == metadata


def test_build_nucleus_metadata_returns_independent_nested_values():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["nucleus_configuration"][
        "feature_weights"
    ]["frame_total"] = 999.0

    metadata["nucleus_configuration"][
        "feature_names"
    ][0] = "modified"

    assert configuration[
        "feature_weights"
    ]["frame_total"] == 1.0

    assert NUCLEUS_FEATURE_NAMES[0] == (
        "frame_total"
    )


@pytest.mark.parametrize(
    "invalid_count",
    [
        True,
        1.5,
        "2",
    ],
    ids=[
        "boolean",
        "float",
        "string",
    ],
)
def test_build_nucleus_metadata_rejects_invalid_excluded_count_type(
    invalid_count: object,
):
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        build_nucleus_metadata(
            source_metadata,
            boundary_metadata,
            configuration,
            candidates,
            boundary_excluded_candidate_count=invalid_count,
        )


def test_build_nucleus_metadata_rejects_negative_excluded_count():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        build_nucleus_metadata(
            source_metadata,
            boundary_metadata,
            configuration,
            candidates,
            boundary_excluded_candidate_count=-1,
        )


#8.19
def test_validate_nucleus_metadata_accepts_valid_metadata():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    validated = validate_nucleus_metadata(
        metadata,
        source_metadata,
        boundary_metadata,
        candidates,
    )

    assert validated == metadata


def test_validate_nucleus_metadata_returns_independent_copy():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    validated = validate_nucleus_metadata(
        metadata,
        source_metadata,
        boundary_metadata,
        candidates,
    )

    assert validated is not metadata

    assert (
        validated["nucleus_configuration"]
        is not metadata["nucleus_configuration"]
    )

    assert (
        validated["nucleus_configuration"][
            "feature_weights"
        ]
        is not metadata["nucleus_configuration"][
            "feature_weights"
        ]
    )

    assert (
        validated["shape_summary"]
        is not metadata["shape_summary"]
    )

    assert (
        validated["nucleus_summary"]
        is not metadata["nucleus_summary"]
    )

    validated["nucleus_configuration"][
        "feature_weights"
    ]["frame_total"] = 999.0

    validated["shape_summary"][
        "number_of_frames"
    ] = 999

    assert metadata["nucleus_configuration"][
        "feature_weights"
    ]["frame_total"] == 1.0

    assert metadata["shape_summary"][
        "number_of_frames"
    ] == 5


def test_validate_nucleus_metadata_accepts_zero_candidates():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        _,
    ) = create_valid_nucleus_metadata_inputs()

    candidates = build_nucleus_candidates(
        np.array(
            [1.0, 0.0, 3.0, 0.0, 2.0],
            dtype=np.float64,
        ),
        np.array(
            [],
            dtype=np.int64,
        ),
    )

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=3,
    )

    validated = validate_nucleus_metadata(
        metadata,
        source_metadata,
        boundary_metadata,
        candidates,
    )

    assert validated["shape_summary"][
        "number_of_candidates"
    ] == 0

    assert validated["nucleus_summary"][
        "minimum_candidate_strength"
    ] == 0.0

    assert validated["nucleus_summary"][
        "maximum_candidate_strength"
    ] == 0.0


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        None,
        [],
        "metadata",
    ],
    ids=[
        "none",
        "list",
        "string",
    ],
)
def test_validate_nucleus_metadata_rejects_non_dictionary_root(
    invalid_metadata: object,
):
    (
        source_metadata,
        boundary_metadata,
        _,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    with pytest.raises(
        TypeError,
        match="root must be a dictionary",
    ):
        validate_nucleus_metadata(
            invalid_metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_missing_top_level_key():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    del metadata["shape_summary"]

    with pytest.raises(
        ValueError,
        match="keys do not match",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_unexpected_top_level_key():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["unexpected"] = {}

    with pytest.raises(
        ValueError,
        match="keys do not match",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_invalid_configuration_section():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["nucleus_configuration"] = []

    with pytest.raises(
        TypeError,
        match="configuration metadata must be a dictionary",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_invalid_summary_section():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["nucleus_summary"] = []

    with pytest.raises(
        TypeError,
        match="summary metadata must be a dictionary",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_missing_excluded_count():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    del metadata["nucleus_summary"][
        "boundary_excluded_candidate_count"
    ]

    with pytest.raises(
        KeyError,
        match="boundary_excluded_candidate_count",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_source_scale_profile_mismatch():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["source_scale_profile"][
        "number_of_frames"
    ] = 999

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_source_boundary_mismatch():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["source_boundary"][
        "number_of_candidates"
    ] = 999

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_configuration_mismatch():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["nucleus_configuration"][
        "threshold"
    ] = 2.0

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_shape_summary_mismatch():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["shape_summary"][
        "number_of_candidates"
    ] = 999

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_strength_summary_mismatch():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["nucleus_summary"][
        "maximum_nucleus_strength"
    ] = 999.0

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


@pytest.mark.parametrize(
    "invalid_count",
    [
        True,
        -1,
        np.nan,
    ],
    ids=[
        "boolean",
        "negative",
        "nan",
    ],
)
def test_validate_nucleus_metadata_rejects_invalid_excluded_count(
    invalid_count: object,
):
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["nucleus_summary"][
        "boundary_excluded_candidate_count"
    ] = invalid_count

    expected_error = (
        TypeError
        if isinstance(invalid_count, bool)
        or not isinstance(
            invalid_count,
            (int, np.integer),
        )
        else ValueError
    )

    with pytest.raises(
        expected_error,
        match="integer|non-negative",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_rejects_configuration_mismatch():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    metadata["nucleus_configuration"][
        "feature_names"
    ][0] = "unexpected_feature"

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_nucleus_metadata(
            metadata,
            source_metadata,
            boundary_metadata,
            candidates,
        )


def test_validate_nucleus_metadata_returns_json_serializable_data():
    (
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
    ) = create_valid_nucleus_metadata_inputs()

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=2,
    )

    validated = validate_nucleus_metadata(
        metadata,
        source_metadata,
        boundary_metadata,
        candidates,
    )

    encoded = json.dumps(
        validated,
        allow_nan=False,
    )

    assert json.loads(encoded) == validated


#8.20
def create_valid_nucleus_extraction_inputs():
    frame_count = 16

    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    coefficients = np.array(
        [
            [
                1.0,
                2.0,
                5.0,
                2.0,
                1.0,
                2.0,
                6.0,
                2.0,
                1.0,
                2.0,
                4.0,
                1.0,
                1.0,
                2.0,
                5.0,
                1.0,
            ],
            [
                1.0,
                2.0,
                4.0,
                2.0,
                1.0,
                2.0,
                5.0,
                2.0,
                1.0,
                2.0,
                3.0,
                1.0,
                1.0,
                2.0,
                4.0,
                1.0,
            ],
            [
                1.0,
                1.5,
                3.0,
                1.5,
                1.0,
                1.5,
                4.0,
                1.5,
                1.0,
                1.5,
                2.5,
                1.0,
                1.0,
                1.5,
                3.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )



    source_cwt_metadata = {
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
            "n_frames": frame_count,
            "frequency_min_hz": 5.0,
            "frequency_max_hz": 20.0,
            "coefficient_dtype": "float64",
        },
    }

    scale_profile, source_metadata = (
        extract_scale_profile(
            coefficients=coefficients,
            frequencies=frequencies,
            source_cwt_metadata=source_cwt_metadata,
        )
    )

    boundary_configuration = (
        build_default_boundary_configuration()
    )

    (
        boundary_candidates,
        boundary_metadata,
    ) = extract_boundary_candidates(
        scale_profile,
        source_metadata,
        boundary_configuration,
    )

    nucleus_configuration = (
        build_default_nucleus_configuration()
    )

    return (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        nucleus_configuration,
    )


def test_extract_nucleus_candidates_returns_valid_outputs():
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    ) = create_valid_nucleus_extraction_inputs()

    candidates, metadata = (
        extract_nucleus_candidates(
            scale_profile,
            source_metadata,
            boundary_candidates,
            boundary_metadata,
            configuration,
        )
    )

    validated_candidates = (
        validate_nucleus_candidates(
            candidates,
            minimum_distance=configuration[
                "minimum_candidate_distance"
            ],
        )
    )

    validated_metadata = validate_nucleus_metadata(
        metadata,
        source_metadata,
        boundary_metadata,
        validated_candidates,
    )

    for key in NUCLEUS_CANDIDATE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            validated_candidates[key],
            candidates[key],
        )

    assert validated_metadata == metadata


def test_extract_nucleus_candidates_preserves_frame_count():
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    ) = create_valid_nucleus_extraction_inputs()

    candidates, metadata = (
        extract_nucleus_candidates(
            scale_profile,
            source_metadata,
            boundary_candidates,
            boundary_metadata,
            configuration,
        )
    )

    frame_count = scale_profile[
        "magnitude"
    ].shape[1]

    assert candidates[
        "nucleus_strength"
    ].shape == (frame_count,)

    assert metadata["shape_summary"][
        "number_of_frames"
    ] == frame_count

    assert metadata["source_scale_profile"][
        "number_of_frames"
    ] == frame_count

    assert metadata["source_boundary"][
        "number_of_frames"
    ] == frame_count


def test_extract_nucleus_candidates_can_disable_boundary_exclusion():
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    ) = create_valid_nucleus_extraction_inputs()

    configuration[
        "boundary_exclusion_distance"
    ] = 0

    candidates, metadata = (
        extract_nucleus_candidates(
            scale_profile,
            source_metadata,
            boundary_candidates,
            boundary_metadata,
            configuration,
        )
    )

    assert metadata["nucleus_configuration"][
        "boundary_exclusion_distance"
    ] == 0

    assert metadata["nucleus_summary"][
        "boundary_excluded_candidate_count"
    ] == 0

    validate_nucleus_candidates(
        candidates,
        minimum_distance=configuration[
            "minimum_candidate_distance"
        ],
    )


def test_extract_nucleus_candidates_accepts_zero_candidates():
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    ) = create_valid_nucleus_extraction_inputs()

    configuration["threshold"] = 1.0e12

    candidates, metadata = (
        extract_nucleus_candidates(
            scale_profile,
            source_metadata,
            boundary_candidates,
            boundary_metadata,
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

    assert metadata["nucleus_summary"][
        "minimum_candidate_strength"
    ] == 0.0

    assert metadata["nucleus_summary"][
        "maximum_candidate_strength"
    ] == 0.0


def test_extract_nucleus_candidates_does_not_modify_inputs():
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    ) = create_valid_nucleus_extraction_inputs()

    original_scale_profile = {
        key: np.array(
            value,
            copy=True,
        )
        for key, value in scale_profile.items()
    }

    original_source_metadata = deepcopy(
        source_metadata
    )

    original_boundary_candidates = {
        key: np.array(
            value,
            copy=True,
        )
        for key, value in boundary_candidates.items()
    }

    original_boundary_metadata = deepcopy(
        boundary_metadata
    )

    original_configuration = deepcopy(
        configuration
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

    for key in scale_profile:
        np.testing.assert_array_equal(
            scale_profile[key],
            original_scale_profile[key],
        )

    for key in boundary_candidates:
        np.testing.assert_array_equal(
            boundary_candidates[key],
            original_boundary_candidates[key],
        )

    assert source_metadata == original_source_metadata
    assert boundary_metadata == original_boundary_metadata
    assert configuration == original_configuration

    for values in candidates.values():
        for source_values in scale_profile.values():
            assert not np.shares_memory(
                values,
                source_values,
            )

        for source_values in boundary_candidates.values():
            assert not np.shares_memory(
                values,
                source_values,
            )

    assert metadata is not source_metadata
    assert metadata is not boundary_metadata


def test_extract_nucleus_candidates_rejects_invalid_scale_profile():
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    ) = create_valid_nucleus_extraction_inputs()

    scale_profile["magnitude"][0, 0] = -1.0

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        extract_nucleus_candidates(
            scale_profile,
            source_metadata,
            boundary_candidates,
            boundary_metadata,
            configuration,
        )


def test_extract_nucleus_candidates_rejects_invalid_boundary_candidates():
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    ) = create_valid_nucleus_extraction_inputs()

    if boundary_candidates[
        "candidate_strengths"
    ].size == 0:
        boundary_candidates[
            "candidate_frame_indices"
        ] = np.array(
            [0],
            dtype=np.int64,
        )
        boundary_candidates[
            "candidate_strengths"
        ] = np.array(
            [999.0],
            dtype=np.float64,
        )
    else:
        boundary_candidates[
            "candidate_strengths"
        ][0] += 1.0

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        extract_nucleus_candidates(
            scale_profile,
            source_metadata,
            boundary_candidates,
            boundary_metadata,
            configuration,
        )


def test_extract_nucleus_candidates_rejects_invalid_boundary_metadata():
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    ) = create_valid_nucleus_extraction_inputs()

    boundary_metadata["shape_summary"][
        "number_of_frames"
    ] += 1

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        extract_nucleus_candidates(
            scale_profile,
            source_metadata,
            boundary_candidates,
            boundary_metadata,
            configuration,
        )


def test_extract_nucleus_candidates_rejects_invalid_configuration():
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    ) = create_valid_nucleus_extraction_inputs()

    configuration["local_contrast_radius"] = 0

    with pytest.raises(
        ValueError,
        match="Local contrast radius must be at least 1",
    ):
        extract_nucleus_candidates(
            scale_profile,
            source_metadata,
            boundary_candidates,
            boundary_metadata,
            configuration,
        )


#8.21
def create_valid_nucleus_persistence_inputs():
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    ) = create_valid_nucleus_extraction_inputs()

    candidates, metadata = extract_nucleus_candidates(
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        configuration,
    )

    return (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    )


def test_save_nucleus_outputs_creates_files(
    tmp_path: Path,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    save_nucleus_outputs(
        output_path,
        metadata_path,
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    )

    assert output_path.is_file()
    assert metadata_path.is_file()


def test_save_nucleus_outputs_persists_candidate_arrays(
    tmp_path: Path,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    save_nucleus_outputs(
        output_path,
        metadata_path,
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    )

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        assert set(archive.files) == set(
            NUCLEUS_CANDIDATE_REQUIRED_KEYS
        )

        for key in NUCLEUS_CANDIDATE_REQUIRED_KEYS:
            np.testing.assert_array_equal(
                archive[key],
                candidates[key],
            )


def test_save_nucleus_outputs_preserves_array_dtypes(
    tmp_path: Path,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    save_nucleus_outputs(
        output_path,
        metadata_path,
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    )

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        assert archive[
            "nucleus_strength"
        ].dtype == np.float64

        assert archive[
            "candidate_frame_indices"
        ].dtype == np.int64

        assert archive[
            "candidate_strengths"
        ].dtype == np.float64


def test_save_nucleus_outputs_persists_metadata(
    tmp_path: Path,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    save_nucleus_outputs(
        output_path,
        metadata_path,
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    )

    loaded_metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8",
        )
    )

    assert loaded_metadata == metadata


def test_save_nucleus_outputs_writes_formatted_json(
    tmp_path: Path,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    save_nucleus_outputs(
        output_path,
        metadata_path,
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    )

    encoded = metadata_path.read_text(
        encoding="utf-8",
    )

    assert encoded.endswith("\n")
    assert "\n  " in encoded

    assert json.loads(encoded) == metadata


def test_save_nucleus_outputs_creates_parent_directories(
    tmp_path: Path,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    output_path = (
        tmp_path
        / "arrays"
        / "nested"
        / "nucleus_candidates.npz"
    )

    metadata_path = (
        tmp_path
        / "metadata"
        / "nested"
        / "nucleus_metadata.json"
    )

    save_nucleus_outputs(
        output_path,
        metadata_path,
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    )

    assert output_path.is_file()
    assert metadata_path.is_file()


def test_save_nucleus_outputs_persists_zero_candidates(
    tmp_path: Path,
):
    (
        _,
        _,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    configuration = (
        build_default_nucleus_configuration()
    )

    strength = np.zeros(
        source_metadata["shape_summary"][
            "number_of_frames"
        ],
        dtype=np.float64,
    )

    candidates = build_nucleus_candidates(
        strength,
        np.array(
            [],
            dtype=np.int64,
        ),
    )

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=0,
    )

    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    save_nucleus_outputs(
        output_path,
        metadata_path,
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    )

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        assert archive[
            "candidate_frame_indices"
        ].shape == (0,)

        assert archive[
            "candidate_frame_indices"
        ].dtype == np.int64

        assert archive[
            "candidate_strengths"
        ].shape == (0,)

        assert archive[
            "candidate_strengths"
        ].dtype == np.float64

    loaded_metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8",
        )
    )

    assert loaded_metadata["shape_summary"][
        "number_of_candidates"
    ] == 0


def test_save_nucleus_outputs_does_not_modify_inputs(
    tmp_path: Path,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    original_candidates = {
        key: np.array(
            value,
            copy=True,
        )
        for key, value in candidates.items()
    }

    original_metadata = deepcopy(
        metadata
    )
    original_source_metadata = deepcopy(
        source_metadata
    )
    original_boundary_metadata = deepcopy(
        boundary_metadata
    )

    save_nucleus_outputs(
        tmp_path / "nucleus_candidates.npz",
        tmp_path / "nucleus_metadata.json",
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    )

    for key in NUCLEUS_CANDIDATE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            candidates[key],
            original_candidates[key],
        )

    assert metadata == original_metadata
    assert source_metadata == original_source_metadata
    assert boundary_metadata == original_boundary_metadata


def test_save_nucleus_outputs_rejects_invalid_candidates(
    tmp_path: Path,
):
    (
        _,
        _,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    configuration = (
        build_default_nucleus_configuration()
    )

    strength = np.zeros(
        source_metadata["shape_summary"][
            "number_of_frames"
        ],
        dtype=np.float64,
    )
    strength[2] = 3.0

    candidates = build_nucleus_candidates(
        strength,
        np.array(
            [2],
            dtype=np.int64,
        ),
    )

    metadata = build_nucleus_metadata(
        source_metadata,
        boundary_metadata,
        configuration,
        candidates,
        boundary_excluded_candidate_count=0,
    )

    candidates["candidate_strengths"][0] = 999.0

    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        save_nucleus_outputs(
            output_path,
            metadata_path,
            candidates,
            metadata,
            source_metadata,
            boundary_metadata,
        )

    assert not output_path.exists()
    assert not metadata_path.exists()


def test_save_nucleus_outputs_rejects_invalid_metadata(
    tmp_path: Path,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    metadata["shape_summary"][
        "number_of_frames"
    ] += 1

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        save_nucleus_outputs(
            tmp_path / "nucleus_candidates.npz",
            tmp_path / "nucleus_metadata.json",
            candidates,
            metadata,
            source_metadata,
            boundary_metadata,
        )

    assert not (
        tmp_path / "nucleus_candidates.npz"
    ).exists()

    assert not (
        tmp_path / "nucleus_metadata.json"
    ).exists()


def test_save_nucleus_outputs_rejects_invalid_configuration_section(
    tmp_path: Path,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    metadata["nucleus_configuration"] = []

    with pytest.raises(
        TypeError,
        match="configuration metadata must be a dictionary",
    ):
        save_nucleus_outputs(
            tmp_path / "nucleus_candidates.npz",
            tmp_path / "nucleus_metadata.json",
            candidates,
            metadata,
            source_metadata,
            boundary_metadata,
        )


def test_save_nucleus_outputs_rejects_missing_minimum_distance(
    tmp_path: Path,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    del metadata["nucleus_configuration"][
        "minimum_candidate_distance"
    ]

    with pytest.raises(
        KeyError,
        match="minimum_candidate_distance",
    ):
        save_nucleus_outputs(
            tmp_path / "nucleus_candidates.npz",
            tmp_path / "nucleus_metadata.json",
            candidates,
            metadata,
            source_metadata,
            boundary_metadata,
        )


@pytest.mark.parametrize(
    "invalid_path",
    [
        "nucleus_candidates.npz",
        None,
    ],
    ids=[
        "string",
        "none",
    ],
)
def test_save_nucleus_outputs_rejects_invalid_output_path(
    tmp_path: Path,
    invalid_path: object,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    with pytest.raises(
        TypeError,
        match="output path must be a Path",
    ):
        save_nucleus_outputs(
            invalid_path,
            tmp_path / "nucleus_metadata.json",
            candidates,
            metadata,
            source_metadata,
            boundary_metadata,
        )


@pytest.mark.parametrize(
    "invalid_path",
    [
        "nucleus_metadata.json",
        None,
    ],
    ids=[
        "string",
        "none",
    ],
)
def test_save_nucleus_outputs_rejects_invalid_metadata_path(
    tmp_path: Path,
    invalid_path: object,
):
    (
        candidates,
        metadata,
        source_metadata,
        boundary_metadata,
    ) = create_valid_nucleus_persistence_inputs()

    with pytest.raises(
        TypeError,
        match="metadata path must be a Path",
    ):
        save_nucleus_outputs(
            tmp_path / "nucleus_candidates.npz",
            invalid_path,
            candidates,
            metadata,
            source_metadata,
            boundary_metadata,
        )


#8.23
def test_nucleus_parser_parses_required_paths():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--input",
            "scale_profile.npz",
            "--metadata-input",
            "scale_profile_metadata.json",
            "--boundary-input",
            "boundary_candidates.npz",
            "--boundary-metadata-input",
            "boundary_metadata.json",
            "--output",
            "nucleus_candidates.npz",
        ]
    )

    assert args.input == Path(
        "scale_profile.npz"
    )
    assert args.metadata_input == Path(
        "scale_profile_metadata.json"
    )
    assert args.boundary_input == Path(
        "boundary_candidates.npz"
    )
    assert args.boundary_metadata_input == Path(
        "boundary_metadata.json"
    )
    assert args.output == Path(
        "nucleus_candidates.npz"
    )
    assert args.metadata_output is None


def test_nucleus_parser_uses_default_configuration_values():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--input",
            "scale_profile.npz",
            "--metadata-input",
            "scale_profile_metadata.json",
            "--boundary-input",
            "boundary_candidates.npz",
            "--boundary-metadata-input",
            "boundary_metadata.json",
            "--output",
            "nucleus_candidates.npz",
        ]
    )

    assert args.local_contrast_radius == (
        DEFAULT_LOCAL_CONTRAST_RADIUS
    )

    assert args.threshold == pytest.approx(
        DEFAULT_NUCLEUS_THRESHOLD
    )

    assert args.minimum_distance == (
        DEFAULT_MINIMUM_NUCLEUS_CANDIDATE_DISTANCE
    )

    assert args.boundary_exclusion_distance == (
        DEFAULT_BOUNDARY_EXCLUSION_DISTANCE
    )


def test_nucleus_parser_uses_default_feature_weights():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--input",
            "scale_profile.npz",
            "--metadata-input",
            "scale_profile_metadata.json",
            "--boundary-input",
            "boundary_candidates.npz",
            "--boundary-metadata-input",
            "boundary_metadata.json",
            "--output",
            "nucleus_candidates.npz",
        ]
    )

    assert args.frame_total_weight == pytest.approx(
        DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
            "frame_total"
        ]
    )

    assert args.frame_mean_weight == pytest.approx(
        DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
            "frame_mean"
        ]
    )

    assert args.frame_max_weight == pytest.approx(
        DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
            "frame_max"
        ]
    )

    assert args.dominant_frequency_weight == (
        pytest.approx(
            DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
                "dominant_frequency_hz"
            ]
        )
    )

    assert args.frequency_centroid_weight == (
        pytest.approx(
            DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
                "frequency_centroid_hz"
            ]
        )
    )

    assert args.frequency_dispersion_weight == (
        pytest.approx(
            DEFAULT_NUCLEUS_FEATURE_WEIGHTS[
                "frequency_dispersion_hz"
            ]
        )
    )


def test_nucleus_parser_accepts_configuration_overrides():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--input",
            "profile.npz",
            "--metadata-input",
            "profile.json",
            "--boundary-input",
            "boundary.npz",
            "--boundary-metadata-input",
            "boundary.json",
            "--output",
            "nucleus.npz",
            "--metadata-output",
            "nucleus.json",
            "--local-contrast-radius",
            "4",
            "--threshold",
            "2.5",
            "--minimum-distance",
            "3",
            "--boundary-exclusion-distance",
            "2",
            "--frame-total-weight",
            "1.5",
            "--frame-mean-weight",
            "2.5",
            "--frame-max-weight",
            "3.5",
            "--dominant-frequency-weight",
            "4.5",
            "--frequency-centroid-weight",
            "5.5",
            "--frequency-dispersion-weight",
            "6.5",
        ]
    )

    assert args.metadata_output == Path(
        "nucleus.json"
    )
    assert args.local_contrast_radius == 4
    assert args.threshold == pytest.approx(2.5)
    assert args.minimum_distance == 3
    assert args.boundary_exclusion_distance == 2

    assert args.frame_total_weight == pytest.approx(
        1.5
    )
    assert args.frame_mean_weight == pytest.approx(
        2.5
    )
    assert args.frame_max_weight == pytest.approx(
        3.5
    )
    assert args.dominant_frequency_weight == (
        pytest.approx(4.5)
    )
    assert args.frequency_centroid_weight == (
        pytest.approx(5.5)
    )
    assert args.frequency_dispersion_weight == (
        pytest.approx(6.5)
    )


@pytest.mark.parametrize(
    (
        "output_path",
        "expected_path",
    ),
    [
        (
            Path("nucleus_candidates.npz"),
            Path(
                "nucleus_candidates_metadata.json"
            ),
        ),
        (
            Path(
                "results/nucleus_output.npz"
            ),
            Path(
                "results/nucleus_output_metadata.json"
            ),
        ),
        (
            Path("result"),
            Path("result_metadata.json"),
        ),
    ],
)
def test_default_nucleus_metadata_path(
    output_path: Path,
    expected_path: Path,
):
    assert default_nucleus_metadata_path(
        output_path
    ) == expected_path


@pytest.mark.parametrize(
    "missing_option",
    [
        "--input",
        "--metadata-input",
        "--boundary-input",
        "--boundary-metadata-input",
        "--output",
    ],
)
def test_nucleus_parser_requires_input_and_output_paths(
    missing_option: str,
):
    parser = build_parser()

    arguments = {
        "--input": "scale_profile.npz",
        "--metadata-input": (
            "scale_profile_metadata.json"
        ),
        "--boundary-input": (
            "boundary_candidates.npz"
        ),
        "--boundary-metadata-input": (
            "boundary_metadata.json"
        ),
        "--output": "nucleus_candidates.npz",
    }

    command_line: list[str] = []

    for option, value in arguments.items():
        if option != missing_option:
            command_line.extend(
                [
                    option,
                    value,
                ]
            )

    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            command_line
        )

    assert error.value.code == 2


@pytest.mark.parametrize(
    (
        "option",
        "invalid_value",
    ),
    [
        (
            "--local-contrast-radius",
            "1.5",
        ),
        (
            "--threshold",
            "invalid",
        ),
        (
            "--minimum-distance",
            "1.5",
        ),
        (
            "--boundary-exclusion-distance",
            "1.5",
        ),
        (
            "--frame-total-weight",
            "invalid",
        ),
    ],
)
def test_nucleus_parser_rejects_invalid_numeric_syntax(
    option: str,
    invalid_value: str,
):
    parser = build_parser()

    command_line = [
        "--input",
        "scale_profile.npz",
        "--metadata-input",
        "scale_profile_metadata.json",
        "--boundary-input",
        "boundary_candidates.npz",
        "--boundary-metadata-input",
        "boundary_metadata.json",
        "--output",
        "nucleus_candidates.npz",
        option,
        invalid_value,
    ]

    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            command_line
        )

    assert error.value.code == 2


def test_nucleus_parser_has_description():
    parser = build_parser()

    assert parser.description is not None
    assert "Nucleus candidates" in (
        parser.description
    )
    assert "ScaleProfile" in (
        parser.description
    )
    assert "Boundary" in (
        parser.description
    )


#8.23
def create_valid_nucleus_cli_arguments(
    tmp_path: Path,
    extra_arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Create parsed Nucleus CLI arguments with valid input files."""

    input_path = tmp_path / "scale_profile.npz"
    metadata_input = (
        tmp_path / "scale_profile_metadata.json"
    )
    boundary_input = (
        tmp_path / "boundary_candidates.npz"
    )
    boundary_metadata_input = (
        tmp_path / "boundary_metadata.json"
    )
    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )

    for path in (
        input_path,
        metadata_input,
        boundary_input,
        boundary_metadata_input,
    ):
        path.touch()

    command_line = [
        "--input",
        str(input_path),
        "--metadata-input",
        str(metadata_input),
        "--boundary-input",
        str(boundary_input),
        "--boundary-metadata-input",
        str(boundary_metadata_input),
        "--output",
        str(output_path),
    ]

    if extra_arguments is not None:
        command_line.extend(
            extra_arguments
        )

    return build_parser().parse_args(
        command_line
    )


def test_validate_arguments_returns_default_metadata_path_and_configuration(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    metadata_output, configuration = (
        validate_arguments(args)
    )

    assert metadata_output == (
        tmp_path
        / "nucleus_candidates_metadata.json"
    )

    assert configuration == (
        build_default_nucleus_configuration()
    )


def test_validate_arguments_accepts_explicit_metadata_output(
    tmp_path: Path,
):
    metadata_output_path = (
        tmp_path / "custom_nucleus_metadata.json"
    )

    args = create_valid_nucleus_cli_arguments(
        tmp_path,
        [
            "--metadata-output",
            str(metadata_output_path),
        ],
    )

    metadata_output, _ = validate_arguments(
        args
    )

    assert metadata_output == (
        metadata_output_path
    )


def test_validate_arguments_builds_overridden_configuration(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path,
        [
            "--local-contrast-radius",
            "4",
            "--threshold",
            "2.5",
            "--minimum-distance",
            "3",
            "--boundary-exclusion-distance",
            "2",
            "--frame-total-weight",
            "1.5",
            "--frame-mean-weight",
            "2.5",
            "--frame-max-weight",
            "3.5",
            "--dominant-frequency-weight",
            "4.5",
            "--frequency-centroid-weight",
            "5.5",
            "--frequency-dispersion-weight",
            "6.5",
        ],
    )

    _, configuration = validate_arguments(
        args
    )

    assert configuration[
        "local_contrast_radius"
    ] == 4

    assert configuration[
        "threshold"
    ] == pytest.approx(2.5)

    assert configuration[
        "minimum_candidate_distance"
    ] == 3

    assert configuration[
        "boundary_exclusion_distance"
    ] == 2

    assert configuration[
        "feature_weights"
    ] == {
        "frame_total": 1.5,
        "frame_mean": 2.5,
        "frame_max": 3.5,
        "dominant_frequency_hz": 4.5,
        "frequency_centroid_hz": 5.5,
        "frequency_dispersion_hz": 6.5,
    }


@pytest.mark.parametrize(
    (
        "attribute",
        "expected_message",
    ),
    [
        (
            "input",
            "ScaleProfile archive does not exist",
        ),
        (
            "metadata_input",
            "ScaleProfile metadata does not exist",
        ),
        (
            "boundary_input",
            "Boundary archive does not exist",
        ),
        (
            "boundary_metadata_input",
            "Boundary metadata does not exist",
        ),
    ],
)
def test_validate_arguments_rejects_missing_input_file(
    tmp_path: Path,
    attribute: str,
    expected_message: str,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    missing_path = getattr(
        args,
        attribute,
    )

    missing_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match=expected_message,
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_invalid_scale_profile_extension(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    invalid_path = (
        tmp_path / "scale_profile.json"
    )
    invalid_path.touch()
    args.input = invalid_path

    with pytest.raises(
        ValueError,
        match="--input must use the .npz extension",
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_invalid_scale_metadata_extension(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    invalid_path = (
        tmp_path / "scale_profile_metadata.npz"
    )
    invalid_path.touch()
    args.metadata_input = invalid_path

    with pytest.raises(
        ValueError,
        match=(
            "--metadata-input must use "
            "the .json extension"
        ),
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_invalid_boundary_extension(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    invalid_path = (
        tmp_path / "boundary_candidates.json"
    )
    invalid_path.touch()
    args.boundary_input = invalid_path

    with pytest.raises(
        ValueError,
        match=(
            "--boundary-input must use "
            "the .npz extension"
        ),
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_invalid_boundary_metadata_extension(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    invalid_path = (
        tmp_path / "boundary_metadata.npz"
    )
    invalid_path.touch()
    args.boundary_metadata_input = invalid_path

    with pytest.raises(
        ValueError,
        match=(
            "--boundary-metadata-input must use "
            "the .json extension"
        ),
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_invalid_output_extension(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    args.output = (
        tmp_path / "nucleus_candidates.json"
    )

    with pytest.raises(
        ValueError,
        match="--output must use the .npz extension",
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_invalid_metadata_output_extension(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path,
        [
            "--metadata-output",
            str(
                tmp_path / "nucleus_metadata.npz"
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "--metadata-output must use "
            "the .json extension"
        ),
    ):
        validate_arguments(
            args
        )


@pytest.mark.parametrize(
    "collision_attribute",
    [
        "input",
        "boundary_input",
    ],
)
def test_validate_arguments_rejects_output_input_collision(
    tmp_path: Path,
    collision_attribute: str,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    args.output = getattr(
        args,
        collision_attribute,
    )

    with pytest.raises(
        ValueError,
        match=(
            "CLI input and output paths "
            "must be different"
        ),
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_output_metadata_collision(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    args.metadata_output = (
        tmp_path
        / "."
        / "nucleus_candidates.npz"
    )

    with pytest.raises(
        ValueError,
        match=(
            "--metadata-output must use "
            "the .json extension"
        ),
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_npz_input_collision(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    args.boundary_input = args.input

    with pytest.raises(
        ValueError,
        match=(
            "CLI input and output paths "
            "must be different"
        ),
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_json_input_collision(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    args.boundary_metadata_input = (
        args.metadata_input
    )

    with pytest.raises(
        ValueError,
        match=(
            "CLI input and output paths "
            "must be different"
        ),
    ):
        validate_arguments(
            args
        )


@pytest.mark.parametrize(
    "invalid_radius",
    [
        0,
        -1,
    ],
    ids=[
        "zero",
        "negative",
    ],
)
def test_validate_arguments_rejects_invalid_local_contrast_radius(
    tmp_path: Path,
    invalid_radius: int,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path,
        [
            "--local-contrast-radius",
            str(invalid_radius),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Local contrast radius "
            "must be at least 1"
        ),
    ):
        validate_arguments(
            args
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
    ids=[
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_validate_arguments_rejects_invalid_threshold(
    tmp_path: Path,
    invalid_threshold: float,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    args.threshold = invalid_threshold

    with pytest.raises(
        ValueError,
        match="finite|non-negative",
    ):
        validate_arguments(
            args
        )


@pytest.mark.parametrize(
    "invalid_distance",
    [
        0,
        -1,
    ],
    ids=[
        "zero",
        "negative",
    ],
)
def test_validate_arguments_rejects_invalid_minimum_distance(
    tmp_path: Path,
    invalid_distance: int,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path,
        [
            "--minimum-distance",
            str(invalid_distance),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Minimum candidate distance "
            "must be at least 1"
        ),
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_negative_boundary_exclusion_distance(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path,
        [
            "--boundary-exclusion-distance",
            "-1",
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Boundary exclusion distance "
            "must be non-negative"
        ),
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_accepts_zero_boundary_exclusion_distance(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path,
        [
            "--boundary-exclusion-distance",
            "0",
        ],
    )

    _, configuration = validate_arguments(
        args
    )

    assert configuration[
        "boundary_exclusion_distance"
    ] == 0


@pytest.mark.parametrize(
    "attribute",
    [
        "frame_total_weight",
        "frame_mean_weight",
        "frame_max_weight",
        "dominant_frequency_weight",
        "frequency_centroid_weight",
        "frequency_dispersion_weight",
    ],
)
def test_validate_arguments_rejects_negative_feature_weight(
    tmp_path: Path,
    attribute: str,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    setattr(
        args,
        attribute,
        -1.0,
    )

    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        validate_arguments(
            args
        )


@pytest.mark.parametrize(
    "invalid_weight",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_validate_arguments_rejects_nonfinite_feature_weight(
    tmp_path: Path,
    invalid_weight: float,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    args.frame_total_weight = invalid_weight

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        validate_arguments(
            args
        )


def test_validate_arguments_rejects_all_zero_feature_weights(
    tmp_path: Path,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    args.frame_total_weight = 0.0
    args.frame_mean_weight = 0.0
    args.frame_max_weight = 0.0
    args.dominant_frequency_weight = 0.0
    args.frequency_centroid_weight = 0.0
    args.frequency_dispersion_weight = 0.0

    with pytest.raises(
        ValueError,
        match=(
            "At least one Nucleus feature weight "
            "must be positive"
        ),
    ):
        validate_arguments(
            args
        )


@pytest.mark.parametrize(
    "collision_attribute",
    [
        "metadata_input",
        "boundary_metadata_input",
    ],
)
def test_validate_arguments_rejects_metadata_output_input_collision(
    tmp_path: Path,
    collision_attribute: str,
):
    args = create_valid_nucleus_cli_arguments(
        tmp_path
    )

    args.metadata_output = getattr(
        args,
        collision_attribute,
    )

    with pytest.raises(
        ValueError,
        match=(
            "CLI input and output paths "
            "must be different"
        ),
    ):
        validate_arguments(
            args
        )


#8.24 E2E CLI test
def test_nucleus_cli_creates_valid_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        _,
    ) = create_valid_nucleus_extraction_inputs()

    scale_profile_path = (
        tmp_path / "scale_profile.npz"
    )
    scale_metadata_path = (
        tmp_path / "scale_profile_metadata.json"
    )
    boundary_path = (
        tmp_path / "boundary_candidates.npz"
    )
    boundary_metadata_path = (
        tmp_path / "boundary_metadata.json"
    )
    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    output_metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    np.savez_compressed(
        scale_profile_path,
        **scale_profile,
    )

    scale_metadata_path.write_text(
        json.dumps(
            source_metadata,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        boundary_path,
        **boundary_candidates,
    )

    boundary_metadata_path.write_text(
        json.dumps(
            boundary_metadata,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_nucleus_candidates.py",
            "--input",
            str(scale_profile_path),
            "--metadata-input",
            str(scale_metadata_path),
            "--boundary-input",
            str(boundary_path),
            "--boundary-metadata-input",
            str(boundary_metadata_path),
            "--output",
            str(output_path),
            "--metadata-output",
            str(output_metadata_path),
        ],
    )

    exit_code = main()

    assert exit_code == 0
    assert output_path.is_file()
    assert output_metadata_path.is_file()

    captured = capsys.readouterr()

    assert (
        "Nucleus candidate extraction: OK"
        in captured.out
    )
    assert "frames:" in captured.out
    assert "candidates:" in captured.out
    assert "boundary excluded:" in captured.out
    assert f"output: {output_path}" in captured.out
    assert (
        f"metadata: {output_metadata_path}"
        in captured.out
    )
    assert captured.err == ""

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        assert set(archive.files) == set(
            NUCLEUS_CANDIDATE_REQUIRED_KEYS
        )

        persisted_candidates = {
            "nucleus_strength": np.array(
                archive["nucleus_strength"],
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

    persisted_metadata = json.loads(
        output_metadata_path.read_text(
            encoding="utf-8",
        )
    )

    configuration = (
        build_default_nucleus_configuration()
    )

    validated_candidates = (
        validate_nucleus_candidates(
            persisted_candidates,
            minimum_distance=configuration[
                "minimum_candidate_distance"
            ],
        )
    )

    validated_metadata = validate_nucleus_metadata(
        persisted_metadata,
        source_metadata,
        boundary_metadata,
        validated_candidates,
    )

    for key in NUCLEUS_CANDIDATE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            validated_candidates[key],
            persisted_candidates[key],
        )

    assert validated_metadata == (
        persisted_metadata
    )


#CLI エラー経路TEST
def test_nucleus_cli_returns_one_for_missing_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    missing_input = (
        tmp_path / "missing_scale_profile.npz"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_nucleus_candidates.py",
            "--input",
            str(missing_input),
            "--metadata-input",
            str(
                tmp_path
                / "scale_profile_metadata.json"
            ),
            "--boundary-input",
            str(
                tmp_path
                / "boundary_candidates.npz"
            ),
            "--boundary-metadata-input",
            str(
                tmp_path
                / "boundary_metadata.json"
            ),
            "--output",
            str(
                tmp_path
                / "nucleus_candidates.npz"
            ),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert (
        "error: ScaleProfile archive "
        "does not exist"
        in captured.err
    )

    assert not (
        tmp_path / "nucleus_candidates.npz"
    ).exists()

    assert not (
        tmp_path
        / "nucleus_candidates_metadata.json"
    ).exists()


def test_nucleus_cli_returns_one_for_invalid_scale_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    (
        scale_profile,
        _,
        boundary_candidates,
        boundary_metadata,
        _,
    ) = create_valid_nucleus_extraction_inputs()

    scale_profile_path = (
        tmp_path / "scale_profile.npz"
    )
    scale_metadata_path = (
        tmp_path / "scale_profile_metadata.json"
    )
    boundary_path = (
        tmp_path / "boundary_candidates.npz"
    )
    boundary_metadata_path = (
        tmp_path / "boundary_metadata.json"
    )
    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    output_metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    np.savez_compressed(
        scale_profile_path,
        **scale_profile,
    )

    scale_metadata_path.write_text(
        json.dumps(
            {
                "format_version": 1,
            }
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        boundary_path,
        **boundary_candidates,
    )

    boundary_metadata_path.write_text(
        json.dumps(
            boundary_metadata,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_nucleus_candidates.py",
            "--input",
            str(scale_profile_path),
            "--metadata-input",
            str(scale_metadata_path),
            "--boundary-input",
            str(boundary_path),
            "--boundary-metadata-input",
            str(boundary_metadata_path),
            "--output",
            str(output_path),
            "--metadata-output",
            str(output_metadata_path),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")

    assert not output_path.exists()
    assert not output_metadata_path.exists()


def test_nucleus_cli_returns_one_for_invalid_boundary_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    (
        scale_profile,
        source_metadata,
        _,
        boundary_metadata,
        _,
    ) = create_valid_nucleus_extraction_inputs()

    scale_profile_path = (
        tmp_path / "scale_profile.npz"
    )
    scale_metadata_path = (
        tmp_path / "scale_profile_metadata.json"
    )
    boundary_path = (
        tmp_path / "boundary_candidates.npz"
    )
    boundary_metadata_path = (
        tmp_path / "boundary_metadata.json"
    )
    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    output_metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    np.savez_compressed(
        scale_profile_path,
        **scale_profile,
    )

    scale_metadata_path.write_text(
        json.dumps(
            source_metadata,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        boundary_path,
        boundary_strength=np.zeros(
            scale_profile["magnitude"].shape[1],
            dtype=np.float64,
        ),
        candidate_frame_indices=np.array(
            [],
            dtype=np.int64,
        ),
        # candidate_strengths is intentionally missing.
    )

    boundary_metadata_path.write_text(
        json.dumps(
            boundary_metadata,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_nucleus_candidates.py",
            "--input",
            str(scale_profile_path),
            "--metadata-input",
            str(scale_metadata_path),
            "--boundary-input",
            str(boundary_path),
            "--boundary-metadata-input",
            str(boundary_metadata_path),
            "--output",
            str(output_path),
            "--metadata-output",
            str(output_metadata_path),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "error: " in captured.err
    assert "candidate_strengths" in captured.err

    assert not output_path.exists()
    assert not output_metadata_path.exists()


def test_nucleus_cli_returns_one_for_invalid_boundary_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        _,
        _,
    ) = create_valid_nucleus_extraction_inputs()

    scale_profile_path = (
        tmp_path / "scale_profile.npz"
    )
    scale_metadata_path = (
        tmp_path / "scale_profile_metadata.json"
    )
    boundary_path = (
        tmp_path / "boundary_candidates.npz"
    )
    boundary_metadata_path = (
        tmp_path / "boundary_metadata.json"
    )
    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    output_metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    np.savez_compressed(
        scale_profile_path,
        **scale_profile,
    )

    scale_metadata_path.write_text(
        json.dumps(
            source_metadata,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        boundary_path,
        **boundary_candidates,
    )

    boundary_metadata_path.write_text(
        json.dumps(
            {
                "format_version": 1,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_nucleus_candidates.py",
            "--input",
            str(scale_profile_path),
            "--metadata-input",
            str(scale_metadata_path),
            "--boundary-input",
            str(boundary_path),
            "--boundary-metadata-input",
            str(boundary_metadata_path),
            "--output",
            str(output_path),
            "--metadata-output",
            str(output_metadata_path),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert (
        "Boundary configuration metadata"
        in captured.err
    )

    assert not output_path.exists()
    assert not output_metadata_path.exists()


def test_nucleus_cli_returns_one_for_invalid_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    (
        scale_profile,
        source_metadata,
        boundary_candidates,
        boundary_metadata,
        _,
    ) = create_valid_nucleus_extraction_inputs()

    scale_profile_path = (
        tmp_path / "scale_profile.npz"
    )
    scale_metadata_path = (
        tmp_path / "scale_profile_metadata.json"
    )
    boundary_path = (
        tmp_path / "boundary_candidates.npz"
    )
    boundary_metadata_path = (
        tmp_path / "boundary_metadata.json"
    )
    output_path = (
        tmp_path / "nucleus_candidates.npz"
    )
    output_metadata_path = (
        tmp_path / "nucleus_metadata.json"
    )

    np.savez_compressed(
        scale_profile_path,
        **scale_profile,
    )

    scale_metadata_path.write_text(
        json.dumps(
            source_metadata,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        boundary_path,
        **boundary_candidates,
    )

    boundary_metadata_path.write_text(
        json.dumps(
            boundary_metadata,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_nucleus_candidates.py",
            "--input",
            str(scale_profile_path),
            "--metadata-input",
            str(scale_metadata_path),
            "--boundary-input",
            str(boundary_path),
            "--boundary-metadata-input",
            str(boundary_metadata_path),
            "--output",
            str(output_path),
            "--metadata-output",
            str(output_metadata_path),
            "--local-contrast-radius",
            "0",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""

    assert (
        "error: Local contrast radius "
        "must be at least 1"
        in captured.err
    )

    assert not output_path.exists()
    assert not output_metadata_path.exists()