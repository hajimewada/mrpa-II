from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.f0.extract_cwt import (
    CWTMetadata,
    MINIMUM_FRAMES,
    MINIMUM_PRECISION,
    build_cwt_metadata,
    validate_cwt_metadata,
)

from analysis.f0.extract_scale_profile import (
    CENTROID_COORDINATE,
    DISPERSION_COORDINATE,
    DOMINANT_SCALE_TIE_POLICY,
    MAGNITUDE_TRANSFORM,
    NUMBER_OF_FRAME_STATISTICS,
    NUMBER_OF_SCALE_STATISTICS,
    SCALE_PROFILE_FORMAT_VERSION,
    STANDARD_DEVIATION_CONVENTION,
    ZERO_ENERGY_POLICY,
    SCALE_PROFILE_REQUIRED_KEYS,
    SOURCE_CWT_REQUIRED_KEYS,
    build_scale_profile,
    build_scale_profile_metadata,
    compute_cwt_magnitude,
    compute_frame_statistics,
    compute_scale_statistics,
    load_cwt_outputs,
    validate_loaded_cwt,
    validate_scale_profile_arrays,
    validate_scale_profile_metadata,
)


def create_persisted_cwt_inputs(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    coefficients = np.arange(
        3 * MINIMUM_FRAMES,
        dtype=np.float64,
    ).reshape(
        3,
        MINIMUM_FRAMES,
    )

    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    metadata: dict[str, object] = {
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
        },
        "cwt_summary": {
            "n_scales": 3,
            "n_frames": MINIMUM_FRAMES,
            "frequency_min_hz": 5.0,
            "frequency_max_hz": 20.0,
        },
    }

    input_path = tmp_path / "cwt.npz"
    metadata_path = tmp_path / "cwt_metadata.json"

    np.savez_compressed(
        input_path,
        coefficients=coefficients,
        frequencies=frequencies,
    )

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    return (
        input_path,
        metadata_path,
        coefficients,
        frequencies,
        metadata,
    )

def test_load_cwt_outputs_loads_persisted_data(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        expected_coefficients,
        expected_frequencies,
        expected_metadata,
    ) = create_persisted_cwt_inputs(
        tmp_path,
    )

    coefficients, frequencies, metadata = (
        load_cwt_outputs(
            input_path,
            metadata_path,
        )
    )

    np.testing.assert_array_equal(
        coefficients,
        expected_coefficients,
    )

    np.testing.assert_array_equal(
        frequencies,
        expected_frequencies,
    )

    assert metadata == expected_metadata

def test_load_cwt_outputs_returns_independent_arrays(
    tmp_path: Path,
) -> None:
    (
        input_path,
        metadata_path,
        expected_coefficients,
        expected_frequencies,
        _,
    ) = create_persisted_cwt_inputs(
        tmp_path,
    )

    coefficients, frequencies, _ = load_cwt_outputs(
        input_path,
        metadata_path,
    )

    coefficients[0, 0] = -999.0
    frequencies[0] = -999.0

    assert expected_coefficients[0, 0] != -999.0
    assert expected_frequencies[0] != -999.0

def test_load_cwt_outputs_rejects_missing_archive(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "cwt_metadata.json"

    metadata_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="CWT archive does not exist",
    ):
        load_cwt_outputs(
            tmp_path / "missing.npz",
            metadata_path,
        )

def test_load_cwt_outputs_rejects_missing_metadata(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "cwt.npz"

    np.savez_compressed(
        input_path,
        coefficients=np.zeros((2, 3)),
        frequencies=np.array([2.0, 1.0]),
    )

    with pytest.raises(
        FileNotFoundError,
        match="CWT metadata file does not exist",
    ):
        load_cwt_outputs(
            input_path,
            tmp_path / "missing.json",
        )

@pytest.mark.parametrize(
    "missing_key",
    SOURCE_CWT_REQUIRED_KEYS,
)
def test_load_cwt_outputs_rejects_missing_archive_key(
    tmp_path: Path,
    missing_key: str,
) -> None:
    input_path = tmp_path / "cwt.npz"
    metadata_path = tmp_path / "cwt_metadata.json"

    arrays = {
        "coefficients": np.zeros((2, 3)),
        "frequencies": np.array([2.0, 1.0]),
    }

    arrays.pop(missing_key)

    np.savez_compressed(
        input_path,
        **arrays,
    )

    metadata_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        KeyError,
        match=missing_key,
    ):
        load_cwt_outputs(
            input_path,
            metadata_path,
        )

def test_load_cwt_outputs_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "cwt.npz"
    metadata_path = tmp_path / "cwt_metadata.json"

    np.savez_compressed(
        input_path,
        coefficients=np.zeros((2, 3)),
        frequencies=np.array([2.0, 1.0]),
    )

    metadata_path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        load_cwt_outputs(
            input_path,
            metadata_path,
        )

from analysis.f0.extract_cwt import (
    MINIMUM_FRAMES,
    build_cwt_metadata,
)

from analysis.f0.extract_scale_profile import (
    SOURCE_CWT_REQUIRED_KEYS,
    load_cwt_outputs,
    validate_loaded_cwt,
)

def create_valid_loaded_cwt() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    number_of_scales = 3
    number_of_frames = MINIMUM_FRAMES

    coefficients = np.arange(
        number_of_scales * number_of_frames,
        dtype=np.float64,
    ).reshape(
        number_of_scales,
        number_of_frames,
    )

    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    metadata = build_cwt_metadata(
        wavelet="mexh",
        sampling_period=0.005,
        method="conv",
        precision=12,
        signal_summary={
            "domain": "log-hz",
            "centering": "median",
            "scaling": "none",
            "center_value": 5.0,
            "scale_value": 1.0,
            "domain_dynamic_range": 0.5,
            "prepared_dynamic_range": 0.5,
            "prepared_signal_mean": 0.0,
            "prepared_signal_median": 0.0,
        },
        scale_summary={
            "wavelet": "mexh",
            "number_of_scales": number_of_scales,
            "minimum_period_seconds": 0.05,
            "maximum_period_seconds": 0.2,
            "minimum_frequency_hz": 5.0,
            "maximum_frequency_hz": 20.0,
            "minimum_scale": 1.0,
            "maximum_scale": 4.0,
            "maximum_absolute_period_error_seconds": 0.0,
            "maximum_relative_period_error": 0.0,
        },
        cwt_summary={
            "n_scales": number_of_scales,
            "n_frames": number_of_frames,
            "frequency_min_hz": 5.0,
            "frequency_max_hz": 20.0,
            "coefficient_dtype": "float64",
            "coefficient_min": 0.0,
            "coefficient_max": float(
                np.max(np.abs(coefficients))
            ),
            "coefficient_mean_abs": float(
                np.mean(np.abs(coefficients))
            ),
        },
    )

    return (
        coefficients,
        frequencies,
        metadata,
    )

def test_validate_loaded_cwt_accepts_valid_input():
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    (
        validated_coefficients,
        validated_frequencies,
        validated_metadata,
    ) = validate_loaded_cwt(
        coefficients,
        frequencies,
        metadata,
    )

    np.testing.assert_array_equal(
        validated_coefficients,
        coefficients,
    )
    np.testing.assert_array_equal(
        validated_frequencies,
        frequencies,
    )

    assert validated_frequencies.dtype == np.float64
    assert validated_metadata == metadata

def test_validate_loaded_cwt_returns_independent_copies():
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    (
        validated_coefficients,
        validated_frequencies,
        validated_metadata,
    ) = validate_loaded_cwt(
        coefficients,
        frequencies,
        metadata,
    )

    validated_coefficients[0, 0] = -999.0
    validated_frequencies[0] = -999.0
    validated_metadata["cwt_summary"]["n_frames"] = 999

    assert coefficients[0, 0] != -999.0
    assert frequencies[0] != -999.0
    assert (
        metadata["cwt_summary"]["n_frames"]
        == MINIMUM_FRAMES
    )

def test_validate_loaded_cwt_accepts_complex_coefficients():
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    complex_coefficients = (
        coefficients.astype(np.complex128)
        + 1.0j
    )

    validated_coefficients, _, _ = (
        validate_loaded_cwt(
            complex_coefficients,
            frequencies,
            metadata,
        )
    )

    assert validated_coefficients.dtype == np.complex128
    np.testing.assert_array_equal(
        validated_coefficients,
        complex_coefficients,
    )

def test_validate_loaded_cwt_rejects_non_2d_coefficients():
    _, frequencies, metadata = create_valid_loaded_cwt()

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        validate_loaded_cwt(
            np.zeros(MINIMUM_FRAMES),
            frequencies,
            metadata,
        )

@pytest.mark.parametrize(
    "coefficients",
    [
        np.full(
            (3, MINIMUM_FRAMES),
            "invalid",
            dtype="<U7",
        ),
        np.zeros(
            (3, MINIMUM_FRAMES),
            dtype=np.bool_,
        ),
    ],
    ids=[
        "string",
        "boolean",
    ],
)
def test_validate_loaded_cwt_rejects_invalid_coefficient_dtype(
    coefficients: np.ndarray,
):
    _, frequencies, metadata = create_valid_loaded_cwt()

    with pytest.raises(
        TypeError,
        match="non-boolean numerical dtype",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            metadata,
        )

def test_validate_loaded_cwt_rejects_too_few_scales():
    _, _, metadata = create_valid_loaded_cwt()

    coefficients = np.zeros(
        (1, MINIMUM_FRAMES),
        dtype=np.float64,
    )

    frequencies = np.array(
        [20.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="at least two scales",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            metadata,
        )

def test_validate_loaded_cwt_rejects_too_few_frames():
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    with pytest.raises(
        ValueError,
        match="at least",
    ):
        validate_loaded_cwt(
            coefficients[:, : MINIMUM_FRAMES - 1],
            frequencies,
            metadata,
        )

@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_loaded_cwt_rejects_nonfinite_coefficients(
    invalid_value: float,
):
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    coefficients[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            metadata,
        )

def test_validate_loaded_cwt_rejects_non_1d_frequencies():
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies.reshape(-1, 1),
            metadata,
        )

def test_validate_loaded_cwt_rejects_frequency_count_mismatch():
    coefficients, _, metadata = create_valid_loaded_cwt()

    with pytest.raises(
        ValueError,
        match="frequency count",
    ):
        validate_loaded_cwt(
            coefficients,
            np.array(
                [20.0, 10.0],
                dtype=np.float64,
            ),
            metadata,
        )

@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_loaded_cwt_rejects_nonfinite_frequencies(
    invalid_value: float,
):
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    frequencies[1] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            metadata,
        )

@pytest.mark.parametrize(
    "invalid_value",
    [
        0.0,
        -1.0,
    ],
)
def test_validate_loaded_cwt_rejects_nonpositive_frequencies(
    invalid_value: float,
):
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    frequencies[-1] = invalid_value

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            metadata,
        )

def test_validate_loaded_cwt_rejects_nondecreasing_frequencies():
    coefficients, _, metadata = create_valid_loaded_cwt()

    frequencies = np.array(
        [20.0, 5.0, 10.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="strictly decreasing",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            metadata,
        )

def test_validate_loaded_cwt_rejects_non_dictionary_metadata():
    coefficients, frequencies, _ = create_valid_loaded_cwt()

    with pytest.raises(
        TypeError,
        match="metadata root must be a dictionary",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            [],  # type: ignore[arg-type]
        )

def test_validate_loaded_cwt_rejects_metadata_scale_count_mismatch():
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    metadata["scale_summary"]["number_of_scales"] = 4
    metadata["cwt_summary"]["n_scales"] = 4

    with pytest.raises(
        ValueError,
        match="metadata scale count",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            metadata,
        )

def test_validate_loaded_cwt_rejects_metadata_frame_count_mismatch():
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    metadata["cwt_summary"]["n_frames"] = (
        MINIMUM_FRAMES + 1
    )

    with pytest.raises(
        ValueError,
        match="metadata frame count",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            metadata,
        )

def test_validate_loaded_cwt_rejects_minimum_frequency_mismatch():
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    metadata["cwt_summary"]["frequency_min_hz"] = 4.0
    metadata["scale_summary"]["minimum_frequency_hz"] = 4.0

    with pytest.raises(
        ValueError,
        match="minimum frequency",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            metadata,
        )

def test_validate_loaded_cwt_rejects_maximum_frequency_mismatch():
    coefficients, frequencies, metadata = (
        create_valid_loaded_cwt()
    )

    metadata["cwt_summary"]["frequency_max_hz"] = 21.0
    metadata["scale_summary"]["maximum_frequency_hz"] = 21.0

    with pytest.raises(
        ValueError,
        match="maximum frequency",
    ):
        validate_loaded_cwt(
            coefficients,
            frequencies,
            metadata,
        )

def test_compute_cwt_magnitude_handles_real_coefficients():
    coefficients = np.array(
        [
            [-3.0, 0.0, 4.0],
            [5.0, -12.0, 8.0],
        ],
        dtype=np.float64,
    )

    magnitude = compute_cwt_magnitude(
        coefficients,
    )

    expected = np.array(
        [
            [3.0, 0.0, 4.0],
            [5.0, 12.0, 8.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        magnitude,
        expected,
    )

def test_compute_cwt_magnitude_handles_complex_coefficients():
    coefficients = np.array(
        [
            [3.0 + 4.0j, 5.0 + 12.0j],
            [0.0 + 0.0j, 8.0 + 15.0j],
        ],
        dtype=np.complex128,
    )

    magnitude = compute_cwt_magnitude(
        coefficients,
    )

    expected = np.array(
        [
            [5.0, 13.0],
            [0.0, 17.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        magnitude,
        expected,
        rtol=0.0,
        atol=0.0,
    )

def test_compute_cwt_magnitude_preserves_shape_and_uses_float64():
    coefficients = np.ones(
        (3, MINIMUM_FRAMES),
        dtype=np.float32,
    )

    magnitude = compute_cwt_magnitude(
        coefficients,
    )

    assert magnitude.shape == coefficients.shape
    assert magnitude.dtype == np.float64

def test_compute_cwt_magnitude_returns_finite_nonnegative_values():
    coefficients = np.linspace(
        -10.0,
        10.0,
        3 * MINIMUM_FRAMES,
        dtype=np.float64,
    ).reshape(
        3,
        MINIMUM_FRAMES,
    )

    magnitude = compute_cwt_magnitude(
        coefficients,
    )

    assert np.all(np.isfinite(magnitude))
    assert np.all(magnitude >= 0.0)

def test_compute_cwt_magnitude_returns_independent_array():
    coefficients = np.array(
        [
            [-1.0, 2.0],
            [-3.0, 4.0],
        ],
        dtype=np.float64,
    )

    original = coefficients.copy()

    magnitude = compute_cwt_magnitude(
        coefficients,
    )

    magnitude[0, 0] = 999.0

    np.testing.assert_array_equal(
        coefficients,
        original,
    )

    assert not np.shares_memory(
        magnitude,
        coefficients,
    )

def test_compute_cwt_magnitude_accepts_zero_coefficients():
    coefficients = np.zeros(
        (3, MINIMUM_FRAMES),
        dtype=np.float64,
    )

    magnitude = compute_cwt_magnitude(
        coefficients,
    )

    np.testing.assert_array_equal(
        magnitude,
        np.zeros_like(magnitude),
    )

def test_compute_cwt_magnitude_rejects_nonfinite_result():
    maximum = np.finfo(np.float64).max

    coefficients = np.array(
        [
            [
                complex(
                    maximum,
                    maximum,
                )
            ]
        ],
        dtype=np.complex128,
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        with pytest.raises(
            ValueError,
            match="finite values",
        ):
            compute_cwt_magnitude(
                coefficients,
            )

def test_compute_scale_statistics_returns_expected_values():
    magnitude = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    statistics = compute_scale_statistics(
        magnitude,
    )

    np.testing.assert_allclose(
        statistics["scale_mean"],
        np.array([2.5, 2.0]),
    )

    np.testing.assert_allclose(
        statistics["scale_std"],
        np.array(
            [
                np.std(
                    [1.0, 2.0, 3.0, 4.0],
                    ddof=0,
                ),
                0.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        statistics["scale_median"],
        np.array([2.5, 2.0]),
    )

    np.testing.assert_allclose(
        statistics["scale_max"],
        np.array([4.0, 2.0]),
    )

    np.testing.assert_allclose(
        statistics["scale_rms"],
        np.array(
            [
                np.sqrt(7.5),
                2.0,
            ]
        ),
    )

def test_compute_scale_statistics_returns_expected_schema():
    magnitude = np.ones(
        (3, MINIMUM_FRAMES),
        dtype=np.float64,
    )

    statistics = compute_scale_statistics(
        magnitude,
    )

    assert set(statistics) == {
        "scale_mean",
        "scale_std",
        "scale_median",
        "scale_max",
        "scale_rms",
    }

    for values in statistics.values():
        assert values.shape == (3,)
        assert values.dtype == np.float64

def test_compute_scale_statistics_accepts_zero_magnitude():
    magnitude = np.zeros(
        (3, MINIMUM_FRAMES),
        dtype=np.float64,
    )

    statistics = compute_scale_statistics(
        magnitude,
    )

    for values in statistics.values():
        np.testing.assert_array_equal(
            values,
            np.zeros(3),
        )

def test_compute_scale_statistics_returns_finite_nonnegative_values():
    magnitude = np.linspace(
        0.0,
        10.0,
        3 * MINIMUM_FRAMES,
        dtype=np.float64,
    ).reshape(
        3,
        MINIMUM_FRAMES,
    )

    statistics = compute_scale_statistics(
        magnitude,
    )

    for values in statistics.values():
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)

def test_compute_scale_statistics_does_not_modify_input():
    magnitude = np.arange(
        3 * MINIMUM_FRAMES,
        dtype=np.float64,
    ).reshape(
        3,
        MINIMUM_FRAMES,
    )

    original = magnitude.copy()

    statistics = compute_scale_statistics(
        magnitude,
    )

    np.testing.assert_array_equal(
        magnitude,
        original,
    )

    for values in statistics.values():
        assert not np.shares_memory(
            values,
            magnitude,
        )

def test_compute_scale_statistics_rejects_non_2d_magnitude():
    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        compute_scale_statistics(
            np.zeros(
                MINIMUM_FRAMES,
                dtype=np.float64,
            )
        )

@pytest.mark.parametrize(
    "shape",
    [
        (0, MINIMUM_FRAMES),
        (3, 0),
    ],
    ids=[
        "no-scales",
        "no-frames",
    ],
)
def test_compute_scale_statistics_rejects_empty_axis(
    shape: tuple[int, int],
):
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        compute_scale_statistics(
            np.zeros(
                shape,
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
def test_compute_scale_statistics_rejects_nonfinite_magnitude(
    invalid_value: float,
):
    magnitude = np.ones(
        (3, MINIMUM_FRAMES),
        dtype=np.float64,
    )

    magnitude[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        compute_scale_statistics(
            magnitude,
        )

def test_compute_scale_statistics_rejects_negative_magnitude():
    magnitude = np.ones(
        (3, MINIMUM_FRAMES),
        dtype=np.float64,
    )

    magnitude[0, 0] = -1.0

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        compute_scale_statistics(
            magnitude,
        )

def test_compute_scale_statistics_rejects_nonfinite_rms():
    magnitude_value = (
        2.0
        * np.sqrt(
            np.finfo(np.float64).max
        )
    )

    magnitude = np.full(
        (2, MINIMUM_FRAMES),
        magnitude_value,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="scale_rms.*finite values",
    ):
        compute_scale_statistics(
            magnitude,
        )

def test_compute_frame_statistics_returns_expected_basic_values():
    magnitude = np.array(
        [
            [1.0, 0.0, 4.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )

    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    statistics = compute_frame_statistics(
        magnitude,
        frequencies,
    )

    np.testing.assert_allclose(
        statistics["frame_total"],
        np.array([6.0, 0.0, 7.0]),
    )

    np.testing.assert_allclose(
        statistics["frame_mean"],
        np.array(
            [
                2.0,
                0.0,
                7.0 / 3.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        statistics["frame_max"],
        np.array([3.0, 0.0, 4.0]),
    )

def test_compute_frame_statistics_reports_dominant_scale():
    magnitude = np.array(
        [
            [1.0, 5.0, 4.0],
            [2.0, 1.0, 4.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )

    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    statistics = compute_frame_statistics(
        magnitude,
        frequencies,
    )

    np.testing.assert_array_equal(
        statistics["dominant_scale_index"],
        np.array(
            [2, 0, 0],
            dtype=np.int64,
        ),
    )

    np.testing.assert_allclose(
        statistics["dominant_frequency_hz"],
        np.array([5.0, 20.0, 20.0]),
    )

def test_compute_frame_statistics_reports_centroid_and_dispersion():
    magnitude = np.array(
        [
            [1.0],
            [1.0],
        ],
        dtype=np.float64,
    )

    frequencies = np.array(
        [20.0, 10.0],
        dtype=np.float64,
    )

    statistics = compute_frame_statistics(
        magnitude,
        frequencies,
    )

    assert statistics[
        "frequency_centroid_hz"
    ][0] == pytest.approx(
        15.0
    )

    assert statistics[
        "frequency_dispersion_hz"
    ][0] == pytest.approx(
        5.0
    )

def test_compute_frame_statistics_applies_zero_energy_policy():
    magnitude = np.zeros(
        (3, 4),
        dtype=np.float64,
    )

    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    statistics = compute_frame_statistics(
        magnitude,
        frequencies,
    )

    np.testing.assert_array_equal(
        statistics["frame_total"],
        np.zeros(4),
    )
    np.testing.assert_array_equal(
        statistics["frame_mean"],
        np.zeros(4),
    )
    np.testing.assert_array_equal(
        statistics["frame_max"],
        np.zeros(4),
    )
    np.testing.assert_array_equal(
        statistics["dominant_scale_index"],
        np.zeros(
            4,
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        statistics["dominant_frequency_hz"],
        np.full(
            4,
            20.0,
        ),
    )
    np.testing.assert_array_equal(
        statistics["frequency_centroid_hz"],
        np.zeros(4),
    )
    np.testing.assert_array_equal(
        statistics["frequency_dispersion_hz"],
        np.zeros(4),
    )

def test_compute_frame_statistics_returns_expected_schema():
    magnitude = np.ones(
        (3, MINIMUM_FRAMES),
        dtype=np.float64,
    )

    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    statistics = compute_frame_statistics(
        magnitude,
        frequencies,
    )

    assert set(statistics) == {
        "frame_total",
        "frame_mean",
        "frame_max",
        "dominant_scale_index",
        "dominant_frequency_hz",
        "frequency_centroid_hz",
        "frequency_dispersion_hz",
    }

    for name, values in statistics.items():
        assert values.shape == (
            MINIMUM_FRAMES,
        )

        if name == "dominant_scale_index":
            assert values.dtype == np.int64
        else:
            assert values.dtype == np.float64

def test_compute_frame_statistics_does_not_modify_inputs():
    magnitude = np.arange(
        3 * MINIMUM_FRAMES,
        dtype=np.float64,
    ).reshape(
        3,
        MINIMUM_FRAMES,
    )

    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    original_magnitude = magnitude.copy()
    original_frequencies = frequencies.copy()

    statistics = compute_frame_statistics(
        magnitude,
        frequencies,
    )

    np.testing.assert_array_equal(
        magnitude,
        original_magnitude,
    )
    np.testing.assert_array_equal(
        frequencies,
        original_frequencies,
    )

    for values in statistics.values():
        assert not np.shares_memory(
            values,
            magnitude,
        )
        assert not np.shares_memory(
            values,
            frequencies,
        )

def test_compute_frame_statistics_rejects_non_2d_magnitude():
    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        compute_frame_statistics(
            np.zeros(
                MINIMUM_FRAMES,
                dtype=np.float64,
            ),
            np.array(
                [20.0],
                dtype=np.float64,
            ),
        )

@pytest.mark.parametrize(
    "shape",
    [
        (0, MINIMUM_FRAMES),
        (3, 0),
    ],
    ids=[
        "no-scales",
        "no-frames",
    ],
)
def test_compute_frame_statistics_rejects_empty_axis(
    shape: tuple[int, int],
):
    frequencies = np.linspace(
        20.0,
        5.0,
        shape[0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        compute_frame_statistics(
            np.zeros(
                shape,
                dtype=np.float64,
            ),
            frequencies,
        )

@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_compute_frame_statistics_rejects_nonfinite_magnitude(
    invalid_value: float,
):
    magnitude = np.ones(
        (3, MINIMUM_FRAMES),
        dtype=np.float64,
    )
    magnitude[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        compute_frame_statistics(
            magnitude,
            np.array(
                [20.0, 10.0, 5.0],
                dtype=np.float64,
            ),
        )

def test_compute_frame_statistics_rejects_negative_magnitude():
    magnitude = np.ones(
        (3, MINIMUM_FRAMES),
        dtype=np.float64,
    )
    magnitude[0, 0] = -1.0

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        compute_frame_statistics(
            magnitude,
            np.array(
                [20.0, 10.0, 5.0],
                dtype=np.float64,
            ),
        )

def test_compute_frame_statistics_rejects_frequency_count_mismatch():
    with pytest.raises(
        ValueError,
        match="frequency count",
    ):
        compute_frame_statistics(
            np.ones(
                (3, MINIMUM_FRAMES),
                dtype=np.float64,
            ),
            np.array(
                [20.0, 10.0],
                dtype=np.float64,
            ),
        )

def test_compute_frame_statistics_rejects_nondecreasing_frequencies():
    with pytest.raises(
        ValueError,
        match="strictly decreasing",
    ):
        compute_frame_statistics(
            np.ones(
                (3, MINIMUM_FRAMES),
                dtype=np.float64,
            ),
            np.array(
                [20.0, 5.0, 10.0],
                dtype=np.float64,
            ),
        )

def create_scale_profile_components() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    magnitude = np.arange(
        3 * MINIMUM_FRAMES,
        dtype=np.float64,
    ).reshape(
        3,
        MINIMUM_FRAMES,
    )

    scale_statistics = compute_scale_statistics(
        magnitude,
    )

    frame_statistics = compute_frame_statistics(
        magnitude,
        frequencies,
    )

    return (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    )

def test_build_scale_profile_returns_required_arrays():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    scale_profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    assert set(scale_profile) == set(
        SCALE_PROFILE_REQUIRED_KEYS
    )

    np.testing.assert_array_equal(
        scale_profile["frequencies"],
        frequencies,
    )

    np.testing.assert_array_equal(
        scale_profile["magnitude"],
        magnitude,
    )

    for key, expected in scale_statistics.items():
        np.testing.assert_array_equal(
            scale_profile[key],
            expected,
        )

    for key, expected in frame_statistics.items():
        np.testing.assert_array_equal(
            scale_profile[key],
            expected,
        )

def test_build_scale_profile_preserves_shapes_and_dtypes():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    scale_profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    assert scale_profile["frequencies"].shape == (3,)
    assert scale_profile["magnitude"].shape == (
        3,
        MINIMUM_FRAMES,
    )

    for key in (
        "scale_mean",
        "scale_std",
        "scale_median",
        "scale_max",
        "scale_rms",
    ):
        assert scale_profile[key].shape == (3,)
        assert scale_profile[key].dtype == np.float64

    for key in (
        "frame_total",
        "frame_mean",
        "frame_max",
        "dominant_frequency_hz",
        "frequency_centroid_hz",
        "frequency_dispersion_hz",
    ):
        assert scale_profile[key].shape == (
            MINIMUM_FRAMES,
        )
        assert scale_profile[key].dtype == np.float64

    assert scale_profile[
        "dominant_scale_index"
    ].shape == (
        MINIMUM_FRAMES,
    )

    assert scale_profile[
        "dominant_scale_index"
    ].dtype == np.int64

def test_build_scale_profile_returns_independent_arrays():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    scale_profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    original_frequency = frequencies[0]
    original_magnitude = magnitude[0, 0]
    original_scale_mean = (
        scale_statistics["scale_mean"][0]
    )
    original_frame_total = (
        frame_statistics["frame_total"][0]
    )

    scale_profile["frequencies"][0] = -999.0
    scale_profile["magnitude"][0, 0] = -999.0
    scale_profile["scale_mean"][0] = -999.0
    scale_profile["frame_total"][0] = -999.0

    assert frequencies[0] == original_frequency
    assert magnitude[0, 0] == original_magnitude
    assert (
        scale_statistics["scale_mean"][0]
        == original_scale_mean
    )
    assert (
        frame_statistics["frame_total"][0]
        == original_frame_total
    )

def test_build_scale_profile_rejects_frequency_shape_mismatch():
    (
        _,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    with pytest.raises(
        ValueError,
        match="frequency shape",
    ):
        build_scale_profile(
            frequencies=np.array(
                [20.0, 10.0],
                dtype=np.float64,
            ),
            magnitude=magnitude,
            scale_statistics=scale_statistics,
            frame_statistics=frame_statistics,
        )

def test_build_scale_profile_rejects_scale_statistic_shape():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    scale_statistics["scale_mean"] = np.zeros(
        2,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="scale_mean.*unexpected shape",
    ):
        build_scale_profile(
            frequencies=frequencies,
            magnitude=magnitude,
            scale_statistics=scale_statistics,
            frame_statistics=frame_statistics,
        )

def test_build_scale_profile_rejects_frame_statistic_shape():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    frame_statistics["frame_total"] = np.zeros(
        MINIMUM_FRAMES - 1,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="frame_total.*unexpected shape",
    ):
        build_scale_profile(
            frequencies=frequencies,
            magnitude=magnitude,
            scale_statistics=scale_statistics,
            frame_statistics=frame_statistics,
        )

@pytest.mark.parametrize(
    (
        "statistics_name",
        "missing_key",
    ),
    [
        (
            "scale",
            "scale_mean",
        ),
        (
            "frame",
            "frame_total",
        ),
    ],
)
def test_build_scale_profile_rejects_missing_statistic(
    statistics_name: str,
    missing_key: str,
):
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    if statistics_name == "scale":
        del scale_statistics[missing_key]
    else:
        del frame_statistics[missing_key]

    with pytest.raises(
        KeyError,
        match=missing_key,
    ):
        build_scale_profile(
            frequencies=frequencies,
            magnitude=magnitude,
            scale_statistics=scale_statistics,
            frame_statistics=frame_statistics,
        )

def test_validate_scale_profile_arrays_accepts_valid_profile():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    validated = validate_scale_profile_arrays(
        profile,
    )

    assert set(validated) == set(
        SCALE_PROFILE_REQUIRED_KEYS
    )

    for key in SCALE_PROFILE_REQUIRED_KEYS:
        np.testing.assert_array_equal(
            validated[key],
            profile[key],
        )

def test_validate_scale_profile_arrays_returns_independent_copies():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    validated = validate_scale_profile_arrays(
        profile,
    )

    validated["frequencies"][0] = -999.0
    validated["magnitude"][0, 0] = -999.0
    validated["frame_total"][0] = -999.0

    assert profile["frequencies"][0] != -999.0
    assert profile["magnitude"][0, 0] != -999.0
    assert profile["frame_total"][0] != -999.0

def test_validate_scale_profile_arrays_rejects_missing_key():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    del profile["scale_mean"]

    with pytest.raises(
        KeyError,
        match="scale_mean",
    ):
        validate_scale_profile_arrays(
            profile,
        )

def test_validate_scale_profile_arrays_rejects_unexpected_key():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    profile["unexpected"] = np.zeros(1)  # type: ignore[typeddict-unknown-key]

    with pytest.raises(
        ValueError,
        match="unexpected arrays",
    ):
        validate_scale_profile_arrays(
            profile,
        )

@pytest.mark.parametrize(
    "key",
    [
        "magnitude",
        "scale_mean",
        "frame_total",
        "frequency_centroid_hz",
    ],
)
def test_validate_scale_profile_arrays_rejects_nonfinite_value(
    key: str,
):
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    profile[key].flat[0] = np.nan

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        validate_scale_profile_arrays(
            profile,
        )

@pytest.mark.parametrize(
    "key",
    [
        "magnitude",
        "scale_std",
        "scale_rms",
        "frame_total",
        "frequency_dispersion_hz",
    ],
)
def test_validate_scale_profile_arrays_rejects_negative_value(
    key: str,
):
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    profile[key].flat[0] = -1.0

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        validate_scale_profile_arrays(
            profile,
        )

def test_validate_scale_profile_arrays_rejects_noninteger_index():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    profile["dominant_scale_index"] = (
        profile["dominant_scale_index"].astype(
            np.float64
        )
    )

    with pytest.raises(
        TypeError,
        match="integer dtype",
    ):
        validate_scale_profile_arrays(
            profile,
        )

def test_validate_scale_profile_arrays_rejects_out_of_range_index():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    profile["dominant_scale_index"][0] = (
        profile["frequencies"].size
    )

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        validate_scale_profile_arrays(
            profile,
        )

def test_validate_scale_profile_arrays_rejects_dominant_frequency_mismatch():
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    profile["dominant_frequency_hz"][0] += 1.0

    with pytest.raises(
        ValueError,
        match="dominant_frequency_hz",
    ):
        validate_scale_profile_arrays(
            profile,
        )

@pytest.mark.parametrize(
    "invalid_centroid",
    [
        4.0,
        21.0,
    ],
)
def test_validate_scale_profile_arrays_rejects_centroid_outside_range(
    invalid_centroid: float,
):
    (
        frequencies,
        magnitude,
        scale_statistics,
        frame_statistics,
    ) = create_scale_profile_components()

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    profile["frequency_centroid_hz"][0] = (
        invalid_centroid
    )

    with pytest.raises(
        ValueError,
        match="frequency range",
    ):
        validate_scale_profile_arrays(
            profile,
        )

@pytest.mark.parametrize(
    "key",
    [
        "frequency_centroid_hz",
        "frequency_dispersion_hz",
    ],
)
def test_validate_scale_profile_arrays_rejects_nonzero_zero_energy_statistic(
    key: str,
):
    frequencies = np.array(
        [20.0, 10.0, 5.0],
        dtype=np.float64,
    )

    magnitude = np.zeros(
        (3, MINIMUM_FRAMES),
        dtype=np.float64,
    )

    scale_statistics = compute_scale_statistics(
        magnitude,
    )
    frame_statistics = compute_frame_statistics(
        magnitude,
        frequencies,
    )

    profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    profile[key][0] = 1.0

    with pytest.raises(
        ValueError,
        match="Zero-energy frames",
    ):
        validate_scale_profile_arrays(
            profile,
        )

def create_scale_profile_metadata_inputs():
    (
        coefficients,
        frequencies,
        source_cwt_metadata,
    ) = create_valid_loaded_cwt()

    magnitude = compute_cwt_magnitude(
        coefficients,
    )

    scale_statistics = compute_scale_statistics(
        magnitude,
    )

    frame_statistics = compute_frame_statistics(
        magnitude,
        frequencies,
    )

    scale_profile = build_scale_profile(
        frequencies=frequencies,
        magnitude=magnitude,
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    scale_profile = validate_scale_profile_arrays(
        scale_profile,
    )

    return (
        source_cwt_metadata,
        scale_profile,
    )

def test_build_scale_profile_metadata_returns_expected_schema():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    assert set(metadata) == {
        "format_version",
        "source_cwt",
        "feature_configuration",
        "shape_summary",
        "scale_statistics_summary",
        "frame_statistics_summary",
    }

    assert (
        metadata["format_version"]
        == SCALE_PROFILE_FORMAT_VERSION
    )

def test_build_scale_profile_metadata_preserves_source_cwt():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    source = metadata["source_cwt"]

    assert source["metadata_format_version"] == 1
    assert source["wavelet"] == "mexh"
    assert source["sampling_period"] == pytest.approx(
        0.005
    )
    assert source["sampling_frequency"] == pytest.approx(
        200.0
    )
    assert source["method"] == "conv"
    assert source["precision"] == 12
    assert source["coefficient_dtype"] == "float64"
    assert source["number_of_scales"] == 3
    assert source["number_of_frames"] == MINIMUM_FRAMES
    assert source["minimum_frequency_hz"] == pytest.approx(
        5.0
    )
    assert source["maximum_frequency_hz"] == pytest.approx(
        20.0
    )

def test_build_scale_profile_metadata_records_feature_configuration():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    configuration = metadata[
        "feature_configuration"
    ]

    assert (
        configuration["magnitude_transform"]
        == MAGNITUDE_TRANSFORM
    )
    assert (
        configuration[
            "standard_deviation_convention"
        ]
        == STANDARD_DEVIATION_CONVENTION
    )
    assert (
        configuration["zero_energy_policy"]
        == ZERO_ENERGY_POLICY
    )
    assert (
        configuration[
            "dominant_scale_tie_policy"
        ]
        == DOMINANT_SCALE_TIE_POLICY
    )
    assert (
        configuration["centroid_coordinate"]
        == CENTROID_COORDINATE
    )
    assert (
        configuration["dispersion_coordinate"]
        == DISPERSION_COORDINATE
    )

def test_build_scale_profile_metadata_reports_shapes():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    shape_summary = metadata["shape_summary"]

    assert shape_summary["number_of_scales"] == 3
    assert (
        shape_summary["number_of_frames"]
        == MINIMUM_FRAMES
    )
    assert shape_summary["frequencies_shape"] == [3]
    assert shape_summary["magnitude_shape"] == [
        3,
        MINIMUM_FRAMES,
    ]
    assert shape_summary["scale_statistic_shape"] == [
        3
    ]
    assert shape_summary["frame_statistic_shape"] == [
        MINIMUM_FRAMES
    ]
    assert (
        shape_summary["number_of_scale_statistics"]
        == NUMBER_OF_SCALE_STATISTICS
    )
    assert (
        shape_summary["number_of_frame_statistics"]
        == NUMBER_OF_FRAME_STATISTICS
    )

def test_build_scale_profile_metadata_reports_scale_statistics():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    summary = metadata[
        "scale_statistics_summary"
    ]

    assert summary["minimum_scale_mean"] == pytest.approx(
        float(np.min(scale_profile["scale_mean"]))
    )
    assert summary["maximum_scale_mean"] == pytest.approx(
        float(np.max(scale_profile["scale_mean"]))
    )
    assert summary["minimum_scale_rms"] == pytest.approx(
        float(np.min(scale_profile["scale_rms"]))
    )
    assert summary["maximum_scale_rms"] == pytest.approx(
        float(np.max(scale_profile["scale_rms"]))
    )

def test_build_scale_profile_metadata_reports_frame_statistics():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    summary = metadata[
        "frame_statistics_summary"
    ]

    assert summary["minimum_frame_total"] == pytest.approx(
        float(np.min(scale_profile["frame_total"]))
    )
    assert summary["maximum_frame_total"] == pytest.approx(
        float(np.max(scale_profile["frame_total"]))
    )
    assert summary[
        "minimum_centroid_frequency_hz"
    ] == pytest.approx(
        float(
            np.min(
                scale_profile[
                    "frequency_centroid_hz"
                ]
            )
        )
    )
    assert summary[
        "maximum_centroid_frequency_hz"
    ] == pytest.approx(
        float(
            np.max(
                scale_profile[
                    "frequency_centroid_hz"
                ]
            )
        )
    )

def test_build_scale_profile_metadata_reports_zero_energy_frames():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    scale_profile["magnitude"][:, :2] = 0.0

    scale_statistics = compute_scale_statistics(
        scale_profile["magnitude"],
    )
    frame_statistics = compute_frame_statistics(
        scale_profile["magnitude"],
        scale_profile["frequencies"],
    )

    scale_profile = build_scale_profile(
        frequencies=scale_profile["frequencies"],
        magnitude=scale_profile["magnitude"],
        scale_statistics=scale_statistics,
        frame_statistics=frame_statistics,
    )

    scale_profile = validate_scale_profile_arrays(
        scale_profile,
    )

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    summary = metadata[
        "frame_statistics_summary"
    ]

    assert summary["zero_energy_frame_count"] == 2
    assert summary["zero_energy_frame_ratio"] == pytest.approx(
        2.0 / MINIMUM_FRAMES
    )

def test_build_scale_profile_metadata_is_json_serializable():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    encoded = json.dumps(
        metadata,
    )

    decoded = json.loads(
        encoded,
    )

    assert decoded == metadata

def test_validate_scale_profile_metadata_accepts_valid_metadata():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    validated_metadata = validate_scale_profile_metadata(
        metadata,
        scale_profile=scale_profile,
    )

    assert validated_metadata == metadata

    assert (
        validated_metadata["format_version"]
        == SCALE_PROFILE_FORMAT_VERSION
    )

    assert (
        validated_metadata["shape_summary"]
        ["number_of_scales"]
        == scale_profile["frequencies"].size
    )

    assert (
        validated_metadata["shape_summary"]
        ["number_of_frames"]
        == scale_profile["magnitude"].shape[1]
    )

    assert (
        validated_metadata["source_cwt"]
        ["minimum_frequency_hz"]
        == pytest.approx(
            float(
                np.min(
                    scale_profile["frequencies"]
                )
            )
        )
    )

    assert (
        validated_metadata["source_cwt"]
        ["maximum_frequency_hz"]
        == pytest.approx(
            float(
                np.max(
                    scale_profile["frequencies"]
                )
            )
        )
    )

    assert validated_metadata is not metadata

    assert (
        validated_metadata["source_cwt"]
        is not metadata["source_cwt"]
    )

    assert (
        validated_metadata["shape_summary"]
        is not metadata["shape_summary"]
    )

    validated_metadata["shape_summary"][
        "number_of_frames"
    ] = 999

    assert (
        metadata["shape_summary"]["number_of_frames"]
        == scale_profile["magnitude"].shape[1]
    )

def test_validate_scale_profile_metadata_returns_independent_copy():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    validated_metadata = validate_scale_profile_metadata(
        metadata,
        scale_profile=scale_profile,
    )

    assert validated_metadata is not metadata

    assert (
        validated_metadata["source_cwt"]
        is not metadata["source_cwt"]
    )

    assert (
        validated_metadata["feature_configuration"]
        is not metadata["feature_configuration"]
    )

    assert (
        validated_metadata["shape_summary"]
        is not metadata["shape_summary"]
    )

    validated_metadata["shape_summary"][
        "number_of_frames"
    ] = 999

    assert (
        metadata["shape_summary"]["number_of_frames"]
        == scale_profile["magnitude"].shape[1]
    )

@pytest.mark.parametrize(
    "invalid_metadata",
    [
        [],
        "invalid",
        1,
        None,
    ],
    ids=[
        "list",
        "string",
        "integer",
        "none",
    ],
)
def test_validate_scale_profile_metadata_rejects_non_dictionary_root(
    invalid_metadata: object,
):
    (
        _,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    with pytest.raises(
        TypeError,
        match="metadata root.*mapping",
    ):
        validate_scale_profile_metadata(
            invalid_metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "missing_key",
    [
        "format_version",
        "source_cwt",
        "feature_configuration",
        "shape_summary",
        "scale_statistics_summary",
        "frame_statistics_summary",
    ],
)
def test_validate_scale_profile_metadata_rejects_missing_top_level_key(
    missing_key: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    del metadata[missing_key]  # type: ignore[misc]

    with pytest.raises(
        KeyError,
        match=missing_key,
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_unexpected_top_level_key():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["unexpected"] = "invalid"  # type: ignore[typeddict-unknown-key]

    with pytest.raises(
        ValueError,
        match="unexpected fields.*unexpected",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_version",
    [
        0,
        2,
        -1,
    ],
)
def test_validate_scale_profile_metadata_rejects_bad_format_version(
    invalid_version: int,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["format_version"] = invalid_version

    with pytest.raises(
        ValueError,
        match=(
            "format_version must be at least 1"
            if invalid_version < 1
            else "Unsupported ScaleProfile metadata format version"
        ),
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_version",
    [
        True,
        False,
    ],
)
def test_validate_scale_profile_metadata_rejects_boolean_format_version(
    invalid_version: bool,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["format_version"] = invalid_version

    with pytest.raises(
        TypeError,
        match="integer.*must not be boolean",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "missing_key",
    [
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
    ],
)
def test_validate_scale_profile_metadata_rejects_missing_source_cwt_key(
    missing_key: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    del metadata["source_cwt"][missing_key]  # type: ignore[misc]

    with pytest.raises(
        KeyError,
        match=missing_key,
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_unexpected_source_cwt_key():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"]["unexpected"] = "invalid"  # type: ignore[typeddict-unknown-key]

    with pytest.raises(
        ValueError,
        match="source_cwt contains unexpected fields.*unexpected",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_version",
    [
        0,
        2,
        -1,
    ],
)
def test_validate_scale_profile_metadata_rejects_bad_source_format_version(
    invalid_version: int,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][
        "metadata_format_version"
    ] = invalid_version

    expected_message = (
        "metadata_format_version must be at least 1"
        if invalid_version < 1
        else "Unsupported source CWT metadata format version"
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_version",
    [
        True,
        False,
    ],
)
def test_validate_scale_profile_metadata_rejects_boolean_source_format_version(
    invalid_version: bool,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][
        "metadata_format_version"
    ] = invalid_version

    with pytest.raises(
        TypeError,
        match="integer.*must not be boolean",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "wavelet",
        "coefficient_dtype",
    ],
)
def test_validate_scale_profile_metadata_rejects_empty_source_string(
    field_name: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][field_name] = "   "

    with pytest.raises(
        ValueError,
        match=f"source_cwt.{field_name} must not be empty",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_value",
    [
        0.0,
        -0.005,
        np.nan,
        np.inf,
        -np.inf,
    ],
    ids=[
        "zero",
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_validate_scale_profile_metadata_rejects_invalid_sampling_period(
    invalid_value: float,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][
        "sampling_period"
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match="source_cwt.sampling_period",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_value",
    [
        0.0,
        -200.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
    ids=[
        "zero",
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_validate_scale_profile_metadata_rejects_invalid_sampling_frequency(
    invalid_value: float,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][
        "sampling_frequency"
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match="source_cwt.sampling_frequency",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_sampling_frequency_mismatch():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][
        "sampling_frequency"
    ] = 100.0

    with pytest.raises(
        ValueError,
        match="sampling_frequency is inconsistent",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_method",
    [
        "",
        "auto",
        "invalid",
    ],
)
def test_validate_scale_profile_metadata_rejects_invalid_source_method(
    invalid_method: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"]["method"] = invalid_method

    expected_message = (
        "must not be empty"
        if not invalid_method
        else "must be one of"
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_precision",
    [
        MINIMUM_PRECISION - 1,
        0,
        -1,
    ],
)
def test_validate_scale_profile_metadata_rejects_low_precision(
    invalid_precision: int,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][
        "precision"
    ] = invalid_precision

    with pytest.raises(
        ValueError,
        match="precision must be at least",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_boolean_precision():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"]["precision"] = True

    with pytest.raises(
        TypeError,
        match="integer.*must not be boolean",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_source_scale_count_mismatch():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][
        "number_of_scales"
    ] += 1

    with pytest.raises(
        ValueError,
        match="metadata scale count",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_source_frame_count_mismatch():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][
        "number_of_frames"
    ] += 1

    with pytest.raises(
        ValueError,
        match="metadata frame count",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_message",
    ),
    [
        (
            "minimum_frequency_hz",
            4.0,
            "minimum frequency does not match",
        ),
        (
            "maximum_frequency_hz",
            21.0,
            "maximum frequency does not match",
        ),
    ],
)
def test_validate_scale_profile_metadata_rejects_source_frequency_range_mismatch(
    field_name: str,
    invalid_value: float,
    expected_message: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][
        field_name
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_reversed_source_frequency_range():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["source_cwt"][
        "minimum_frequency_hz"
    ] = 25.0

    with pytest.raises(
        ValueError,
        match="frequency range must be increasing",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

#stage 4
@pytest.mark.parametrize(
    "missing_key",
    [
        "magnitude_transform",
        "standard_deviation_convention",
        "zero_energy_policy",
        "dominant_scale_tie_policy",
        "centroid_coordinate",
        "dispersion_coordinate",
    ],
)
def test_validate_scale_profile_metadata_rejects_missing_feature_configuration_key(
    missing_key: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    del metadata["feature_configuration"][
        missing_key
    ]  # type: ignore[misc]

    with pytest.raises(
        KeyError,
        match=missing_key,
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_unexpected_feature_configuration_key():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["feature_configuration"][
        "unexpected"
    ] = "invalid"  # type: ignore[typeddict-unknown-key]

    with pytest.raises(
        ValueError,
        match=(
            "feature_configuration contains "
            "unexpected fields.*unexpected"
        ),
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
    ),
    [
        (
            "magnitude_transform",
            "power",
        ),
        (
            "standard_deviation_convention",
            "sample",
        ),
        (
            "zero_energy_policy",
            "nan-statistics",
        ),
        (
            "dominant_scale_tie_policy",
            "last-index",
        ),
        (
            "centroid_coordinate",
            "scale_index",
        ),
        (
            "dispersion_coordinate",
            "scale_index",
        ),
    ],
)
def test_validate_scale_profile_metadata_rejects_unsupported_feature_configuration(
    field_name: str,
    invalid_value: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["feature_configuration"][
        field_name
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported ScaleProfile "
            "feature configuration"
        ),
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        1,
        True,
        [],
    ],
    ids=[
        "none",
        "integer",
        "boolean",
        "list",
    ],
)
def test_validate_scale_profile_metadata_rejects_nonstring_feature_configuration(
    invalid_value: object,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["feature_configuration"][
        "magnitude_transform"
    ] = invalid_value  # type: ignore[typeddict-item]

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported ScaleProfile "
            "feature configuration"
        ),
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

#stage 5
@pytest.mark.parametrize(
    "missing_key",
    [
        "number_of_scales",
        "number_of_frames",
        "frequencies_shape",
        "magnitude_shape",
        "scale_statistic_shape",
        "frame_statistic_shape",
        "number_of_scale_statistics",
        "number_of_frame_statistics",
    ],
)
def test_validate_scale_profile_metadata_rejects_missing_shape_summary_key(
    missing_key: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    del metadata["shape_summary"][
        missing_key
    ]  # type: ignore[misc]

    with pytest.raises(
        KeyError,
        match=missing_key,
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_unexpected_shape_summary_key():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["shape_summary"][
        "unexpected"
    ] = 1  # type: ignore[typeddict-unknown-key]

    with pytest.raises(
        ValueError,
        match=(
            "shape_summary contains "
            "unexpected fields.*unexpected"
        ),
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_scale_count_shape_mismatch():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["shape_summary"][
        "number_of_scales"
    ] += 1

    with pytest.raises(
        ValueError,
        match="frequencies_shape is inconsistent",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_scale_count_array_mismatch():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    invalid_scale_count = (
        scale_profile["frequencies"].size + 1
    )

    metadata["shape_summary"][
        "number_of_scales"
    ] = invalid_scale_count

    metadata["shape_summary"][
        "frequencies_shape"
    ] = [invalid_scale_count]

    metadata["shape_summary"][
        "magnitude_shape"
    ][0] = invalid_scale_count

    metadata["shape_summary"][
        "scale_statistic_shape"
    ] = [invalid_scale_count]

    with pytest.raises(
        ValueError,
        match="metadata scale count",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_frame_count_array_mismatch():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    invalid_frame_count = (
        scale_profile["magnitude"].shape[1] + 1
    )

    metadata["shape_summary"][
        "number_of_frames"
    ] = invalid_frame_count

    metadata["shape_summary"][
        "magnitude_shape"
    ][1] = invalid_frame_count

    metadata["shape_summary"][
        "frame_statistic_shape"
    ] = [invalid_frame_count]

    with pytest.raises(
        ValueError,
        match="metadata frame count",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "number_of_scales",
        "number_of_frames",
    ],
)
def test_validate_scale_profile_metadata_rejects_boolean_dimension(
    field_name: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["shape_summary"][
        field_name
    ] = True

    with pytest.raises(
        TypeError,
        match="integer.*must not be boolean",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
    ),
    [
        ("number_of_scales", 0),
        ("number_of_scales", -1),
        ("number_of_frames", 0),
        ("number_of_frames", -1),
    ],
)
def test_validate_scale_profile_metadata_rejects_nonpositive_dimension(
    field_name: str,
    invalid_value: int,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["shape_summary"][
        field_name
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be at least 1",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_shape",
    ),
    [
        (
            "frequencies_shape",
            (3,),
        ),
        (
            "magnitude_shape",
            (3, MINIMUM_FRAMES),
        ),
        (
            "scale_statistic_shape",
            np.array([3]),
        ),
        (
            "frame_statistic_shape",
            str(MINIMUM_FRAMES),
        ),
    ],
    ids=[
        "frequency-tuple",
        "magnitude-tuple",
        "scale-numpy-array",
        "frame-string",
    ],
)
def test_validate_scale_profile_metadata_rejects_nonlist_shape(
    field_name: str,
    invalid_shape: object,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["shape_summary"][
        field_name
    ] = invalid_shape  # type: ignore[typeddict-item]

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be a list",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_shape",
    ),
    [
        (
            "frequencies_shape",
            [4],
        ),
        (
            "magnitude_shape",
            [3, MINIMUM_FRAMES + 1],
        ),
        (
            "scale_statistic_shape",
            [4],
        ),
        (
            "frame_statistic_shape",
            [MINIMUM_FRAMES + 1],
        ),
    ],
    ids=[
        "frequencies",
        "magnitude",
        "scale-statistic",
        "frame-statistic",
    ],
)
def test_validate_scale_profile_metadata_rejects_shape_mismatch(
    field_name: str,
    invalid_shape: list[int],
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["shape_summary"][
        field_name
    ] = invalid_shape  # type: ignore[typeddict-item]

    with pytest.raises(
        ValueError,
        match=f"{field_name} is inconsistent",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_count",
    [
        NUMBER_OF_SCALE_STATISTICS - 1,
        NUMBER_OF_SCALE_STATISTICS + 1,
    ],
)
def test_validate_scale_profile_metadata_rejects_scale_statistic_count_mismatch(
    invalid_count: int,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["shape_summary"][
        "number_of_scale_statistics"
    ] = invalid_count

    with pytest.raises(
        ValueError,
        match="Unexpected number of scale statistics",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_count",
    [
        NUMBER_OF_FRAME_STATISTICS - 1,
        NUMBER_OF_FRAME_STATISTICS + 1,
    ],
)
def test_validate_scale_profile_metadata_rejects_frame_statistic_count_mismatch(
    invalid_count: int,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["shape_summary"][
        "number_of_frame_statistics"
    ] = invalid_count

    with pytest.raises(
        ValueError,
        match="Unexpected number of frame statistics",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "number_of_scale_statistics",
        "number_of_frame_statistics",
    ],
)
def test_validate_scale_profile_metadata_rejects_boolean_statistic_count(
    field_name: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["shape_summary"][
        field_name
    ] = True

    with pytest.raises(
        TypeError,
        match="integer.*must not be boolean",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

#stage 6
@pytest.mark.parametrize(
    "missing_key",
    [
        "minimum_scale_mean",
        "maximum_scale_mean",
        "minimum_scale_rms",
        "maximum_scale_rms",
    ],
)
def test_validate_scale_profile_metadata_rejects_missing_scale_summary_key(
    missing_key: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    del metadata["scale_statistics_summary"][
        missing_key
    ]  # type: ignore[misc]

    with pytest.raises(
        KeyError,
        match=missing_key,
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_unexpected_scale_summary_key():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["scale_statistics_summary"][
        "unexpected"
    ] = 0.0  # type: ignore[typeddict-unknown-key]

    with pytest.raises(
        ValueError,
        match=(
            "scale_statistics_summary contains "
            "unexpected fields.*unexpected"
        ),
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "minimum_scale_mean",
        "maximum_scale_mean",
        "minimum_scale_rms",
        "maximum_scale_rms",
    ],
)
def test_validate_scale_profile_metadata_rejects_nonnumeric_scale_summary(
    field_name: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["scale_statistics_summary"][
        field_name
    ] = "invalid"  # type: ignore[typeddict-item]

    with pytest.raises(
        TypeError,
        match="real number",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_boolean_scale_summary():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["scale_statistics_summary"][
        "minimum_scale_mean"
    ] = True

    with pytest.raises(
        TypeError,
        match="real number.*must not be boolean",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "minimum_scale_mean",
        "maximum_scale_mean",
        "minimum_scale_rms",
        "maximum_scale_rms",
    ],
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
def test_validate_scale_profile_metadata_rejects_nonfinite_scale_summary(
    field_name: str,
    invalid_value: float,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["scale_statistics_summary"][
        field_name
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "minimum_scale_mean",
        "maximum_scale_mean",
        "minimum_scale_rms",
        "maximum_scale_rms",
    ],
)
def test_validate_scale_profile_metadata_rejects_negative_scale_summary(
    field_name: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["scale_statistics_summary"][
        field_name
    ] = -1.0

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be at least 0.0",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "minimum_scale_mean",
        "maximum_scale_mean",
        "minimum_scale_rms",
        "maximum_scale_rms",
    ],
)
def test_validate_scale_profile_metadata_rejects_scale_summary_mismatch(
    field_name: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    original_value = metadata[
        "scale_statistics_summary"
    ][field_name]

    metadata["scale_statistics_summary"][
        field_name
    ] = original_value + 0.01

    with pytest.raises(
        ValueError,
        match=(
            f"scale_statistics_summary.{field_name} "
            "does not match"
        ),
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    (
        "minimum_field",
        "maximum_field",
    ),
    [
        (
            "minimum_scale_mean",
            "maximum_scale_mean",
        ),
        (
            "minimum_scale_rms",
            "maximum_scale_rms",
        ),
    ],
    ids=[
        "mean",
        "rms",
    ],
)
def test_validate_scale_profile_metadata_rejects_reversed_scale_summary_range(
    minimum_field: str,
    maximum_field: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    maximum_value = metadata[
        "scale_statistics_summary"
    ][maximum_field]

    metadata["scale_statistics_summary"][
        minimum_field
    ] = maximum_value + 1.0

    with pytest.raises(
        ValueError,
        match="must not exceed",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

#stage 7
@pytest.mark.parametrize(
    "missing_key",
    [
        "minimum_frame_total",
        "maximum_frame_total",
        "minimum_centroid_frequency_hz",
        "maximum_centroid_frequency_hz",
        "zero_energy_frame_count",
        "zero_energy_frame_ratio",
    ],
)
def test_validate_scale_profile_metadata_rejects_missing_frame_summary_key(
    missing_key: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    del metadata["frame_statistics_summary"][
        missing_key
    ]  # type: ignore[misc]

    with pytest.raises(
        KeyError,
        match=missing_key,
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_unexpected_frame_summary_key():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        "unexpected"
    ] = 0.0  # type: ignore[typeddict-unknown-key]

    with pytest.raises(
        ValueError,
        match=(
            "frame_statistics_summary contains "
            "unexpected fields.*unexpected"
        ),
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "minimum_frame_total",
        "maximum_frame_total",
        "minimum_centroid_frequency_hz",
        "maximum_centroid_frequency_hz",
        "zero_energy_frame_ratio",
    ],
)
def test_validate_scale_profile_metadata_rejects_nonnumeric_frame_summary(
    field_name: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        field_name
    ] = "invalid"  # type: ignore[typeddict-item]

    with pytest.raises(
        TypeError,
        match="real number",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "minimum_frame_total",
        "maximum_frame_total",
        "minimum_centroid_frequency_hz",
        "maximum_centroid_frequency_hz",
        "zero_energy_frame_ratio",
    ],
)
def test_validate_scale_profile_metadata_rejects_boolean_frame_summary(
    field_name: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        field_name
    ] = True  # type: ignore[typeddict-item]

    with pytest.raises(
        TypeError,
        match="real number.*must not be boolean",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "minimum_frame_total",
        "maximum_frame_total",
        "minimum_centroid_frequency_hz",
        "maximum_centroid_frequency_hz",
        "zero_energy_frame_ratio",
    ],
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
def test_validate_scale_profile_metadata_rejects_nonfinite_frame_summary(
    field_name: str,
    invalid_value: float,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        field_name
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "minimum_frame_total",
        "maximum_frame_total",
        "minimum_centroid_frequency_hz",
        "maximum_centroid_frequency_hz",
        "zero_energy_frame_ratio",
    ],
)
def test_validate_scale_profile_metadata_rejects_negative_frame_summary(
    field_name: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        field_name
    ] = -1.0

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be at least 0.0",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_count",
    [
        1.5,
        "1",
        True,
        None,
    ],
    ids=[
        "float",
        "string",
        "boolean",
        "none",
    ],
)
def test_validate_scale_profile_metadata_rejects_invalid_zero_energy_count_type(
    invalid_count: object,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        "zero_energy_frame_count"
    ] = invalid_count  # type: ignore[typeddict-item]

    with pytest.raises(
        TypeError,
        match="integer.*must not be boolean",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "invalid_count",
    [
        -1,
        MINIMUM_FRAMES + 1,
    ],
    ids=[
        "negative",
        "above-frame-count",
    ],
)
def test_validate_scale_profile_metadata_rejects_invalid_zero_energy_count(
    invalid_count: int,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        "zero_energy_frame_count"
    ] = invalid_count

    expected_message = (
        "must be at least 0"
        if invalid_count < 0
        else "exceeds the number of frames"
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_zero_energy_ratio_above_one():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        "zero_energy_frame_ratio"
    ] = 1.1

    with pytest.raises(
        ValueError,
        match="must not exceed 1.0",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    "field_name",
    [
        "minimum_frame_total",
        "maximum_frame_total",
        "minimum_centroid_frequency_hz",
        "maximum_centroid_frequency_hz",
    ],
)
def test_validate_scale_profile_metadata_rejects_frame_summary_mismatch(
    field_name: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        field_name
    ] += 0.01

    with pytest.raises(
        ValueError,
        match=(
            f"frame_statistics_summary.{field_name} "
            "does not match"
        ),
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_zero_energy_count_mismatch():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        "zero_energy_frame_count"
    ] += 1

    with pytest.raises(
        ValueError,
        match="zero_energy_frame_count does not match",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

def test_validate_scale_profile_metadata_rejects_zero_energy_ratio_mismatch():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["frame_statistics_summary"][
        "zero_energy_frame_ratio"
    ] = 0.25

    with pytest.raises(
        ValueError,
        match="zero_energy_frame_ratio does not match",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

@pytest.mark.parametrize(
    (
        "minimum_field",
        "maximum_field",
    ),
    [
        (
            "minimum_frame_total",
            "maximum_frame_total",
        ),
        (
            "minimum_centroid_frequency_hz",
            "maximum_centroid_frequency_hz",
        ),
    ],
    ids=[
        "frame-total",
        "centroid-frequency",
    ],
)
def test_validate_scale_profile_metadata_rejects_reversed_frame_summary_range(
    minimum_field: str,
    maximum_field: str,
):
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    maximum_value = metadata[
        "frame_statistics_summary"
    ][maximum_field]

    metadata["frame_statistics_summary"][
        minimum_field
    ] = maximum_value + 1.0

    with pytest.raises(
        ValueError,
        match="must not exceed",
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )

#Stage 8
def test_validate_scale_profile_metadata_rejects_non_json_value():
    (
        source_cwt_metadata,
        scale_profile,
    ) = create_scale_profile_metadata_inputs()

    metadata = build_scale_profile_metadata(
        source_cwt_metadata=source_cwt_metadata,
        scale_profile=scale_profile,
    )

    metadata["format_version"] = np.int64(
        SCALE_PROFILE_FORMAT_VERSION
    )

    with pytest.raises(
        ValueError,
        match=(
            "ScaleProfile metadata must be "
            "JSON serializable"
        ),
    ):
        validate_scale_profile_metadata(
            metadata,
            scale_profile=scale_profile,
        )