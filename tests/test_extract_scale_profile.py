from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.f0.extract_cwt import (
    CWTMetadata,
    MINIMUM_FRAMES,
    validate_cwt_metadata,
)

from analysis.f0.extract_scale_profile import (
    SOURCE_CWT_REQUIRED_KEYS,
    compute_cwt_magnitude,
    load_cwt_outputs,
    validate_loaded_cwt,
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