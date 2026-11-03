from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from analysis.f0.extract_cwt import (
    MINIMUM_FRAMES,
    center_signal,
    convert_f0_domain,
    default_metadata_path,
    generate_period_grid,
    generate_physical_scales,
    infer_sampling_period,
    load_core_inputs,
    load_interpolation_metadata,
    load_optional_mask,
    prepare_cwt_signal,
    periods_to_frequencies,
    resolve_period_range,
    scale_signal,
    scales_to_physical_coordinates,
    summarize_masks,
    validate_arguments,
    validate_core_arrays,
    validate_interpolation_metadata,
    validate_interpolation_ratio,
    validate_metadata_sampling_period,
    validate_continuous_wavelet,
    validate_positive_real,
    validate_requested_periods,
    validate_signal_variation,
    frequencies_to_scales,
)


def make_valid_arrays(
    number_of_frames: int = 201,
) -> tuple[np.ndarray, np.ndarray]:
    time_axis = np.arange(number_of_frames, dtype=np.float64) * 0.005
    f0 = 220.0 * np.exp(
        0.05 * np.sin(2.0 * np.pi * 2.0 * time_axis)
    )
    return f0, time_axis


def make_args(
    *,
    input_path,
    time_path,
    output_path,
    metadata_output=None,
    voicing_input=None,
    interpolation_mask_input=None,
    interpolation_metadata=None,
) -> Namespace:
    return Namespace(
        input=input_path,
        time_input=time_path,
        output=output_path,
        voicing_input=voicing_input,
        interpolation_mask_input=interpolation_mask_input,
        interpolation_metadata=interpolation_metadata,
        metadata_output=metadata_output,
        domain="log-hz",
        centering="median",
        scaling="none",
        wavelet="morl",
        method="conv",
        precision=12,
        number_of_scales=64,
        minimum_period=0.05,
        maximum_period=None,
        scale_spacing="log",
        max_interpolation_ratio=1.0,
        time_rtol=1e-6,
        time_atol=1e-9,
        minimum_dynamic_range=1e-12,
    )


def test_validate_core_arrays_accepts_valid_input() -> None:
    f0, time_axis = make_valid_arrays()

    loaded_f0, loaded_time = validate_core_arrays(f0, time_axis)

    assert loaded_f0.dtype == np.float64
    assert loaded_time.dtype == np.float64
    assert loaded_f0.shape == loaded_time.shape
    assert np.array_equal(loaded_f0, f0)
    assert np.array_equal(loaded_time, time_axis)


@pytest.mark.parametrize(
    "invalid_f0",
    [
        np.zeros(16, dtype=np.float64),
        np.full(16, -1.0, dtype=np.float64),
        np.full(16, np.nan, dtype=np.float64),
        np.full(16, np.inf, dtype=np.float64),
    ],
)
def test_validate_core_arrays_rejects_invalid_f0(
    invalid_f0: np.ndarray,
) -> None:
    time_axis = np.arange(16, dtype=np.float64) * 0.005

    with pytest.raises(ValueError):
        validate_core_arrays(invalid_f0, time_axis)


def test_validate_core_arrays_rejects_non_1d_f0() -> None:
    f0 = np.ones((4, 4), dtype=np.float64)
    time_axis = np.arange(16, dtype=np.float64) * 0.005

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_core_arrays(f0, time_axis)


def test_validate_core_arrays_rejects_shape_mismatch() -> None:
    f0 = np.full(16, 220.0)
    time_axis = np.arange(17, dtype=np.float64) * 0.005

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        validate_core_arrays(f0, time_axis)


def test_validate_core_arrays_rejects_short_input() -> None:
    f0 = np.full(MINIMUM_FRAMES - 1, 220.0)
    time_axis = (
        np.arange(MINIMUM_FRAMES - 1, dtype=np.float64) * 0.005
    )

    with pytest.raises(
        ValueError,
        match="at least 16 frames",
    ):
        validate_core_arrays(f0, time_axis)


def test_validate_core_arrays_rejects_non_increasing_time() -> None:
    f0 = np.full(16, 220.0)
    time_axis = np.arange(16, dtype=np.float64) * 0.005
    time_axis[8] = time_axis[7]

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        validate_core_arrays(f0, time_axis)


def test_infer_sampling_period_accepts_uniform_time() -> None:
    time_axis = np.arange(201, dtype=np.float64) * 0.005

    sampling_period = infer_sampling_period(time_axis)

    assert np.isclose(
        sampling_period,
        0.005,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(
        1.0 / sampling_period,
        200.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_infer_sampling_period_rejects_irregular_time() -> None:
    time_axis = np.arange(20, dtype=np.float64) * 0.005
    time_axis[10:] += 0.001

    with pytest.raises(
        ValueError,
        match="not uniformly sampled",
    ):
        infer_sampling_period(time_axis)


def test_load_core_inputs(tmp_path) -> None:
    f0, time_axis = make_valid_arrays()

    f0_path = tmp_path / "f0.npy"
    time_path = tmp_path / "time.npy"

    np.save(f0_path, f0)
    np.save(time_path, time_axis)

    loaded_f0, loaded_time = load_core_inputs(
        f0_path,
        time_path,
    )

    assert np.array_equal(loaded_f0, f0)
    assert np.array_equal(loaded_time, time_axis)


def test_load_optional_mask_accepts_bool_mask(tmp_path) -> None:
    mask = np.array([True, False, True], dtype=bool)
    path = tmp_path / "mask.npy"
    np.save(path, mask)

    loaded = load_optional_mask(
        path,
        expected_shape=(3,),
        name="test mask",
    )

    assert loaded is not None
    assert loaded.dtype == np.bool_
    assert np.array_equal(loaded, mask)


def test_load_optional_mask_rejects_integer_mask(tmp_path) -> None:
    mask = np.array([1, 0, 1], dtype=np.int64)
    path = tmp_path / "mask.npy"
    np.save(path, mask)

    with pytest.raises(
        ValueError,
        match="boolean dtype",
    ):
        load_optional_mask(
            path,
            expected_shape=(3,),
            name="test mask",
        )


def test_load_optional_mask_rejects_shape_mismatch(tmp_path) -> None:
    mask = np.array([True, False], dtype=bool)
    path = tmp_path / "mask.npy"
    np.save(path, mask)

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        load_optional_mask(
            path,
            expected_shape=(3,),
            name="test mask",
        )


def test_summarize_masks_reports_complementarity() -> None:
    voicing = np.array([False, True, True], dtype=bool)
    interpolation = np.array([True, False, False], dtype=bool)

    summary = summarize_masks(
        voicing_mask=voicing,
        interpolation_mask=interpolation,
    )

    assert summary["mask_overlap_frames"] == 0
    assert summary["masks_are_complementary"] is True
    assert summary["voiced_frames"] == 2
    assert summary["interpolated_frames"] == 1
    assert np.isclose(summary["interpolation_ratio"], 1.0 / 3.0)


def test_validate_interpolation_ratio_rejects_excess() -> None:
    interpolation = np.array(
        [True, True, False, False],
        dtype=bool,
    )

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        validate_interpolation_ratio(
            interpolation,
            maximum_ratio=0.25,
        )


def test_load_and_validate_interpolation_metadata(tmp_path) -> None:
    metadata = {
        "total_frames": 4,
        "observed_frames": 3,
        "interpolated_frames": 1,
        "observed_values_preserved": True,
        "frame_period_seconds": 0.005,
    }

    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    loaded = load_interpolation_metadata(path)

    assert loaded == metadata

    voicing = np.array([False, True, True, True], dtype=bool)
    interpolation = np.array(
        [True, False, False, False],
        dtype=bool,
    )

    validate_interpolation_metadata(
        loaded,
        number_of_frames=4,
        voicing_mask=voicing,
        interpolation_mask=interpolation,
    )

    validate_metadata_sampling_period(
        loaded,
        sampling_period=0.005,
        rtol=1e-6,
        atol=1e-9,
    )


def test_metadata_frame_count_mismatch_is_rejected() -> None:
    metadata = {
        "total_frames": 5,
        "observed_frames": 4,
        "interpolated_frames": 1,
        "observed_values_preserved": True,
    }

    with pytest.raises(
        ValueError,
        match="total_frames",
    ):
        validate_interpolation_metadata(
            metadata,
            number_of_frames=4,
            voicing_mask=None,
            interpolation_mask=None,
        )


def test_metadata_requires_observed_values_preserved() -> None:
    metadata = {
        "total_frames": 4,
        "observed_frames": 3,
        "interpolated_frames": 1,
        "observed_values_preserved": False,
    }

    with pytest.raises(
        ValueError,
        match="observed_values_preserved",
    ):
        validate_interpolation_metadata(
            metadata,
            number_of_frames=4,
            voicing_mask=None,
            interpolation_mask=None,
        )


def test_validate_arguments_returns_default_metadata_path(
    tmp_path,
) -> None:
    f0, time_axis = make_valid_arrays()

    f0_path = tmp_path / "f0.npy"
    time_path = tmp_path / "time.npy"
    output_path = tmp_path / "cwt.npz"

    np.save(f0_path, f0)
    np.save(time_path, time_axis)

    args = make_args(
        input_path=f0_path,
        time_path=time_path,
        output_path=output_path,
    )

    metadata_path = validate_arguments(args)

    assert metadata_path == default_metadata_path(output_path)
    assert metadata_path.name == "cwt_metadata.json"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("precision", 11, "precision"),
        ("number_of_scales", 1, "number-of-scales"),
        ("minimum_period", 0.0, "minimum-period"),
        ("max_interpolation_ratio", 1.1, "max-interpolation-ratio"),
    ],
)
def test_validate_arguments_rejects_invalid_numeric_values(
    tmp_path,
    field: str,
    value: float | int,
    message: str,
) -> None:
    f0, time_axis = make_valid_arrays()

    f0_path = tmp_path / "f0.npy"
    time_path = tmp_path / "time.npy"
    output_path = tmp_path / "cwt.npz"

    np.save(f0_path, f0)
    np.save(time_path, time_axis)

    args = make_args(
        input_path=f0_path,
        time_path=time_path,
        output_path=output_path,
    )
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        validate_arguments(args)

def test_validate_arguments_accepts_default_dynamic_range(
    tmp_path,
) -> None:
    f0, time_axis = make_valid_arrays()

    f0_path = tmp_path / "f0.npy"
    time_path = tmp_path / "time.npy"
    output_path = tmp_path / "cwt.npz"

    np.save(f0_path, f0)
    np.save(time_path, time_axis)

    args = make_args(
        input_path=f0_path,
        time_path=time_path,
        output_path=output_path,
    )

    metadata_path = validate_arguments(args)

    assert metadata_path == default_metadata_path(output_path)
    assert args.minimum_dynamic_range == 1e-12


def test_validate_arguments_accepts_zero_dynamic_range_threshold(
    tmp_path,
) -> None:
    f0, time_axis = make_valid_arrays()

    f0_path = tmp_path / "f0.npy"
    time_path = tmp_path / "time.npy"
    output_path = tmp_path / "cwt.npz"

    np.save(f0_path, f0)
    np.save(time_path, time_axis)

    args = make_args(
        input_path=f0_path,
        time_path=time_path,
        output_path=output_path,
    )
    args.minimum_dynamic_range = 0.0

    validate_arguments(args)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_validate_arguments_rejects_invalid_dynamic_range(
    tmp_path,
    invalid_value: float,
) -> None:
    f0, time_axis = make_valid_arrays()

    f0_path = tmp_path / "f0.npy"
    time_path = tmp_path / "time.npy"
    output_path = tmp_path / "cwt.npz"

    np.save(f0_path, f0)
    np.save(time_path, time_axis)

    args = make_args(
        input_path=f0_path,
        time_path=time_path,
        output_path=output_path,
    )
    args.minimum_dynamic_range = invalid_value

    with pytest.raises(
        ValueError,
        match="minimum-dynamic-range",
    ):
        validate_arguments(args)


def test_convert_f0_domain_log_hz() -> None:
    f0 = np.array(
        [100.0, 200.0, 400.0],
        dtype=np.float64,
    )
    original = f0.copy()

    converted = convert_f0_domain(
        f0,
        domain="log-hz",
    )

    np.testing.assert_allclose(
        converted,
        np.log(f0),
        rtol=1e-12,
        atol=1e-12,
    )

    assert converted.dtype == np.float64
    assert converted.shape == f0.shape
    assert np.array_equal(f0, original)
    assert not np.shares_memory(converted, f0)


def test_convert_f0_domain_hz_returns_independent_copy() -> None:
    f0 = np.array(
        [100.0, 200.0, 400.0],
        dtype=np.float64,
    )

    converted = convert_f0_domain(
        f0,
        domain="hz",
    )

    assert np.array_equal(converted, f0)
    assert converted.dtype == np.float64
    assert not np.shares_memory(converted, f0)


def test_convert_f0_domain_rejects_unsupported_domain() -> None:
    f0 = np.array([100.0, 200.0], dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="Unsupported CWT signal domain",
    ):
        convert_f0_domain(
            f0,
            domain="invalid",
        )


@pytest.mark.parametrize(
    "invalid_f0",
    [
        np.array([100.0, 0.0, 200.0]),
        np.array([100.0, -1.0, 200.0]),
        np.array([100.0, np.nan, 200.0]),
        np.array([100.0, np.inf, 200.0]),
    ],
)
def test_convert_f0_domain_rejects_invalid_values(
    invalid_f0: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        convert_f0_domain(
            invalid_f0,
            domain="log-hz",
        )


def test_convert_f0_domain_rejects_multidimensional_input() -> None:
    f0 = np.full((2, 2), 220.0, dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        convert_f0_domain(
            f0,
            domain="log-hz",
        )


def test_center_signal_median() -> None:
    signal = np.array(
        [1.0, 2.0, 10.0],
        dtype=np.float64,
    )
    original = signal.copy()

    centered, center_value = center_signal(
        signal,
        centering="median",
    )

    expected = np.array(
        [-1.0, 0.0, 8.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        centered,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(
        center_value,
        2.0,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(
        np.median(centered),
        0.0,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.array_equal(signal, original)
    assert not np.shares_memory(centered, signal)


def test_center_signal_mean() -> None:
    signal = np.array(
        [1.0, 2.0, 10.0],
        dtype=np.float64,
    )

    centered, center_value = center_signal(
        signal,
        centering="mean",
    )

    expected_center = 13.0 / 3.0
    expected = signal - expected_center

    np.testing.assert_allclose(
        centered,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(
        center_value,
        expected_center,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(
        np.mean(centered),
        0.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_center_signal_none_returns_independent_copy() -> None:
    signal = np.array(
        [1.0, 2.0, 10.0],
        dtype=np.float64,
    )

    centered, center_value = center_signal(
        signal,
        centering="none",
    )

    assert np.array_equal(centered, signal)
    assert center_value == 0.0
    assert not np.shares_memory(centered, signal)


def test_center_signal_rejects_unsupported_method() -> None:
    signal = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="Unsupported CWT signal centering",
    ):
        center_signal(
            signal,
            centering="invalid",
        )


@pytest.mark.parametrize(
    "invalid_signal",
    [
        np.array([1.0, np.nan, 2.0]),
        np.array([1.0, np.inf, 2.0]),
    ],
)
def test_center_signal_rejects_non_finite_values(
    invalid_signal: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        center_signal(
            invalid_signal,
            centering="median",
        )


def test_scale_signal_none_returns_independent_copy() -> None:
    signal = np.array(
        [-0.5, 0.0, 0.25, 1.0],
        dtype=np.float64,
    )
    original = signal.copy()

    scaled, scale_value = scale_signal(
        signal,
        scaling="none",
    )

    assert np.array_equal(scaled, signal)
    assert np.array_equal(signal, original)
    assert scaled.dtype == np.float64
    assert scaled.shape == signal.shape
    assert scale_value == 1.0
    assert not np.shares_memory(scaled, signal)


def test_scale_signal_none_converts_numeric_input_to_float64() -> None:
    signal = np.array(
        [-2, 0, 3],
        dtype=np.int64,
    )

    scaled, scale_value = scale_signal(
        signal,
        scaling="none",
    )

    expected = np.array(
        [-2.0, 0.0, 3.0],
        dtype=np.float64,
    )

    assert np.array_equal(scaled, expected)
    assert scaled.dtype == np.float64
    assert scale_value == 1.0
    assert not np.shares_memory(scaled, signal)


def test_scale_signal_rejects_unsupported_method() -> None:
    signal = np.array(
        [-1.0, 0.0, 1.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported CWT signal scaling",
    ):
        scale_signal(
            signal,
            scaling="standard",
        )


@pytest.mark.parametrize(
    "invalid_signal",
    [
        np.array([1.0, np.nan, 2.0]),
        np.array([1.0, np.inf, 2.0]),
        np.array([1.0, -np.inf, 2.0]),
    ],
)
def test_scale_signal_rejects_non_finite_values(
    invalid_signal: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        scale_signal(
            invalid_signal,
            scaling="none",
        )


def test_scale_signal_rejects_empty_input() -> None:
    signal = np.array([], dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        scale_signal(
            signal,
            scaling="none",
        )


def test_scale_signal_rejects_multidimensional_input() -> None:
    signal = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        scale_signal(
            signal,
            scaling="none",
        )


def test_validate_signal_variation_returns_dynamic_range() -> None:
    signal = np.array(
        [-1.0, 0.0, 2.0],
        dtype=np.float64,
    )
    original = signal.copy()

    dynamic_range = validate_signal_variation(
        signal,
        minimum_dynamic_range=1e-12,
    )

    assert np.isclose(
        dynamic_range,
        3.0,
        rtol=1e-12,
        atol=1e-12,
    )
    assert isinstance(dynamic_range, float)
    assert np.array_equal(signal, original)


def test_validate_signal_variation_accepts_zero_threshold() -> None:
    signal = np.array(
        [0.0, 1e-15],
        dtype=np.float64,
    )

    dynamic_range = validate_signal_variation(
        signal,
        minimum_dynamic_range=0.0,
    )

    assert dynamic_range > 0.0
    assert np.isclose(
        dynamic_range,
        1e-15,
        rtol=1e-12,
        atol=1e-18,
    )


def test_validate_signal_variation_accepts_range_above_threshold() -> None:
    signal = np.array(
        [0.0, 0.5001],
        dtype=np.float64,
    )

    dynamic_range = validate_signal_variation(
        signal,
        minimum_dynamic_range=0.5,
    )

    assert np.isclose(
        dynamic_range,
        0.5001,
        rtol=1e-12,
        atol=1e-12,
    )


def test_validate_signal_variation_rejects_constant_signal() -> None:
    signal = np.array(
        [1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="no measurable variation",
    ):
        validate_signal_variation(
            signal,
            minimum_dynamic_range=1e-12,
        )


def test_validate_signal_variation_rejects_range_equal_to_threshold() -> None:
    signal = np.array(
        [0.0, 0.5],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match=r"required>0\.5",
    ):
        validate_signal_variation(
            signal,
            minimum_dynamic_range=0.5,
        )


def test_validate_signal_variation_rejects_range_below_threshold() -> None:
    signal = np.array(
        [0.0, 0.25],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="no measurable variation",
    ):
        validate_signal_variation(
            signal,
            minimum_dynamic_range=0.5,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        "1e-12",
        None,
    ],
)
def test_validate_signal_variation_rejects_invalid_threshold(
    invalid_threshold,
) -> None:
    signal = np.array(
        [-1.0, 0.0, 1.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="finite non-negative number",
    ):
        validate_signal_variation(
            signal,
            minimum_dynamic_range=invalid_threshold,
        )


@pytest.mark.parametrize(
    "invalid_signal",
    [
        np.array([0.0, np.nan, 1.0]),
        np.array([0.0, np.inf, 1.0]),
        np.array([0.0, -np.inf, 1.0]),
    ],
)
def test_validate_signal_variation_rejects_non_finite_signal(
    invalid_signal: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        validate_signal_variation(
            invalid_signal,
            minimum_dynamic_range=1e-12,
        )


def test_validate_signal_variation_rejects_empty_signal() -> None:
    signal = np.array([], dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        validate_signal_variation(
            signal,
            minimum_dynamic_range=1e-12,
        )


def test_validate_signal_variation_rejects_multidimensional_signal() -> None:
    signal = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_signal_variation(
            signal,
            minimum_dynamic_range=1e-12,
        )


def test_validate_signal_variation_accepts_numpy_float_threshold() -> None:
    signal = np.array(
        [-1.0, 0.0, 1.0],
        dtype=np.float64,
    )

    dynamic_range = validate_signal_variation(
        signal,
        minimum_dynamic_range=np.float64(1e-12),
    )

    assert np.isclose(
        dynamic_range,
        2.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_validate_signal_variation_does_not_modify_input() -> None:
    signal = np.array(
        [-2.0, -0.5, 0.0, 3.0],
        dtype=np.float64,
    )
    original = signal.copy()

    validate_signal_variation(
        signal,
        minimum_dynamic_range=1e-12,
    )

    assert np.array_equal(signal, original)


def test_prepare_cwt_signal_default_pipeline() -> None:
    time_axis = np.arange(800, dtype=np.float64) * 0.005
    f0 = 220.0 * np.exp(
        0.1 * np.sin(
            2.0 * np.pi * 2.0 * time_axis
        )
    )
    original = f0.copy()

    prepared, summary = prepare_cwt_signal(f0)

    assert prepared.shape == f0.shape
    assert prepared.dtype == np.float64
    assert np.all(np.isfinite(prepared))
    assert np.array_equal(f0, original)
    assert not np.shares_memory(prepared, f0)

    assert summary["domain"] == "log-hz"
    assert summary["centering"] == "median"
    assert summary["scaling"] == "none"
    assert summary["number_of_frames"] == f0.size
    assert summary["scale_value"] == 1.0

    assert np.isclose(
        np.median(prepared),
        0.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_prepare_cwt_signal_matches_known_log_f0_modulation() -> None:
    time_axis = np.arange(800, dtype=np.float64) * 0.005

    expected = 0.1 * np.sin(
        2.0 * np.pi * 2.0 * time_axis
    )
    f0 = 220.0 * np.exp(expected)

    prepared, summary = prepare_cwt_signal(
        f0,
        domain="log-hz",
        centering="median",
        scaling="none",
    )

    np.testing.assert_allclose(
        prepared,
        expected,
        rtol=1e-11,
        atol=1e-11,
    )

    assert np.isclose(
        summary["center_value"],
        np.log(220.0),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(
        summary["prepared_dynamic_range"],
        0.2,
        rtol=1e-12,
        atol=1e-12,
    )


def test_prepare_cwt_signal_hz_mean_centering() -> None:
    f0 = np.array(
        [100.0, 120.0, 140.0],
        dtype=np.float64,
    )

    prepared, summary = prepare_cwt_signal(
        f0,
        domain="hz",
        centering="mean",
        scaling="none",
    )

    expected = np.array(
        [-20.0, 0.0, 20.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        prepared,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.isclose(summary["center_value"], 120.0)
    assert np.isclose(summary["prepared_signal_mean"], 0.0)
    assert np.isclose(summary["prepared_dynamic_range"], 40.0)


def test_prepare_cwt_signal_without_centering() -> None:
    f0 = np.array(
        [100.0, 120.0, 140.0],
        dtype=np.float64,
    )

    prepared, summary = prepare_cwt_signal(
        f0,
        domain="hz",
        centering="none",
        scaling="none",
    )

    assert np.array_equal(prepared, f0)
    assert not np.shares_memory(prepared, f0)
    assert summary["center_value"] == 0.0
    assert summary["scale_value"] == 1.0


def test_prepare_cwt_signal_rejects_constant_f0() -> None:
    f0 = np.full(201, 220.0, dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="no measurable variation",
    ):
        prepare_cwt_signal(f0)


def test_prepare_cwt_signal_rejects_near_constant_f0() -> None:
    f0 = np.array(
        [
            220.0,
            220.0 + 1e-14,
            220.0,
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="no measurable variation",
    ):
        prepare_cwt_signal(
            f0,
            domain="hz",
            minimum_dynamic_range=1e-12,
        )


def test_prepare_cwt_signal_respects_zero_dynamic_range_threshold() -> None:
    f0 = np.array(
        [220.0, 220.0 + 1e-10],
        dtype=np.float64,
    )

    prepared, summary = prepare_cwt_signal(
        f0,
        domain="hz",
        centering="none",
        minimum_dynamic_range=0.0,
    )

    assert prepared.shape == f0.shape
    assert summary["prepared_dynamic_range"] > 0.0
    assert summary["minimum_dynamic_range"] == 0.0


@pytest.mark.parametrize(
    ("domain", "centering", "scaling", "message"),
    [
        (
            "invalid",
            "median",
            "none",
            "Unsupported CWT signal domain",
        ),
        (
            "log-hz",
            "invalid",
            "none",
            "Unsupported CWT signal centering",
        ),
        (
            "log-hz",
            "median",
            "standard",
            "Unsupported CWT signal scaling",
        ),
    ],
)
def test_prepare_cwt_signal_rejects_unsupported_options(
    domain: str,
    centering: str,
    scaling: str,
    message: str,
) -> None:
    f0 = np.array(
        [100.0, 110.0, 120.0],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match=message):
        prepare_cwt_signal(
            f0,
            domain=domain,
            centering=centering,
            scaling=scaling,
        )


@pytest.mark.parametrize(
    "invalid_f0",
    [
        np.array([100.0, 0.0, 120.0]),
        np.array([100.0, -1.0, 120.0]),
        np.array([100.0, np.nan, 120.0]),
        np.array([100.0, np.inf, 120.0]),
    ],
)
def test_prepare_cwt_signal_rejects_invalid_f0(
    invalid_f0: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        prepare_cwt_signal(invalid_f0)


def test_prepare_cwt_signal_summary_is_complete() -> None:
    f0 = np.array(
        [100.0, 110.0, 120.0],
        dtype=np.float64,
    )

    _, summary = prepare_cwt_signal(
        f0,
        domain="log-hz",
        centering="median",
        scaling="none",
    )

    required_keys = {
        "domain",
        "centering",
        "scaling",
        "center_value",
        "scale_value",
        "number_of_frames",
        "input_f0_min_hz",
        "input_f0_max_hz",
        "input_f0_mean_hz",
        "input_f0_median_hz",
        "domain_signal_min",
        "domain_signal_max",
        "domain_dynamic_range",
        "prepared_signal_min",
        "prepared_signal_max",
        "prepared_signal_mean",
        "prepared_signal_median",
        "prepared_dynamic_range",
        "minimum_dynamic_range",
    }

    assert required_keys == set(summary)


def test_prepare_cwt_signal_summary_is_json_serializable() -> None:
    f0 = np.array(
        [100.0, 110.0, 120.0],
        dtype=np.float64,
    )

    _, summary = prepare_cwt_signal(f0)

    encoded = json.dumps(summary)
    decoded = json.loads(encoded)

    assert decoded["domain"] == "log-hz"
    assert decoded["number_of_frames"] == 3


def test_prepare_cwt_signal_preserves_input_exactly() -> None:
    f0 = np.array(
        [100.0, 110.0, 120.0],
        dtype=np.float64,
    )
    original = f0.copy()

    prepare_cwt_signal(f0)

    assert np.array_equal(f0, original)


@pytest.mark.parametrize(
    "value",
    [
        1,
        0.5,
        np.int64(2),
        np.float64(0.25),
    ],
)
def test_validate_positive_real_accepts_numeric_scalars(value) -> None:
    result = validate_positive_real(
        value,
        name="test_value",
    )

    assert isinstance(result, float)
    assert result > 0.0
    assert np.isclose(result, float(value))


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        0.0,
        -1,
        -0.5,
        np.nan,
        np.inf,
        -np.inf,
        True,
        False,
        np.bool_(True),
        "0.5",
        None,
    ],
)
def test_validate_positive_real_rejects_invalid_values(
    invalid_value,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite positive number",
    ):
        validate_positive_real(
            invalid_value,
            name="test_value",
        )


def test_resolve_period_range_uses_duration_half_by_default() -> None:
    minimum, maximum, summary = resolve_period_range(
        minimum_period=0.05,
        maximum_period=None,
        sampling_period=0.005,
        duration=1.005,
    )

    assert np.isclose(minimum, 0.05)
    assert np.isclose(maximum, 0.5025)

    assert summary["maximum_period_was_automatic"] is True
    assert summary["requested_maximum_period_seconds"] is None
    assert np.isclose(
        summary["sampling_minimum_period_seconds"],
        0.02,
    )
    assert np.isclose(
        summary["duration_maximum_period_seconds"],
        0.5025,
    )
    assert np.isclose(
        summary["minimum_period_frames"],
        10.0,
    )
    assert np.isclose(
        summary["maximum_period_frames"],
        100.5,
    )
    assert np.isclose(
        summary["minimum_cycles_in_signal"],
        20.1,
    )
    assert np.isclose(
        summary["maximum_cycles_in_signal"],
        2.0,
    )


def test_resolve_period_range_applies_automatic_cap() -> None:
    minimum, maximum, summary = resolve_period_range(
        minimum_period=0.05,
        maximum_period=None,
        sampling_period=0.005,
        duration=10.0,
    )

    assert np.isclose(minimum, 0.05)
    assert np.isclose(maximum, 2.0)
    assert summary["maximum_period_was_automatic"] is True
    assert np.isclose(
        summary["maximum_period_cap_seconds"],
        2.0,
    )


def test_resolve_period_range_accepts_explicit_maximum() -> None:
    minimum, maximum, summary = resolve_period_range(
        minimum_period=0.05,
        maximum_period=1.0,
        sampling_period=0.005,
        duration=4.0,
    )

    assert np.isclose(minimum, 0.05)
    assert np.isclose(maximum, 1.0)
    assert summary["maximum_period_was_automatic"] is False
    assert np.isclose(
        summary["requested_maximum_period_seconds"],
        1.0,
    )


def test_resolve_period_range_accepts_exact_four_frame_minimum() -> None:
    minimum, maximum, summary = resolve_period_range(
        minimum_period=0.020,
        maximum_period=0.5,
        sampling_period=0.005,
        duration=2.0,
    )

    assert np.isclose(minimum, 0.020)
    assert np.isclose(maximum, 0.5)
    assert np.isclose(
        summary["minimum_period_frames"],
        4.0,
    )


def test_resolve_period_range_normalizes_close_four_frame_boundary() -> None:
    sampling_period = 0.005
    minimum_allowed = 4.0 * sampling_period

    minimum, _, summary = resolve_period_range(
        minimum_period=minimum_allowed - 1e-14,
        maximum_period=0.5,
        sampling_period=sampling_period,
        duration=2.0,
    )

    assert minimum == minimum_allowed
    assert summary["resolved_minimum_period_seconds"] == minimum_allowed


def test_resolve_period_range_rejects_less_than_four_frames() -> None:
    with pytest.raises(
        ValueError,
        match="at least four frames",
    ):
        resolve_period_range(
            minimum_period=0.019,
            maximum_period=0.5,
            sampling_period=0.005,
            duration=2.0,
        )


def test_resolve_period_range_rejects_explicit_maximum_above_half_duration() -> None:
    with pytest.raises(
        ValueError,
        match="exceeds half",
    ):
        resolve_period_range(
            minimum_period=0.05,
            maximum_period=0.8,
            sampling_period=0.005,
            duration=1.005,
        )


def test_resolve_period_range_accepts_maximum_at_half_duration() -> None:
    _, maximum, summary = resolve_period_range(
        minimum_period=0.05,
        maximum_period=0.5025,
        sampling_period=0.005,
        duration=1.005,
    )

    assert np.isclose(maximum, 0.5025)
    assert summary["maximum_period_was_automatic"] is False


@pytest.mark.parametrize(
    ("minimum_period", "maximum_period"),
    [
        (0.5, 0.5),
        (0.6, 0.5),
    ],
)
def test_resolve_period_range_rejects_non_increasing_range(
    minimum_period: float,
    maximum_period: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="too short for the requested period range",
    ):
        resolve_period_range(
            minimum_period=minimum_period,
            maximum_period=maximum_period,
            sampling_period=0.1,
            duration=2.0,
        )


def test_resolve_period_range_rejects_short_duration_for_automatic_maximum() -> None:
    with pytest.raises(
        ValueError,
        match="too short for the requested period range",
    ):
        resolve_period_range(
            minimum_period=0.05,
            maximum_period=None,
            sampling_period=0.005,
            duration=0.08,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("minimum_period", 0.0),
        ("minimum_period", np.nan),
        ("minimum_period", True),
        ("minimum_period", "0.05"),
        ("sampling_period", 0.0),
        ("sampling_period", np.inf),
        ("sampling_period", False),
        ("duration", 0.0),
        ("duration", -1.0),
        ("duration", "1.0"),
        ("maximum_period_cap", 0.0),
        ("maximum_period_cap", np.nan),
        ("maximum_period_cap", True),
    ],
)
def test_resolve_period_range_rejects_invalid_required_values(
    field: str,
    invalid_value,
) -> None:
    arguments = {
        "minimum_period": 0.05,
        "maximum_period": None,
        "sampling_period": 0.005,
        "duration": 1.005,
        "maximum_period_cap": 2.0,
    }
    arguments[field] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite positive number",
    ):
        resolve_period_range(**arguments)


@pytest.mark.parametrize(
    "invalid_maximum",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        True,
        "0.5",
    ],
)
def test_resolve_period_range_rejects_invalid_explicit_maximum(
    invalid_maximum,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite positive number",
    ):
        resolve_period_range(
            minimum_period=0.05,
            maximum_period=invalid_maximum,
            sampling_period=0.005,
            duration=2.0,
        )


def test_resolve_period_range_summary_is_json_serializable() -> None:
    _, _, summary = resolve_period_range(
        minimum_period=0.05,
        maximum_period=None,
        sampling_period=0.005,
        duration=1.005,
    )

    encoded = json.dumps(summary)
    decoded = json.loads(encoded)

    assert decoded["maximum_period_was_automatic"] is True
    assert np.isclose(
        decoded["resolved_maximum_period_seconds"],
        0.5025,
    )


def test_resolve_period_range_summary_is_complete() -> None:
    _, _, summary = resolve_period_range(
        minimum_period=0.05,
        maximum_period=None,
        sampling_period=0.005,
        duration=1.005,
    )

    expected_keys = {
        "requested_minimum_period_seconds",
        "requested_maximum_period_seconds",
        "sampling_minimum_period_seconds",
        "duration_maximum_period_seconds",
        "maximum_period_cap_seconds",
        "resolved_minimum_period_seconds",
        "resolved_maximum_period_seconds",
        "maximum_period_was_automatic",
        "minimum_period_frames",
        "maximum_period_frames",
        "minimum_cycles_in_signal",
        "maximum_cycles_in_signal",
    }

    assert set(summary) == expected_keys


def test_generate_period_grid_log_spacing() -> None:
    periods = generate_period_grid(
        minimum_period=0.05,
        maximum_period=0.8,
        number_of_scales=5,
        spacing="log",
    )

    expected = np.geomspace(
        0.05,
        0.8,
        num=5,
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        periods,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    assert periods.shape == (5,)
    assert periods.dtype == np.float64
    assert np.all(np.diff(periods) > 0.0)


def test_generate_period_grid_linear_spacing() -> None:
    periods = generate_period_grid(
        minimum_period=0.05,
        maximum_period=0.25,
        number_of_scales=5,
        spacing="linear",
    )

    expected = np.array(
        [0.05, 0.10, 0.15, 0.20, 0.25],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        periods,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "spacing",
    [
        "log",
        "linear",
    ],
)
def test_generate_period_grid_preserves_endpoints(
    spacing: str,
) -> None:
    minimum = 0.05
    maximum = 0.5025

    periods = generate_period_grid(
        minimum_period=minimum,
        maximum_period=maximum,
        number_of_scales=64,
        spacing=spacing,
    )

    assert periods[0] == minimum
    assert periods[-1] == maximum


@pytest.mark.parametrize(
    "spacing",
    [
        "log",
        "linear",
    ],
)
def test_generate_period_grid_values_are_valid(
    spacing: str,
) -> None:
    periods = generate_period_grid(
        minimum_period=0.05,
        maximum_period=0.5,
        number_of_scales=64,
        spacing=spacing,
    )

    assert periods.shape == (64,)
    assert periods.dtype == np.float64
    assert np.all(np.isfinite(periods))
    assert np.all(periods > 0.0)
    assert np.all(np.diff(periods) > 0.0)


def test_generate_period_grid_log_has_constant_ratio() -> None:
    periods = generate_period_grid(
        minimum_period=0.05,
        maximum_period=0.8,
        number_of_scales=17,
        spacing="log",
    )

    ratios = periods[1:] / periods[:-1]

    np.testing.assert_allclose(
        ratios,
        np.full_like(ratios, ratios[0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_generate_period_grid_linear_has_constant_difference() -> None:
    periods = generate_period_grid(
        minimum_period=0.05,
        maximum_period=0.8,
        number_of_scales=17,
        spacing="linear",
    )

    differences = np.diff(periods)

    np.testing.assert_allclose(
        differences,
        np.full_like(differences, differences[0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_generate_period_grid_two_scales() -> None:
    periods = generate_period_grid(
        minimum_period=0.05,
        maximum_period=0.5,
        number_of_scales=2,
        spacing="log",
    )

    assert np.array_equal(
        periods,
        np.array([0.05, 0.5], dtype=np.float64),
    )


@pytest.mark.parametrize(
    "invalid_count",
    [
        0,
        1,
        -1,
        2.5,
        True,
        False,
        np.bool_(True),
        "64",
        None,
    ],
)
def test_generate_period_grid_rejects_invalid_scale_count(
    invalid_count,
) -> None:
    with pytest.raises(
        ValueError,
        match="integer of at least 2",
    ):
        generate_period_grid(
            minimum_period=0.05,
            maximum_period=0.5,
            number_of_scales=invalid_count,
            spacing="log",
        )


@pytest.mark.parametrize(
    "invalid_spacing",
    [
        "geometric",
        "logarithmic",
        "LINEAR",
        "",
        1,
        None,
    ],
)
def test_generate_period_grid_rejects_invalid_spacing(
    invalid_spacing,
) -> None:
    with pytest.raises(
        ValueError,
        match="spacing",
    ):
        generate_period_grid(
            minimum_period=0.05,
            maximum_period=0.5,
            number_of_scales=64,
            spacing=invalid_spacing,
        )


@pytest.mark.parametrize(
    ("minimum_period", "maximum_period"),
    [
        (0.5, 0.5),
        (0.6, 0.5),
        (0.0, 0.5),
        (-0.1, 0.5),
        (0.05, 0.0),
        (0.05, -0.5),
        (np.nan, 0.5),
        (0.05, np.inf),
        (True, 0.5),
        (0.05, False),
        ("0.05", 0.5),
        (0.05, "0.5"),
    ],
)
def test_generate_period_grid_rejects_invalid_period_range(
    minimum_period,
    maximum_period,
) -> None:
    with pytest.raises(ValueError):
        generate_period_grid(
            minimum_period=minimum_period,
            maximum_period=maximum_period,
            number_of_scales=64,
            spacing="log",
        )


def test_generate_period_grid_accepts_numpy_scalars() -> None:
    periods = generate_period_grid(
        minimum_period=np.float64(0.05),
        maximum_period=np.float64(0.5),
        number_of_scales=np.int64(8),
        spacing="log",
    )

    assert periods.shape == (8,)
    assert periods.dtype == np.float64
    assert periods[0] == 0.05
    assert periods[-1] == 0.5


def test_generate_period_grid_returns_independent_arrays() -> None:
    first = generate_period_grid(
        minimum_period=0.05,
        maximum_period=0.5,
        number_of_scales=8,
        spacing="log",
    )
    second = generate_period_grid(
        minimum_period=0.05,
        maximum_period=0.5,
        number_of_scales=8,
        spacing="log",
    )

    assert np.array_equal(first, second)
    assert not np.shares_memory(first, second)


def test_periods_to_frequencies() -> None:
    periods = np.array(
        [0.05, 0.10, 0.20, 0.50],
        dtype=np.float64,
    )

    frequencies = periods_to_frequencies(periods)

    expected = np.array(
        [20.0, 10.0, 5.0, 2.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        frequencies,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.all(np.diff(frequencies) < 0.0)
    assert not np.shares_memory(frequencies, periods)


def test_validate_requested_periods_returns_independent_copy() -> None:
    periods = np.array(
        [0.05, 0.10, 0.20],
        dtype=np.float64,
    )

    validated = validate_requested_periods(periods)

    assert np.array_equal(validated, periods)
    assert validated.dtype == np.float64
    assert not np.shares_memory(validated, periods)


@pytest.mark.parametrize(
    "invalid_periods",
    [
        np.array([0.05]),
        np.array([0.05, 0.05]),
        np.array([0.10, 0.05]),
        np.array([0.05, 0.0]),
        np.array([0.05, -0.1]),
        np.array([0.05, np.nan]),
        np.array([0.05, np.inf]),
        np.ones((2, 2), dtype=np.float64),
    ],
)
def test_validate_requested_periods_rejects_invalid_grid(
    invalid_periods: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        validate_requested_periods(invalid_periods)


def test_validate_continuous_wavelet_accepts_morl() -> None:
    wavelet = validate_continuous_wavelet("morl")

    assert wavelet.name == "morl"


@pytest.mark.parametrize(
    "invalid_wavelet",
    [
        "",
        "not-a-wavelet",
        1,
        None,
    ],
)
def test_validate_continuous_wavelet_rejects_invalid_name(
    invalid_wavelet,
) -> None:
    with pytest.raises(
        ValueError,
        match="wavelet",
    ):
        validate_continuous_wavelet(invalid_wavelet)


def test_frequencies_to_scales_morl() -> None:
    periods = np.geomspace(0.05, 0.5, 16)
    frequencies = 1.0 / periods

    scales, normalized = frequencies_to_scales(
        frequencies,
        wavelet="morl",
        sampling_period=0.005,
    )

    assert scales.shape == periods.shape
    assert normalized.shape == periods.shape
    assert scales.dtype == np.float64
    assert normalized.dtype == np.float64
    assert np.all(scales > 0.0)
    assert np.all(np.diff(scales) > 0.0)

    np.testing.assert_allclose(
        normalized,
        frequencies * 0.005,
        rtol=1e-12,
        atol=1e-12,
    )


def test_frequencies_to_scales_rejects_nyquist_violation() -> None:
    frequencies = np.array(
        [120.0, 100.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Nyquist",
    ):
        frequencies_to_scales(
            frequencies,
            wavelet="morl",
            sampling_period=0.005,
        )


def test_scales_to_physical_coordinates() -> None:
    periods = np.geomspace(0.05, 0.5, 16)
    frequencies = 1.0 / periods

    scales, _ = frequencies_to_scales(
        frequencies,
        wavelet="morl",
        sampling_period=0.005,
    )

    actual_frequencies, actual_periods = (
        scales_to_physical_coordinates(
            scales,
            wavelet="morl",
            sampling_period=0.005,
        )
    )

    np.testing.assert_allclose(
        actual_frequencies,
        frequencies,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual_periods,
        periods,
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "spacing",
    [
        "log",
        "linear",
    ],
)
def test_generate_physical_scales_round_trip(
    spacing: str,
) -> None:
    requested_periods = generate_period_grid(
        minimum_period=0.05,
        maximum_period=0.5025,
        number_of_scales=64,
        spacing=spacing,
    )

    (
        scales,
        frequencies,
        actual_periods,
        summary,
    ) = generate_physical_scales(
        periods_seconds=requested_periods,
        wavelet="morl",
        sampling_period=0.005,
    )

    assert scales.shape == (64,)
    assert frequencies.shape == (64,)
    assert actual_periods.shape == (64,)

    assert np.all(np.diff(scales) > 0.0)
    assert np.all(np.diff(frequencies) < 0.0)
    assert np.all(np.diff(actual_periods) > 0.0)

    np.testing.assert_allclose(
        actual_periods,
        requested_periods,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual_periods,
        1.0 / frequencies,
        rtol=1e-12,
        atol=1e-12,
    )

    assert summary["wavelet"] == "morl"
    assert summary["number_of_scales"] == 64
    assert summary["maximum_relative_period_error"] <= 1e-10


def test_generate_physical_scales_summary_is_json_serializable() -> None:
    requested_periods = np.geomspace(
        0.05,
        0.5,
        num=16,
        dtype=np.float64,
    )

    _, _, _, summary = generate_physical_scales(
        periods_seconds=requested_periods,
        wavelet="morl",
        sampling_period=0.005,
    )

    decoded = json.loads(json.dumps(summary))

    assert decoded["wavelet"] == "morl"
    assert decoded["number_of_scales"] == 16


def test_generate_physical_scales_preserves_requested_periods() -> None:
    requested_periods = np.geomspace(
        0.05,
        0.5,
        num=16,
        dtype=np.float64,
    )
    original = requested_periods.copy()

    generate_physical_scales(
        periods_seconds=requested_periods,
        wavelet="morl",
        sampling_period=0.005,
    )

    assert np.array_equal(requested_periods, original)


def test_generate_physical_scales_summary_is_complete() -> None:
    requested_periods = np.geomspace(
        0.05,
        0.5,
        num=16,
        dtype=np.float64,
    )

    _, _, _, summary = generate_physical_scales(
        periods_seconds=requested_periods,
        wavelet="morl",
        sampling_period=0.005,
    )

    expected_keys = {
        "wavelet",
        "number_of_scales",
        "sampling_period_seconds",
        "sampling_frequency_hz",
        "requested_minimum_period_seconds",
        "requested_maximum_period_seconds",
        "actual_minimum_period_seconds",
        "actual_maximum_period_seconds",
        "maximum_frequency_hz",
        "minimum_frequency_hz",
        "minimum_normalized_frequency",
        "maximum_normalized_frequency",
        "minimum_scale",
        "maximum_scale",
        "maximum_absolute_period_error_seconds",
        "maximum_relative_period_error",
    }

    assert set(summary) == expected_keys


def create_cli_cwt_inputs(
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Create a valid variable-F0 fixture for CLI error-path tests."""

    time_axis = np.arange(
        201,
        dtype=np.float64,
    ) * 0.005

    f0 = 220.0 * np.exp(
        0.05 * np.sin(
            2.0 * np.pi * 2.0 * time_axis
        )
    )

    f0_path = tmp_path / "continuous_f0.npy"
    time_path = tmp_path / "time.npy"

    np.save(f0_path, f0)
    np.save(time_path, time_axis)

    return f0_path, time_path


def run_extract_cwt_cli(
    *,
    f0_path: Path,
    time_path: Path,
    output_path: Path,
    extra_arguments: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run extract_cwt.py as a real CLI subprocess."""

    repository_root = Path(__file__).resolve().parents[1]
    script_path = (
        repository_root
        / "analysis"
        / "f0"
        / "extract_cwt.py"
    )

    command = [
        sys.executable,
        str(script_path),
        "--input",
        str(f0_path),
        "--time-input",
        str(time_path),
        "--output",
        str(output_path),
    ]

    if extra_arguments:
        command.extend(extra_arguments)

    return subprocess.run(
        command,
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("extra_arguments", "expected_message"),
    [
        (
            ["--minimum-period", "0.019"],
            "minimum period must span at least four frames",
        ),
        (
            ["--maximum-period", "0.8"],
            "maximum period exceeds half of the input duration",
        ),
        (
            [
                "--minimum-period",
                "0.5",
                "--maximum-period",
                "0.5",
            ],
            "maximum-period must be greater than",
        ),
        (
            ["--wavelet", "not-a-wavelet"],
            "Unsupported continuous wavelet",
        ),
        (
            ["--number-of-scales", "1"],
            "number-of-scales must be at least 2",
        ),
    ],
)
def test_cwt_cli_rejects_invalid_semantic_configuration(
    tmp_path: Path,
    extra_arguments: list[str],
    expected_message: str,
) -> None:
    f0_path, time_path = create_cli_cwt_inputs(tmp_path)
    output_path = tmp_path / "invalid_cwt.npz"
    metadata_path = tmp_path / "invalid_cwt_metadata.json"

    result = run_extract_cwt_cli(
        f0_path=f0_path,
        time_path=time_path,
        output_path=output_path,
        extra_arguments=extra_arguments,
    )

    assert result.returncode == 1
    assert expected_message in result.stderr
    assert not output_path.exists()
    assert not metadata_path.exists()


def test_cwt_cli_rejects_invalid_scale_spacing(
    tmp_path: Path,
) -> None:
    """Argparse choice errors use the standard exit code 2."""

    f0_path, time_path = create_cli_cwt_inputs(tmp_path)
    output_path = tmp_path / "invalid_spacing_cwt.npz"

    result = run_extract_cwt_cli(
        f0_path=f0_path,
        time_path=time_path,
        output_path=output_path,
        extra_arguments=[
            "--scale-spacing",
            "invalid",
        ],
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert "--scale-spacing" in result.stderr
    assert not output_path.exists()


def test_cwt_cli_rejects_period_below_operational_limit(
    tmp_path: Path,
) -> None:
    """The four-frame limit rejects this before Nyquist conversion."""

    f0_path, time_path = create_cli_cwt_inputs(tmp_path)
    output_path = tmp_path / "nyquist_guard_cwt.npz"

    result = run_extract_cwt_cli(
        f0_path=f0_path,
        time_path=time_path,
        output_path=output_path,
        extra_arguments=[
            "--minimum-period",
            "0.005",
        ],
    )

    assert result.returncode == 1
    assert (
        "minimum period must span at least four frames"
        in result.stderr
    )
    assert not output_path.exists()


def test_cwt_cli_rejects_constant_signal_without_output(
    tmp_path: Path,
) -> None:
    time_axis = np.arange(
        201,
        dtype=np.float64,
    ) * 0.005
    constant_f0 = np.full(
        201,
        220.0,
        dtype=np.float64,
    )

    f0_path = tmp_path / "constant_f0.npy"
    time_path = tmp_path / "time.npy"
    output_path = tmp_path / "constant_cwt.npz"
    metadata_path = tmp_path / "constant_cwt_metadata.json"

    np.save(f0_path, constant_f0)
    np.save(time_path, time_axis)

    result = run_extract_cwt_cli(
        f0_path=f0_path,
        time_path=time_path,
        output_path=output_path,
    )

    assert result.returncode == 1
    assert "no measurable variation" in result.stderr
    assert not output_path.exists()
    assert not metadata_path.exists()


def test_cwt_cli_error_message_is_written_to_stderr(
    tmp_path: Path,
) -> None:
    f0_path, time_path = create_cli_cwt_inputs(tmp_path)
    output_path = tmp_path / "stderr_test_cwt.npz"

    result = run_extract_cwt_cli(
        f0_path=f0_path,
        time_path=time_path,
        output_path=output_path,
        extra_arguments=[
            "--wavelet",
            "not-a-wavelet",
        ],
    )

    assert result.returncode == 1
    assert result.stderr.startswith("error:")
    assert "Unsupported continuous wavelet" in result.stderr
    assert not output_path.exists()
