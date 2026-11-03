from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest

from analysis.f0.extract_cwt import (
    MINIMUM_FRAMES,
    center_signal,
    convert_f0_domain,
    default_metadata_path,
    infer_sampling_period,
    load_core_inputs,
    load_interpolation_metadata,
    load_optional_mask,
    prepare_cwt_signal,
    scale_signal,
    summarize_masks,
    validate_arguments,
    validate_core_arrays,
    validate_interpolation_metadata,
    validate_interpolation_ratio,
    validate_metadata_sampling_period,
    validate_signal_variation,
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
