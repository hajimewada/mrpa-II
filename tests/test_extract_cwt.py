from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest

from analysis.f0.extract_cwt import (
    MINIMUM_FRAMES,
    default_metadata_path,
    infer_sampling_period,
    load_core_inputs,
    load_interpolation_metadata,
    load_optional_mask,
    summarize_masks,
    validate_arguments,
    validate_core_arrays,
    validate_interpolation_metadata,
    validate_interpolation_ratio,
    validate_metadata_sampling_period,
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