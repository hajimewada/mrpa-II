from __future__ import annotations

import numpy as np
import pytest

from analysis.f0.interpolate_f0 import interpolate_f0


RTOL = 1e-12
ATOL = 1e-12


def test_internal_gap_in_log_hz_domain() -> None:
    f0 = np.array([100.0, 0.0, 0.0, 200.0])
    time_axis = np.array([0.0, 1.0, 2.0, 3.0])

    interpolated, voicing, interpolation_mask = interpolate_f0(
        f0,
        time_axis,
        method="linear",
        domain="log-hz",
        edge_policy="nearest",
    )

    expected = np.exp(
        np.interp(
            time_axis,
            time_axis[f0 > 0.0],
            np.log(f0[f0 > 0.0]),
        )
    )

    # Observed values must be preserved exactly.
    assert np.array_equal(
        interpolated[voicing],
        f0[voicing],
    )

    # Computed interpolation values use tolerance-based comparison.
    np.testing.assert_allclose(
        interpolated,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_array_equal(
        voicing,
        np.array([True, False, False, True]),
    )
    np.testing.assert_array_equal(
        interpolation_mask,
        np.array([False, True, True, False]),
    )


def test_leading_gap_uses_nearest_value() -> None:
    f0 = np.array([0.0, 0.0, 100.0, 110.0])
    time_axis = np.array([0.0, 1.0, 2.0, 3.0])

    interpolated, voicing, interpolation_mask = interpolate_f0(
        f0,
        time_axis,
        method="linear",
        domain="log-hz",
        edge_policy="nearest",
    )

    expected = np.array([100.0, 100.0, 100.0, 110.0])

    np.testing.assert_allclose(
        interpolated,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    assert np.array_equal(
        interpolated[voicing],
        f0[voicing],
    )

    np.testing.assert_array_equal(
        voicing,
        np.array([False, False, True, True]),
    )
    np.testing.assert_array_equal(
        interpolation_mask,
        np.array([True, True, False, False]),
    )


def test_trailing_gap_uses_nearest_value() -> None:
    f0 = np.array([100.0, 110.0, 0.0, 0.0])
    time_axis = np.array([0.0, 1.0, 2.0, 3.0])

    interpolated, voicing, interpolation_mask = interpolate_f0(
        f0,
        time_axis,
        method="linear",
        domain="log-hz",
        edge_policy="nearest",
    )

    expected = np.array([100.0, 110.0, 110.0, 110.0])

    np.testing.assert_allclose(
        interpolated,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    assert np.array_equal(
        interpolated[voicing],
        f0[voicing],
    )

    np.testing.assert_array_equal(
        voicing,
        np.array([True, True, False, False]),
    )
    np.testing.assert_array_equal(
        interpolation_mask,
        np.array([False, False, True, True]),
    )


def test_all_voiced_values_are_copied_exactly() -> None:
    f0 = np.array([100.0, 110.0, 120.0])
    time_axis = np.array([0.0, 1.0, 2.0])

    interpolated, voicing, interpolation_mask = interpolate_f0(
        f0,
        time_axis,
    )

    assert np.array_equal(interpolated, f0)
    assert np.all(voicing)
    assert not np.any(interpolation_mask)


def test_all_unvoiced_input_is_rejected() -> None:
    f0 = np.array([0.0, 0.0, 0.0])
    time_axis = np.array([0.0, 1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="contains no voiced frames",
    ):
        interpolate_f0(f0, time_axis)


def test_single_voiced_frame_is_rejected() -> None:
    f0 = np.array([0.0, 100.0, 0.0])
    time_axis = np.array([0.0, 1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="At least two voiced frames",
    ):
        interpolate_f0(f0, time_axis)


@pytest.mark.parametrize(
    "f0",
    [
        np.array([100.0, np.nan, 120.0]),
        np.array([100.0, np.inf, 120.0]),
        np.array([100.0, -1.0, 120.0]),
    ],
)
def test_invalid_f0_values_are_rejected(f0: np.ndarray) -> None:
    time_axis = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        interpolate_f0(f0, time_axis)


def test_non_increasing_time_axis_is_rejected() -> None:
    f0 = np.array([100.0, 0.0, 120.0])
    time_axis = np.array([0.0, 1.0, 1.0])

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        interpolate_f0(f0, time_axis)


def test_shape_mismatch_is_rejected() -> None:
    f0 = np.array([100.0, 0.0, 120.0])
    time_axis = np.array([0.0, 1.0])

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        interpolate_f0(f0, time_axis)


def test_edge_policy_error_rejects_leading_gap() -> None:
    f0 = np.array([0.0, 100.0, 110.0])
    time_axis = np.array([0.0, 1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="Leading or trailing missing frames",
    ):
        interpolate_f0(
            f0,
            time_axis,
            edge_policy="error",
        )


def test_220_hz_extraction_output(tmp_path) -> None:
    """Validate interpolation behavior on a realistic extracted contour."""

    raw = np.array(
        [
            0.0,
            218.02042213565952,
            219.89047860,
            219.89517854,
            219.89174650,
        ],
        dtype=np.float64,
    )
    time_axis = np.arange(raw.size, dtype=np.float64) * 0.005

    interpolated, voicing, interpolation_mask = interpolate_f0(
        raw,
        time_axis,
        method="linear",
        domain="log-hz",
        edge_policy="nearest",
    )

    assert interpolated.shape == raw.shape
    assert np.all(np.isfinite(interpolated))
    assert np.all(interpolated > 0.0)

    # Raw observations remain bit-for-bit unchanged.
    assert np.array_equal(
        interpolated[voicing],
        raw[voicing],
    )

    # Edge extension passes through log/exp, so compare with tolerance.
    assert np.isclose(
        interpolated[0],
        interpolated[1],
        rtol=RTOL,
        atol=ATOL,
    )

    assert interpolation_mask[0]
    assert np.count_nonzero(interpolation_mask) == 1
