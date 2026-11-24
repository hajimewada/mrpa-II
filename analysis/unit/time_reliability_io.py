"""Time, voicing, and interpolation-mask input loader for MRPA-II."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class TimeReliabilityInput:
    """Validated time axis and optional reliability masks.

    Missing optional masks are represented by ``None``. This is distinct
    from an all-false mask, which means the input was supplied and no frame
    was marked true.
    """

    time: FloatArray
    voicing: BoolArray | None
    interpolation_mask: BoolArray | None

    time_path: Path
    voicing_path: Path | None
    interpolation_mask_path: Path | None

    def __post_init__(self) -> None:
        time = _require_float64_vector("time", self.time)
        if time.size == 0:
            raise ValueError("time cannot be empty")
        _require_finite_array("time", time)
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time must be strictly increasing")
        if time[0] < 0.0:
            raise ValueError("time cannot start before 0 seconds")
        if not self.time_path.is_absolute():
            raise ValueError("time_path must be absolute")

        for name, values, path in (
            ("voicing", self.voicing, self.voicing_path),
            (
                "interpolation_mask",
                self.interpolation_mask,
                self.interpolation_mask_path,
            ),
        ):
            if values is None:
                if path is not None:
                    raise ValueError(
                        f"{name}_path must be None when {name} is unavailable"
                    )
                continue

            mask = _require_bool_vector(name, values)
            if mask.shape != time.shape:
                raise ValueError(
                    f"{name} shape mismatch: {mask.shape} != {time.shape}"
                )
            if path is None or not path.is_absolute():
                raise ValueError(
                    f"{name}_path must be an absolute path when available"
                )

    @property
    def frame_count(self) -> int:
        return int(self.time.size)

    @property
    def sampling_period_seconds(self) -> float:
        """Median adjacent-frame interval."""

        if self.time.size < 2:
            raise ValueError(
                "At least two frames are required to infer sampling period"
            )
        return float(np.median(np.diff(self.time)))

    @property
    def sampling_frequency_hz(self) -> float:
        return 1.0 / self.sampling_period_seconds

    @property
    def voicing_available(self) -> bool:
        return self.voicing is not None

    @property
    def interpolation_mask_available(self) -> bool:
        return self.interpolation_mask is not None

    @property
    def voiced_frame_count(self) -> int | None:
        if self.voicing is None:
            return None
        return int(np.count_nonzero(self.voicing))

    @property
    def interpolated_frame_count(self) -> int | None:
        if self.interpolation_mask is None:
            return None
        return int(np.count_nonzero(self.interpolation_mask))

    @property
    def overlap_frame_count(self) -> int | None:
        """Frames marked both voiced and interpolated, for diagnostics."""

        if self.voicing is None or self.interpolation_mask is None:
            return None
        return int(
            np.count_nonzero(self.voicing & self.interpolation_mask)
        )


def load_time_reliability_input(
    time_path: str | Path,
    *,
    voicing_path: str | Path | None = None,
    interpolation_mask_path: str | Path | None = None,
    expected_frame_count: int | None = None,
    expected_sampling_period_seconds: float | None = None,
    sampling_period_rtol: float = 1.0e-9,
    sampling_period_atol: float = 1.0e-12,
) -> TimeReliabilityInput:
    """Load and validate time, voicing, and interpolation arrays.

    The time axis is required. Reliability masks are optional. Numeric masks
    are accepted only when every value is exactly 0 or 1, then converted to
    ``numpy.bool_``.
    """

    resolved_time = _resolve_existing_file(time_path, "Time axis")
    time = _load_float64_npy(resolved_time, "time")

    if expected_frame_count is not None:
        if expected_frame_count <= 0:
            raise ValueError("expected_frame_count must be > 0")
        if time.size != expected_frame_count:
            raise ValueError(
                "Time-axis frame count mismatch: "
                f"expected={expected_frame_count}, actual={time.size}"
            )

    resolved_voicing, voicing = _load_optional_mask(
        voicing_path,
        "Voicing mask",
        "voicing",
    )
    resolved_interpolation, interpolation = _load_optional_mask(
        interpolation_mask_path,
        "Interpolation mask",
        "interpolation_mask",
    )

    result = TimeReliabilityInput(
        time=time,
        voicing=voicing,
        interpolation_mask=interpolation,
        time_path=resolved_time,
        voicing_path=resolved_voicing,
        interpolation_mask_path=resolved_interpolation,
    )

    if expected_sampling_period_seconds is not None:
        if (
            not np.isfinite(expected_sampling_period_seconds)
            or expected_sampling_period_seconds <= 0.0
        ):
            raise ValueError(
                "expected_sampling_period_seconds must be finite and > 0"
            )
        actual = result.sampling_period_seconds
        if not np.isclose(
            actual,
            expected_sampling_period_seconds,
            rtol=sampling_period_rtol,
            atol=sampling_period_atol,
        ):
            raise ValueError(
                "Sampling-period mismatch: "
                f"expected={expected_sampling_period_seconds!r}, "
                f"actual={actual!r}"
            )

    return result


def _load_optional_mask(
    path: str | Path | None,
    label: str,
    array_name: str,
) -> tuple[Path | None, BoolArray | None]:
    if path is None:
        return None, None

    resolved = _resolve_existing_file(path, label)
    return resolved, _load_bool_mask_npy(resolved, array_name)


def _load_float64_npy(path: Path, name: str) -> FloatArray:
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Failed to read {name} from {path}: {error}") from error

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric; got {array.dtype}")

    result = np.ascontiguousarray(array, dtype=np.float64)
    _require_finite_array(name, result)
    return result


def _load_bool_mask_npy(path: Path, name: str) -> BoolArray:
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Failed to read {name} from {path}: {error}") from error

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; got shape {array.shape}"
        )

    if np.issubdtype(array.dtype, np.bool_):
        return np.ascontiguousarray(array, dtype=np.bool_)

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(
            f"{name} must be boolean or numeric binary; got {array.dtype}"
        )
    _require_finite_array(name, array)

    valid = (array == 0) | (array == 1)
    if not np.all(valid):
        invalid_values = np.unique(array[~valid])
        raise ValueError(
            f"{name} must contain only 0 or 1; "
            f"invalid values={invalid_values.tolist()}"
        )

    return np.ascontiguousarray(array.astype(np.bool_, copy=False))


def _resolve_existing_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _require_float64_vector(name: str, values: FloatArray) -> FloatArray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64; got {array.dtype}")
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; got shape {array.shape}"
        )
    return array


def _require_bool_vector(name: str, values: BoolArray) -> BoolArray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must have dtype bool; got {array.dtype}")
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; got shape {array.shape}"
        )
    return array


def _require_finite_array(
    name: str,
    values: NDArray[np.generic],
) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
