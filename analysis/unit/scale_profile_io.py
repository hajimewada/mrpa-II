"""ScaleProfile input loader for MRPA-II Boundary Region construction.

Append this module content to boundary_region_io.py, or import it as a
separate module during development.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Int64Array = NDArray[np.int64]

_REQUIRED_SCALE_PROFILE_KEYS = frozenset(
    {
        "frequencies",
        "magnitude",
        "scale_mean",
        "scale_std",
        "scale_median",
        "scale_max",
        "scale_rms",
        "frame_total",
        "frame_mean",
        "frame_max",
        "dominant_scale_index",
        "dominant_frequency_hz",
        "frequency_centroid_hz",
        "frequency_dispersion_hz",
    }
)


@dataclass(frozen=True, slots=True)
class ScaleProfileInput:
    """Validated ScaleProfile input for Boundary Region construction."""

    frequencies: FloatArray
    magnitude: FloatArray

    scale_mean: FloatArray
    scale_std: FloatArray
    scale_median: FloatArray
    scale_max: FloatArray
    scale_rms: FloatArray

    frame_total: FloatArray
    frame_mean: FloatArray
    frame_max: FloatArray
    dominant_scale_index: Int64Array
    dominant_frequency_hz: FloatArray
    frequency_centroid_hz: FloatArray
    frequency_dispersion_hz: FloatArray

    metadata: Mapping[str, Any]
    archive_path: Path
    metadata_path: Path

    def __post_init__(self) -> None:
        frequencies = _require_float64_vector(
            "frequencies", self.frequencies
        )
        magnitude = _require_float64_matrix(
            "magnitude", self.magnitude
        )

        if frequencies.size == 0:
            raise ValueError("frequencies cannot be empty")
        if magnitude.shape[0] != frequencies.size:
            raise ValueError(
                "magnitude scale count does not match frequencies: "
                f"{magnitude.shape[0]} != {frequencies.size}"
            )
        if magnitude.shape[1] == 0:
            raise ValueError("magnitude must contain at least one frame")

        _require_finite_array("frequencies", frequencies)
        _require_finite_array("magnitude", magnitude)
        if np.any(frequencies <= 0.0):
            raise ValueError("frequencies must be strictly positive")
        if np.any(magnitude < 0.0):
            raise ValueError("magnitude must be nonnegative")

        scale_count, frame_count = magnitude.shape

        scale_arrays = {
            "scale_mean": self.scale_mean,
            "scale_std": self.scale_std,
            "scale_median": self.scale_median,
            "scale_max": self.scale_max,
            "scale_rms": self.scale_rms,
        }
        for name, values in scale_arrays.items():
            array = _require_float64_vector(name, values)
            if array.shape != (scale_count,):
                raise ValueError(
                    f"{name} must have shape ({scale_count},); "
                    f"got {array.shape}"
                )
            _require_finite_array(name, array)
            if np.any(array < 0.0):
                raise ValueError(f"{name} must be nonnegative")

        frame_float_arrays = {
            "frame_total": self.frame_total,
            "frame_mean": self.frame_mean,
            "frame_max": self.frame_max,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "frequency_centroid_hz": self.frequency_centroid_hz,
            "frequency_dispersion_hz": self.frequency_dispersion_hz,
        }
        for name, values in frame_float_arrays.items():
            array = _require_float64_vector(name, values)
            if array.shape != (frame_count,):
                raise ValueError(
                    f"{name} must have shape ({frame_count},); "
                    f"got {array.shape}"
                )
            _require_finite_array(name, array)
            if np.any(array < 0.0):
                raise ValueError(f"{name} must be nonnegative")

        dominant_indices = _require_int64_vector(
            "dominant_scale_index", self.dominant_scale_index
        )
        if dominant_indices.shape != (frame_count,):
            raise ValueError(
                "dominant_scale_index must have shape "
                f"({frame_count},); got {dominant_indices.shape}"
            )
        if np.any(dominant_indices < 0) or np.any(
            dominant_indices >= scale_count
        ):
            raise ValueError(
                "dominant_scale_index contains out-of-range indices"
            )

        _validate_derived_statistics(
            frequencies=frequencies,
            magnitude=magnitude,
            scale_mean=self.scale_mean,
            scale_std=self.scale_std,
            scale_median=self.scale_median,
            scale_max=self.scale_max,
            scale_rms=self.scale_rms,
            frame_total=self.frame_total,
            frame_mean=self.frame_mean,
            frame_max=self.frame_max,
            dominant_scale_index=dominant_indices,
            dominant_frequency_hz=self.dominant_frequency_hz,
            frequency_centroid_hz=self.frequency_centroid_hz,
            frequency_dispersion_hz=self.frequency_dispersion_hz,
        )

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if not self.archive_path.is_absolute():
            raise ValueError("archive_path must be absolute")
        if not self.metadata_path.is_absolute():
            raise ValueError("metadata_path must be absolute")

    @property
    def scale_count(self) -> int:
        return int(self.magnitude.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.magnitude.shape[1])


def load_scale_profile_input(
    archive_path: str | Path,
    metadata_path: str | Path,
    *,
    expected_frame_count: int | None = None,
) -> ScaleProfileInput:
    """Load and validate a ScaleProfile artifact.

    Parameters
    ----------
    archive_path:
        Path to ``*_scale_profile.npz``.
    metadata_path:
        Path to ``*_scale_profile_metadata.json``.
    expected_frame_count:
        Optional frame count from the Boundary input. When supplied, the
        loader rejects mismatched artifacts.
    """

    resolved_archive = _resolve_existing_file(
        archive_path, "ScaleProfile archive"
    )
    resolved_metadata = _resolve_existing_file(
        metadata_path, "ScaleProfile metadata"
    )
    metadata = _load_json_object(resolved_metadata)
    arrays = _load_scale_profile_archive(resolved_archive)

    scale_count, frame_count = arrays["magnitude"].shape
    if expected_frame_count is not None:
        if expected_frame_count <= 0:
            raise ValueError("expected_frame_count must be > 0")
        if frame_count != expected_frame_count:
            raise ValueError(
                "ScaleProfile frame count mismatch: "
                f"expected={expected_frame_count}, archive={frame_count}"
            )

    _validate_scale_profile_metadata(
        metadata=metadata,
        scale_count=scale_count,
        frame_count=frame_count,
    )

    return ScaleProfileInput(
        frequencies=arrays["frequencies"],
        magnitude=arrays["magnitude"],
        scale_mean=arrays["scale_mean"],
        scale_std=arrays["scale_std"],
        scale_median=arrays["scale_median"],
        scale_max=arrays["scale_max"],
        scale_rms=arrays["scale_rms"],
        frame_total=arrays["frame_total"],
        frame_mean=arrays["frame_mean"],
        frame_max=arrays["frame_max"],
        dominant_scale_index=arrays["dominant_scale_index"],
        dominant_frequency_hz=arrays["dominant_frequency_hz"],
        frequency_centroid_hz=arrays["frequency_centroid_hz"],
        frequency_dispersion_hz=arrays["frequency_dispersion_hz"],
        metadata=metadata,
        archive_path=resolved_archive,
        metadata_path=resolved_metadata,
    )


def _load_scale_profile_archive(
    archive_path: Path,
) -> dict[str, FloatArray | Int64Array]:
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            available = frozenset(archive.files)
            missing = _REQUIRED_SCALE_PROFILE_KEYS - available
            if missing:
                raise ValueError(
                    "ScaleProfile archive is missing required keys: "
                    f"{sorted(missing)}; available={sorted(available)}"
                )

            result: dict[str, FloatArray | Int64Array] = {}
            for name in _REQUIRED_SCALE_PROFILE_KEYS:
                values = archive[name]
                if name == "magnitude":
                    result[name] = _coerce_float64_matrix(name, values)
                elif name == "dominant_scale_index":
                    result[name] = _coerce_int64_vector(name, values)
                else:
                    result[name] = _coerce_float64_vector(name, values)
    except (OSError, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and "missing required keys" in str(error):
            raise
        raise ValueError(
            f"Failed to read ScaleProfile archive {archive_path}: {error}"
        ) from error
    return result


def _validate_scale_profile_metadata(
    *,
    metadata: Mapping[str, Any],
    scale_count: int,
    frame_count: int,
) -> None:
    format_version = metadata.get("format_version")
    if isinstance(format_version, bool) or not isinstance(
        format_version, (int, np.integer)
    ):
        raise TypeError("metadata format_version must be an integer")
    if int(format_version) < 1:
        raise ValueError("metadata format_version must be >= 1")

    shape_summary = metadata.get("shape_summary")
    if not isinstance(shape_summary, Mapping):
        raise ValueError("metadata.shape_summary must be an object")

    _require_metadata_integer(
        shape_summary, "number_of_scales", scale_count
    )
    _require_metadata_integer(
        shape_summary, "number_of_frames", frame_count
    )
    _require_metadata_shape(
        shape_summary, "frequencies_shape", (scale_count,)
    )
    _require_metadata_shape(
        shape_summary, "magnitude_shape", (scale_count, frame_count)
    )
    _require_metadata_shape(
        shape_summary, "scale_statistic_shape", (scale_count,)
    )
    _require_metadata_shape(
        shape_summary, "frame_statistic_shape", (frame_count,)
    )
    _require_metadata_integer(
        shape_summary, "number_of_scale_statistics", 5
    )
    _require_metadata_integer(
        shape_summary, "number_of_frame_statistics", 7
    )

    feature_configuration = metadata.get("feature_configuration")
    if not isinstance(feature_configuration, Mapping):
        raise ValueError("metadata.feature_configuration must be an object")
    if feature_configuration.get("magnitude_transform") != "absolute":
        raise ValueError(
            "ScaleProfile magnitude_transform must be 'absolute'"
        )
    if feature_configuration.get("dominant_scale_tie_policy") != "first-index":
        raise ValueError(
            "dominant_scale_tie_policy must be 'first-index'"
        )
    if feature_configuration.get("standard_deviation_convention") != "population":
        raise ValueError(
            "standard_deviation_convention must be 'population'"
        )

    source_cwt = metadata.get("source_cwt")
    if not isinstance(source_cwt, Mapping):
        raise ValueError("metadata.source_cwt must be an object")
    _require_metadata_integer(source_cwt, "number_of_scales", scale_count)
    _require_metadata_integer(source_cwt, "number_of_frames", frame_count)

    sampling_period = source_cwt.get("sampling_period")
    sampling_frequency = source_cwt.get("sampling_frequency")
    if not isinstance(sampling_period, (int, float)) or isinstance(
        sampling_period, bool
    ):
        raise TypeError("source_cwt.sampling_period must be numeric")
    if not isinstance(sampling_frequency, (int, float)) or isinstance(
        sampling_frequency, bool
    ):
        raise TypeError("source_cwt.sampling_frequency must be numeric")
    if not np.isfinite(sampling_period) or sampling_period <= 0.0:
        raise ValueError("source_cwt.sampling_period must be finite and > 0")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0.0:
        raise ValueError(
            "source_cwt.sampling_frequency must be finite and > 0"
        )
    if not np.isclose(
        float(sampling_period) * float(sampling_frequency),
        1.0,
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        raise ValueError(
            "sampling_period and sampling_frequency are inconsistent"
        )


def _validate_derived_statistics(
    *,
    frequencies: FloatArray,
    magnitude: FloatArray,
    scale_mean: FloatArray,
    scale_std: FloatArray,
    scale_median: FloatArray,
    scale_max: FloatArray,
    scale_rms: FloatArray,
    frame_total: FloatArray,
    frame_mean: FloatArray,
    frame_max: FloatArray,
    dominant_scale_index: Int64Array,
    dominant_frequency_hz: FloatArray,
    frequency_centroid_hz: FloatArray,
    frequency_dispersion_hz: FloatArray,
) -> None:
    rtol = 1.0e-10
    atol = 1.0e-12

    expected = {
        "scale_mean": np.mean(magnitude, axis=1),
        "scale_std": np.std(magnitude, axis=1, ddof=0),
        "scale_median": np.median(magnitude, axis=1),
        "scale_max": np.max(magnitude, axis=1),
        "scale_rms": np.sqrt(np.mean(np.square(magnitude), axis=1)),
        "frame_total": np.sum(magnitude, axis=0),
        "frame_mean": np.mean(magnitude, axis=0),
        "frame_max": np.max(magnitude, axis=0),
        "dominant_scale_index": np.argmax(magnitude, axis=0),
    }
    actual = {
        "scale_mean": scale_mean,
        "scale_std": scale_std,
        "scale_median": scale_median,
        "scale_max": scale_max,
        "scale_rms": scale_rms,
        "frame_total": frame_total,
        "frame_mean": frame_mean,
        "frame_max": frame_max,
        "dominant_scale_index": dominant_scale_index,
    }
    for name, expected_values in expected.items():
        actual_values = actual[name]
        if name == "dominant_scale_index":
            if not np.array_equal(actual_values, expected_values):
                raise ValueError(f"{name} does not match magnitude")
        elif not np.allclose(actual_values, expected_values, rtol=rtol, atol=atol):
            raise ValueError(f"{name} does not match magnitude")

    expected_dominant_frequency = frequencies[dominant_scale_index]
    if not np.allclose(
        dominant_frequency_hz,
        expected_dominant_frequency,
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError("dominant_frequency_hz does not match frequencies")

    totals = np.sum(magnitude, axis=0)
    weighted = frequencies[:, None] * magnitude
    expected_centroid = np.divide(
        np.sum(weighted, axis=0),
        totals,
        out=np.zeros_like(totals),
        where=totals > 0.0,
    )
    if not np.allclose(
        frequency_centroid_hz, expected_centroid, rtol=rtol, atol=atol
    ):
        raise ValueError("frequency_centroid_hz does not match magnitude")

    centered = frequencies[:, None] - expected_centroid[None, :]
    variance = np.divide(
        np.sum(magnitude * np.square(centered), axis=0),
        totals,
        out=np.zeros_like(totals),
        where=totals > 0.0,
    )
    expected_dispersion = np.sqrt(np.maximum(variance, 0.0))
    if not np.allclose(
        frequency_dispersion_hz,
        expected_dispersion,
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError("frequency_dispersion_hz does not match magnitude")


def _require_metadata_integer(
    mapping: Mapping[str, Any], name: str, expected: int
) -> None:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"metadata field {name} must be an integer")
    if int(value) != expected:
        raise ValueError(
            f"metadata field {name} mismatch: {value} != {expected}"
        )


def _require_metadata_shape(
    mapping: Mapping[str, Any], name: str, expected: tuple[int, ...]
) -> None:
    value = mapping.get(name)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer))
        for item in value
    ):
        raise TypeError(f"metadata field {name} must be an integer list")
    actual = tuple(int(item) for item in value)
    if actual != expected:
        raise ValueError(
            f"metadata field {name} mismatch: {actual} != {expected}"
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"Metadata is not valid UTF-8: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Metadata is not valid JSON: {path}: {error}") from error
    except OSError as error:
        raise ValueError(f"Failed to read metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError("ScaleProfile metadata root must be a JSON object")
    return value


def _resolve_existing_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _coerce_float64_vector(name: str, values: NDArray[Any]) -> FloatArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric; got {array.dtype}")
    result = np.ascontiguousarray(array, dtype=np.float64)
    _require_finite_array(name, result)
    return result


def _coerce_float64_matrix(name: str, values: NDArray[Any]) -> FloatArray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric; got {array.dtype}")
    result = np.ascontiguousarray(array, dtype=np.float64)
    _require_finite_array(name, result)
    return result


def _coerce_int64_vector(name: str, values: NDArray[Any]) -> Int64Array:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers; got {array.dtype}")
    return np.ascontiguousarray(array, dtype=np.int64)


def _require_float64_vector(name: str, values: FloatArray) -> FloatArray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64; got {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {array.shape}")
    return array


def _require_float64_matrix(name: str, values: FloatArray) -> FloatArray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64; got {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got {array.shape}")
    return array


def _require_int64_vector(name: str, values: Int64Array) -> Int64Array:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.int64):
        raise TypeError(f"{name} must have dtype int64; got {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {array.shape}")
    return array


def _require_finite_array(name: str, values: NDArray[Any]) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
