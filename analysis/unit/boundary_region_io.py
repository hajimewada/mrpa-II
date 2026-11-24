"""Input/output helpers for MRPA-II Boundary Region construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from analysis.unit.boundary_region import (
    stable_sort_boundary_candidates,
)


FloatArray = NDArray[np.float64]
Int64Array = NDArray[np.int64]

_REQUIRED_BOUNDARY_KEYS = frozenset(
    {
        "boundary_strength",
        "candidate_frame_indices",
        "candidate_strengths",
    }
)


@dataclass(frozen=True, slots=True)
class BoundaryInput:
    """Validated Boundary candidate input."""

    boundary_strength: FloatArray
    candidate_frame_indices: Int64Array
    candidate_strengths: FloatArray
    metadata: Mapping[str, Any]
    archive_path: Path
    metadata_path: Path

    def __post_init__(self) -> None:
        boundary_strength = _require_float64_vector(
            "boundary_strength", self.boundary_strength
        )
        candidate_frames = _require_int64_vector(
            "candidate_frame_indices", self.candidate_frame_indices
        )
        candidate_strengths = _require_float64_vector(
            "candidate_strengths", self.candidate_strengths
        )

        if boundary_strength.size == 0:
            raise ValueError("boundary_strength cannot be empty")
        _require_finite_array("boundary_strength", boundary_strength)

        if candidate_frames.size != candidate_strengths.size:
            raise ValueError(
                "candidate_frame_indices and candidate_strengths "
                "have different lengths"
            )
        _require_finite_array("candidate_strengths", candidate_strengths)

        if candidate_frames.size:
            if np.any(candidate_frames < 0):
                raise ValueError("candidate_frame_indices contains negative indices")
            if np.any(candidate_frames >= boundary_strength.size):
                raise ValueError("candidate_frame_indices contains out-of-range indices")
            if np.any(np.diff(candidate_frames) <= 0):
                raise ValueError(
                    "candidate_frame_indices must be strictly increasing"
                )

            expected = boundary_strength[candidate_frames]
            if not np.allclose(
                candidate_strengths,
                expected,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                mismatch = np.flatnonzero(
                    ~np.isclose(
                        candidate_strengths,
                        expected,
                        rtol=1.0e-10,
                        atol=1.0e-12,
                    )
                )
                first = int(mismatch[0])
                raise ValueError(
                    "candidate_strengths does not match boundary_strength "
                    f"at candidate {first}, frame={candidate_frames[first]}"
                )

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if not self.archive_path.is_absolute():
            raise ValueError("archive_path must be absolute")
        if not self.metadata_path.is_absolute():
            raise ValueError("metadata_path must be absolute")

    @property
    def frame_count(self) -> int:
        return int(self.boundary_strength.size)

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_frame_indices.size)


def load_boundary_input(
    archive_path: str | Path,
    metadata_path: str | Path,
) -> BoundaryInput:
    """Load, normalize, sort, and validate a Boundary candidate artifact."""

    resolved_archive = _resolve_existing_file(
        archive_path, "Boundary archive"
    )
    resolved_metadata = _resolve_existing_file(
        metadata_path, "Boundary metadata"
    )
    metadata = _load_json_object(resolved_metadata)

    boundary_strength, candidate_frames, candidate_strengths = (
        _load_boundary_archive(resolved_archive)
    )
    sorted_candidates = stable_sort_boundary_candidates(
        candidate_frames,
        candidate_strengths,
        duplicate_frame_policy="error",
    )

    sorted_frames = sorted_candidates.frame_indices
    sorted_strengths = sorted_candidates.strengths
    _validate_metadata_consistency(
        metadata=metadata,
        frame_count=int(boundary_strength.size),
        candidate_count=int(sorted_frames.size),
    )

    return BoundaryInput(
        boundary_strength=boundary_strength,
        candidate_frame_indices=sorted_frames,
        candidate_strengths=sorted_strengths,
        metadata=metadata,
        archive_path=resolved_archive,
        metadata_path=resolved_metadata,
    )


def _load_boundary_archive(
    archive_path: Path,
) -> tuple[FloatArray, Int64Array, FloatArray]:
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            available = frozenset(archive.files)
            missing = _REQUIRED_BOUNDARY_KEYS - available
            if missing:
                raise ValueError(
                    "Boundary archive is missing required keys: "
                    f"{sorted(missing)}; available={sorted(available)}"
                )
            boundary_strength = _coerce_float64_vector(
                "boundary_strength", archive["boundary_strength"]
            )
            candidate_frames = _coerce_int64_vector(
                "candidate_frame_indices", archive["candidate_frame_indices"]
            )
            candidate_strengths = _coerce_float64_vector(
                "candidate_strengths", archive["candidate_strengths"]
            )
    except (OSError, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and "missing required keys" in str(error):
            raise
        raise ValueError(
            f"Failed to read Boundary archive {archive_path}: {error}"
        ) from error

    return boundary_strength, candidate_frames, candidate_strengths


def _validate_metadata_consistency(
    *,
    metadata: Mapping[str, Any],
    frame_count: int,
    candidate_count: int,
) -> None:
    format_version = metadata.get("format_version")
    if format_version is not None:
        if isinstance(format_version, bool) or not isinstance(
            format_version, (int, np.integer)
        ):
            raise TypeError("metadata format_version must be an integer")
        if int(format_version) < 1:
            raise ValueError("metadata format_version must be >= 1")

    metadata_frame_count = _find_first_integer(
        metadata,
        (
            ("shape_summary", "number_of_frames"),
            ("boundary_summary", "number_of_frames"),
            ("source_scale_profile", "number_of_frames"),
            ("number_of_frames",),
            ("frame_count",),
        ),
    )
    if metadata_frame_count is not None and metadata_frame_count != frame_count:
        raise ValueError(
            "Boundary metadata frame count mismatch: "
            f"metadata={metadata_frame_count}, archive={frame_count}"
        )

    metadata_candidate_count = _find_first_integer(
        metadata,
        (
            ("shape_summary", "candidate_count"),
            ("boundary_summary", "candidate_count"),
            ("boundary_summary", "boundary_candidate_count"),
            ("candidate_count",),
            ("boundary_candidate_count",),
        ),
    )
    if (
        metadata_candidate_count is not None
        and metadata_candidate_count != candidate_count
    ):
        raise ValueError(
            "Boundary metadata candidate count mismatch: "
            f"metadata={metadata_candidate_count}, archive={candidate_count}"
        )


def _find_first_integer(
    metadata: Mapping[str, Any],
    paths: tuple[tuple[str, ...], ...],
) -> int | None:
    for path in paths:
        value: Any = metadata
        for component in path:
            if not isinstance(value, Mapping) or component not in value:
                value = None
                break
            value = value[component]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(
                "Metadata field must be an integer: " + ".".join(path)
            )
        return int(value)
    return None


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
        raise TypeError("Boundary metadata root must be a JSON object")
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
