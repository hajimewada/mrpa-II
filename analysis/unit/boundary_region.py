"""Data models for MRPA-II Milestone 9.1 Boundary Regions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from math import isfinite
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray


FORMAT_VERSION: Final[int] = 1
ARTIFACT_TYPE: Final[str] = "mrpa_boundary_regions"

RepresentativePolicy = Literal["strongest_boundary_evidence"]
DuplicateFramePolicy = Literal["error"]
TiePolicy = Literal["first-index"]
NormalizationPolicy = Literal["median_mad"]
ChangeOperator = Literal["absolute_first_difference"]
ZeroMadPolicy = Literal["zero_activity"]
IntermediateProminencePolicy = Literal["prevent_merge"]
ProminenceSource = Literal[
    "nucleus_strength",
    "scale_profile_aggregate",
    "interpolated_f0_contrast",
    "unavailable",
]


FloatArray = NDArray[np.float64]
Int64Array = NDArray[np.int64]
Int8Array = NDArray[np.int8]
BoolArray = NDArray[np.bool_]


class UnitBoundaryKind(IntEnum):
    """Codes stored in unit_boundary_kinds."""

    UTTERANCE_START = 0
    DETECTED_BOUNDARY_REGION = 1
    UTTERANCE_END = 2


@dataclass(frozen=True, slots=True)
class SeedConfig:
    """Configuration for raw Boundary candidate clustering."""

    link_distance_frames: int
    minimum_strength: float = 0.0
    duplicate_frame_policy: DuplicateFramePolicy = "error"
    tie_policy: TiePolicy = "first-index"

    def __post_init__(self) -> None:
        if self.link_distance_frames < 0:
            raise ValueError(
                "link_distance_frames must be >= 0"
            )
        _require_finite(
            "minimum_strength",
            self.minimum_strength,
        )

        if self.duplicate_frame_policy != "error":
            raise ValueError(
                "Unsupported duplicate_frame_policy: "
                f"{self.duplicate_frame_policy!r}"
            )

        if self.tie_policy != "first-index":
            raise ValueError(
                f"Unsupported tie_policy: {self.tie_policy!r}"
            )


@dataclass(frozen=True, slots=True)
class SupportConfig:
    """Configuration for temporal Boundary support estimation."""

    relative_level: float
    absolute_floor: float
    maximum_radius_frames: int
    interval_convention: str = "half-open"

    def __post_init__(self) -> None:
        _require_probability(
            "relative_level",
            self.relative_level,
        )
        _require_finite(
            "absolute_floor",
            self.absolute_floor,
        )

        if self.maximum_radius_frames < 0:
            raise ValueError(
                "maximum_radius_frames must be >= 0"
            )

        if self.interval_convention != "half-open":
            raise ValueError(
                "Only half-open intervals are supported"
            )


@dataclass(frozen=True, slots=True)
class ScaleChangeConfig:
    """Configuration for cross-scale Boundary evidence."""

    activity_threshold: float
    minimum_active_scale_count: int
    minimum_contiguous_scale_run: int
    normalization: NormalizationPolicy = "median_mad"
    change_operator: ChangeOperator = (
        "absolute_first_difference"
    )
    zero_mad_policy: ZeroMadPolicy = "zero_activity"

    def __post_init__(self) -> None:
        _require_finite(
            "activity_threshold",
            self.activity_threshold,
        )

        if self.activity_threshold < 0.0:
            raise ValueError(
                "activity_threshold must be >= 0"
            )

        if self.minimum_active_scale_count < 0:
            raise ValueError(
                "minimum_active_scale_count must be >= 0"
            )

        if self.minimum_contiguous_scale_run < 0:
            raise ValueError(
                "minimum_contiguous_scale_run must be >= 0"
            )

        if self.normalization != "median_mad":
            raise ValueError(
                "Unsupported normalization: "
                f"{self.normalization!r}"
            )

        if self.change_operator != (
            "absolute_first_difference"
        ):
            raise ValueError(
                "Unsupported change_operator: "
                f"{self.change_operator!r}"
            )

        if self.zero_mad_policy != "zero_activity":
            raise ValueError(
                "Unsupported zero_mad_policy: "
                f"{self.zero_mad_policy!r}"
            )


@dataclass(frozen=True, slots=True)
class MergeConfig:
    """Configuration for merging Initial Boundary Regions."""

    gap_frames: int
    overlap_ratio: float
    scale_similarity_threshold: float
    intermediate_prominence_policy: (
        IntermediateProminencePolicy
    ) = "prevent_merge"

    def __post_init__(self) -> None:
        if self.gap_frames < 0:
            raise ValueError("gap_frames must be >= 0")

        _require_probability(
            "overlap_ratio",
            self.overlap_ratio,
        )
        _require_probability(
            "scale_similarity_threshold",
            self.scale_similarity_threshold,
        )

        if self.intermediate_prominence_policy != (
            "prevent_merge"
        ):
            raise ValueError(
                "Unsupported intermediate_prominence_policy: "
                f"{self.intermediate_prominence_policy!r}"
            )


@dataclass(frozen=True, slots=True)
class ProminenceConfig:
    """Configuration for left/right prominence search."""

    search_radius_frames: int
    source: ProminenceSource = "unavailable"

    def __post_init__(self) -> None:
        if self.search_radius_frames < 0:
            raise ValueError(
                "search_radius_frames must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class ConfidenceWeights:
    """Weights for composite Boundary Region confidence."""

    strength: float
    temporal: float
    cross_scale: float
    context: float
    reliability: float

    def __post_init__(self) -> None:
        values = (
            self.strength,
            self.temporal,
            self.cross_scale,
            self.context,
            self.reliability,
        )

        for name, value in zip(
            (
                "strength",
                "temporal",
                "cross_scale",
                "context",
                "reliability",
            ),
            values,
            strict=True,
        ):
            _require_probability(name, value)

        total = sum(values)

        if not np.isclose(
            total,
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Confidence weights must sum to 1.0; "
                f"got {total:.17g}"
            )


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    """Boundary Region acceptance and confidence settings."""

    weights: ConfidenceWeights
    acceptance_threshold: float

    def __post_init__(self) -> None:
        _require_probability(
            "acceptance_threshold",
            self.acceptance_threshold,
        )


@dataclass(frozen=True, slots=True)
class UnitBridgeConfig:
    """Configuration for the Milestone 9.2 bridge."""

    include_utterance_edges: bool = True
    include_only_accepted_regions: bool = True
    representative_policy: RepresentativePolicy = (
        "strongest_boundary_evidence"
    )

    def __post_init__(self) -> None:
        if self.representative_policy != (
            "strongest_boundary_evidence"
        ):
            raise ValueError(
                "Unsupported representative_policy: "
                f"{self.representative_policy!r}"
            )


@dataclass(frozen=True, slots=True)
class BoundaryRegionConfig:
    """Complete configuration for Boundary Region construction."""

    seed: SeedConfig
    support: SupportConfig
    scale_change: ScaleChangeConfig
    merge: MergeConfig
    prominence: ProminenceConfig
    confidence: ConfidenceConfig
    unit_bridge: UnitBridgeConfig = field(
        default_factory=UnitBridgeConfig
    )


@dataclass(frozen=True, slots=True)
class BoundarySeedCluster:
    """A deterministic cluster of raw Boundary candidates."""

    seed_index: int
    member_frame_indices: Int64Array
    member_strengths: FloatArray

    first_frame: int
    last_frame: int

    peak_frame: int
    peak_strength: float

    def __post_init__(self) -> None:
        if self.seed_index < 0:
            raise ValueError("seed_index must be >= 0")

        frames = _require_int64_vector(
            "member_frame_indices",
            self.member_frame_indices,
        )
        strengths = _require_float64_vector(
            "member_strengths",
            self.member_strengths,
        )

        if frames.size == 0:
            raise ValueError(
                "BoundarySeedCluster cannot be empty"
            )

        if frames.size != strengths.size:
            raise ValueError(
                "member frame and strength lengths differ"
            )

        _require_strictly_increasing(
            "member_frame_indices",
            frames,
        )
        _require_finite_array(
            "member_strengths",
            strengths,
        )

        if self.first_frame != int(frames[0]):
            raise ValueError(
                "first_frame does not match first member"
            )

        if self.last_frame != int(frames[-1]):
            raise ValueError(
                "last_frame does not match last member"
            )

        matching = np.flatnonzero(
            frames == self.peak_frame
        )

        if matching.size != 1:
            raise ValueError(
                "peak_frame must be exactly one member"
            )

        actual_strength = float(
            strengths[int(matching[0])]
        )

        if actual_strength != self.peak_strength:
            raise ValueError(
                "peak_strength does not match member strength"
            )

        maximum = float(np.max(strengths))

        if self.peak_strength != maximum:
            raise ValueError(
                "peak_strength must be the maximum "
                "member strength"
            )

        expected_peak = int(
            frames[
                int(
                    np.flatnonzero(
                        strengths == maximum
                    )[0]
                )
            ]
        )

        if self.peak_frame != expected_peak:
            raise ValueError(
                "Tie policy requires the first-index peak"
            )

    @property
    def member_count(self) -> int:
        return int(self.member_frame_indices.size)


@dataclass(frozen=True, slots=True)
class InitialBoundaryRegion:
    """A pre-merge region generated from one seed cluster."""

    initial_region_index: int
    seed_index: int

    support_start_frame: int
    support_end_frame: int

    representative_frame: int
    representative_strength: float

    plateau_start_frame: int
    plateau_end_frame: int
    plateau_center_frame: float

    def __post_init__(self) -> None:
        if self.initial_region_index < 0:
            raise ValueError(
                "initial_region_index must be >= 0"
            )

        if self.seed_index < 0:
            raise ValueError("seed_index must be >= 0")

        _require_half_open_interval(
            self.support_start_frame,
            self.support_end_frame,
        )

        if not (
            self.support_start_frame
            <= self.representative_frame
            < self.support_end_frame
        ):
            raise ValueError(
                "representative_frame is outside support"
            )

        _require_finite(
            "representative_strength",
            self.representative_strength,
        )

        _require_half_open_interval(
            self.plateau_start_frame,
            self.plateau_end_frame,
        )

        if not (
            self.support_start_frame
            <= self.plateau_start_frame
            < self.plateau_end_frame
            <= self.support_end_frame
        ):
            raise ValueError(
                "plateau must be contained in support"
            )

        _require_finite(
            "plateau_center_frame",
            self.plateau_center_frame,
        )

    @property
    def duration_frames(self) -> int:
        return (
            self.support_end_frame
            - self.support_start_frame
        )


@dataclass(frozen=True, slots=True)
class OptionalFrameMeasurement:
    """Optional frame-aligned measurement without NaN."""

    frame: int
    value: float
    available: bool

    def __post_init__(self) -> None:
        if self.available:
            if self.frame < 0:
                raise ValueError(
                    "Available frame must be >= 0"
                )
            _require_finite("value", self.value)
        else:
            if self.frame != -1:
                raise ValueError(
                    "Unavailable frame must be -1"
                )
            if self.value != 0.0:
                raise ValueError(
                    "Unavailable value must be 0.0"
                )


@dataclass(frozen=True, slots=True)
class OptionalFloatMeasurement:
    """Optional scalar measurement without NaN."""

    value: float
    available: bool

    def __post_init__(self) -> None:
        if self.available:
            _require_finite("value", self.value)
        elif self.value != 0.0:
            raise ValueError(
                "Unavailable value must be 0.0"
            )


@dataclass(frozen=True, slots=True)
class RegionScaleMetrics:
    """Cross-scale Boundary evidence for one final Region."""

    active_scale_count: int
    minimum_active_scale_index: int
    maximum_active_scale_index: int
    scale_span: int
    longest_contiguous_scale_run: int
    scale_change_occupancy: float
    peak_scale_support_count: int

    def __post_init__(self) -> None:
        if self.active_scale_count < 0:
            raise ValueError(
                "active_scale_count must be >= 0"
            )

        if self.longest_contiguous_scale_run < 0:
            raise ValueError(
                "longest_contiguous_scale_run must be >= 0"
            )

        if self.peak_scale_support_count < 0:
            raise ValueError(
                "peak_scale_support_count must be >= 0"
            )

        _require_probability(
            "scale_change_occupancy",
            self.scale_change_occupancy,
        )

        if self.active_scale_count == 0:
            if self.minimum_active_scale_index != -1:
                raise ValueError(
                    "No active scale requires minimum index -1"
                )
            if self.maximum_active_scale_index != -1:
                raise ValueError(
                    "No active scale requires maximum index -1"
                )
            if self.scale_span != 0:
                raise ValueError(
                    "No active scale requires scale_span 0"
                )
            if self.longest_contiguous_scale_run != 0:
                raise ValueError(
                    "No active scale requires run length 0"
                )
            return

        if self.minimum_active_scale_index < 0:
            raise ValueError(
                "minimum_active_scale_index must be >= 0"
            )

        if (
            self.maximum_active_scale_index
            < self.minimum_active_scale_index
        ):
            raise ValueError(
                "maximum active scale precedes minimum"
            )

        expected_span = (
            self.maximum_active_scale_index
            - self.minimum_active_scale_index
            + 1
        )

        if self.scale_span != expected_span:
            raise ValueError(
                "scale_span does not match active indices"
            )

        if (
            self.longest_contiguous_scale_run
            > self.scale_span
        ):
            raise ValueError(
                "contiguous run cannot exceed scale span"
            )


@dataclass(frozen=True, slots=True)
class RegionReliability:
    """Voicing and interpolation diagnostics."""

    observed_voicing_ratio: float
    interpolation_ratio: float
    longest_interpolated_run: int
    available: bool

    def __post_init__(self) -> None:
        if self.available:
            _require_probability(
                "observed_voicing_ratio",
                self.observed_voicing_ratio,
            )
            _require_probability(
                "interpolation_ratio",
                self.interpolation_ratio,
            )
            if self.longest_interpolated_run < 0:
                raise ValueError(
                    "longest_interpolated_run "
                    "must be >= 0"
                )
        else:
            if self.observed_voicing_ratio != 0.0:
                raise ValueError(
                    "Unavailable voicing ratio must be 0.0"
                )
            if self.interpolation_ratio != 0.0:
                raise ValueError(
                    "Unavailable interpolation ratio "
                    "must be 0.0"
                )
            if self.longest_interpolated_run != 0:
                raise ValueError(
                    "Unavailable interpolated run must be 0"
                )


@dataclass(frozen=True, slots=True)
class RegionConfidence:
    """Confidence components for one Boundary Region."""

    strength: float
    temporal: float
    cross_scale: float
    context: float
    reliability: float
    composite: float
    accepted: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("strength", self.strength),
            ("temporal", self.temporal),
            ("cross_scale", self.cross_scale),
            ("context", self.context),
            ("reliability", self.reliability),
            ("composite", self.composite),
        ):
            _require_probability(name, value)


@dataclass(frozen=True, slots=True)
class BoundaryRegion:
    """Final Boundary Region after merge and recomputation."""

    region_index: int

    support_start_frame: int
    support_end_frame: int
    support_start_time: float
    support_end_time: float

    representative_frame: int
    representative_time: float
    representative_strength: float

    weighted_center_frame: float
    support_midpoint_frame: float
    plateau_center_frame: float

    seed_indices: Int64Array
    raw_candidate_indices: Int64Array

    scale_metrics: RegionScaleMetrics

    representative_frame_total: float
    representative_dominant_scale_index: int
    representative_dominant_frequency_hz: float
    representative_frequency_centroid_hz: float
    representative_frequency_dispersion_hz: float

    local_baseline: float
    evidence_depth: float
    prosodic_valley_depth: OptionalFloatMeasurement

    left_prominence: OptionalFrameMeasurement
    right_prominence: OptionalFrameMeasurement

    reliability: RegionReliability
    confidence: RegionConfidence

    def __post_init__(self) -> None:
        if self.region_index < 0:
            raise ValueError("region_index must be >= 0")

        _require_half_open_interval(
            self.support_start_frame,
            self.support_end_frame,
        )

        if not (
            self.support_start_frame
            <= self.representative_frame
            < self.support_end_frame
        ):
            raise ValueError(
                "representative frame is outside support"
            )

        for name, value in (
            ("support_start_time", self.support_start_time),
            ("support_end_time", self.support_end_time),
            ("representative_time", self.representative_time),
            (
                "representative_strength",
                self.representative_strength,
            ),
            (
                "weighted_center_frame",
                self.weighted_center_frame,
            ),
            (
                "support_midpoint_frame",
                self.support_midpoint_frame,
            ),
            (
                "plateau_center_frame",
                self.plateau_center_frame,
            ),
            (
                "representative_frame_total",
                self.representative_frame_total,
            ),
            (
                "representative_dominant_frequency_hz",
                self.representative_dominant_frequency_hz,
            ),
            (
                "representative_frequency_centroid_hz",
                self.representative_frequency_centroid_hz,
            ),
            (
                "representative_frequency_dispersion_hz",
                self.representative_frequency_dispersion_hz,
            ),
            ("local_baseline", self.local_baseline),
            ("evidence_depth", self.evidence_depth),
        ):
            _require_finite(name, value)

        if self.support_end_time <= self.support_start_time:
            raise ValueError(
                "support_end_time must exceed start time"
            )

        if self.representative_dominant_scale_index < 0:
            raise ValueError(
                "representative dominant scale must be >= 0"
            )

        seeds = _require_int64_vector(
            "seed_indices",
            self.seed_indices,
        )
        candidates = _require_int64_vector(
            "raw_candidate_indices",
            self.raw_candidate_indices,
        )

        if seeds.size == 0:
            raise ValueError(
                "Final BoundaryRegion requires a seed"
            )

        if candidates.size == 0:
            raise ValueError(
                "Final BoundaryRegion requires a "
                "raw candidate"
            )

        _require_strictly_increasing(
            "seed_indices",
            seeds,
        )
        _require_strictly_increasing(
            "raw_candidate_indices",
            candidates,
        )

    @property
    def duration_frames(self) -> int:
        return (
            self.support_end_frame
            - self.support_start_frame
        )

    @property
    def duration_seconds(self) -> float:
        return (
            self.support_end_time
            - self.support_start_time
        )

    @property
    def both_sides_supported(self) -> bool:
        return (
            self.left_prominence.available
            and self.right_prominence.available
        )


@dataclass(frozen=True, slots=True)
class UnitBoundary:
    """One coordinate in the Milestone 9.2 bridge."""

    frame_index: int
    time: float
    kind: UnitBoundaryKind
    region_index: int

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(
                "frame_index must be >= 0"
            )

        _require_finite("time", self.time)

        if self.kind in (
            UnitBoundaryKind.UTTERANCE_START,
            UnitBoundaryKind.UTTERANCE_END,
        ):
            if self.region_index != -1:
                raise ValueError(
                    "Utterance edge must have "
                    "region_index -1"
                )
        elif self.kind is (
            UnitBoundaryKind.DETECTED_BOUNDARY_REGION
        ):
            if self.region_index < 0:
                raise ValueError(
                    "Detected Boundary requires "
                    "a valid region index"
                )


@dataclass(frozen=True, slots=True)
class ScaleProfileEvidence:
    """Cross-scale evidence stored in the final NPZ."""

    frequencies: FloatArray
    magnitude: FloatArray
    magnitude_median: FloatArray
    magnitude_mad: FloatArray
    change_evidence: FloatArray
    change_active: BoolArray

    def __post_init__(self) -> None:
        frequencies = _require_float64_vector(
            "frequencies",
            self.frequencies,
        )
        medians = _require_float64_vector(
            "magnitude_median",
            self.magnitude_median,
        )
        mads = _require_float64_vector(
            "magnitude_mad",
            self.magnitude_mad,
        )

        magnitude = _require_float64_matrix(
            "magnitude",
            self.magnitude,
        )
        changes = _require_float64_matrix(
            "change_evidence",
            self.change_evidence,
        )
        active = _require_bool_matrix(
            "change_active",
            self.change_active,
        )

        scale_count = frequencies.size

        if magnitude.shape[0] != scale_count:
            raise ValueError(
                "magnitude scale count mismatch"
            )

        if medians.shape != (scale_count,):
            raise ValueError(
                "magnitude_median shape mismatch"
            )

        if mads.shape != (scale_count,):
            raise ValueError(
                "magnitude_mad shape mismatch"
            )

        if changes.shape != magnitude.shape:
            raise ValueError(
                "change_evidence shape mismatch"
            )

        if active.shape != magnitude.shape:
            raise ValueError(
                "change_active shape mismatch"
            )

        _require_finite_array(
            "frequencies",
            frequencies,
        )
        _require_finite_array(
            "magnitude",
            magnitude,
        )
        _require_finite_array(
            "magnitude_median",
            medians,
        )
        _require_finite_array(
            "magnitude_mad",
            mads,
        )
        _require_finite_array(
            "change_evidence",
            changes,
        )

        if np.any(magnitude < 0.0):
            raise ValueError(
                "Scale magnitude must be nonnegative"
            )

        if np.any(mads < 0.0):
            raise ValueError(
                "MAD values must be nonnegative"
            )

        if np.any(changes < 0.0):
            raise ValueError(
                "Scale change evidence must be nonnegative"
            )

    @property
    def scale_count(self) -> int:
        return int(self.magnitude.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.magnitude.shape[1])


@dataclass(frozen=True, slots=True)
class BoundaryRegionResult:
    """Complete in-memory representation of the v1 artifact."""

    utterance_id: str
    time: FloatArray
    boundary_strength: FloatArray

    raw_candidate_frame_indices: Int64Array
    raw_candidate_strengths: FloatArray

    scale_profile: ScaleProfileEvidence

    seeds: tuple[BoundarySeedCluster, ...]
    initial_regions: tuple[InitialBoundaryRegion, ...]
    regions: tuple[BoundaryRegion, ...]
    unit_boundaries: tuple[UnitBoundary, ...]

    config: BoundaryRegionConfig

    format_version: int = FORMAT_VERSION
    artifact_type: str = ARTIFACT_TYPE

    def __post_init__(self) -> None:
        if not self.utterance_id:
            raise ValueError("utterance_id cannot be empty")

        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format_version: "
                f"{self.format_version}"
            )

        if self.artifact_type != ARTIFACT_TYPE:
            raise ValueError(
                f"Unsupported artifact_type: "
                f"{self.artifact_type!r}"
            )

        time = _require_float64_vector(
            "time",
            self.time,
        )
        boundary_strength = _require_float64_vector(
            "boundary_strength",
            self.boundary_strength,
        )

        if time.size == 0:
            raise ValueError("time cannot be empty")

        if time.shape != boundary_strength.shape:
            raise ValueError(
                "time and boundary_strength shapes differ"
            )

        _require_finite_array("time", time)
        _require_finite_array(
            "boundary_strength",
            boundary_strength,
        )
        _require_strictly_increasing("time", time)

        if self.scale_profile.frame_count != time.size:
            raise ValueError(
                "ScaleProfile frame count mismatch"
            )

        raw_frames = _require_int64_vector(
            "raw_candidate_frame_indices",
            self.raw_candidate_frame_indices,
        )
        raw_strengths = _require_float64_vector(
            "raw_candidate_strengths",
            self.raw_candidate_strengths,
        )

        if raw_frames.size != raw_strengths.size:
            raise ValueError(
                "Raw candidate array lengths differ"
            )

        if raw_frames.size:
            _require_strictly_increasing(
                "raw_candidate_frame_indices",
                raw_frames,
            )

            if np.any(raw_frames < 0):
                raise ValueError(
                    "Raw candidate frame is negative"
                )

            if np.any(raw_frames >= time.size):
                raise ValueError(
                    "Raw candidate frame is out of range"
                )

        _require_finite_array(
            "raw_candidate_strengths",
            raw_strengths,
        )

        self._validate_seed_sequence()
        self._validate_region_sequence()
        self._validate_unit_boundaries()

    @property
    def frame_count(self) -> int:
        return int(self.time.size)

    @property
    def scale_count(self) -> int:
        return self.scale_profile.scale_count

    @property
    def raw_candidate_count(self) -> int:
        return int(
            self.raw_candidate_frame_indices.size
        )

    @property
    def accepted_region_count(self) -> int:
        return sum(
            region.confidence.accepted
            for region in self.regions
        )

    def _validate_seed_sequence(self) -> None:
        expected = tuple(range(len(self.seeds)))
        actual = tuple(
            seed.seed_index
            for seed in self.seeds
        )

        if actual != expected:
            raise ValueError(
                "Seed indices must be contiguous"
            )

        member_count = sum(
            seed.member_count
            for seed in self.seeds
        )

        if member_count != self.raw_candidate_count:
            raise ValueError(
                "Seed membership does not cover "
                "all raw candidates"
            )

    def _validate_region_sequence(self) -> None:
        expected = tuple(range(len(self.regions)))
        actual = tuple(
            region.region_index
            for region in self.regions
        )

        if actual != expected:
            raise ValueError(
                "Region indices must be contiguous"
            )

        previous_end = 0
        previous_representative = -1

        for region in self.regions:
            if region.support_end_frame > self.frame_count:
                raise ValueError(
                    "Region support exceeds frame count"
                )

            if (
                region.support_start_frame
                < previous_end
            ):
                raise ValueError(
                    "Final Boundary Regions overlap"
                )

            if (
                region.representative_frame
                <= previous_representative
            ):
                raise ValueError(
                    "Representatives must be "
                    "strictly increasing"
                )

            previous_end = region.support_end_frame
            previous_representative = (
                region.representative_frame
            )

    def _validate_unit_boundaries(self) -> None:
        if len(self.unit_boundaries) < 2:
            raise ValueError(
                "At least two Unit Boundaries are required"
            )

        frames = np.asarray(
            [
                boundary.frame_index
                for boundary in self.unit_boundaries
            ],
            dtype=np.int64,
        )

        _require_strictly_increasing(
            "unit_boundary_frame_indices",
            frames,
        )

        first = self.unit_boundaries[0]
        last = self.unit_boundaries[-1]

        if first.frame_index != 0:
            raise ValueError(
                "First Unit Boundary must be frame 0"
            )

        if first.kind is not (
            UnitBoundaryKind.UTTERANCE_START
        ):
            raise ValueError(
                "First Unit Boundary must be "
                "utterance start"
            )

        if last.frame_index != self.frame_count:
            raise ValueError(
                "Last Unit Boundary must equal frame count"
            )

        if last.kind is not UnitBoundaryKind.UTTERANCE_END:
            raise ValueError(
                "Last Unit Boundary must be utterance end"
            )

        for boundary in self.unit_boundaries[1:-1]:
            if boundary.kind is not (
                UnitBoundaryKind.DETECTED_BOUNDARY_REGION
            ):
                raise ValueError(
                    "Interior Unit Boundaries must be "
                    "detected Boundary Regions"
                )

            region = self.regions[
                boundary.region_index
            ]

            if not region.confidence.accepted:
                raise ValueError(
                    "Unit bridge may contain only "
                    "accepted Regions"
                )

            if (
                boundary.frame_index
                != region.representative_frame
            ):
                raise ValueError(
                    "Unit Boundary frame does not match "
                    "Region representative"
                )


def _require_finite(
    name: str,
    value: float,
) -> None:
    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite; got {value!r}"
        )


def _require_probability(
    name: str,
    value: float,
) -> None:
    _require_finite(name, value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be in [0, 1]; got {value}"
        )


def _require_half_open_interval(
    start: int,
    end: int,
) -> None:
    if start < 0:
        raise ValueError(
            "Interval start must be >= 0"
        )

    if end <= start:
        raise ValueError(
            "Half-open interval must satisfy start < end"
        )


def _require_finite_array(
    name: str,
    values: NDArray[np.generic],
) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} contains non-finite values"
        )


def _require_strictly_increasing(
    name: str,
    values: Int64Array | FloatArray,
) -> None:
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ValueError(
            f"{name} must be strictly increasing"
        )


def _require_int64_vector(
    name: str,
    values: Int64Array,
) -> Int64Array:
    array = np.asarray(values)

    if array.dtype != np.dtype(np.int64):
        raise TypeError(
            f"{name} must have dtype int64; "
            f"got {array.dtype}"
        )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; "
            f"got shape {array.shape}"
        )

    return array


def _require_float64_vector(
    name: str,
    values: FloatArray,
) -> FloatArray:
    array = np.asarray(values)

    if array.dtype != np.dtype(np.float64):
        raise TypeError(
            f"{name} must have dtype float64; "
            f"got {array.dtype}"
        )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; "
            f"got shape {array.shape}"
        )

    return array


def _require_float64_matrix(
    name: str,
    values: FloatArray,
) -> FloatArray:
    array = np.asarray(values)

    if array.dtype != np.dtype(np.float64):
        raise TypeError(
            f"{name} must have dtype float64; "
            f"got {array.dtype}"
        )

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional; "
            f"got shape {array.shape}"
        )

    return array


def _require_bool_matrix(
    name: str,
    values: BoolArray,
) -> BoolArray:
    array = np.asarray(values)

    if array.dtype != np.dtype(np.bool_):
        raise TypeError(
            f"{name} must have dtype bool; "
            f"got {array.dtype}"
        )

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional; "
            f"got shape {array.shape}"
        )

    return array

#9.19
@dataclass(frozen=True, slots=True)
class SortedBoundaryCandidates:
    """Boundary candidates sorted by frame using a stable sort.

    ``source_indices`` maps each sorted candidate back to its position
    in the original unsorted input.
    """

    frame_indices: Int64Array
    strengths: FloatArray
    source_indices: Int64Array

    def __post_init__(self) -> None:
        frames = _require_int64_vector(
            "frame_indices",
            self.frame_indices,
        )
        strengths = _require_float64_vector(
            "strengths",
            self.strengths,
        )
        source_indices = _require_int64_vector(
            "source_indices",
            self.source_indices,
        )

        if not (
            frames.size
            == strengths.size
            == source_indices.size
        ):
            raise ValueError(
                "Sorted candidate arrays must have equal lengths"
            )

        _require_finite_array(
            "strengths",
            strengths,
        )

        if frames.size == 0:
            return

        if np.any(frames < 0):
            raise ValueError(
                "frame_indices contains negative values"
            )

        if np.any(source_indices < 0):
            raise ValueError(
                "source_indices contains negative values"
            )

        if np.any(np.diff(frames) < 0):
            raise ValueError(
                "frame_indices must be nondecreasing"
            )

        if np.unique(source_indices).size != source_indices.size:
            raise ValueError(
                "source_indices must be unique"
            )

        expected_source_indices = np.arange(
            source_indices.size,
            dtype=np.int64,
        )

        if not np.array_equal(
            np.sort(source_indices),
            expected_source_indices,
        ):
            raise ValueError(
                "source_indices must be a permutation of "
                "the original candidate positions"
            )

    @property
    def candidate_count(self) -> int:
        return int(self.frame_indices.size)

    @property
    def has_duplicate_frames(self) -> bool:
        return bool(
            self.frame_indices.size > 1
            and np.any(
                np.diff(self.frame_indices) == 0
            )
        )

    @property
    def duplicate_frames(self) -> Int64Array:
        if self.frame_indices.size < 2:
            return np.empty(
                0,
                dtype=np.int64,
            )

        duplicate_positions = np.flatnonzero(
            np.diff(self.frame_indices) == 0
        )

        return np.unique(
            self.frame_indices[
                duplicate_positions + 1
            ]
        ).astype(
            np.int64,
            copy=False,
        )


def stable_sort_boundary_candidates(
    frame_indices: Int64Array,
    strengths: FloatArray,
    *,
    duplicate_frame_policy: DuplicateFramePolicy = "error",
) -> SortedBoundaryCandidates:
    """Stably sort Boundary candidates by frame index.

    Equal frame indices preserve their original relative order because
    NumPy's stable sorting algorithm is used. Under the Milestone 9.1
    version-1 contract, duplicate frame indices are rejected after
    sorting.

    Parameters
    ----------
    frame_indices:
        One-dimensional integer candidate-frame array.
    strengths:
        One-dimensional numeric candidate-strength array.
    duplicate_frame_policy:
        Version 1 supports only ``"error"``.

    Returns
    -------
    SortedBoundaryCandidates
        Contiguous ``int64`` and ``float64`` arrays plus source indices.

    Raises
    ------
    TypeError
        If frame indices are non-integral or strengths are non-numeric.
    ValueError
        If shapes differ, values are invalid, or duplicates are found.
    """

    if duplicate_frame_policy != "error":
        raise ValueError(
            "Unsupported duplicate_frame_policy: "
            f"{duplicate_frame_policy!r}"
        )

    raw_frames = np.asarray(frame_indices)
    raw_strengths = np.asarray(strengths)

    if raw_frames.ndim != 1:
        raise ValueError(
            "frame_indices must be one-dimensional; "
            f"got shape {raw_frames.shape}"
        )

    if raw_strengths.ndim != 1:
        raise ValueError(
            "strengths must be one-dimensional; "
            f"got shape {raw_strengths.shape}"
        )

    if raw_frames.size != raw_strengths.size:
        raise ValueError(
            "frame_indices and strengths must have equal "
            f"lengths: {raw_frames.size} != "
            f"{raw_strengths.size}"
        )

    if not np.issubdtype(
        raw_frames.dtype,
        np.integer,
    ):
        raise TypeError(
            "frame_indices must contain integers; "
            f"got dtype {raw_frames.dtype}"
        )

    if not np.issubdtype(
        raw_strengths.dtype,
        np.number,
    ):
        raise TypeError(
            "strengths must be numeric; "
            f"got dtype {raw_strengths.dtype}"
        )

    frames = np.ascontiguousarray(
        raw_frames,
        dtype=np.int64,
    )
    candidate_strengths = np.ascontiguousarray(
        raw_strengths,
        dtype=np.float64,
    )

    _require_finite_array(
        "strengths",
        candidate_strengths,
    )

    if np.any(frames < 0):
        invalid_frames = np.unique(
            frames[frames < 0]
        )

        raise ValueError(
            "frame_indices contains negative values: "
            f"{invalid_frames.tolist()}"
        )

    source_indices = np.arange(
        frames.size,
        dtype=np.int64,
    )

    if frames.size == 0:
        return SortedBoundaryCandidates(
            frame_indices=frames,
            strengths=candidate_strengths,
            source_indices=source_indices,
        )

    order = np.argsort(
        frames,
        kind="stable",
    )

    result = SortedBoundaryCandidates(
        frame_indices=np.ascontiguousarray(
            frames[order],
            dtype=np.int64,
        ),
        strengths=np.ascontiguousarray(
            candidate_strengths[order],
            dtype=np.float64,
        ),
        source_indices=np.ascontiguousarray(
            source_indices[order],
            dtype=np.int64,
        ),
    )

    if result.has_duplicate_frames:
        raise ValueError(
            "Duplicate Boundary candidate frames are not "
            "allowed under duplicate_frame_policy='error': "
            f"{result.duplicate_frames.tolist()}"
        )

    return result


#9.1.10
def cluster_boundary_seeds(
    candidates: SortedBoundaryCandidates,
    *,
    link_distance_frames: int,
) -> tuple[BoundarySeedCluster, ...]:
    """Group sorted Boundary candidates into transitive seed clusters.

    Consecutive candidates belong to the same cluster when their frame
    distance is less than or equal to ``link_distance_frames``.

    Clustering is transitive. For example, frames ``[10, 12, 14]`` form
    one cluster when ``link_distance_frames == 2``, even though the
    distance between the first and last frames is 4.

    Parameters
    ----------
    candidates:
        Candidates sorted by ``stable_sort_boundary_candidates``.
        Duplicate frames must already have been rejected.
    link_distance_frames:
        Maximum distance between consecutive candidates in one cluster.

    Returns
    -------
    tuple[BoundarySeedCluster, ...]
        Seed clusters ordered by frame. Seed indices are contiguous from
        zero.

    Raises
    ------
    ValueError
        If the link distance is negative, the candidate sequence is not
        strictly increasing, or candidate arrays are inconsistent.
    """

    if link_distance_frames < 0:
        raise ValueError(
            "link_distance_frames must be >= 0"
        )

    frames = candidates.frame_indices
    strengths = candidates.strengths

    if frames.size != strengths.size:
        raise ValueError(
            "Candidate frame and strength lengths differ"
        )

    if frames.size == 0:
        return ()

    if np.any(np.diff(frames) <= 0):
        raise ValueError(
            "Seed clustering requires strictly increasing "
            "candidate frames"
        )

    cluster_starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(
                np.diff(frames) > link_distance_frames
            ).astype(np.int64)
            + 1,
        )
    )

    cluster_ends = np.concatenate(
        (
            cluster_starts[1:],
            np.asarray(
                [frames.size],
                dtype=np.int64,
            ),
        )
    )

    clusters: list[BoundarySeedCluster] = []

    for seed_index, (start, end) in enumerate(
        zip(
            cluster_starts,
            cluster_ends,
            strict=True,
        )
    ):
        start_index = int(start)
        end_index = int(end)

        member_frames = np.ascontiguousarray(
            frames[start_index:end_index],
            dtype=np.int64,
        )
        member_strengths = np.ascontiguousarray(
            strengths[start_index:end_index],
            dtype=np.float64,
        )

        peak_offset = _select_first_maximum_index(
            member_strengths
        )
        peak_frame = int(
            member_frames[peak_offset]
        )
        peak_strength = float(
            member_strengths[peak_offset]
        )

        clusters.append(
            BoundarySeedCluster(
                seed_index=seed_index,
                member_frame_indices=member_frames,
                member_strengths=member_strengths,
                first_frame=int(member_frames[0]),
                last_frame=int(member_frames[-1]),
                peak_frame=peak_frame,
                peak_strength=peak_strength,
            )
        )

    result = tuple(clusters)

    _validate_boundary_seed_clusters(
        result,
        candidates=candidates,
        link_distance_frames=link_distance_frames,
    )

    return result


def _select_first_maximum_index(
    values: FloatArray,
) -> int:
    """Return the first index holding the maximum finite value."""

    array = _require_float64_vector(
        "values",
        values,
    )

    if array.size == 0:
        raise ValueError(
            "Cannot select a maximum from an empty array"
        )

    _require_finite_array(
        "values",
        array,
    )

    maximum = float(np.max(array))
    matches = np.flatnonzero(
        array == maximum
    )

    if matches.size == 0:
        raise RuntimeError(
            "Maximum value was not found in its source array"
        )

    return int(matches[0])


def _validate_boundary_seed_clusters(
    clusters: tuple[BoundarySeedCluster, ...],
    *,
    candidates: SortedBoundaryCandidates,
    link_distance_frames: int,
) -> None:
    """Validate coverage, ordering, and cluster separation."""

    if not clusters:
        if candidates.candidate_count != 0:
            raise ValueError(
                "Non-empty candidates produced no clusters"
            )
        return

    expected_seed_indices = tuple(
        range(len(clusters))
    )
    actual_seed_indices = tuple(
        cluster.seed_index
        for cluster in clusters
    )

    if actual_seed_indices != expected_seed_indices:
        raise