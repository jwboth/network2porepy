from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NormalParams:
    """Defines a normal distribution with mean and standard deviation."""

    mean: float
    stddev: float


@dataclass
class DomainConfig:
    """2D domain bounds."""

    xmin: float = -100.0
    ymin: float = -100.0
    xmax: float = 100.0
    ymax: float = 100.0


@dataclass
class FamilyConfig:
    """Configuration for a fracture family."""

    target_num: int
    major_axis_length: NormalParams
    rotation_deg: NormalParams


@dataclass
class ConstraintsConfig:
    """Configuration for constraints during fracture generation."""

    min_distance: float
    min_intersecting_angle_deg_self: float
    min_intersecting_angle_deg_other: float
    min_intersection_distance: float


@dataclass
class PostprocessingConfig:
    extension_threshold: float
    """Distance below which tips are extended to touch other family lines."""
    extension_max_iterations: int
    """Maximum iteration number."""
    branch_proximity_tolerance: float
    """Tolerance for checking if branches are close enough to trim."""
    trim_short_branch_length: float
    """If intersection branch is shorter than this, trim to Y-node."""


@dataclass
class OutputConfig:
    """Configuration for output settings."""

    raw_csv: str = "raw_fractures.csv"
    y_node_csv: str = "y_node_fractures.csv"
    families_csv: str = "families_metadata.csv"


@dataclass
class Config:
    """Configuration for fracture generation."""

    domain: DomainConfig
    subdomain: DomainConfig
    families: list[FamilyConfig]
    constraints: ConstraintsConfig
    postprocessing: PostprocessingConfig
    output: OutputConfig | None = None
    seed: int = 0
    max_iterations: int = 100

    def __post_init__(self):
        """Set defaults for optional fields."""
        if self.output is None:
            self.output = OutputConfig()
