"""Configuration loading from TOML files."""

from __future__ import annotations

from pathlib import Path
import tomllib

from .config import (
    Config,
    DomainConfig,
    FamilyConfig,
    ConstraintsConfig,
    PostprocessingConfig,
    NormalParams,
    OutputConfig,
)


class ConfigLoader:
    """Load configuration from TOML files matching the existing schema."""

    @staticmethod
    def load(toml_path: Path | str) -> Config:
        """Load configuration from TOML file.

        Expected structure:
        ```toml
        [domain]
        xmin, ymin, xmax, ymax

        [subdomain]
        xmin, ymin, xmax, ymax

        [constraints]
        min_distance, min_intersecting_angle_deg_self, etc.

        [sampler]
        num = 2

        [sampler.1]
        target_num = 5
        type = "main"

        [sampler.1.major_axis_length]
        mean = 50.0
        stddev = 10.0

        [sampler.1.rotation_deg]
        mean = 60.0
        stddev = 7.5

        [sampler.2]
        # ... family 2 ...
        ```

        Args:
            toml_path: Path to TOML configuration file

        Returns:
            Parsed Config object

        Raises:
            FileNotFoundError: If TOML file doesn't exist
            ValueError: If TOML structure is invalid
        """
        toml_path = Path(toml_path)

        if not toml_path.exists():
            raise FileNotFoundError(f"Config file not found: {toml_path}")

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        print(f"✓ Loaded configuration from {toml_path}")

        return ConfigLoader._parse_config(data)

    @staticmethod
    def _parse_config(data: dict) -> Config:
        """Parse raw TOML data into Config object."""

        # Parse domain
        domain_data = data.get("domain", {})
        domain = DomainConfig(
            xmin=domain_data["xmin"],
            ymin=domain_data["ymin"],
            xmax=domain_data["xmax"],
            ymax=domain_data["ymax"],
        )

        # Parse subdomain
        subdomain_data = data.get("subdomain", {})
        subdomain = DomainConfig(
            xmin=subdomain_data["xmin"],
            ymin=subdomain_data["ymin"],
            xmax=subdomain_data["xmax"],
            ymax=subdomain_data["ymax"],
        )

        # Parse families from sampler section
        sampler_data = data.get("sampler", {})
        num_families = sampler_data.get("num", 1)

        if num_families < 1:
            raise ValueError("num_families must be >= 1")

        families = []
        for family_idx in range(1, num_families + 1):
            family_key = str(family_idx)

            if family_key not in sampler_data:
                raise ValueError(
                    f"Family {family_idx} defined in sampler.num but not in sampler.{family_idx}"
                )

            family = ConfigLoader._parse_family(sampler_data[family_key])
            families.append(family)

        # Parse constraints
        constraints_data = data.get("constraints", {})
        constraints = ConstraintsConfig(
            min_distance=constraints_data["min_distance"],
            min_intersecting_angle_deg_self=constraints_data[
                "min_intersecting_angle_deg_self"
            ],
            min_intersecting_angle_deg_other=constraints_data[
                "min_intersecting_angle_deg_other"
            ],
            min_intersection_distance=constraints_data["min_intersection_distance"],
        )

        # Parse postprocessing (optional)
        postprocessing_data = data.get("postprocessing", {})
        postprocessing = PostprocessingConfig(
            extension_threshold=postprocessing_data.get("extension_threshold", 5.0),
            extension_max_iterations=postprocessing_data.get(
                "extension_max_iterations", 100
            ),
            branch_proximity_tolerance=postprocessing_data.get(
                "branch_proximity_tolerance", 3.0
            ),
            trim_short_branch_length=postprocessing_data.get(
                "trim_short_branch_length", 2.0
            ),
        )

        # Parse output (optional)
        output_data = data.get("output", {})
        output = OutputConfig(
            raw_csv=output_data.get("raw_csv", "raw_fractures.csv"),
            y_node_csv=output_data.get("y_node_csv", "y_node_fractures.csv"),
            families_csv=output_data.get("families_csv", "families_metadata.csv"),
        )

        # Parse top-level options
        seed = data.get("seed", 0)
        max_iterations = data.get("max_iterations", 20000)

        return Config(
            domain=domain,
            subdomain=subdomain,
            families=families,
            constraints=constraints,
            postprocessing=postprocessing,
            output=output,
            seed=seed,
            max_iterations=max_iterations,
        )

    @staticmethod
    def _parse_family(family_data: dict) -> FamilyConfig:
        """Parse a single family configuration from sampler.i section."""

        # Parse major_axis_length
        major_axis_data = family_data.get("major_axis_length", {})
        major_axis_length = NormalParams(
            mean=major_axis_data.get("mean", 50.0),
            stddev=major_axis_data.get("stddev", 10.0),
        )

        # Parse rotation_deg
        rotation_data = family_data.get("rotation_deg", {})
        rotation_deg = NormalParams(
            mean=rotation_data.get("mean", 0.0),
            stddev=rotation_data.get("stddev", 7.5),
        )

        return FamilyConfig(
            target_num=family_data.get("target_num", 5),
            major_axis_length=major_axis_length,
            rotation_deg=rotation_deg,
            # type=family_data.get("type", "main"),
        )

    @staticmethod
    def save_template(output_path: Path | str) -> None:
        """Save a template configuration file.

        Args:
            output_path: Path where to save the template config
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        template = """# Biconjugate Fracture Network Generator Configuration

# Box domain dimensions (outer bounds)
[domain]
xmin = -500.0
ymin = -3000.0
xmax = 500.0
ymax = -2000.0

# Box subdomain dimensions (fracture generation region)
[subdomain]
xmin = -50.0
ymin = -2550.0
xmax = 50.0
ymax = -2450.0

# Main constraints
[constraints]
min_distance = 5.0
min_intersecting_angle_deg_self = 20.0
min_intersecting_angle_deg_other = 25.0
min_intersection_distance = 5.0

# Fracture properties
[sampler]
num = 2

# Family 1
[sampler.1]
target_num = 5
type = "main"

[sampler.1.major_axis_length]
mean = 50.0
stddev = 10.0

[sampler.1.rotation_deg]
mean = 60.0
stddev = 7.5

# Family 2
[sampler.2]
target_num = 5
type = "main"

[sampler.2.major_axis_length]
mean = 50.0
stddev = 10.0

[sampler.2.rotation_deg]
mean = 30.0
stddev = 7.5

# Post-processing (optional)
[postprocessing]
extension_threshold = 2.5
extension_max_iterations = 100
extension_movement_threshold = 1e-6

branch_shortening_distance = 2.5
branch_proximity_tolerance = 3.0
trim_short_branch_length = 5.0

geometric_tolerance = 1e-8

# Output (optional)
[output]
raw_csv = "raw_fractures.csv"
extended_csv = "extended_fractures.csv"
y_node_csv = "y_node_fractures.csv"
families_csv = "families_metadata.csv"

# Top-level options
seed = 0
max_iterations = 20000
"""

        output_path.write_text(template)
        print(f"✓ Template configuration saved to {output_path}")
