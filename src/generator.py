"""Self-contained fracture generator for 2D biconjugate fracture networks.

This module provides a FractureGenerator class that encapsulates all fracture
generation, processing, visualization, and I/O operations for 2D conjugate
fracture networks.

Example usage:
    from ncp_mechanics.geometry.dim_2d.fracture_generator import FractureGenerator, FractureConfig

    # Configure fracture generation
    config = FractureConfig(
        seed=42,
        num_fractures=[3, 3],
        length=[50.0, 50.0],
        dip=[30.0, 60.0],
    )

    # Generate fractures
    generator = FractureGenerator(config)
    raw_network, extended_network, y_node_network = generator.generate()

    # Save to disk
    generator.save("output_folder")

    # Visualize
    generator.visualize_all("output_folder")

    # Convert to PorePy LineFractures
    fractures = generator.to_line_fractures()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from .network import FractureNetwork
from .config import Config, ConstraintsConfig, PostprocessingConfig
from .distances import (
    line_distance,
    segment_intersection_point,
    closest_extension_point,
)

StageIdentifier = Literal["raw", "extended", "y_node_processed"]


@dataclass
class ConstraintChecker:
    """Encapsulates constraint evaluation logic (mirrors 3D ConstraintsMatrix)."""

    constraints: ConstraintsConfig

    @staticmethod
    def _angle_between_lines(line1: np.ndarray, line2: np.ndarray) -> float:
        """Compute acute angle between two line segments in degrees.

        Returns angle in range [0, 90] degrees (acute angle only).

        Args:
            line1: Shape (2, 2), points [[x0, y0], [x1, y1]]
            line2: Shape (2, 2), points [[x0, y0], [x1, y1]]

        Returns:
            Angle in degrees [0, 90]
        """
        # Direction vectors
        dir1 = line1[1] - line1[0]
        dir2 = line2[1] - line2[0]

        # Normalize
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = dir2 / np.linalg.norm(dir2)

        # Compute angle from dot product
        cos_angle = np.abs(np.dot(dir1, dir2))  # abs for acute angle
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical safety
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def check_distance_constraint(
        self,
        line: np.ndarray,
        existing_lines: list[np.ndarray],
    ) -> tuple[bool, str]:
        """Check minimum distance constraint.

        Returns (passes_constraint, violation_reason or "")
        """
        if not existing_lines:
            return True, ""

        dists = [line_distance(line, existing) for existing in existing_lines]
        min_dist = min(dists)

        if min_dist < self.constraints.min_distance:
            return (
                False,
                f"distance_{min_dist:.2f}_below_{self.constraints.min_distance}",
            )

        return True, ""

    def check_no_self_intersection(
        self,
        line: np.ndarray,
        existing_lines: list[np.ndarray],
    ) -> tuple[bool, str]:
        """Check that line doesn't hard-intersect with existing lines in same family.

        Returns (passes_constraint, violation_reason or "")
        """
        for existing_line in existing_lines:
            if FractureNetwork._check_intersection(line, existing_line):
                return False, "hard_intersection"

        return True, ""

    def check_intersection_angle_constraint(
        self, line: np.ndarray, existing_lines: list[np.ndarray], min_angle_deg: float
    ) -> tuple[bool, str]:
        """Check intersection angle constraint for lines that DO intersect.

        Only enforced if lines actually intersect.
        Returns (passes_constraint, violation_reason or "")
        """
        for existing_line in existing_lines:
            # Check if lines intersect
            intersection_pt = segment_intersection_point(line, existing_line, tol=1e-2)

            if intersection_pt is not None:  # Lines DO intersect
                angle_deg = self._angle_between_lines(line, existing_line)

                if angle_deg < min_angle_deg:
                    return (
                        False,
                        f"intersection_angle_{angle_deg:.1f}_deg_below_{min_angle_deg}",
                    )

        return True, ""

    def check_intersection_distance_constraint(
        self,
        line: np.ndarray,
        existing_lines: list[np.ndarray],
        min_intersection_distance: float,
    ) -> tuple[bool, str]:
        """Check distance from interection to end points for lines that DO intersect.

        Only enforced if lines actually intersect.
        Returns (passes_constraint, violation_reason or "")
        """
        for existing_line in existing_lines:
            # Check if lines intersect
            intersection_pt = segment_intersection_point(line, existing_line, tol=1e-2)

            if intersection_pt is not None:  # Lines DO intersect
                intersection_distance = min(
                    [np.linalg.norm(intersection_pt - pt) for pt in existing_line]
                )
                if intersection_distance < min_intersection_distance:
                    return (
                        False,
                        f"intersection_distance_{intersection_distance:.1f}_below_{min_intersection_distance}",
                    )

        return True, ""

    def evaluate_self_family(
        self,
        line: np.ndarray,
        family_lines: list[np.ndarray],
    ) -> tuple[bool, str]:
        """Evaluate self-family constraints.

        Returns (accepted, reason)
        """
        # Check 1: No hard intersections
        passed, reason = self.check_no_self_intersection(line, family_lines)
        if not passed:
            return False, reason

        # Check 2: Minimum distance
        passed, reason = self.check_distance_constraint(line, family_lines)
        if not passed:
            return False, reason

        # Check 3: Intersection angle (if intersecting)
        passed, reason = self.check_intersection_angle_constraint(
            line, family_lines, self.constraints.min_intersecting_angle_deg_self
        )
        if not passed:
            return False, reason

        # Check 4: Intersection distance (if intersecting)
        passed, reason = self.check_intersection_distance_constraint(
            line, family_lines, self.constraints.min_intersection_distance
        )
        if not passed:
            return False, reason

        return True, "accepted"

    def evaluate_cross_family(
        self,
        line: np.ndarray,
        other_family_lines: list[np.ndarray],
    ) -> tuple[bool, str]:
        """Evaluate cross-family constraints.

        Returns (accepted, reason)
        """
        if not other_family_lines:
            return True, ""

        # Check 1: Intersection angle (if intersecting)
        passed, reason = self.check_intersection_angle_constraint(
            line, other_family_lines, self.constraints.min_intersecting_angle_deg_other
        )
        if not passed:
            return False, f"cross_family_{reason}"

        # Check 2: Intersection distance (if intersecting)
        passed, reason = self.check_intersection_distance_constraint(
            line, other_family_lines, self.constraints.min_intersection_distance
        )
        if not passed:
            return False, reason

        return True, ""


# @dataclass
# class ConstraintChecker:
#     """Encapsulates constraint evaluation logic (mirrors 3D ConstraintsMatrix)."""

#     constraints: ConstraintsConfig

#     def check_distance_constraint(
#         self, line: np.ndarray, existing_lines: list[np.ndarray]
#     ) -> bool:
#         """Check if line respects distance constraints vs existing lines."""
#         if not existing_lines:
#             return True
#         dists = [line_distance(line, existing) for existing in existing_lines]
#         return all(self.min_dist <= d <= self.max_dist for d in dists)

#     def check_intersection_constraint(
#         self, line: np.ndarray, existing_lines: list[np.ndarray]
#     ) -> bool:
#         """Check if line doesn't intersect with existing lines in same family."""
#         # TODO replace with check using min_intersecting_angle_deg_self
#         return not any(
#             FractureNetwork._check_intersection(line, existing)
#             for existing in existing_lines
#         )

#     def evaluate(
#         self, line: np.ndarray, family_lines: list[np.ndarray]
#     ) -> tuple[bool, str]:
#         """Evaluate all constraints. Returns (accepted, reason)."""
#         if not self.check_intersection_constraint(line, family_lines):
#             return False, "self_intersection"
#         if not self.check_distance_constraint(line, family_lines):
#             return False, "distance_violation"
#         return True, "accepted"


@dataclass
class ConstraintMatrix:
    """Encapsulates self-family and cross-family constraint evaluation."""

    checker: ConstraintChecker
    num_families: int

    def evaluate(
        self,
        line: np.ndarray,
        family_idx: int,
        family_lines_all: list[list[np.ndarray]],
    ) -> tuple[bool, str]:
        """Evaluate line against all constraint types.

        Checks:
        1. Self-family constraints (same family)
        2. Cross-family constraints (other families)

        Args:
            line: Candidate line segment
            family_idx: Family index for this line
            family_lines_all: All lines organized by family

        Returns:
            (accepted, reason)
        """
        # Self-family constraints
        self_passed, self_reason = self.checker.evaluate_self_family(
            line, family_lines_all[family_idx]
        )
        if not self_passed:
            return False, self_reason

        # Cross-family constraints
        for other_family_idx in range(self.num_families):
            if other_family_idx == family_idx:
                continue  # Skip self-family (already checked)

            cross_passed, cross_reason = self.checker.evaluate_cross_family(
                line, family_lines_all[other_family_idx]
            )
            if not cross_passed:
                return False, cross_reason

        return True, "accepted"


@dataclass
class SamplingStatus:
    """Tracks sampling progress and rejections (mirrors 3D SamplingStatus)."""

    target_counts: dict[int, int]  # family_idx -> target
    current_counts: dict[int, int] = field(default_factory=dict)
    rejection_counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.current_counts = {fam: 0 for fam in self.target_counts}

    def set_target_count(self, family_idx: int, target: int) -> None:
        self.target_counts[family_idx] = target

    def increase_counter(self, family_idx: int) -> None:
        self.current_counts[family_idx] += 1

    def increase_rejected_counter(self, reason: str) -> None:
        self.rejection_counters[reason] = self.rejection_counters.get(reason, 0) + 1

    def finished(self) -> bool:
        return all(
            self.current_counts.get(fam, 0) >= target
            for fam, target in self.target_counts.items()
        )

    def print(self) -> None:
        total = sum(self.current_counts.values())
        target_total = sum(self.target_counts.values())
        print(f"[monitor] Progress: {total}/{target_total}")
        for fam, count in self.current_counts.items():
            target = self.target_counts[fam]
            print(f"  Family {fam}: {count}/{target}")
        if self.rejection_counters:
            print("  Rejections:")
            for reason, count in sorted(
                self.rejection_counters.items(), key=lambda x: -x[1]
            ):
                print(f"    {reason}: {count}")


@dataclass
class PostprocessingStats:
    """Track post-processing operations."""

    num_extensions: int = 0
    num_trims: int = 0
    num_iterations: int = 0
    total_distance_extended: float = 0.0
    num_branch_shortenings: int = 0
    total_distance_shortened: float = 0.0


class Postprocessor:
    """Encapsulates post-processing logic (extension + trimming)."""

    def __init__(self, config: PostprocessingConfig, num_families: int):
        self.config = config
        self.num_families = num_families
        self.stats = PostprocessingStats()

    @staticmethod
    def _angle_between_lines(line1: np.ndarray, line2: np.ndarray) -> float:
        """Compute acute angle between two line segments in degrees."""
        dir1 = line1[1] - line1[0]
        dir2 = line2[1] - line2[0]

        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = dir2 / np.linalg.norm(dir2)

        cos_angle = np.abs(np.dot(dir1, dir2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def extend_fractures(self, lines: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
        """Extend fracture tips to touch other family lines (direction-preserving).

        Implements iterative extension until convergence:
        - For each family, extend tips ONLY along their current direction
        - Only extend toward lines in OTHER families (skip self-family)
        - If extension distance <= extension_threshold, extend to touch
        - Repeat until no more extensions occur (convergence)

        Key constraint: Endpoints move only along the line's direction vector.
        This preserves fracture orientation and prevents arbitrary direction changes.

        Args:
            lines: Fracture lines organized by family

        Returns:
            Extended lines (same structure, directions preserved)
        """

        current_lines = [[line.copy() for line in family] for family in lines]

        for iteration in range(self.config.extension_max_iterations):
            extended_lines = [[] for _ in range(self.num_families)]
            num_extensions_this_iter = 0

            for family_idx in range(self.num_families):
                for line_idx, line in enumerate(current_lines[family_idx]):
                    extended_line = line.copy()

                    # Check both endpoints
                    for endpoint_idx in [0, 1]:
                        endpoint = extended_line[endpoint_idx]

                        # Extract direction vector (normalized)
                        direction = (
                            extended_line[endpoint_idx]
                            - extended_line[(endpoint_idx + 1) % 2]
                        )
                        line_length = np.linalg.norm(direction)

                        direction_normalized = direction / line_length

                        # Find closest point on lines in OTHER families only
                        (min_closest, min_distance, min_extension_needed) = (
                            self._find_closest_point_on_other_families(
                                endpoint,
                                family_idx,
                                current_lines,
                                direction_normalized,
                            )
                        )

                        # Extend if:
                        # 1. There is a closest point found
                        # 2. Extension distance is within threshold
                        if (
                            min_closest is not None
                            and min_extension_needed is not None
                            and min_extension_needed <= self.config.extension_threshold
                        ):
                            # Move endpoint along direction to touch closest point
                            extended_line[endpoint_idx] = min_closest
                            num_extensions_this_iter += 1
                            self.stats.total_distance_extended += min_extension_needed

                    extended_lines[family_idx].append(extended_line)

            current_lines = extended_lines
            self.stats.num_iterations += 1
            self.stats.num_extensions += num_extensions_this_iter

            print(
                f"[postprocess-extend] Iteration {iteration + 1}: "
                f"{num_extensions_this_iter} extensions"
            )

            # Convergence: no extensions this iteration
            if num_extensions_this_iter == 0:
                print(f"[postprocess-extend] Converged at iteration {iteration + 1}")
                break

        return current_lines

    def _find_closest_point_on_other_families(
        self,
        endpoint: np.ndarray,
        family_idx: int,
        all_lines: list[list[np.ndarray]],
        direction_normalized: np.ndarray,
    ) -> tuple[np.ndarray | None, float, float | None]:
        """Find closest point on OTHER families to endpoint, considering direction.

        Returns closest point that lies on the line's extension direction,
        checking only lines in OTHER families.

        Args:
            endpoint: Current endpoint position
            family_idx: Index of current family (to skip self)
            all_lines: All lines organized by family
            direction_normalized: Unit direction vector from endpoint

        Returns:
            Tuple of (closest_point, distance_to_endpoint, extension_needed)
            Returns (None, inf, None) if no closest point found
        """
        min_closest_pt = None
        min_perpendicular_distance = float("inf")
        min_extension_distance = None

        # Check against ALL OTHER families (skip self-family)
        for other_family_idx in range(self.num_families):
            if other_family_idx == family_idx:
                continue  # Skip self-family

            other_family_lines = all_lines[other_family_idx]

            for other_line in other_family_lines:
                extension_dist, closest_on_current, perp_dist = (
                    self._compute_extension_to_touch_segment(
                        endpoint, direction_normalized, other_line
                    )
                )

                # Only consider forward extensions (t > 0)
                if extension_dist > 0:
                    # Prioritize: minimize perpendicular distance first
                    if perp_dist < min_perpendicular_distance:
                        min_closest_pt = closest_on_current
                        min_perpendicular_distance = perp_dist
                        min_extension_distance = extension_dist

        return min_closest_pt, min_perpendicular_distance, min_extension_distance

    @staticmethod
    def _compute_extension_to_touch_segment(
        endpoint: np.ndarray,
        direction: np.ndarray,
        segment: np.ndarray,
    ) -> tuple[float, np.ndarray, float]:
        """Compute extension distance to get closest to a segment.

        Finds 't' such that point P = endpoint + t * direction is closest to segment.

        This is: minimize ||P - Q|| where Q is on segment and P = endpoint + t * direction

        Args:
            endpoint: Starting point on current line
            direction: Unit direction vector (normalized)
            segment: The other line segment [[x0, y0], [x1, y1]]

        Returns:
            (extension_distance, closest_point_on_current_line, perpendicular_distance_to_segment)
        """
        assert np.isclose(np.linalg.norm(direction), 1)
        seg_start, seg_end = segment[0], segment[1]
        seg_vec = seg_end - seg_start
        seg_length_sq = np.dot(seg_vec, seg_vec)

        if seg_length_sq < 1e-10:  # Degenerate segment
            # Just compute distance to a point
            to_point = seg_start - endpoint
            t = np.dot(to_point, direction)
            closest_pt = endpoint + t * direction
            distance = np.linalg.norm(seg_start - closest_pt)
            return t, closest_pt, distance

        # For a point P = endpoint + t * direction on the current line,
        # and a point Q = seg_start + s * seg_vec on the segment (0 <= s <= 1),
        # minimize ||P - Q||^2

        # This gives us two equations (one for t, one for s).
        # We solve for the closest pair.

        # Vector from segment start to endpoint
        ep_to_seg = endpoint - seg_start

        # Compute closest parameter s on segment (clamped to [0, 1])
        # s = dot(ep_to_seg + t*direction, seg_vec) / seg_length_sq
        # But we need to solve both simultaneously.

        # Expand: minimize ||ep_to_seg + t*direction - s*seg_vec||^2
        # Let u = ep_to_seg, d = direction, v = seg_vec
        # minimize ||u + t*d - s*v||^2

        dir_dot_seg = np.dot(direction, seg_vec)
        dir_dot_ep = np.dot(direction, ep_to_seg)
        seg_dot_ep = np.dot(seg_vec, ep_to_seg)

        denom = seg_length_sq - dir_dot_seg**2

        if abs(denom) < 1e-10:  # Lines are parallel
            # Just project endpoint onto segment
            s = 0.5  # Middle of segment
            closest_on_seg = seg_start + s * seg_vec
            to_seg = closest_on_seg - endpoint
            t = np.dot(to_seg, direction)
            closest_pt = endpoint + t * direction
            distance = np.linalg.norm(closest_on_seg - closest_pt)
            return t, closest_pt, distance

        # Solve for s
        s = (seg_dot_ep - dir_dot_ep * dir_dot_seg) / denom
        s = np.clip(s, 0.0, 1.0)  # Clamp to segment

        # Solve for t from (1)
        t = -dir_dot_ep + s * dir_dot_seg

        # Compute points
        closest_on_current = endpoint + t * direction
        closest_on_segment = seg_start + s * seg_vec
        perpendicular_distance = np.linalg.norm(closest_on_segment - closest_on_current)
        return t, closest_on_current, perpendicular_distance

    def shorten_near_miss_branches(
        self, extended_lines: list[list[np.ndarray]]
    ) -> list[list[np.ndarray]]:
        """Shorten branches for near-miss fractures.

        For each endpoint in each line:
        1. Find nearby endpoints from other families
        2. If perpendicular distance < branch_proximity_tolerance:
            - Compute optimal shortening direction
            - Move endpoint inward by branch_shortening_distance
        3. Only shorten if line remains valid (length > min_length)

        This handles cases where extension didn't work due to
        perpendicular offset or other geometric constraints.

        Args:
            extended_lines: Lines after extension phase

        Returns:
            Shortened lines (same structure)
        """
        shortened_lines = [
            [line.copy() for line in family] for family in extended_lines
        ]
        min_line_length = 0.5  # Minimum acceptable length

        for family_idx in range(self.num_families):
            for line_idx, line in enumerate(shortened_lines[family_idx]):
                updated_line = line.copy()
                line_length = np.linalg.norm(updated_line[1] - updated_line[0])

                if line_length < min_line_length:
                    continue  # Skip degenerate lines

                direction = (updated_line[1] - updated_line[0]) / line_length

                # Check both endpoints
                for endpoint_idx in [0, 1]:
                    endpoint = updated_line[endpoint_idx]

                    # Find nearby endpoints in other families
                    nearby_endpoints = self._find_nearby_endpoints(
                        endpoint,
                        family_idx,
                        extended_lines,
                        self.config.branch_proximity_tolerance,
                    )

                    if nearby_endpoints:
                        # Found near-miss: shorten this branch toward the gap
                        shortened_endpoint = self._shorten_endpoint_toward_nearby(
                            endpoint,
                            nearby_endpoints,
                            direction,
                            self.config.branch_proximity_tolerance,
                        )

                        # Validate shortening doesn't create degenerate line
                        if shortened_endpoint is not None:
                            test_line = updated_line.copy()
                            test_line[endpoint_idx] = shortened_endpoint
                            test_length = np.linalg.norm(test_line[1] - test_line[0])

                            if test_length >= min_line_length:
                                updated_line[endpoint_idx] = shortened_endpoint
                                self.stats.num_branch_shortenings += 1
                                displacement = np.linalg.norm(
                                    shortened_endpoint - endpoint
                                )
                                self.stats.total_distance_shortened += displacement

                                print(
                                    f"[shorten] Family {family_idx}, Line {line_idx}, "
                                    f"Endpoint {endpoint_idx}: shortened by {displacement:.4f}"
                                )

                shortened_lines[family_idx][line_idx] = updated_line

        print(
            f"[shorten] Applied {self.stats.num_branch_shortenings} branch shortenings"
        )
        return shortened_lines

    def _find_nearby_endpoints(
        self,
        endpoint: np.ndarray,
        family_idx: int,
        all_lines: list[list[np.ndarray]],
        proximity_tolerance: float,
    ) -> list[np.ndarray]:
        """Find endpoints in other families within proximity_tolerance.

        Args:
            endpoint: Reference endpoint
            family_idx: Family index to skip
            all_lines: All lines organized by family
            proximity_tolerance: Max distance to consider as "nearby"

        Returns:
            List of nearby endpoints from other families
        """
        nearby = []

        for other_family_idx in range(self.num_families):
            if other_family_idx == family_idx:
                continue  # Skip self-family

            for other_line in all_lines[other_family_idx]:
                # Check both endpoints of other line
                for other_endpoint in [other_line[0], other_line[1]]:
                    distance = np.linalg.norm(endpoint - other_endpoint)

                    if distance < proximity_tolerance:
                        nearby.append(other_endpoint)

        return nearby

    def _shorten_endpoint_toward_nearby(
        self,
        endpoint: np.ndarray,
        nearby_endpoints: list[np.ndarray],
        line_direction: np.ndarray,
        shorten_distance: float,
    ) -> np.ndarray | None:
        """Compute shortened endpoint position toward nearby endpoints.

        Strategy: Move endpoint inward along the line direction
        toward the center of mass of nearby endpoints.

        Args:
            endpoint: Current endpoint
            nearby_endpoints: List of nearby endpoints from other families
            line_direction: Unit direction vector of current line
            shorten_distance: How much to move inward

        Returns:
            New endpoint position, or None if shortening not recommended
        """
        if not nearby_endpoints:
            return None

        # Compute center of mass of nearby endpoints
        center_of_mass = np.mean(nearby_endpoints, axis=0)

        # Vector from endpoint to center of nearby endpoints
        to_center = center_of_mass - endpoint
        to_center_dist = np.linalg.norm(to_center)

        if to_center_dist < 1e-10:
            return None  # Endpoint already at center

        to_center_normalized = to_center / to_center_dist

        # Move endpoint inward along line direction toward the gap
        # Use component of to_center that's opposite to line direction
        inward_component = -line_direction  # Move opposite to line direction

        # Blend: prioritize inward movement, consider toward-center direction
        blend_direction = 0.7 * inward_component + 0.3 * to_center_normalized
        blend_direction = blend_direction / np.linalg.norm(blend_direction)

        # Compute new endpoint
        shortened = endpoint + shorten_distance * blend_direction

        return shortened

    def trim_to_y_nodes(
        self, extended_lines: list[list[np.ndarray]]
    ) -> list[list[np.ndarray]]:
        """Trim intersecting fractures to create Y-nodes.

        For each line in each family:
        1. Find ALL intersection points with ALL other families
        2. For each endpoint, check if there's an exceeding branch < trim_short_branch_length
        3. If yes, move endpoint inward to nearest intersection point
        4. This works for any number of families (N >= 2)

        An "exceeding branch" is the distance from an endpoint to the nearest
        intersection point on the line. If this distance is small, the endpoint
        is trimmed inward to the intersection.

        Args:
            extended_lines: Already-extended lines, organized by family

        Returns:
            Trimmed lines (same structure)

        """
        updated_lines = [[line.copy() for line in family] for family in extended_lines]

        # Iterate over each family
        for family_idx in range(self.num_families):
            for line_idx, line in enumerate(updated_lines[family_idx]):
                # Step 1: Find ALL intersections with ALL OTHER families
                all_intersections = self._find_all_intersections(
                    line, family_idx, extended_lines
                )

                if len(all_intersections) == 0:
                    # No intersections, no trimming needed
                    continue

                # Step 2: Analyze both endpoints
                updated_line = line.copy()
                start = line[0]
                end = line[1]

                # Trim start endpoint if exceeding branch is short
                start_trimmed = self._trim_endpoint_if_short_branch(
                    start, all_intersections, trim_endpoint_idx=0, line=updated_line
                )
                if start_trimmed is not None:
                    updated_line[0] = start_trimmed
                    self.stats.num_trims += 1

                # Trim end endpoint if exceeding branch is short
                end_trimmed = self._trim_endpoint_if_short_branch(
                    end, all_intersections, trim_endpoint_idx=1, line=updated_line
                )
                if end_trimmed is not None:
                    updated_line[1] = end_trimmed
                    self.stats.num_trims += 1

                updated_lines[family_idx][line_idx] = updated_line

        print(
            f"[postprocess-trim] Applied {self.stats.num_trims} trims across all families"
        )
        return updated_lines

    def _find_all_intersections(
        self,
        line: np.ndarray,
        family_idx: int,
        all_lines: list[list[np.ndarray]],
    ) -> list[np.ndarray]:
        """Find ALL intersection points between line and all other families.

        Args:
            line: The line to check intersections for
            family_idx: Index of the family this line belongs to
            all_lines: All lines organized by family

        Returns:
            List of unique intersection points (deduplicated within tolerance)
        """
        intersections = []

        # Check against all other families
        for other_family_idx in range(self.num_families):
            if other_family_idx == family_idx:
                continue  # Skip self-family

            other_family_lines = all_lines[other_family_idx]

            for other_line in other_family_lines:
                intersection_pt = segment_intersection_point(line, other_line, tol=1e-2)

                if intersection_pt is not None:
                    intersections.append(intersection_pt)

        # Deduplicate intersections within tolerance
        unique_intersections = self._deduplicate_points(intersections, tol=1e-2)

        return unique_intersections

    def _trim_endpoint_if_short_branch(
        self,
        endpoint: np.ndarray,
        all_intersections: list[np.ndarray],
        trim_endpoint_idx: int,
        line: np.ndarray,
    ) -> np.ndarray | None:
        """Check if endpoint has short exceeding branch; trim if necessary.

        The "exceeding branch" is the segment of the line that extends beyond
        the nearest intersection point. If this segment is shorter than
        trim_short_branch_length, we trim the endpoint to the intersection.

        Args:
            endpoint: The endpoint to check (start or end)
            all_intersections: All intersection points on this line
            trim_endpoint_idx: 0 for start, 1 for end
            line: The full line (for reference)

        Returns:
            New endpoint position if trimmed, None if not trimmed
        """
        if len(all_intersections) == 0:
            return None

        # Find nearest intersection to this endpoint
        distances_to_intersections = [
            np.linalg.norm(endpoint - intr_pt) for intr_pt in all_intersections
        ]
        min_dist_idx = np.argmin(distances_to_intersections)
        nearest_intersection = all_intersections[min_dist_idx]
        exceeding_branch_length = distances_to_intersections[min_dist_idx]

        # Trim if exceeding branch is short
        if exceeding_branch_length < self.config.trim_short_branch_length:
            # Move endpoint inward to the intersection, with clamping
            return self._clamp_endpoint_move(
                endpoint, nearest_intersection, self.config.trim_short_branch_length
            )

        return None

    @staticmethod
    def _deduplicate_points(points: list[np.ndarray], tol: float) -> list[np.ndarray]:
        """Deduplicate points within tolerance."""
        unique: list[np.ndarray] = []
        for point in points:
            if not any(np.linalg.norm(point - u) <= tol for u in unique):
                unique.append(point)
        return unique

    @staticmethod
    def _clamp_endpoint_move(
        endpoint: np.ndarray, target: np.ndarray, max_move: float
    ) -> np.ndarray:
        """Clamp endpoint move toward target."""
        displacement = target - endpoint
        distance = np.linalg.norm(displacement)
        if distance <= max_move:
            return target.copy()
        return endpoint + max_move * (displacement / distance)


class FractureGenerator:
    """Generate 2D biconjugate fracture networks."""

    def __init__(self, config: Config | None = None):
        """Initialize the FractureGenerator."""
        # Config
        self.config = config or Config()
        self._print_config()

        # Random number generator
        self._rng: np.random.Generator = np.random.default_rng(self.config.seed)

        # Constraints
        self._constraint_checker = ConstraintChecker(
            constraints=self.config.constraints
        )
        self._constraint_matrix = ConstraintMatrix(
            checker=self._constraint_checker, num_families=len(self.config.families)
        )

        # Status tracking
        self._status = SamplingStatus(
            target_counts={
                i: fam.target_num for i, fam in enumerate(self.config.families)
            }
        )

        # Postprocessing
        self._post_processor = Postprocessor(
            config=self.config.postprocessing,
            num_families=len(self.config.families),
        )

        # Fracture storage (for N families)
        self._reset()

    def _print_config(self) -> None:
        """Print configuration summary (mirrors 3D printConfig())."""
        cfg = self.config
        print(
            f"\n[config] domain: [{cfg.domain.xmin}, {cfg.domain.ymin}] -> "
            f"[{cfg.domain.xmax}, {cfg.domain.ymax}]"
        )
        print(
            f"[config] subdomain: [{cfg.subdomain.xmin}, {cfg.subdomain.ymin}] -> "
            f"[{cfg.subdomain.xmax}, {cfg.subdomain.ymax}]"
        )
        print(f"[config] families: {len(cfg.families)}")
        for i, fam in enumerate(cfg.families):
            print(f"[config] family.{i}.target_num={fam.target_num}")
            print(
                f"[config] family.{i}.sampler.major_axis_length.mean={fam.major_axis_length.mean}"
            )
            print(
                f"[config] family.{i}.sampler.rotation_deg.mean={fam.rotation_deg.mean}"
            )
        print(f"[config] constraints:")
        print(f"[config]   min_distance={cfg.constraints.min_distance}")
        print(
            f"[config]   min_intersecting_angle_deg_self={cfg.constraints.min_intersecting_angle_deg_self}"
        )
        print(
            f"[config]   min_intersecting_angle_deg_other={cfg.constraints.min_intersecting_angle_deg_other}"
        )
        print(
            f"[config]   min_intersecting_distance={cfg.constraints.min_intersection_distance}"
        )
        print()

    def _reset(self) -> None:
        """Reset internal state for new generation run."""
        self._lines: list[list[np.ndarray]] = [[] for _ in self.config.families]
        self._dips: list[np.ndarray] = [
            np.array([], dtype=float) for _ in self.config.families
        ]

    def generate(self) -> FractureNetwork:
        """Generate a fracture network based on the configuration."""
        self._reset()
        self._place_fractures()

        raw_lines = self._copy_lines(self._lines)
        dips = [d.copy() for d in self._dips]

        raw_network = FractureNetwork(
            domain=self.config.domain,
            lines=raw_lines,
            identifier="raw",
            dips=dips,
            colors=["green", "red"],  # TODO self.config.colors,
        )

        return raw_network

    def postprocess(self, network: FractureNetwork) -> FractureNetwork:
        """Post-process raw network: extend and trim to create Y-nodes.

        Pipeline:
        1. Extension phase: extend tips to touch near-by lines
        2. Trimming phase: trim short branches at intersections to create Y-nodes

        Args:
            network: Raw fracture network from generate()

        Returns:
            Post-processed network with identifier "y_node_processed"
        """
        print("\n[postprocess] Starting extension phase...")
        extended_lines = self._post_processor.extend_fractures(network.lines)

        print("\n[postprocess] Branch Shortening: Handling near-miss fractures")
        shortened_lines = self._post_processor.shorten_near_miss_branches(
            extended_lines
        )

        print("\n[postprocess] Starting trimming phase...")
        processed_lines = self._post_processor.trim_to_y_nodes(shortened_lines)

        print(f"\n[postprocess] Summary:")
        print(f"  Extensions: {self._post_processor.stats.num_extensions}")
        print(f"  Trims: {self._post_processor.stats.num_trims}")
        print(
            f"  Total distance extended: {self._post_processor.stats.total_distance_extended:.2f}"
        )

        return FractureNetwork(
            domain=self.config.domain,
            lines=processed_lines,
            identifier="y_node_processed",
            dips=network.dips,
            colors=network.colors,
        )

    def _random_generator(self) -> None:
        """Generate length and dip angles with perturbations."""
        cfg = self.config
        length_perturbations = [
            self._rng.normal(0, lv, nf)
            for nf, lv in zip(cfg.num_fractures, cfg.length_variation)
        ]
        dip_perturbations = [
            self._rng.normal(0, dv, nf)
            for nf, dv in zip(cfg.num_fractures, cfg.dip_variation)
        ]
        self._lengths = [cfg.length[i] + length_perturbations[i] for i in range(2)]
        self._dips = [cfg.dip[i] + dip_perturbations[i] for i in range(2)]

    def _place_fractures(self) -> None:
        """Place fractures according to distance and intersection constraints."""
        cfg = self.config

        # Place first fracture at center of subdomain - guaranteed to cross
        subdomain_center = np.mean(
            np.array(
                [
                    [cfg.subdomain.xmin, cfg.subdomain.ymin],
                    [cfg.subdomain.xmax, cfg.subdomain.ymax],
                ]
            ),
            axis=0,
        )

        for family_idx, fam_cfg in enumerate(cfg.families):
            # Special case of no-fracture
            if fam_cfg.target_num == 0:
                continue

            dip_deg = self._rng.normal(
                fam_cfg.rotation_deg.mean, fam_cfg.rotation_deg.stddev
            )
            length = self._rng.normal(
                fam_cfg.major_axis_length.mean, fam_cfg.major_axis_length.stddev
            )

            dip_rad = np.radians(dip_deg)
            dir_vec = np.array([np.cos(dip_rad), np.sin(dip_rad)])
            start = subdomain_center - 0.5 * length * dir_vec
            end = subdomain_center + 0.5 * length * dir_vec
            first_line = np.array([start, end])

            self._lines[family_idx].append(first_line)
            self._dips[family_idx] = np.append(self._dips[family_idx], dip_deg)
            self._status.increase_counter(family_idx)

        # Main loop: random family selection
        iteration = 0
        exponent = 0.0
        while not self._status.finished() and iteration < cfg.max_iterations:
            # Random family selection (mirrors 3D multiSampler(id))
            family_idx = self._rng.integers(0, len(cfg.families))

            # Skip if family is complete
            if (
                self._status.current_counts[family_idx]
                >= self._status.target_counts[family_idx]
            ):
                iteration += 1
                continue

            # Increase range.
            if iteration % (cfg.max_iterations // 10) == 0:
                exponent += 0.1

            # Attempt placement
            accepted = self._place_single_fracture(
                family_idx, self._status.current_counts[family_idx], exponent
            )

            if accepted:
                self._status.increase_counter(family_idx)

            iteration += 1

            # Periodic status output
            if iteration % 100 == 0:
                self._status.print()

        # Final status report
        if not self._status.finished():
            print(
                f"\n[warning] Max iterations ({cfg.max_iterations}) reached without convergence"
            )
        self._status.print()

    def _place_single_fracture(
        self, family_idx: int, fracture_idx: int, exponent: float
    ) -> bool:
        """Place a single fracture with distance and intersection constraints.

        Retruns:
            True if fracture was placed, False if rejected.

        """
        cfg = self.config
        fam_cfg = cfg.families[family_idx]

        max_attempts = 10
        for attempt in range(max_attempts):
            # Sample random parameters
            dip_deg = self._rng.normal(
                fam_cfg.rotation_deg.mean, fam_cfg.rotation_deg.stddev
            )
            length = self._rng.normal(
                fam_cfg.major_axis_length.mean, fam_cfg.major_axis_length.stddev
            )

            # Sample random center
            # TODO increase exponent? Works also without...
            center = self._sample_random_center_in_subdomain(exponent)

            # Generate trial line
            dip_rad = np.radians(dip_deg)
            dir_vec = np.array([np.cos(dip_rad), np.sin(dip_rad)])
            start = center - 0.5 * length * dir_vec
            end = center + 0.5 * length * dir_vec
            trial_line = np.array([start, end])

            # Evaluate ALL constraints via constraint matrix
            accepted, reason = self._constraint_matrix.evaluate(
                trial_line, family_idx, self._lines
            )

            if accepted:
                self._lines[family_idx].append(trial_line)
                self._dips[family_idx] = np.append(self._dips[family_idx], dip_deg)
                return True
            else:
                self._status.increase_rejected_counter(reason)

        return False

    def _sample_random_center_in_subdomain(self, exponent: int) -> np.ndarray:
        """Sample random center point within subdomain."""
        cfg = self.config
        xmin, xmax = cfg.subdomain.xmin, cfg.subdomain.xmax
        ymin, ymax = cfg.subdomain.ymin, cfg.subdomain.ymax

        xcenter = 0.5 * (xmin + xmax)
        ycenter = 0.5 * (ymin + ymax)

        xdist = 0.5 * 2**exponent * (xmax - xmin)
        ydist = 0.5 * 2**exponent * (ymax - ymin)

        xmin = xcenter - xdist
        xmax = xcenter + xdist
        ymin = ycenter - ydist
        ymax = ycenter + ydist

        center = np.array(
            [
                self._rng.uniform(xmin, xmax),
                self._rng.uniform(ymin, ymax),
            ]
        )
        return center

    # ! ---- Geometry Utilities ---- ! #

    @staticmethod
    def _copy_lines(lines: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
        """Deep-copy nested fracture-line structure."""
        return [[line.copy() for line in family] for family in lines]
