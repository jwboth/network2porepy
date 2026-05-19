from __future__ import annotations

from pathlib import Path

import numpy as np
from .config import DomainConfig


class FractureNetwork:
    """Container for a 2D biconjugate fracture network with analysis methods.

    This class holds fracture line data for two conjugate sets and provides
    methods for analyzing network properties like connectivity, complexity,
    and geometric bounds.

    Attributes:
        lines: List of two lists containing fracture lines for each set.
        dips: List of two arrays containing dip angles for each fracture.
        colors: Colors for visualization of each set.
    """

    def __init__(
        self,
        domain: DomainConfig,
        lines: list[list[np.ndarray]] | None = None,
        dips: list[np.ndarray] | None = None,
        colors: list[str] | None = None,
    ):
        """Initialize the FractureNetwork.

        Args:
            lines: Fracture lines for each set.
            dips: Dip angles for each fracture in each set.
            colors: Colors for plotting each set.
        """
        self.domain: DomainConfig = domain
        self.lines: list[list[np.ndarray]] = lines if lines is not None else [[], []]
        self.dips: list[np.ndarray] = (
            dips
            if dips is not None
            else [np.array([], dtype=float), np.array([], dtype=float)]
        )
        self.colors: list[str] = colors if colors is not None else ["green", "red"]

    @property
    def num_fractures(self) -> list[int]:
        """Number of fractures in each set."""
        return [len(self.lines[0]), len(self.lines[1])]

    @property
    def total_fractures(self) -> int:
        """Total number of fractures across both sets."""
        return sum(self.num_fractures)

    def get_lines(self) -> list[list[np.ndarray]]:
        """Get fracture lines.

        Returns:
            List of fracture lines for each set.
        """
        return self.lines

    # ! ---- Geometry Utilities ---- ! #

    @staticmethod
    def _check_intersection(line1: np.ndarray, line2: np.ndarray) -> bool:
        """Check if two line segments intersect.

        Args:
            line1: First line segment as array of shape (2, 2).
            line2: Second line segment as array of shape (2, 2).

        Returns:
            True if the lines intersect, False otherwise.
        """

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        a, b = line1[0], line1[1]
        c, d = line2[0], line2[1]
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    # ! ---- Analysis ---- ! #

    def analyze_complexity(self) -> dict:
        """Analyze the complexity of the fracture network.

        Returns:
            Dictionary with complexity metrics:
            - num_fractures: Number of fractures [set1, set2].
            - total_length: Total fracture length [set1, set2].
            - connectivity: Number of fracture-fracture intersections.
            - bounding_box: Bounding box of the network.
        """
        lines = self.get_lines()

        metrics = {
            "num_fractures": [len(lines[0]), len(lines[1])],
            "total_length": [0.0, 0.0],
            "connectivity": 0,
            "bounding_box": None,
        }

        all_points = []
        for i in range(2):
            for line in lines[i]:
                metrics["total_length"][i] += np.linalg.norm(line[1] - line[0])
                all_points.extend([line[0], line[1]])

        # Count intersections between sets
        for line1 in lines[0]:
            for line2 in lines[1]:
                if self._check_intersection(line1, line2):
                    metrics["connectivity"] += 1

        if all_points:
            all_points = np.array(all_points)
            metrics["bounding_box"] = {
                "xmin": all_points[:, 0].min(),
                "xmax": all_points[:, 0].max(),
                "ymin": all_points[:, 1].min(),
                "ymax": all_points[:, 1].max(),
            }

        return metrics

    def compute_intersection_matrix(self) -> np.ndarray:
        """Compute the intersection matrix between fracture sets.

        Returns:
            Boolean matrix of shape (n0, n1) where entry (i,j) is True
            if fracture i from set 0 intersects fracture j from set 1.
        """
        assert len(self.lines) == 2, (
            "Expected exactly two sets of fractures for intersection analysis."
        )

        lines = self.get_lines()
        n0, n1 = len(lines[0]), len(lines[1])

        if n0 == 0 or n1 == 0:
            return np.zeros((n0, n1), dtype=bool)

        intersects = np.zeros((n0, n1), dtype=bool)
        for i, line0 in enumerate(lines[0]):
            for j, line1 in enumerate(lines[1]):
                intersects[i, j] = self._check_intersection(line0, line1)

        return intersects

    def sort(self) -> None:
        """Sort fractures to maximize connectivity when adding incrementally.

        Reorders fractures in both sets such that:
        1. The first fracture of each set intersects each other.
        2. Each subsequent fracture maximizes the increase in intersections.
        3. If no fracture increases connectivity, any remaining is added.

        This enables taking subsets of the network (e.g., first k fractures
        from each set) that have maximal connectivity for that size.

        """
        assert len(self.lines) == 2, (
            "Expected exactly two sets of fractures for intersection analysis."
        )

        intersects = self.compute_intersection_matrix()
        n0, n1 = intersects.shape

        if n0 == 0 or n1 == 0:
            return

        # Greedy ordering
        sorted_idx_0 = [0]
        sorted_idx_1 = [0]
        remaining_0 = set(range(n0)) - {0}
        remaining_1 = set(range(n1)) - {0}

        def count_new_intersections(idx: int, from_set: int) -> int:
            if from_set == 0:
                return sum(intersects[idx, j] for j in sorted_idx_1)
            return sum(intersects[i, idx] for i in sorted_idx_0)

        while remaining_0 or remaining_1:
            best_candidate = None
            best_gain = -1
            best_set = None

            for idx in remaining_0:
                gain = count_new_intersections(idx, 0)
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = idx
                    best_set = 0

            for idx in remaining_1:
                gain = count_new_intersections(idx, 1)
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = idx
                    best_set = 1

            if best_candidate is None or best_gain == 0:
                if remaining_0:
                    sorted_idx_0.append(remaining_0.pop())
                elif remaining_1:
                    sorted_idx_1.append(remaining_1.pop())
                continue

            if best_set == 0:
                sorted_idx_0.append(best_candidate)
                remaining_0.remove(best_candidate)
            else:
                sorted_idx_1.append(best_candidate)
                remaining_1.remove(best_candidate)

        # Reorder all data
        self._reorder(sorted_idx_0, sorted_idx_1)

    def _reorder(self, idx_0: list[int], idx_1: list[int]) -> None:
        """Reorder fractures according to given indices.

        Args:
            idx_0: New order for set 0.
            idx_1: New order for set 1.
        """
        self.lines[0] = [self.lines[0][i] for i in idx_0]
        self.lines[1] = [self.lines[1][i] for i in idx_1]

        if self.dips:
            d0, d1 = self.dips[0], self.dips[1]
            self.dips = [
                d0[idx_0] if len(d0) > 0 else d0,
                d1[idx_1] if len(d1) > 0 else d1,
            ]

    def get_subset(self, n0: int, n1: int) -> FractureNetwork:
        """Get a subset of fractures from each set as a new FractureNetwork.

        Args:
            n0: Number of fractures to take from set 0.
            n1: Number of fractures to take from set 1.

        Returns:
            New FractureNetwork containing only the first n0 and n1 fractures.
        """
        domain = self.domain

        lines = [self.lines[0][:n0], self.lines[1][:n1]]

        dips: list[np.ndarray] = []
        if self.dips:
            dips = [self.dips[0][:n0], self.dips[1][:n1]]

        return FractureNetwork(
            domain=domain,
            lines=lines,
            dips=dips,
            colors=self.colors,
        )

    # ! ---- I/O ---- ! #

    def save(self, path: Path | str) -> None:
        """Save fracture network to CSV files.

        Args:
            path: Directory path to save files.
        """
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        # Main fracture CSV (PorePy format)
        csv_path = folder / "fractures.csv"
        with open(csv_path, "w") as f:
            # Write domain in the format
            # "DOMAIN_XMIN, DOMAIN_YMIN, DOMAIN_XMAX, DOMAIN_YMAX \n";
            f.write(
                f"{self.domain.xmin}, {self.domain.ymin}, {self.domain.xmax}, {self.domain.ymax}\n"
            )

            # Write line fracture coordinates in the format:
            # P0_X, P0_Y, P0_Z, ..., PN_X, PN_Y, PN_Z
            fracture_id = 0
            for family_idx, family_lines in enumerate(self.lines):
                for line in family_lines:
                    f.write(f"{line[0, 0]},{line[0, 1]},{line[1, 0]},{line[1, 1]}\n")

        # Write metadata with more info for later statistical analysis, in the format:
        # fracture_id, start_x, start_y, end_x, end_y, dip_deg, length, family_id
        meta_csv_path = folder / "fractures_metadata.csv"
        with open(meta_csv_path, "w") as f:
            f.write(
                "fracture_id,start_x,start_y,end_x,end_y,dip_deg,length,family_id\n"
            )
            fracture_id = 0
            for family_idx, family_lines in enumerate(self.lines):
                for line in family_lines:
                    length = np.linalg.norm(line[1] - line[0])
                    f.write(
                        f"{fracture_id},"
                        f"{line[0, 0]},{line[0, 1]},"
                        f"{line[1, 0]},{line[1, 1]},"
                        f"{self.dips[family_idx]},"
                        f"{length},"
                        f"{family_idx}\n"
                    )
                    fracture_id += 1

        print(f"Fracture network saved to {folder}.")

    @classmethod
    def load(cls, path: Path | str) -> FractureNetwork:
        """Load fracture network from CSV files.

        Args:
            path: Directory path containing fracture files.

        Returns:
            FractureNetwork instance with loaded fractures.
        """
        raise NotImplementedError
        # folder = Path(path)
        # network = cls()

    # ! ---- Visualization ---- ! #

    def plot_network(
        self,
        save_path: Path | str | None = None,
        show: bool = False,
    ) -> None:
        """Plot the fracture network.

        Args:
            save_path: Path to save the plot.
            show: If True, display the plot interactively.
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        lines = self.get_lines()
        title = f"Fracture network"

        plt.figure()
        for i in range(2):
            n_lines = len(lines[i])
            if n_lines > 0:
                # Create color gradient from base color (darker to lighter)
                base_rgb = mcolors.to_rgb(self.colors[i])
                for line_counter, line in enumerate(lines[i]):
                    # Vary lightness: darker for early lines, lighter for later
                    factor = 0.1 + 0.9 * (line_counter / max(1, n_lines - 1))
                    color = tuple(c * factor for c in base_rgb)
                    plt.plot(line[:, 0], line[:, 1], color=color, linestyle="-")
        plt.axis("equal")
        plt.title(title)
        plt.tight_layout()

        if save_path:
            fname = "network.png"
            plt.savefig(Path(save_path) / fname)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_stereonet(
        self, save_path: Path | str | None = None, show: bool = False
    ) -> None:
        """Plot stereonet of fracture orientations.

        Args:
            save_path: Path to save the plot.
            show: If True, display the plot interactively.
        """
        import matplotlib.pyplot as plt

        try:
            import mplstereonet  # noqa: F401
        except ImportError:
            print("mplstereonet required. Install with: pip install mplstereonet")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="stereonet")

        strike = 90  # Hardcoded.

        for i in range(2):
            for dip in zip(self.dips[i]):
                ax.plane(strike, dip, color=self.colors[i], linestyle="-", linewidth=2)
        for i in range(2):
            for dip in self.dips[i]:
                ax.pole(strike, dip, color=self.colors[i], marker="o", markersize=10)
        ax.grid()

        if save_path:
            plt.savefig(Path(save_path) / "stereonet.png")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_rose_diagram(
        self, save_path: Path | str | None = None, show: bool = False
    ) -> None:
        """Plot rose diagram of dip directions.

        Args:
            save_path: Path to save the plot.
            show: If True, display the plot interactively.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        for i in range(2):
            angles = np.deg2rad(self.dips[i])
            angles_extended = np.concatenate([angles, angles + np.pi])
            ax.hist(
                angles_extended,
                bins=32,
                density=True,
                color=self.colors[i],
                alpha=0.5,
                label=f"Set {i + 1}",
            )
        ax.legend(loc="upper right", fontsize=12)
        ax.set_title("Rose diagram of dip directions", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(Path(save_path) / "rose_dip_directions.png")

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_all(
        self, save_path: Path | str | None = None, show: bool = False
    ) -> None:
        """Generate all visualization plots.

        Args:
            save_path: Path to save plots.
            show: If True, display plots interactively.
        """
        self.plot_network(save_path, show=show)
        self.plot_stereonet(save_path, show=show)
        self.plot_rose_diagram(save_path, show=show)

    # ! ---- PorePy Integration ---- ! #

    def to_line_fractures(self) -> list[pp.LineFracture]:
        """Convert fracture network to PorePy LineFracture objects.

        Returns:
            List of PorePy LineFracture objects.
        """
        import porepy as pp

        lines = self.get_lines()
        fractures = []
        for i in range(2):
            for line in lines[i]:
                pts = line.T
                fractures.append(pp.LineFracture(pts))
        return fractures
