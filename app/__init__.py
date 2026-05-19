"""Biconjugate fracture network generator application."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.generator import FractureGenerator
from src.config import (
    Config,
    DomainConfig,
    FamilyConfig,
    ConstraintsConfig,
    PostprocessingConfig,
    NormalParams,
    OutputConfig,
)
from src.network import FractureNetwork


output = Path(__file__).parent / "output"


def main():
    """Generate a biconjugate fracture network and save results."""

    regenerate_networks = True
    N_seeds = 5
    N_fractures = 10
    # dips = [
    #     [30 + 0 * 45, 60 + 0 * 45],
    #     # [30 + 1 * 45, 60 + 1 * 45],
    #     # [30 + 2 * 45, 60 + 2 * 45],
    #     # [30 + 3 * 45, 60 + 3 * 45],
    # ]

    # Generate networks for multiple seeds
    analysis_results = []
    for seed in range(N_seeds):
        config = Config(
            domain=DomainConfig(xmin=-500, xmax=500, ymin=-3000, ymax=-2000),
            subdomain=DomainConfig(xmin=-25, xmax=25, ymin=-2525, ymax=-2475),
            families=[
                # Family 1
                FamilyConfig(
                    target_num=N_fractures,
                    major_axis_length=NormalParams(50, 5),
                    rotation_deg=NormalParams(30, 5),
                ),
                # Family 2
                FamilyConfig(
                    target_num=N_fractures,
                    major_axis_length=NormalParams(50, 5),
                    rotation_deg=NormalParams(60, 5),
                ),
            ],
            constraints=ConstraintsConfig(
                min_distance=5,
                min_intersecting_angle_deg_self=10,
                min_intersecting_angle_deg_other=20,
                min_intersection_distance=5,
            ),
            postprocessing=PostprocessingConfig(
                extension_threshold=5,
                extension_max_iterations=10,
                branch_proximity_tolerance=5,
                trim_short_branch_length=5,
            ),
            seed=seed,
            max_iterations=100,
        )

        output_network = output / f"seed_{seed}"
        output_raw = output_network / "raw"
        output_y_node = output_network / "y_node_processed"

        if regenerate_networks:
            generator = FractureGenerator(config)

            print("Generating fractures...")
            # raw_network, extended_network, y_node_network = generator.generate()
            raw_network = generator.generate()
            y_node_network = generator.postprocess(raw_network)

            # Sort fractures to maximize connectivity
            raw_network.sort()
            y_node_network.sort()

            # Analyze
            # metrics = raw_network.analyze_complexity()
            metrics = y_node_network.analyze_complexity()

            num_fracs = sum(metrics["num_fractures"])
            connectivity = metrics["connectivity"]
            print(
                f"\nGenerated {num_fracs} fractures with {connectivity} intersections"
            )

            # Data management
            output_raw.mkdir(exist_ok=True, parents=True)
            output_y_node.mkdir(exist_ok=True, parents=True)

            raw_network.save(output_raw)
            y_node_network.save(output_y_node)

            # Visualization
            raw_network.visualize_all(output_raw, show=False)
            y_node_network.visualize_all(output_y_node, show=False)

            # User feedback
            print(f"\nResults saved to {output_network}")

        else:
            raw_network = FractureNetwork.load(output_raw, identifier="raw")
            y_node_network = FractureNetwork.load(
                output_y_node, identifier="y_node_processed"
            )

        # assert False, "continue here"

        # Overall analysis for sub-networks
        for n_fracs in range(1, N_fractures + 1):
            y_node_sub_network = y_node_network.get_subset(n_fracs, n_fracs)
            metrics = y_node_sub_network.analyze_complexity()
            analysis_results.append({"seed": seed, "n_fracs": n_fracs, **metrics})

    # Collect statistics
    df = pd.DataFrame(analysis_results)
    df.to_csv(output / "statistics.csv", index=False)

    # Visualize analysis results - plot number fractures vs connectivity using means and std devs
    plt.figure()

    df_mean = df.groupby("n_fracs")["connectivity"].mean()
    df_std = df.groupby("n_fracs")["connectivity"].std()
    plt.errorbar(
        df_mean.index,
        df_mean.values,
        yerr=df_std.values,
        fmt="o-",
        color="black",
        ecolor="lightgray",
        elinewidth=3,
        capsize=0,
        label="Mean ± Std Dev",
    )

    plt.xlabel("Number of Fractures")
    plt.ylabel("Connectivity")
    plt.title("Fracture Network Connectivity vs Number of Fractures")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output / "connectivity.png", dpi=300)
    plt.show()


__all__ = ["main"]
