"""Biconjugate fracture network generator application."""

from __future__ import annotations

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from src.generator import FractureGenerator
from src.loader import ConfigLoader
from src.network import FractureNetwork


output = Path(__file__).parent / "output"


def main():
    """Generate a biconjugate fracture network and save results."""

    parser = argparse.ArgumentParser(description="Fracture Network Generator")
    parser.add_argument("--load", action="store_true", help="Load existing networks")
    parser.add_argument(
        "--n_seeds", type=int, default=20, help="Number of random seeds"
    )

    args = parser.parse_args()

    load_networks = args.load
    N_seeds = args.n_seeds

    config = ConfigLoader.load(Path(__file__).parent / f"config.toml")

    # Generate networks for multiple seeds
    analysis_results = []
    for seed in range(N_seeds):
        # Overwrite seed
        config.seed = seed

        # Data management
        output_network = output / f"seed_{seed}"
        output_raw = output_network / "raw"
        output_y_node = output_network / "y_node_processed"

        if load_networks:
            raw_network = FractureNetwork.load(output_raw)
            y_node_network = FractureNetwork.load(output_y_node)
        else:
            generator = FractureGenerator(config)

            print("Generating fractures...")
            raw_network = generator.generate()
            y_node_network = generator.postprocess(raw_network)

            # Sort fractures to maximize connectivity
            raw_network.sort()
            y_node_network.sort()

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

        # Overall analysis for sub-networks
        # Extract N_fractures from config.
        N_fractures = max(
            config.families[i].target_num for i in range(len(config.families))
        )
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
