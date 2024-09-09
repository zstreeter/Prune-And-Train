import argparse
from src.main import run_pruning_process
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Prune and retrain AlexNet using JAX.")
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.01,
        help="Pruning threshold for weight pruning. Lower values prune more aggressively.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot training accuracy and pruning statistics.",
    )
    parser.add_argument(
        "--show-pruning-stats",
        action="store_true",
        help="Show how much pruning was done after each iteration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Run the main pruning process
    accuracy_stats, pruning_stats = run_pruning_process(prune_ratio=args.prune_ratio)

    if args.plot:
        # Plot the accuracy and pruning stats
        plt.figure()
        plt.plot(accuracy_stats, label="Accuracy")
        plt.plot(pruning_stats, label="Pruned Weights")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    if args.show_pruning_stats:
        print(f"Final Pruning Stats: {pruning_stats}")


if __name__ == "__main__":
    main()
