import jax
import jax.numpy as jnp
from src.dataset_loader import load_tiny_imagenet
from src.alexnet import AlexNet
from src.train import create_train_state, train_and_evaluate
from src.prune import prune_weights


def run_pruning_process(prune_ratio=0.01):
    # Load dataset
    train_ds, val_ds = load_tiny_imagenet()

    # Initialize AlexNet model
    model = AlexNet(num_classes=200)
    rng = jax.random.PRNGKey(0)

    # Create training state
    state = create_train_state(rng, model, learning_rate=0.001)

    # Track accuracy and pruning stats
    accuracy_stats = []
    pruning_stats = []

    # Train, prune, retrain
    num_iterations = 5
    for iteration in range(num_iterations):
        print(f"Prune-Retrain Iteration {iteration+1}")

        # Train and prune
        val_acc = train_and_evaluate(
            state,
            train_ds,
            val_ds,
            num_epochs=10,
            prune_fn=lambda params: prune_weights(params, threshold=prune_ratio),
        )

        accuracy_stats.append(val_acc)
        # Collect some dummy pruning stats (number of non-zero weights, etc.)
        pruning_stats.append(jnp.sum(jnp.abs(state.params) > prune_ratio))

    return accuracy_stats, pruning_stats
