import jax
import jax.numpy as jnp


def prune_weights(model_params, threshold=0.01):
    """
    Prune model weights by setting those below a threshold to zero.
    """

    def prune(param):
        return jnp.where(jnp.abs(param) > threshold, param, 0)

    pruned_params = jax.tree_util.tree_map(prune, model_params)
    return pruned_params
