import jax
import jax.numpy as jnp
from flax.training import train_state
import optax


def create_train_state(rng, model, learning_rate=0.001):
    params = model.init(rng, jnp.ones([1, 64, 64, 3]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_step(state, batch):
    # Batch is a tuple: (images, labels)
    images, labels = batch

    # Convert TensorFlow tensors to JAX arrays
    images = jnp.array(images)
    labels = jnp.array(labels)

    # One-hot encode labels (assuming num_classes is 200 for Tiny ImageNet)
    labels = jax.nn.one_hot(labels, num_classes=200)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


def evaluate(state, dataset):
    acc = []
    for batch in dataset:
        images, labels = batch
        # Convert TensorFlow tensors to JAX arrays
        images = jnp.array(images)
        labels = jnp.array(labels)

        logits = state.apply_fn({"params": state.params}, images)
        acc.append(
            jnp.mean(jnp.argmax(logits, -1) == labels)
        )  # No need for one-hot here
    return jnp.mean(jnp.stack(acc))


def train_and_evaluate(state, train_ds, val_ds, num_epochs=10, prune_fn=None):
    for epoch in range(num_epochs):
        for batch in train_ds:
            state = train_step(state, batch)

        val_acc = evaluate(state, val_ds)
        print(f"Epoch {epoch}, Validation accuracy: {val_acc}")

        # Prune after every epoch
        if prune_fn:
            state = state.replace(params=prune_fn(state.params))
