import tensorflow as tf
from pathlib import Path


def load_tiny_imagenet(data_dir="~/.cache/datasets"):
    data_dir = Path(data_dir).expanduser().resolve()

    # Load Tiny ImageNet dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir / "tiny-imagenet-200/train", image_size=(64, 64), batch_size=32
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir / "tiny-imagenet-200/val", image_size=(64, 64), batch_size=32
    )

    return train_ds, val_ds
