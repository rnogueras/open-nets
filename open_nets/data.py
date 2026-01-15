"""Data utils."""

import numpy as np


def create_dataset(
    size: int = 10000, target_prob: float = 0.5, noise_level: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic dataset."""
    noise = np.random.rand(size) * noise_level
    noise = 2 * ((noise - noise.min()) / (noise.max() - noise.min())) - 1

    is_target = np.random.rand(size) < target_prob
    x = np.where(is_target, 1 + noise * noise_level, 0 + noise * noise_level)
    x = x[:, np.newaxis]

    y = is_target.astype(int)
    y = y[:, np.newaxis]

    return x, y


def split_batches(
    x: np.ndarray, y: np.ndarray, batch_size: int = 50, shuffle: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Split input vectors into batches."""
    assert len(x) == len(y), f"Input length mismatch: {len(x)} != {len(y)}"

    num_samples = x.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    batches_x = np.array_split(x[indices], np.ceil(num_samples / batch_size))
    batches_y = np.array_split(y[indices], np.ceil(num_samples / batch_size))

    return batches_x, batches_y
