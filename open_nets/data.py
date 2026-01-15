"""Data utils."""

import numpy as np


def create_dataset(size=10000, target_prob=0.5, noise_level=0.5):
    noise = np.random.rand(size) * noise_level
    noise = 2 * ((noise - noise.min()) / (noise.max() - noise.min())) - 1

    is_target = np.random.rand(size) < target_prob
    x = np.where(is_target, 1 + noise * noise_level, 0 + noise * noise_level)
    x = x[:, np.newaxis]

    y = is_target.astype(int)
    y = y[:, np.newaxis]

    return x, y


def split_batches(x, y, batch_size=50, shuffle=False):
    num_samples = x.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    batches_x = np.array_split(x[indices], np.ceil(num_samples / batch_size))
    batches_y = np.array_split(y[indices], np.ceil(num_samples / batch_size))

    return batches_x, batches_y
