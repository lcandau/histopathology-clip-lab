"""Seeds + op determinism."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def enable_op_determinism() -> None:
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
