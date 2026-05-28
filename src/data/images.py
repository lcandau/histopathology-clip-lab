"""Image loaders matched to the keras_hub ResNet50 backbone.

The keras_hub `resnet_50_imagenet` preset expects RGB inputs scaled to [0, 1].
This is NOT the same convention as `keras.applications.ResNet50`, which uses
Caffe-style preprocessing (BGR channel swap + ImageNet mean subtraction in the
[0, 255] range). Earlier notebooks in this project accidentally fed Caffe-
preprocessed inputs into the keras_hub backbone, which capped the frozen-
feature ceiling at ~0.64 macro-F1 on LC25000 when the correct ceiling is ~0.97.

All new notebooks should import `load_image` (TensorFlow pipeline) or
`load_image_pil` (PIL / NumPy pipeline, used by exp_05 alongside Macenko) from
this module rather than redefining preprocessing locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


DEFAULT_IMG_SIZE: int = 224


def load_image(path, img_size: int = DEFAULT_IMG_SIZE) -> tf.Tensor:
    """Read a JPEG/PNG and return a float32 [0, 1] RGB tensor of shape (H, W, 3).

    Use this inside `tf.data.Dataset.map(...)` pipelines.
    """
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def load_image_pil(path, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Read a JPEG/PNG via PIL and return a float32 [0, 1] RGB array of shape (H, W, 3).

    Used by exp_05 because Macenko's `fitter.transform` operates on NumPy arrays
    in [0, 255] uint8 space — callers should do `fitter.transform(uint8_array)`
    BEFORE this function is applied, then pass the normalised uint8 array here.
    """
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr
