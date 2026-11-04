"""Residual U-Net with three task heads, losses and metrics (Keras 3 / TensorFlow).

Architecture (default: base_filters=24, depth=4 -> filters 24-48-96-192-384)

    input (H, W, 1)
      |-- ResBlock(24) ---------------------------------------------- skip --.
      MaxPool                                                                  |
      |-- ResBlock(48) ------------------------------------- skip --.         |
      MaxPool                                                        |         |
      |-- ResBlock(96)  + SpatialDropout ----------- skip --.        |         |
      MaxPool                                               |        |         |
      |-- ResBlock(192) + SpatialDropout -- skip --.        |        |         |
      MaxPool                                      |        |        |         |
      ResBlock(384) + SpatialDropout  (bottleneck) |        |        |         |
      Upsample+concat --------------------------- ' ResBlock(192)    |         |
      Upsample+concat ------------------------------------ ' ResBlock(96)      |
      Upsample+concat ---------------------------------------------- ' ResBlock(48)
      Upsample+concat ------------------------------------------------------- ' ResBlock(24)
      |-> 1x1 conv, sigmoid  ->  "mask"         P(pixel belongs to a wire)
      |-> 1x1 conv, sigmoid  ->  "centerline"   P(pixel lies on a wire axis)
      '-> 1x1 conv, tanh     ->  "orientation"  (cos 2theta, sin 2theta)

Design choices
* Residual blocks (He et al. 2016) ease optimisation of the deeper network.
* Bilinear up-sampling + convolution instead of transposed convolutions avoids
  checkerboard artefacts (Odena et al. 2016), which would bias edge positions
  and therefore width measurements.
* Receptive field ~ 200 px for depth 4: large enough to see *both* arms of a
  crossing, which is required to decide which arm continues which wire.
* Regularisation against over-fitting: BatchNorm, SpatialDropout2D in the deep
  (high-capacity) levels, decoupled weight decay (AdamW), data augmentation
  and early stopping on an independent validation set.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers, ops

from .config import ModelConfig, TrainConfig

OUTPUTS = ("mask", "centerline", "orientation")


# --------------------------------------------------------------------------- building blocks
def res_block(x, filters: int, name: str, dropout: float = 0.0):
    """Two 3x3 conv-BN layers with an identity (or 1x1 projection) shortcut."""
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, use_bias=False, name=f"{name}_proj")(x)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_drop")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(x)


def build_unet(cfg: ModelConfig) -> keras.Model:
    """Fully convolutional: trained on tiles, applicable to any size divisible by 2**depth."""
    inputs = keras.Input(shape=(None, None, 1), name="sem_image")
    f = cfg.filters
    deep = lambda level: cfg.dropout if level >= cfg.depth - 1 else 0.0

    # Encoder: features at decreasing resolution, kept for the skip connections
    x, skips = inputs, []
    for level in range(cfg.depth):
        x = res_block(x, f[level], f"enc{level}", deep(level))
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{level}")(x)

    x = res_block(x, f[cfg.depth], "bottleneck", cfg.dropout)

    # Decoder: up-sample, fuse with the encoder features of the same resolution
    for level in reversed(range(cfg.depth)):
        x = layers.UpSampling2D(2, interpolation="bilinear", name=f"up{level}")(x)
        x = layers.Concatenate(name=f"cat{level}")([x, skips[level]])
        x = res_block(x, f[level], f"dec{level}")

    # Task heads (float32 so that mixed precision never degrades the outputs)
    outputs = {
        "mask": layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="mask")(x),
        "centerline": layers.Conv2D(1, 1, activation="sigmoid", dtype="float32",
                                    name="centerline")(x),
        "orientation": layers.Conv2D(2, 1, activation="tanh", dtype="float32",
                                     name="orientation")(x),
    }
    return keras.Model(inputs, outputs, name=f"AuNW_ResUNet_d{cfg.depth}_f{cfg.base_filters}")


def receptive_field(cfg: ModelConfig) -> int:
    """Theoretical receptive field (px) of one output pixel (encoder + bottleneck + decoder).

    Recursion r <- r + (k - 1) * j, j <- j * stride over all conv/pool layers,
    where j is the current pixel spacing ("jump") in input pixels.
    """
    r, j = 1, 1
    for _ in range(cfg.depth):
        r += 2 * 2 * j          # two 3x3 convs
        r += 1 * j              # 2x2 max-pool
        j *= 2
    r += 2 * 2 * j              # bottleneck
    for level in reversed(range(cfg.depth)):
        r += 2 * 2 * 2**level   # two 3x3 convs per decoder level, at spacing 2**level
    return r


# --------------------------------------------------------------------------- losses
@keras.saving.register_keras_serializable(package="aunw")
def dice_bce_loss(y_true, y_pred, eps: float = 1.0):
    """Binary cross-entropy + soft Dice.

    BCE gives well-calibrated per-pixel probabilities; the Dice term
        1 - (2 sum(y p) + eps) / (sum(y) + sum(p) + eps)
    is insensitive to class imbalance (thin centrelines cover only a few %
    of the image), so the two complement each other.
    """
    bce = ops.mean(keras.losses.binary_crossentropy(y_true, y_pred))
    axes = (1, 2, 3)
    inter = ops.sum(y_true * y_pred, axis=axes)
    union = ops.sum(y_true, axis=axes) + ops.sum(y_pred, axis=axes)
    dice = 1.0 - (2.0 * inter + eps) / (union + eps)
    return bce + ops.mean(dice)


@keras.saving.register_keras_serializable(package="aunw")
def masked_orientation_loss(y_true, y_pred):
    """Weighted MSE on the double-angle vector; y_true = (cos2t, sin2t, weight)."""
    w = y_true[..., 2:3]
    err = ops.sum(ops.square(y_true[..., :2] - y_pred), axis=-1, keepdims=True)
    return ops.sum(w * err) / (ops.sum(w) + 1.0)


# --------------------------------------------------------------------------- metrics
@keras.saving.register_keras_serializable(package="aunw")
def dice_coef(y_true, y_pred):
    """Hard Dice (= F1 score) at threshold 0.5."""
    p = ops.cast(y_pred > 0.5, "float32")
    inter = ops.sum(y_true * p)
    return (2.0 * inter + 1.0) / (ops.sum(y_true) + ops.sum(p) + 1.0)


@keras.saving.register_keras_serializable(package="aunw")
def angle_error_deg(y_true, y_pred):
    """Mean absolute axial angle error (degrees) on labelled single-wire pixels."""
    w = y_true[..., 2]
    t_true = 0.5 * ops.arctan2(y_true[..., 1], y_true[..., 0])
    t_pred = 0.5 * ops.arctan2(y_pred[..., 1], y_pred[..., 0])
    d = ops.abs(t_true - t_pred)
    d = ops.minimum(d, 3.141592653589793 - d)        # axial: 170 deg error == 10 deg error
    return ops.sum(w * d) / (ops.sum(w) + 1.0) * (180.0 / 3.141592653589793)


def compile_model(model: keras.Model, cfg: TrainConfig) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.learning_rate,
                                         weight_decay=cfg.weight_decay, clipnorm=1.0),
        loss={"mask": dice_bce_loss, "centerline": dice_bce_loss,
              "orientation": masked_orientation_loss},
        loss_weights=cfg.loss_weights,
        metrics={"mask": [dice_coef], "centerline": [dice_coef],
                 "orientation": [angle_error_deg]},
        jit_compile=False,
    )
    return model


def load_model(path) -> keras.Model:
    return keras.models.load_model(path, compile=False)


def layer_table(model: keras.Model) -> list[dict]:
    """Layer name / type / output shape / parameter count, for display."""
    rows = []
    for layer in model.layers:
        try:
            shape = str(layer.output.shape)
        except Exception:
            shape = "multiple"
        rows.append({"layer": layer.name, "type": layer.__class__.__name__,
                     "output shape": shape, "parameters": layer.count_params()})
    return rows


def enable_mixed_precision(on: bool) -> None:
    keras.mixed_precision.set_global_policy("mixed_float16" if on else "float32")
    if on and not tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("float32")   # no benefit on CPU
