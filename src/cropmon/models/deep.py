"""Optional hybrid CNN-LSTM for raw spectral cubes.

TensorFlow is an optional dependency (``pip install -e ".[deep]"``). The rest of
the platform works without it; this module only builds and trains the spatial
feature extractor plus temporal head described in the problem statement.
"""

from __future__ import annotations

from typing import Any


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised only without TF
        raise ImportError(
            'TensorFlow is not installed. Install the deep extra: pip install -e ".[deep]"'
        ) from exc
    return tf


def build_hybrid_cnn_lstm(
    timesteps: int = 6,
    height: int = 32,
    width: int = 32,
    bands: int = 12,
    sensor_dim: int = 7,
    num_classes: int = 3,
) -> Any:
    """Build a per-timestep CNN encoder followed by an LSTM over time.

    Inputs are two tensors: ``cube`` ``(batch, timesteps, height, width, bands)``
    and ``sensor`` ``(batch, timesteps, sensor_dim)``.
    """
    tf = _require_tensorflow()
    keras = tf.keras

    cube_input = keras.Input(shape=(timesteps, height, width, bands), name="cube")
    sensor_input = keras.Input(shape=(timesteps, sensor_dim), name="sensor")

    encoder = keras.Sequential(
        [
            keras.layers.Input(shape=(height, width, bands)),
            keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.GlobalAveragePooling2D(),
        ],
        name="cnn_encoder",
    )

    frames = keras.layers.TimeDistributed(encoder)(cube_input)
    fused = keras.layers.Concatenate()([frames, sensor_input])
    temporal = keras.layers.LSTM(32, return_sequences=False)(fused)
    dense = keras.layers.Dense(32, activation="relu")(temporal)
    output = keras.layers.Dense(num_classes, activation="softmax", name="risk")(dense)

    model = keras.Model(inputs=[cube_input, sensor_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
