"""
Definición de arquitecturas de modelos: clasificador con
transfer learning (VGG16) y convnet personalizado desde cero.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_transfer_model(pretrained_base_path: str, augment: bool = False):
    """
    Construye un clasificador binario usando una base convolucional
    VGG16 pre-entrenada (congelada) y un head denso propio.

    Parameters
    ----------
    pretrained_base_path : str
        Ruta al modelo base pre-entrenado guardado (formato Keras).
    augment : bool
        Si True, añade capas de data augmentation (RandomFlip,
        RandomContrast) antes de la base pre-entrenada.

    Returns
    -------
    tf.keras.Model
        Modelo sin compilar, listo para `model.compile(...)`.
    """
    pretrained_base = tf.keras.models.load_model(pretrained_base_path)
    pretrained_base.trainable = False

    layers_list = []

    if augment:
        layers_list += [
            layers.RandomFlip("horizontal"),
            layers.RandomContrast(0.5),
        ]

    layers_list += [
        pretrained_base,
        layers.Flatten(),
        layers.Dense(6, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]

    model = keras.Sequential(layers_list)
    return model


def build_custom_convnet(input_shape=(128, 128, 3)):
    """
    Construye un convnet propio desde cero, compuesto por tres
    bloques convolucionales (Conv2D + MaxPool2D) seguidos de un
    head denso de clasificación binaria.

    Parameters
    ----------
    input_shape : tuple
        Forma de entrada (alto, ancho, canales).

    Returns
    -------
    tf.keras.Model
        Modelo sin compilar, listo para `model.compile(...)`.
    """
    model = keras.Sequential([
        # Bloque convolucional 1
        layers.Conv2D(
            filters=32, kernel_size=5, activation="relu",
            padding="same", input_shape=input_shape,
        ),
        layers.MaxPool2D(),

        # Bloque convolucional 2
        layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPool2D(),

        # Bloque convolucional 3
        layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPool2D(),

        # Head de clasificación
        layers.Flatten(),
        layers.Dense(units=6, activation="relu"),
        layers.Dense(units=1, activation="sigmoid"),
    ])
    return model


def compile_and_train(model, ds_train, ds_valid, epochs=30, optimizer="adam"):
    """
    Compila un modelo para clasificación binaria y lo entrena.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo sin compilar.
    ds_train, ds_valid : tf.data.Dataset
        Datasets de entrenamiento y validación.
    epochs : int
        Número de épocas.
    optimizer : str or tf.keras.optimizers.Optimizer
        Optimizador a usar.

    Returns
    -------
    tf.keras.callbacks.History
        Historial del entrenamiento.
    """
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    history = model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=epochs,
        verbose=0,
    )
    return history

def build_vgg16_base(input_shape=(128, 128, 3)):
    """
    Alternativa a cargar un modelo pre-entrenado desde disco:
    construye la base VGG16 directamente desde Keras Applications,
    pre-entrenada en ImageNet y sin el head de clasificación.

    Parameters
    ----------
    input_shape : tuple
        Forma de entrada (alto, ancho, canales).

    Returns
    -------
    tf.keras.Model
        Base convolucional VGG16 (congelable con .trainable = False).
    """
    base = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = False
    return base