"""
Carga y construcción del pipeline de datos (tf.data) para el
dataset Car-or-Truck.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def load_datasets(
    train_dir: str = "../data/train",
    valid_dir: str = "../data/valid",
    image_size=(128, 128),
    batch_size: int = 64,
):
    """
    Carga los conjuntos de entrenamiento y validación desde
    directorios de imágenes organizados por clase.

    Parameters
    ----------
    train_dir : str
        Ruta al directorio de entrenamiento.
    valid_dir : str
        Ruta al directorio de validación.
    image_size : tuple
        Tamaño (alto, ancho) al que se redimensionan las imágenes.
    batch_size : int
        Tamaño de lote.

    Returns
    -------
    ds_train_, ds_valid_ : tf.data.Dataset
        Datasets crudos (sin normalizar) de entrenamiento y validación.
    """
    ds_train_ = image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        image_size=list(image_size),
        interpolation="nearest",
        batch_size=batch_size,
        shuffle=True,
    )
    ds_valid_ = image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="binary",
        image_size=list(image_size),
        interpolation="nearest",
        batch_size=batch_size,
        shuffle=False,
    )
    return ds_train_, ds_valid_


def convert_to_float(image, label):
    """Convierte una imagen a float32 en el rango [0, 1]."""
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


def build_pipeline(ds_train_, ds_valid_):
    """
    Aplica normalización, cache y prefetch a los datasets crudos.

    Parameters
    ----------
    ds_train_, ds_valid_ : tf.data.Dataset
        Datasets crudos devueltos por `load_datasets`.

    Returns
    -------
    ds_train, ds_valid : tf.data.Dataset
        Datasets listos para entrenar.
    """
    autotune = tf.data.experimental.AUTOTUNE

    ds_train = (
        ds_train_
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=autotune)
    )
    ds_valid = (
        ds_valid_
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=autotune)
    )
    return ds_train, ds_valid