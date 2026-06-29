"""
src/utils.py
============
Funciones auxiliares para visualización y análisis del modelo.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# Visualización de datos
def show_sample_images(dataset: tf.data.Dataset,
                        class_names: list,
                        n_images: int = 9) -> None:
    """
    Muestra una cuadrícula de imágenes de muestra del dataset.

    Args:
        dataset:     Dataset de TensorFlow.
        class_names: Lista de nombres de clase ['Car', 'Truck'].
        n_images:    Número de imágenes a mostrar.
    """
    images, labels = next(iter(dataset))
    n = min(n_images, len(images))
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(images[i].numpy().astype('uint8'))
        label_idx = int(labels[i].numpy())
        axes[i].set_title(class_names[label_idx], fontsize=12)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Muestras del dataset', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def show_feature_maps(model: keras.Model, image: np.ndarray, layer_name: str) -> None:
    """
    Visualiza los feature maps generados por una capa específica.

    Args:
        model:      Modelo Keras entrenado.
        image:      Array de imagen con shape (H, W, C).
        layer_name: Nombre de la capa convolucional a visualizar.
    """
    # Modelo auxiliar que devuelve la activación de la capa indicada
    feature_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)

    # Preparación de la imagen
    img_array = tf.expand_dims(image, axis=0)
    feature_maps = feature_model.predict(img_array, verbose=0)

    n_filters = feature_maps.shape[-1]
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    fig.suptitle(f'Feature Maps — Capa: {layer_name}', fontsize=13)

    for i in range(n_rows * n_cols):
        ax = axes.flat[i]
        if i < n_filters:
            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_augmentation_examples(image: np.ndarray, augmentation_layer, n_examples: int = 9) -> None:
    """
    Muestra múltiples versiones aumentadas de una misma imagen.

    Args:
        image:             Array de imagen con shape (H, W, C).
        augmentation_layer: Capa de aumentación de Keras.
        n_examples:        Número de versiones a generar.
    """
    img_tensor = tf.cast(tf.expand_dims(image, 0), tf.float32)
    cols = 3
    rows = (n_examples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i in range(n_examples):
        augmented = augmentation_layer(img_tensor, training=True)
        axes[i].imshow(augmented[0].numpy().astype('uint8'))
        axes[i].set_title(f'Versión {i + 1}')
        axes[i].axis('off')

    for i in range(n_examples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Ejemplos de Data Augmentation', fontsize=14)
    plt.tight_layout()
    plt.show()


# Evaluación del modelo 
def predict_image(model: keras.Model, image_path: str, class_names: list, img_size: tuple = (128, 128)) -> dict:
    """
    Realiza una predicción sobre una imagen desde disco.

    Args:
        model:       Modelo Keras entrenado.
        image_path:  Ruta a la imagen.
        class_names: Lista de nombres de clase.
        img_size:    Tamaño de imagen esperado por el modelo.

    Returns:
        Diccionario con clase predicha, probabilidad y confianza.
    """
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prob = float(model.predict(img_array, verbose=0)[0][0])
    class_idx = int(prob > 0.5)

    # Visualización
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title( f"Predicción: {class_names[class_idx]}\n" f"Confianza: {max(prob, 1 - prob):.1%}", fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return {
        'clase':         class_names[class_idx],
        'probabilidad':  prob,
        'confianza':     max(prob, 1 - prob),
    }