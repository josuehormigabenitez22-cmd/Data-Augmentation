"""
src/models.py
=============
Definición de arquitecturas CNN para clasificación de imágenes.

Contiene:
  - build_transfer_model()   → VGG16 + Transfer Learning
  - build_custom_convnet()   → Red personalizada desde cero
  - build_augmented_model()  → VGG16 + Data Augmentation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


# Constantes 
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
NUM_CLASSES = 1   # Clasificación binaria: Coche (0) vs Camión (1)


# Modelos  
def build_transfer_model(pretrained_base_path: str) -> keras.Model:
    """
    Construye un clasificador basado en Transfer Learning con VGG16.

    La base convolucional de VGG16 (preentrenada en ImageNet) se congela.
    Solo se entrena la cabeza clasificadora añadida.

    Args:
        pretrained_base_path: Ruta al modelo VGG16 preentrenado guardado.

    Returns:
        Modelo Keras compilado listo para entrenar.
    """
    # Carga de la base preentrenada
    pretrained_base = tf.keras.models.load_model(pretrained_base_path)
    pretrained_base.trainable = False   # Congelar pesos

    model = keras.Sequential([
        # Cabeza clasificadora
        pretrained_base,
        layers.Flatten(),
        layers.Dense(6, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='sigmoid'),
    ], name='transfer_learning_vgg16')

    return model


def build_custom_convnet(input_shape: tuple = (128, 128, 3)) -> keras.Model:
    """
    Construye una red convolucional personalizada desde cero.

    Arquitectura de 3 bloques convolucionales con número de filtros
    creciente (32 → 64 → 128) y Max Pooling entre bloques.

    Args:
        input_shape: Dimensiones de la imagen de entrada (H, W, C).

    Returns:
        Modelo Keras sin compilar.
    """
    model = keras.Sequential([

        # Bloque 1: características simples -> (bordes, texturas) 
        layers.Conv2D(
            filters=32,
            kernel_size=5,
            activation='relu',
            padding='same',
            input_shape=input_shape,
            name='conv1',
        ),
        layers.MaxPool2D(pool_size=2, name='pool1'),

        # Bloque 2: características intermedias -> (formas, partes) 
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='conv2',
        ),
        layers.MaxPool2D(pool_size=2, name='pool2'),

        # Bloque 3: características complejas -> (objetos reconocibles) 
        layers.Conv2D(
            filters=128,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='conv3',
        ),
        layers.MaxPool2D(pool_size=2, name='pool3'),

        # Cabeza clasificadora 
        layers.Flatten(name='flatten'),
        layers.Dense(units=6, activation='relu', name='dense_hidden'),
        layers.Dense(units=NUM_CLASSES, activation='sigmoid', name='output'),

    ], name='custom_convnet')

    return model


def build_augmented_model(pretrained_base_path: str) -> keras.Model:
    """
    Construye el modelo con Transfer Learning y Data Augmentation integrada.

    La aumentación se aplica directamente en el modelo como capas de
    preprocesamiento, lo que permite ejecutarla en GPU durante el training.

    Args:
        pretrained_base_path: Ruta al modelo VGG16 preentrenado guardado.

    Returns:
        Modelo Keras sin compilar.
    """
    pretrained_base = tf.keras.models.load_model(pretrained_base_path)
    pretrained_base.trainable = False

    model = keras.Sequential([

        # Data Augmentation (solo activa durante training) 
        preprocessing.RandomFlip('horizontal'),
        preprocessing.RandomContrast(0.5),

        # Base convolucional 
        pretrained_base,

        # Cabeza clasificadora  
        layers.Flatten(),
        layers.Dense(6, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='sigmoid'),

    ], name='augmented_transfer_model')

    return model


# Compilación  
def compile_binary_classifier(model: keras.Model,
                               learning_rate: float = 1e-4) -> keras.Model:
    """
    Compila un modelo para clasificación binaria.

    Args:
        model:         Modelo Keras a compilar.
        learning_rate: Tasa de aprendizaje del optimizador Adam.

    Returns:
        El mismo modelo compilado.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )
    return model


# Punto de entrada  
if __name__ == '__main__':
    # Inspección rápida de la arquitectura personalizada
    model = build_custom_convnet()
    model.summary()