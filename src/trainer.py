"""
src/trainer.py
==============
Lógica de entrenamiento para los clasificadores CNN.

Uso desde línea de comandos:
    python src/trainer.py --model transfer --epochs 30
    python src/trainer.py --model custom   --epochs 40
    python src/trainer.py --model augmented --epochs 30
"""

import argparse
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

from models import (
    build_transfer_model,
    build_custom_convnet,
    build_augmented_model,
    compile_binary_classifier,
)

# Configuración 

DATA_DIR = 'data/'
MODELS_DIR = 'models/'
RESULTS_DIR = 'results/training_curves/'

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


# Pipeline de datos 
def load_dataset(split: str) -> tf.data.Dataset:
    """
    Carga y prepara un split del dataset desde directorio.

    Args:
        split: 'train' o 'valid'

    Returns:
        Dataset de TensorFlow optimizado con cache y prefetch.
    """
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        labels='inferred',
        label_mode='binary',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split == 'train'),
        seed=42,
    )
    return ds.cache().prefetch(buffer_size=AUTOTUNE)


# Callbacks
def get_callbacks(model_name: str) -> list:
    """Devuelve los callbacks estándar para el entrenamiento."""
    return [
        keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor='val_binary_accuracy',
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            monitor='val_loss',
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f'{model_name}_best.keras'),
            save_best_only=True,
            monitor='val_binary_accuracy',
            verbose=1,
        ),
    ]


# Visualización 
def plot_history(history: keras.callbacks.History, model_name: str) -> None:
    """Guarda las curvas de entrenamiento como imagen."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(history.history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Curvas de Entrenamiento — {model_name}', fontsize=14)

    df[['loss', 'val_loss']].plot(ax=ax1, color=['steelblue', 'tomato'])
    ax1.set_title('Pérdida')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Binary Crossentropy')
    ax1.legend(['Entrenamiento', 'Validación'])
    ax1.grid(alpha=0.3)

    df[['binary_accuracy', 'val_binary_accuracy']].plot(
        ax=ax2, color=['steelblue', 'tomato']
    )
    ax2.set_title('Precisión')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Binary Accuracy')
    ax2.legend(['Entrenamiento', 'Validación'])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, f'{model_name}_history.png')
    plt.savefig(output_path, dpi=150)
    print(f"  Curvas guardadas en: {output_path}")


# Entrenamiento  
def train(model_name: str, epochs: int, learning_rate: float) -> None:
    """
    Ejecuta el pipeline completo de entrenamiento.

    Args:
        model_name:    'transfer', 'custom' o 'augmented'
        epochs:        Número máximo de épocas
        learning_rate: Tasa de aprendizaje
    """
    PRETRAINED_PATH = os.path.join(MODELS_DIR, 'vgg16-pretrained-base')

    # Selección de arquitectura
    if model_name == 'transfer':
        model = build_transfer_model(PRETRAINED_PATH)
    elif model_name == 'custom':
        model = build_custom_convnet()
    elif model_name == 'augmented':
        model = build_augmented_model(PRETRAINED_PATH)
    else:
        raise ValueError(f"Modelo desconocido: '{model_name}'")

    model = compile_binary_classifier(model, learning_rate)
    model.summary()

    # Carga de datos
    print("\nCargando datos...")
    ds_train = load_dataset('train')
    ds_valid = load_dataset('valid')

    # Entrenamiento
    print(f"\nEntrenando '{model_name}' durante máximo {epochs} épocas...\n")
    history = model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=epochs,
        callbacks=get_callbacks(model_name),
    )

    # Resultados finales
    loss, acc = model.evaluate(ds_valid, verbose=0)
    print(f"\n{'='*40}")
    print(f"  Modelo:               {model_name}")
    print(f"  Pérdida validación:   {loss:.4f}")
    print(f"  Precisión validación: {acc:.4f} ({acc*100:.1f}%)")
    print(f"{'='*40}\n")

    plot_history(history, model_name)


# CLI 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrenamiento de clasificadores CNN (Coche vs Camión)'
    )
    parser.add_argument(
        '--model', type=str, default='augmented',
        choices=['transfer', 'custom', 'augmented'],
        help='Arquitectura a entrenar (default: augmented)',
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='Número máximo de épocas (default: 30)',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Tasa de aprendizaje Adam (default: 1e-4)',
    )
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
    )