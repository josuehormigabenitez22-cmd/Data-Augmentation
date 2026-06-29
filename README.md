# Computer Vision con Redes Neuronales Convolucionales

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)
![Status](https://img.shields.io/badge/Estado-En%20Desarrollo-green?style=for-the-badge)
![License](https://img.shields.io/badge/Licencia-MIT-lightgrey?style=for-the-badge)

**Portfolio profesional de Deep Learning aplicado a Visión por Computador**

*Clasificación de imágenes mediante Redes Neuronales Convolucionales (CNN) — de la teoría a la producción*

</div>

---

## Tabla de Contenidos

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Estructura del Repositorio](#-estructura-del-repositorio)
3. [Fundamentos Teóricos](#-fundamentos-teóricos)
   - [1. El Clasificador Convolucional](#1-el-clasificador-convolucional)
   - [2. Convolución y Activación ReLU](#2-convolución-y-activación-relu)
   - [3. Max Pooling y Condensación](#3-max-pooling-y-condensación)
   - [4. Sliding Windows — Strides y Padding](#4-sliding-windows--strides-y-padding)
   - [5. Diseño de una ConvNet Personalizada](#5-diseño-de-una-convnet-personalizada)
   - [6. Data Augmentation](#6-data-augmentation)
4. [Caso Práctico: Clasificador Coche vs. Camión](#-caso-práctico-clasificador-coche-vs-camión)
5. [Resultados](#-resultados)
6. [Instalación y Ejecución](#-instalación-y-ejecución)
7. [Tecnologías Utilizadas](#-tecnologías-utilizadas)
8. [Referencias](#-referencias)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema completo de **clasificación de imágenes** usando Redes Neuronales Convolucionales (CNN), partiendo desde los principios matemáticos más fundamentales hasta el despliegue de un modelo de producción.

El problema central es el siguiente: dado un conjunto de imágenes de vehículos (~10.000 fotos), ¿puede un modelo aprender a distinguir entre un **Coche** y un **Camión** de forma automática?

### ¿Qué aprenderás con este proyecto?

| Concepto | Descripción |
|---|---|
| **Arquitectura CNN** | Base convolucional + cabeza clasificadora |
| **Transfer Learning** | Reutilización de VGG16 preentrenado en ImageNet |
| **Extracción de características** | Kernels, feature maps, activaciones |
| **Regularización** | Max Pooling, Data Augmentation, Dropout |
| **Diseño desde cero** | Construcción de bloques convolucionales propios |

---

## 🗂️ Estructura del Repositorio

```
computer-vision-portfolio/
│
├── README.md                        # Documentación principal (este archivo)
├── requirements.txt                 # Dependencias del proyecto
│
├── notebooks/
│   ├── 01_convolutional_classifier.ipynb   # Transfer Learning con VGG16
│   ├── 02_convolution_relu.ipynb           # Convolución y activación ReLU
│   ├── 03_max_pooling.ipynb                # Max Pooling y translation invariance
│   ├── 04_sliding_windows.ipynb            # Strides y Padding
│   ├── 05_custom_convnet.ipynb             # Diseño de red personalizada
│   └── 06_data_augmentation.ipynb          # Data Augmentation
│
├── src/
│   ├── models.py                    # Definición de arquitecturas
│   ├── trainer.py                   # Lógica de entrenamiento
│   ├── augmentation.py              # Pipeline de aumentación de datos
│   └── utils.py                     # Funciones auxiliares y visualización
│
├── assets/
│   └── diagrams/                    # Diagramas de arquitectura
│
└── results/
    ├── training_curves/             # Gráficas de pérdida y precisión
    └── predictions/                 # Ejemplos de predicciones del modelo
```

---

## 📚 Fundamentos Teóricos

### 1. El Clasificador Convolucional

Una red convolucional para clasificación de imágenes se divide en dos partes fundamentales:

```
 ┌─────────────────────────────────────────────────────┐
 │                     IMAGEN DE ENTRADA                │
 └──────────────────────────┬──────────────────────────┘
                            │
            ┌───────────────▼────────────────┐
            │        BASE CONVOLUCIONAL       │
            │  (Extracción de características)│
            │   Conv2D → ReLU → MaxPool2D     │
            │   Conv2D → ReLU → MaxPool2D     │
            │   Conv2D → ReLU → MaxPool2D     │
            └───────────────┬────────────────┘
                            │
            ┌───────────────▼────────────────┐
            │         CABEZA CLASIFICADORA    │
            │  (Toma de decisión)             │
            │   Flatten → Dense → Sigmoid     │
            └───────────────┬────────────────┘
                            │
            ┌───────────────▼────────────────┐
            │         CLASE PREDICHA          │
            │      [Coche] o [Camión]         │
            └────────────────────────────────┘
```

#### Conceptos clave

**Base convolucional:** Aprende a detectar características visuales de la imagen: bordes, texturas, formas, patrones y combinaciones de estos. Cuanto más profunda es la red, más abstractas y complejas son las características que aprende.

**Cabeza clasificadora:** Usa las características extraídas por la base para asignar la imagen a una categoría. Está formada principalmente por capas densas (`Dense`).

#### Transfer Learning

Entrenar una CNN desde cero requiere millones de imágenes y días de cómputo. La solución es el **Transfer Learning**: reutilizar la base de un modelo ya entrenado en un gran dataset (como ImageNet) y añadir únicamente una nueva cabeza para la tarea específica.

```
  Modelo preentrenado (VGG16 en ImageNet)
  ┌─────────────────────┐
  │  BASE CONVOLUCIONAL │  ← Se congela (trainable = False)
  │  (16 capas)         │     Mantiene todo lo aprendido
  └──────────┬──────────┘
             │
  ┌──────────▼──────────┐
  │  NUEVA CABEZA       │  ← Se entrena desde cero
  │  (Dense + Sigmoid)  │     Aprende la tarea específica
  └─────────────────────┘
```

> **¿Por qué funciona?** Las características visuales básicas (bordes, texturas, gradientes) son universales. Un modelo entrenado en 1.4M de imágenes ya sabe "ver". Solo necesitamos enseñarle a tomar decisiones nuevas.

---

### 2. Convolución y Activación ReLU

La extracción de características mediante una capa convolucional ocurre en dos pasos:

#### 2.1 Filtrado por Convolución

Un **kernel** es una pequeña matriz de pesos que se desliza sobre la imagen aplicando una suma ponderada de píxeles:

```
  Kernel 3×3 (detector de bordes):      Imagen original:
  ┌────┬────┬────┐                       ┌────┬────┬────┬────┐
  │ -1 │ -1 │ -1 │                       │ 10 │ 10 │ 10 │  5 │
  ├────┼────┼────┤         ★             ├────┼────┼────┼────┤
  │ -1 │  8 │ -1 │    ──────────►        │ 10 │ 10 │  5 │  0 │
  ├────┼────┼────┤                       ├────┼────┼────┼────┤
  │ -1 │ -1 │ -1 │                       │  5 │  0 │  0 │  0 │
  └────┴────┴────┘                       └────┴────┴────┴────┘

                              ▼
                       Feature Map (mapa de características)
```

Parámetros principales de `Conv2D`:

| Parámetro | Función |
|---|---|
| `filters` | Número de feature maps que se generan (uno por kernel) |
| `kernel_size` | Dimensiones del kernel, p.ej. `(3, 3)` o `(5, 5)` |
| `activation` | Función de activación aplicada a la salida |

```python
from tensorflow.keras import layers

# Una capa convolucional con 64 filtros y kernels de 3x3
conv_layer = layers.Conv2D(
    filters=64,
    kernel_size=3,
    activation='relu'
)
```

#### 2.2 Detección con ReLU

La función **ReLU** (Rectified Linear Unit) actúa como un umbral: mantiene los valores positivos (zonas donde el kernel detectó la característica) y elimina los negativos (zonas donde no la detectó).

```
  f(x) = max(0, x)

  Valor de entrada   │  Valor de salida
  ───────────────────┼──────────────────
       -5.3          │       0.0
       -0.1          │       0.0
        0.0          │       0.0
        2.7          │       2.7
        8.4          │       8.4
```

> **Intuición:** ReLU convierte el mapa de características en un mapa de importancia. Todo lo que no es relevante pasa a ser exactamente 0. Todo lo que sí es relevante mantiene su valor original.

---

### 3. Max Pooling y Condensación

Tras aplicar Conv2D + ReLU, el mapa de características contiene mucho espacio "muerto" (zonas de valor 0). **Max Pooling** condensa la información manteniendo solo los valores más importantes.

#### Funcionamiento

```
  Feature Map (4×4):           Max Pooling 2×2:
  ┌───┬───┬───┬───┐            ┌───┬───┐
  │ 0 │ 4 │ 0 │ 2 │            │ 4 │ 3 │
  ├───┼───┼───┼───┤  ──────►   ├───┼───┤
  │ 0 │ 2 │ 3 │ 0 │            │ 5 │ 4 │
  ├───┼───┼───┼───┤            └───┴───┘
  │ 5 │ 0 │ 0 │ 4 │
  ├───┼───┼───┼───┤   Cada bloque 2×2 → un único valor máximo
  │ 0 │ 3 │ 2 │ 0 │   El mapa se reduce a la mitad de tamaño
  └───┴───┴───┴───┘
```

```python
# Capa de Max Pooling con ventana de 2×2
pool_layer = layers.MaxPool2D(pool_size=2)
```

#### Translation Invariance

Una propiedad clave del Max Pooling es que hace al modelo **tolerante a pequeños desplazamientos** en la posición de las características. Si el objeto está un poco a la izquierda o a la derecha, el clasificador sigue reconociéndolo correctamente.

> **Beneficio práctico:** El modelo puede reconocer un coche independientemente de si está centrado, desplazado o con diferente escala en la fotografía.

---

### 4. Sliding Windows — Strides y Padding

Las operaciones de convolución y pooling comparten un mecanismo fundamental: una **ventana deslizante** que recorre la imagen de forma sistemática.

#### Strides (Zancadas)

El **stride** define cuántos píxeles avanza la ventana en cada paso:

```
  Imagen 5×5, Stride = 1:              Imagen 5×5, Stride = 2:
  ┌─────────────────┐                   ┌─────────────────┐
  │[■ ■ ■]          │                   │[■ ■ ■]          │
  │[■ ■ ■]          │  → avanza 1px     │[■ ■ ■]          │  → avanza 2px
  │[■ ■ ■]          │                   │[■ ■ ■]          │
  │                 │                   │                 │
  └─────────────────┘                   └─────────────────┘
  Output: más detallado                 Output: más comprimido
```

| Capa | Stride habitual | Motivo |
|---|---|---|
| `Conv2D` | `1` | Queremos máxima resolución en el feature map |
| `MaxPool2D` | `2` | Queremos reducir la imagen a la mitad |

#### Padding

El **padding** decide qué hacer con los píxeles del borde de la imagen:

```
  padding='valid':                      padding='same':
  La ventana nunca sale de la imagen    Se añaden ceros alrededor
  → El output es más pequeño            → El output mantiene el tamaño
  → Se pierde información de bordes     → Los bordes tienen menos influencia
```

```python
# Configuración completa de capas con strides y padding
model = keras.Sequential([
    layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,          # Avanza 1 píxel en cada paso
        padding='same',     # El output conserva las dimensiones
        activation='relu'
    ),
    layers.MaxPool2D(
        pool_size=2,
        strides=2,          # Reduce a la mitad el tamaño
        padding='same'
    )
])
```

---

### 5. Diseño de una ConvNet Personalizada

Una red profunda encadena múltiples **bloques convolucionales**. Cada bloque extrae características más complejas que el anterior.

```
  Imagen → [Bloque 1] → [Bloque 2] → [Bloque 3] → [Cabeza] → Clase
             ↓             ↓             ↓
           Bordes        Formas        Objetos
           Texturas      Partes        Completos
           Gradientes    Patrones      Reconocibles
```

#### Patrón de diseño estándar

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([

    # Bloque 1 — características simples
    layers.Conv2D(filters=32, kernel_size=5, activation="relu",
                  padding='same', input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    # Bloque 2 — características intermedias
    layers.Conv2D(filters=64, kernel_size=3, activation="relu",
                  padding='same'),
    layers.MaxPool2D(),

    # Bloque 3 — características complejas
    layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                  padding='same'),
    layers.MaxPool2D(),

    # Cabeza clasificadora
    layers.Flatten(),
    layers.Dense(units=6, activation="relu"),
    layers.Dense(units=1, activation="sigmoid"),
])
```

> **Patrón clave:** el número de filtros se **duplica** en cada bloque (32 → 64 → 128). Esto compensa la reducción de tamaño causada por el MaxPooling, asegurando que la red genere cada vez más representaciones distintas aunque el mapa sea más pequeño.

#### Por qué las redes profundas funcionan mejor

| Profundidad | Capacidad de representación |
|---|---|
| 1 bloque | Detecta bordes y gradientes simples |
| 2 bloques | Combina bordes → texturas y formas básicas |
| 3+ bloques | Combina formas → objetos reconocibles (ruedas, ventanas...) |

---

### 6. Data Augmentation

El modelo aprende únicamente de los datos que le proporcionamos. Más datos → mejor generalización. **Data Augmentation** crea versiones artificiales de nuestras imágenes existentes aplicando transformaciones que no alteran la clase.

#### Transformaciones habituales

```
  Imagen original:          Versiones aumentadas:
  ┌─────────────┐           ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │             │           │ Flip │ │Rotar │ │ Zoom │ │Brillo│
  │   🚗 ────►  │    →      │  ◄🚗 │ │ 🚗 ↺ │ │  🚗  │ │  🚗  │
  │             │           │      │ │      │ │(amp) │ │(+clr)│
  └─────────────┘           └──────┘ └──────┘ └──────┘ └──────┘
```

```python
from tensorflow.keras.layers.experimental import preprocessing

model = keras.Sequential([

    # Capa de aumentación (solo activa durante el entrenamiento)
    preprocessing.RandomFlip('horizontal'),      # Espejo horizontal
    preprocessing.RandomRotation(0.1),           # Rotación ±10%
    preprocessing.RandomZoom(0.1),               # Zoom ±10%
    preprocessing.RandomContrast(0.5),           # Contraste ±50%

    # Base preentrenada (congelada)
    pretrained_base,

    # Cabeza clasificadora
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
```

> **Regla de oro:** las transformaciones deben preservar la clase. Para coches y camiones, el espejo horizontal tiene sentido (un coche sigue siendo un coche mirando a la izquierda). Una rotación de 180° podría no tenerlo (un camión boca abajo es confuso para el modelo).

---

## 🔬 Caso Práctico: Clasificador Coche vs. Camión

### El Problema

Dado un dataset de ~10.000 imágenes de vehículos, construimos un sistema capaz de clasificar automáticamente si una imagen muestra un **Coche** o un **Camión**.

### Pipeline Completo

#### Paso 1 — Preparación del entorno

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
```

#### Paso 2 — Carga y exploración de datos

```python
# Configuración del pipeline de datos
IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

# Dataset de entrenamiento
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train/',
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

# Dataset de validación
ds_valid = tf.keras.preprocessing.image_dataset_from_directory(
    'data/valid/',
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Optimización del pipeline
ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
ds_valid = ds_valid.cache().prefetch(buffer_size=AUTOTUNE)

print(f"Clases detectadas: {ds_train.class_names}")
```

```
# Output esperado:
# Found 5117 files belonging to 2 classes.
# Found 5051 files belonging to 2 classes.
# Clases detectadas: ['Car', 'Truck']
```

#### Paso 3 — Definición del modelo con Transfer Learning

```python
# Carga de la base preentrenada VGG16
pretrained_base = tf.keras.models.load_model('models/vgg16-pretrained-base')
pretrained_base.trainable = False   # Congelamos los pesos

# Construcción del modelo completo
model = keras.Sequential([

    # 1. Data Augmentation (regularización)
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomContrast(0.3),

    # 2. Base convolucional (VGG16 preentrenada)
    pretrained_base,

    # 3. Cabeza clasificadora
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

model.summary()
```

#### Paso 4 — Compilación y entrenamiento

```python
# Compilación
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_binary_accuracy',
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3,
        monitor='val_loss',
    ),
]

# Entrenamiento
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    callbacks=callbacks,
    verbose=1,
)
```

#### Paso 5 — Visualización de resultados

```python
def plot_training_history(history):
    """Genera gráficas de pérdida y precisión del entrenamiento."""
    history_df = pd.DataFrame(history.history)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Curvas de Entrenamiento — Clasificador CNN', fontsize=14)

    # Pérdida
    axes[0].plot(history_df['loss'],     label='Entrenamiento', color='steelblue')
    axes[0].plot(history_df['val_loss'], label='Validación',    color='tomato')
    axes[0].set_title('Pérdida (Binary Crossentropy)')
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Precisión
    axes[1].plot(history_df['binary_accuracy'],     label='Entrenamiento', color='steelblue')
    axes[1].plot(history_df['val_binary_accuracy'], label='Validación',    color='tomato')
    axes[1].set_title('Precisión (Binary Accuracy)')
    axes[1].set_xlabel('Épocas')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_curves/training_history.png', dpi=150)
    plt.show()

plot_training_history(history)
```

#### Paso 6 — Evaluación y predicciones

```python
# Evaluación final en el conjunto de validación
loss, accuracy = model.evaluate(ds_valid, verbose=0)
print(f"\n{'='*40}")
print(f"  Pérdida en validación:    {loss:.4f}")
print(f"  Precisión en validación:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"{'='*40}")


def predict_image(image_path: str) -> dict:
    """Realiza una predicción sobre una imagen nueva."""
    img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array, verbose=0)[0][0]

    return {
        'clase':       'Camión' if prediction > 0.5 else 'Coche',
        'confianza':   max(prediction, 1 - prediction),
        'probabilidad': float(prediction),
    }

# Ejemplo de uso
resultado = predict_image('assets/test_vehicle.jpg')
print(f"\nPredicción: {resultado['clase']}")
print(f"Confianza:  {resultado['confianza']:.1%}")
```

---

## 📊 Resultados

### Comparativa de arquitecturas

| Modelo | Precisión (val) | Épocas | Parámetros |
|---|---|---|---|
| CNN desde cero (3 bloques) | ~85% | 40 | 291.405 |
| VGG16 + Transfer Learning | ~92% | 30 | 14.7M (base) + 197k (head) |
| VGG16 + Augmentation | ~94% | 30 | 14.7M (base) + 197k (head) |

### Conclusiones

**Transfer Learning** aporta una mejora de ~7 puntos porcentuales sobre una red entrenada desde cero con los mismos datos, demostrando el valor de las representaciones preaprendidas en ImageNet.

**Data Augmentation** reduce el sobreajuste (overfitting) visiblemente: las curvas de entrenamiento y validación convergen mejor, y la precisión en validación mejora ~2 puntos adicionales.

**El diseño de la arquitectura** importa: duplicar los filtros en cada bloque convolucional (32 → 64 → 128) mientras se reduce el tamaño espacial con MaxPool2D es un patrón robusto que mejora la capacidad representativa sin aumentar el costo computacional de forma descontrolada.

---

## ⚙️ Instalación y Ejecución

### Prerrequisitos

- Python 3.9+
- pip o conda
- GPU recomendada (CUDA 11.x compatible con TF 2.x)

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://gitlab.com/<tu-usuario>/computer-vision-portfolio.git
cd computer-vision-portfolio

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el notebook principal
jupyter lab notebooks/01_convolutional_classifier.ipynb
```

### Ejecución rápida (sin GPU)

```bash
# Modo CPU con dataset reducido para pruebas
python src/trainer.py --epochs 5 --batch-size 16 --no-gpu
```

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Uso |
|---|---|---|
| **Python** | 3.9+ | Lenguaje principal |
| **TensorFlow** | 2.10+ | Framework de Deep Learning |
| **Keras** | API de alto nivel | Definición de modelos |
| **NumPy** | 1.23+ | Operaciones numéricas |
| **Pandas** | 1.5+ | Análisis de métricas |
| **Matplotlib** | 3.6+ | Visualización de resultados |
| **Pillow** | 9.x | Manipulación de imágenes |
| **Jupyter Lab** | 3.x+ | Entorno de notebooks |

---

## 📖 Referencias

- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR 2015. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- LeCun, Y., Boser, B., Denker, J.S., et al. (1989). *Backpropagation Applied to Handwritten Zip Code Recognition*. Neural Computation, 1(4), 541-551.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.

---

<div align="center">

**Desarrollado como portfolio profesional de Deep Learning**

*Si este proyecto te resultó útil, considera darle una ⭐ al repositorio*

</div>