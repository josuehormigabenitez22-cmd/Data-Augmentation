"""
Funciones de visualización: kernels, extracción de features paso a
paso, e historial de entrenamiento.
"""

from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf


def show_kernel(plt, kernel, label: bool = True, digits=None, text_size: int = 28):
    """
    Dibuja un kernel de convolución como un mapa de calor con sus
    valores numéricos superpuestos.

    Parameters
    ----------
    plt : module
        matplotlib.pyplot ya importado.
    kernel : array-like
        Kernel 2D a visualizar.
    label : bool
        Si se muestran los valores numéricos sobre cada celda.
    digits : int or None
        Número de decimales a redondear (None = sin redondear).
    text_size : int
        Tamaño de fuente de las etiquetas.
    """
    kernel = np.array(kernel)
    if digits is not None:
        kernel = kernel.round(digits)

    cmap = plt.get_cmap("Blues_r")
    plt.imshow(kernel, cmap=cmap)
    rows, cols = kernel.shape
    thresh = (kernel.max() + kernel.min()) / 2

    if label:
        for i, j in product(range(rows), range(cols)):
            val = kernel[i, j]
            color = cmap(0) if val > thresh else cmap(255)
            plt.text(
                j, i, val,
                color=color, size=text_size,
                horizontalalignment="center",
                verticalalignment="center",
            )
    plt.xticks([])
    plt.yticks([])


def show_extraction(
    plt,
    image,
    kernel,
    conv_stride=1,
    conv_padding="valid",
    activation="relu",
    pool_size=2,
    pool_stride=2,
    pool_padding="same",
    figsize=(10, 10),
    subplot_shape=(2, 2),
    ops=("Input", "Filter", "Detect", "Condense"),
    gamma=1.0,
):
    """
    Aplica y visualiza los tres pasos de extracción de features
    (filtrar, detectar, condensar) sobre una imagen, usando un
    kernel dado.

    Parameters
    ----------
    plt : module
        matplotlib.pyplot ya importado.
    image : tf.Tensor
        Imagen de entrada (H, W, C).
    kernel : tf.Tensor
        Kernel 2D a aplicar como filtro convolucional.
    conv_stride, conv_padding : parámetros de la capa Conv2D.
    activation : str
        Función de activación tras la convolución.
    pool_size, pool_stride, pool_padding : parámetros de MaxPool2D.
    figsize : tuple
        Tamaño de la figura.
    subplot_shape : tuple
        Distribución de subplots (filas, columnas).
    ops : tuple[str]
        Subconjunto y orden de pasos a mostrar:
        'Input', 'Filter', 'Detect', 'Condense'.
    gamma : float
        Corrección gamma aplicada a las imágenes de Detect/Condense.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel.shape,
            strides=conv_stride,
            padding=conv_padding,
            use_bias=False,
            input_shape=image.shape,
        ),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.MaxPool2D(
            pool_size=pool_size,
            strides=pool_stride,
            padding=pool_padding,
        ),
    ])

    layer_filter, layer_detect, layer_condense = model.layers
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    layer_filter.set_weights([kernel])

    image = tf.expand_dims(image, axis=0)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image_filter = layer_filter(image)
    image_detect = layer_detect(image_filter)
    image_condense = layer_condense(image_detect)

    images = {}
    if "Input" in ops:
        images["Input"] = (image, 1.0)
    if "Filter" in ops:
        images["Filter"] = (image_filter, 1.0)
    if "Detect" in ops:
        images["Detect"] = (image_detect, gamma)
    if "Condense" in ops:
        images["Condense"] = (image_condense, gamma)

    plt.figure(figsize=figsize)
    for i, title in enumerate(ops):
        img, g = images[title]
        plt.subplot(*subplot_shape, i + 1)
        plt.imshow(tf.image.adjust_gamma(tf.squeeze(img), g))
        plt.axis("off")
        plt.title(title)


def plot_history(history):
    """
    Grafica loss y binary_accuracy (train vs validación) a partir
    del objeto History devuelto por model.fit().

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Resultado de model.fit().
    """
    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ["loss", "val_loss"]].plot(title="Loss")
    history_frame.loc[:, ["binary_accuracy", "val_binary_accuracy"]].plot(
        title="Binary Accuracy"
    )

def circle(size, val=None, r_shrink=0):
    """
    Genera una imagen sintética con un círculo (contorno) para
    pruebas de extracción de features con distintos strides.

    Parameters
    ----------
    size : tuple[int, int]
        Dimensiones (alto, ancho) de la imagen.
    val : float or None
        Valor fijo para los píxeles del círculo. Si es None,
        se usan valores aleatorios uniformes.
    r_shrink : int
        Reducción del radio respecto a la mitad de `size`.

    Returns
    -------
    np.ndarray
        Imagen 2D con el círculo dibujado.
    """
    from skimage import draw, transform

    img = np.zeros([size[0] + 1, size[1] + 1])
    rr, cc = draw.circle_perimeter(
        size[0] // 2, size[1] // 2,
        radius=size[0] // 2 - r_shrink,
        shape=[size[0] + 1, size[1] + 1],
    )
    if val is None:
        img[rr, cc] = np.random.uniform(size=img.shape)[rr, cc]
    else:
        img[rr, cc] = val
    img = transform.resize(img, size, order=0)
    return img