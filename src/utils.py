"""
Utilidades generales del proyecto.
"""

import os
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 31415) -> None:
    """
    Fija las semillas de numpy, tensorflow y variables de entorno
    para asegurar reproducibilidad entre ejecuciones.

    Parameters
    ----------
    seed : int
        Semilla a utilizar.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def set_matplotlib_defaults(plt) -> None:
    """
    Configura los valores por defecto de Matplotlib usados en todo
    el proyecto (tipografía, tamaños, colormap).

    Parameters
    ----------
    plt : module
        El módulo matplotlib.pyplot ya importado en el notebook.
    """
    plt.rc("figure", autolayout=True)
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=18,
        titlepad=10,
    )
    plt.rc("image", cmap="magma")