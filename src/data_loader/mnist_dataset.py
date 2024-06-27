from typing import Tuple
from keras.datasets import mnist
import numpy as np


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return mnist.load_data()
