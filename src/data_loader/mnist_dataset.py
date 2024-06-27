from typing import Tuple
from keras.datasets import mnist
import numpy as np


def load_mnist_data() -> (
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    return (x_train, y_train), (x_test, y_test)
