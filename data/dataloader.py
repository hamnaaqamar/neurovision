from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist['data'] / 255.0
    y = mnist['target'].astype(int)
    return train_test_split(X, y, test_size=0.2)
