import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

def load_mnist():
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)  

    X_train, X_test = X[:56000], X[56000:]
    y_train, y_test = y[:56000], y[56000:]

    return X_train, y_train, X_test, y_test
