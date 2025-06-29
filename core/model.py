import numpy as np
from core.layers import Layer_Dense, Activation_ReLU, Activation_Softmax

class TrainedModel:
    def __init__(self):

        self.dense1 = Layer_Dense(784, 128)
        self.activation1 = Activation_ReLU()
        self.dense2 = Layer_Dense(128, 10)
        self.softmax = Activation_Softmax()


        self.dense1.weights = np.load("weights/weights_dense1.npy")
        self.dense1.biases = np.load("weights/biases_dense1.npy")
        self.dense2.weights = np.load("weights/weights_dense2.npy")
        self.dense2.biases = np.load("weights/biases_dense2.npy")

    def predict(self, x, return_confidence=False):

        self.dense1.forward(x[np.newaxis, :]) 
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.softmax.forward(self.dense2.output)

        prediction = int(np.argmax(self.softmax.output))
        if return_confidence:
            return prediction, self.softmax.output[0]
        return prediction
