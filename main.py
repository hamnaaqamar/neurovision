from data.dataloader import load_mnist
from core.layers import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy
from core.optimizer import Optimizer_Adam

X_train, y_train, X_test, y_test = load_mnist()

dense1 = Layer_Dense(784, 128)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-5)

# Training loop...
