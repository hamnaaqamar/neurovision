
from data.dataloader import load_mnist
from core.layers import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy
from core.optimizer import Optimizer_Adam
import numpy as np

X_train, y_train, X_test, y_test = load_mnist()

dense1 = Layer_Dense(784, 128)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-5)

for epoch in range(10001):
    dense1.forward(X)
    
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    
    loss = loss_activation.forward(dense2.output, y)
    
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()