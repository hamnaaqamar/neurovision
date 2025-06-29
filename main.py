from data.dataloader import load_mnist
from core.layers import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy
from core.optimizer import Optimizer_Adam
import numpy as np

X_train, y_train, X_test, y_test = load_mnist()

y_train = np.array(y_train)
y_test = np.array(y_test)

dense1 = Layer_Dense(784, 128)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-5)

if len(y_train.shape) == 2:
    y_labels = np.argmax(y_train, axis=1)
else:
    y_labels = y_train

for epoch in range(10001):
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_train)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y_labels)

    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y_train)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss_activation.forward(dense2.output, y_test)

test_preds = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test

test_acc = np.mean(test_preds == y_test_labels)
print(f"Final Test Accuracy: {test_acc:.3f}")
