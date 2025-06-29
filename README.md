
# NeuroVision: Neural Network from Scratch for Digit Recognition

NeuroVision is a custom-built neural network framework implemented entirely with **NumPy**, showcasing the full forward and backward pass mechanics — from dense layers to softmax activation — trained to classify handwritten digits from the **MNIST** dataset.

It includes:

- A modular **deep learning engine** (like a mini PyTorch/Keras)
- **From-scratch training** using backpropagation, Adam optimizer, and custom loss functions
- A **Streamlit app** with drawing canvas to predict digits in real-time
- Support for **model weight saving/loading** via `.npy` files


## Project Highlights

| Module        | Description |
|---------------|-------------|
| `core/`       | Custom neural network layers, activation functions, optimizer, and loss functions |
| `data/`       | MNIST dataset loader |
| `weights/`    | Folder to store trained model weights |
| `main.py`     | Training script for the model |
| `app.py`      | Interactive web app built using Streamlit |
| `README.md`   | You’re here! |

---

## How it Works

### Neural Network Architecture

- **Layer 1:** Dense (784 → 128)
- **Activation:** ReLU
- **Layer 2:** Dense (128 → 10)
- **Activation + Loss:** Softmax + Categorical Cross-Entropy
- **Optimizer:** Adam with learning rate decay


  ![neurovision](https://github.com/user-attachments/assets/20febdb9-7479-4305-8493-e2002d097725)



## Training the Model

```bash
python main.py


