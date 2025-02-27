import numpy as np
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mlp import *

def download_and_split_data(val_size=0.2):
    # Download MNIST dataset if not present
    if not os.path.exists("mnist.npz"):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print("Downloading MNIST dataset...")
        response = requests.get(url)
        with open("mnist.npz", "wb") as f:
            f.write(response.content)
    
    # Load dataset from npz file
    with np.load("mnist.npz") as data:
        X_train = data['x_train']
        y_train = data['y_train']
        X_test = data['x_test']
        y_test = data['y_test']
    
    # Flatten images: (N,28,28) -> (N,784) and normalize
    X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

    # One-hot encode labels (10 classes)
    num_classes = 10
    y_train_oh = np.eye(num_classes)[y_train]
    y_test_oh = np.eye(num_classes)[y_test]

    # Split training set into training and validation
    X_train, X_val, y_train_oh, y_val_oh = train_test_split(
        X_train, y_train_oh, test_size=val_size, random_state=42)
    
    return X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_oh, y_test

def create_and_train_mlp(X_train, y_train, X_val, y_val, epochs=20, learning_rate=0.001, batch_size=32,
                         hidden_layers=[512, 256, 128], dropout_rate=0.2):
    input_dim = X_train.shape[1]
    layers = []
    prev_dim = input_dim

    # Build hidden layers with ReLU activation
    for dim in hidden_layers:
        layers.append(Layer(prev_dim, dim, Relu()))
        prev_dim = dim

    # Build output layer: 10 neurons with Softmax activation for classification
    layers.append(Layer(prev_dim, 10, Softmax()))
    
    mlp = MultilayerPerceptron(layers)
    
    loss_function = CrossEntropy()
    train_losses, val_losses = mlp.train(
        X_train, y_train, X_val, y_val,
        loss_func=loss_function,
        rmsprop=False,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        dropout_rate=dropout_rate
    )
    return mlp, train_losses, val_losses

def evaluate_accuracy(mlp, X_test, y_test_oh, y_test_labels):
    y_pred = mlp.forward(X_test, training=False)
    pred_labels = np.argmax(y_pred, axis=1)
    accuracy = np.mean(pred_labels == y_test_labels)
    return accuracy, pred_labels

if __name__ == "__main__":
    # Download and split data; y_test_labels holds the original labels.
    X_train, y_train, X_val, y_val, X_test, y_test_oh, y_test_labels = download_and_split_data(val_size=0.2)
    
    # Create and train the MLP
    mlp, train_losses, val_losses = create_and_train_mlp(
        X_train, y_train, X_val, y_val,
        epochs=20,
        learning_rate=0.001,
        batch_size=64,
        hidden_layers=[512, 256, 128],
        dropout_rate=0.0  # Set dropout to 0 if you want to test without dropout
    )
    
    # Evaluate overall test accuracy
    test_accuracy, pred_labels = evaluate_accuracy(mlp, X_test, y_test_oh, y_test_labels)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Plot the loss curves
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.show()
    
    # For each class (0-9), select one sample from the test set and display the image with predicted class.
    selected_indices = []
    for digit in range(10):
        indices = np.where(y_test_labels == digit)[0]
        if len(indices) > 0:
            selected_indices.append(indices[0])
    
    plt.figure(figsize=(12,4))
    for i, idx in enumerate(selected_indices):
        img = X_test[idx].reshape(28, 28)  # Reshape flattened vector back to 28x28 image
        true_label = y_test_labels[idx]
        predicted_label = pred_labels[idx]
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
