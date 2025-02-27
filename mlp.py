import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple

def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    num_samples = train_x.shape[0]
    for i in range(0, num_samples, batch_size):
        yield train_x[i:i + batch_size], train_y[i:i + batch_size]

class ActivationFunction(ABC):
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """

        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """"
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)

class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)

class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log(1 + np.exp(x)))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sp = np.log(1 + np.exp(x))
        tanh_sp = np.tanh(sp)
        return tanh_sp + x * (1 - tanh_sp ** 2) / (1 + np.exp(-x))

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * np.mean((y_true - y_pred) ** 2)
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Ensure both are column vectors.
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        return (y_pred - y_true) / y_true.shape[0]

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return y_pred - y_true

class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        # print(f"Initializing layer with {fan_in} inputs and {fan_out} outputs")
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        # print(f"Initialized weights with shape {self.W.shape}")
        self.b = np.zeros((1, fan_out))
        self.activations = None
        self.delta = None
        self.h = None
        self.z = None
        self.dropout_rate = 0.0
        self.dropout_mask = None

    def forward(self, h: np.ndarray):
        """Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        self.h = h
        # Apply dropout only if dropout_rate > 0.
        if self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape)
            h = h * self.dropout_mask
        self.z = np.dot(h, self.W) + self.b
        self.activations = self.activation_function.forward(self.z)
        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        dz = delta * self.activation_function.derivative(self.z)
        # Apply dropout mask only if its shape matches dz.
        if self.dropout_rate > 0 and self.dropout_mask.shape == dz.shape:
            dz = dz * self.dropout_mask
        dL_dW = np.dot(h.T, dz)
        dL_db = np.sum(dz, axis=0, keepdims=True)
        self.delta = np.dot(dz, self.W.T)
        return dL_dW, dL_db

class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """

        self.layers = layers

    def forward(self, x: np.ndarray, training: bool=True) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        h = x
        for layer in self.layers:
            # For evaluation, disable dropout.
            if not training:
                dropout_rate = layer.dropout_rate
                layer.dropout_rate = 0.0
                h = layer.forward(h)
                layer.dropout_rate = dropout_rate
            else:
                h = layer.forward(h)
        return h

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []
        delta = loss_grad
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            h_input = input_data if i == 0 else self.layers[i - 1].activations
            dw, db = layer.backward(h_input, delta)
            dl_dw_all.insert(0, dw)
            dl_db_all.insert(0, db)
            delta = layer.delta
        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, rmsprop: bool=False, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, dropout_rate: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        training_losses = []
        validation_losses = []
        # Apply dropout only on hidden layers (exclude output layer).
        for layer in self.layers[:-1]:
            layer.dropout_rate = dropout_rate
        # Ensure the output layer has no dropout.
        self.layers[-1].dropout_rate = 0.0

        if rmsprop:
            beta = 0.9
            ms_w = [np.zeros_like(layer.W) for layer in self.layers]
            ms_b = [np.zeros_like(layer.b) for layer in self.layers]
            
        for epoch in range(epochs):
            epoch_training_loss = 0.0
            indices = np.random.permutation(train_x.shape[0])
            train_x = train_x[indices]
            train_y = train_y[indices]
            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                y_pred = self.forward(batch_x)
                loss_grad = loss_func.derivative(batch_y, y_pred)
                dl_dw_all, dl_db_all = self.backward(loss_grad, batch_x)
                for j, layer in enumerate(self.layers):
                    if rmsprop:
                        ms_w[j] = beta * ms_w[j] + (1 - beta) * dl_dw_all[j] ** 2
                        ms_b[j] = beta * ms_b[j] + (1 - beta) * dl_db_all[j] ** 2
                        layer.W -= learning_rate * dl_dw_all[j] / np.sqrt(ms_w[j] + 1e-8)
                        layer.b -= learning_rate * dl_db_all[j] / np.sqrt(ms_b[j] + 1e-8)
                    else:
                        layer.W -= learning_rate * dl_dw_all[j]
                        layer.b -= learning_rate * dl_db_all[j]
                epoch_training_loss += loss_func.loss(batch_y, y_pred)
            avg_training_loss = epoch_training_loss / (train_x.shape[0] / batch_size)
            training_losses.append(avg_training_loss)
            y_val_pred = self.forward(val_x, training=False)
            val_loss = loss_func.loss(val_y, y_val_pred)
            validation_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_training_loss:.4f} - Validation Loss: {val_loss:.4f}")
        return training_losses, validation_losses
