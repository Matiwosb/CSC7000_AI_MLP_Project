import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from io import StringIO

from m import *

def download_and_split_data(test_size=0.2, val_size=0.2):
    """
    Downloads and preprocesses the MPG dataset.
    Returns training, validation, and test sets along with the target scaler.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                    'acceleration', 'model_year', 'origin', 'car_name']
    
    df = pd.read_csv(url, names=column_names, delim_whitespace=True)
    df = df.replace('?', np.nan)
    
    numeric_columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 
                       'weight', 'acceleration']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    X = df[['cylinders', 'displacement', 'horsepower', 'weight', 
            'acceleration', 'model_year', 'origin']].values
    y = df['mpg'].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_y

def create_and_train_mlp(X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.001, batch_size=32, hidden_layers=[64, 128], activation=Relu(), loss_function=SquaredError(), rmsprop=False, dropout_rate=0.0):
    input_dim = X_train.shape[1]
    layers = []
    prev_dim = input_dim

    for dim in hidden_layers:
        layers.append(Layer(prev_dim, dim, activation))
        prev_dim = dim

    layers.append(Layer(prev_dim, 1, Linear()))
    mlp = MultilayerPerceptron(layers)

    train_losses, val_losses = mlp.train(
        X_train, y_train, X_val, y_val,
        loss_func=loss_function,
        rmsprop=rmsprop,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        dropout_rate=dropout_rate
    )
    return mlp, train_losses, val_losses

if __name__ == "__main__":
    # Download and preprocess data (using 15% for test, ~17.6% of remaining for validation)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_y = download_and_split_data(test_size=0.15, val_size=0.176)

    # Train the MLP
    mlp, train_losses, val_losses = create_and_train_mlp(
        X_train, y_train, X_val, y_val,
        epochs=30,
        learning_rate=0.01,
        batch_size=64,
        hidden_layers=[64, 128, 256, 512],
        activation=Linear(),  # Using Linear activation throughout here.
        dropout_rate=0.0,
        rmsprop=False 
    )

    # Evaluate on the test set
    y_pred_test = mlp.forward(X_test)
    test_loss = SquaredError().loss(y_test, y_pred_test)
    print(f"Test Loss: {test_loss}")

    # Plot training and validation loss curves
    epochs_range = range(1, len(train_losses)+1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Training Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (0.5 x MSE)")
    plt.title("MPG Regression Loss Curves")
    plt.legend()
    plt.show()

    # Inverse-transform the predictions and true values to original MPG scale
    y_pred_orig = scaler_y.inverse_transform(y_pred_test)
    y_test_orig = scaler_y.inverse_transform(y_test)

    # Select 10 different samples from testing randomly
    np.random.seed(42)
    sample_indices = np.random.choice(range(X_test.shape[0]), size=10, replace=False)
    sample_preds = y_pred_orig[sample_indices].flatten()
    sample_trues = y_test_orig[sample_indices].flatten()

    # Create and display a table with predicted MPG vs. true MPG
    table = pd.DataFrame({
        "Predicted MPG": sample_preds,
        "True MPG": sample_trues
    })
    print("\nSample Predictions vs. True MPG:")
    print(table)
