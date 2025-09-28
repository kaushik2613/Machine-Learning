import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# # Load the Iris dataset
# iris = datasets.load_iris()
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df.head())
# sns.jointplot(x = 'sepal length (cm)', y = 'sepal width (cm)', data = df)
# plt.show()
# sns.pairplot(df, kind='scatter', plot_kws={'alpha':0.4}, diag_kws={'alpha':0.55, 'bins':40})
# plt.show()


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """


        # Split data into training and validation sets (90% train, 10% validation)
        validation_split = 0.1
        validation_size = int(len(X) * validation_split)

        X_train, X_val = X[:-validation_size], X[-validation_size:]
        y_train, y_val = y[:-validation_size], y[-validation_size:]
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience


        # TODO: Initialize the weights and bias based on the shape of X and y.
        self.weights = None
        self.bias = None
        # Initialize model parameters based on shape of X and y
        # n_features = X_train.shape[1]  # Number of input features
        n_outputs, n_features = X_train.shape
        # n_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1  # Number of target variables

        # Initialize weights and biases
        self.weights = np.zeros(n_features)  # Small random values
        # self.weights = np.random.randn(n_features, n_outputs) * 0.01
        # self.biases = np.zeros((1, n_outputs))  # Bias initialized to 0
        self.bias = 0
        # self.bias = np.zeros((1,))
        # self.weights = np.random.randn(n_features, 1) * 0.01


        # Initialize variables for early stopping
        # best_weights = self.weights.copy()
        # best_biases = self.biases.copy()
        best_weights = []
        best_biases = []
        best_val_loss = float('inf')
        patience_counter = 0
        overall_loss = []

        # TODO: Implement the training loop.
        # Training Loop
        for epoch in range(max_epochs):
            # Shuffle the data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]


            # Mini-batch gradient descent
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                # Forward pass
                predictions = np.dot(X_batch, self.weights) + self.bias

                # Compute the error
                error = predictions - y_batch

                # Compute gradients for weights and biases
                grad_weights = (2 / len(X_batch)) * np.dot(X_batch.T, error) + 2 * regularization * self.weights
                grad_biases = (2 / len(X_batch)) * np.sum(error, axis=0, keepdims=True)

                # Update parameters using gradient descent
                self.weights -= 0.01 * grad_weights  # learning rate = 0.01
                self.bias -= 0.01 * grad_biases  # learning rate = 0.01

            # Evaluate the loss on the validation set after each epoch
            val_predictions = np.dot(X_val, self.weights) + self.bias
            val_loss = np.mean((val_predictions - y_val) ** 2)  # Mean Squared Error
            overall_loss.append(val_loss)

            print(f"Epoch {epoch + 1}/{max_epochs}, Validation Loss: {val_loss}")

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.weights.copy()
                best_biases = self.bias.copy()
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Set the model parameters to the best found during training
        self.weights = best_weights
        self.bias = best_biases
        return overall_loss



    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        X = np.array(X,dtype=float)
        # print(X)
        print(self.weights['weights'])
        print(self.weights['bias'])

        return np.dot(X, self.weights['weights']) + self.weights['bias']

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        # Get predictions from the model
        predicted_values = self.predict(X)

        # Calculate the Mean Squared Error
        mse = np.mean((y - predicted_values) ** 2)
        return mse

    def save(self, file_path):
        """
        Saves the model parameters (weights) to a file.
        Args:
            file_path (str): Path to the file where model parameters will be saved.
        """
        if self.weights is None:
            raise ValueError("Model weights are not set. Cannot save.")
        np.savez(file_path, weights = self.weights, bias = self.bias)
        print(f"Model weights saved to {file_path}")

    def load(self, file_path):
        """
        Loads the model parameters (weights) from a file.
        Args:
            file_path (str): Path to the file from which model parameters will be loaded.
        """
        try:
            self.weights = np.load(file_path)
            print(f"Model weights loaded from {file_path}")
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
