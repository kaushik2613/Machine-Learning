import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import os


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n, d = X.shape
        # Initialize weights and bias
        self.W = np.zeros(d)
        self.b = 0

        # Gradient descent loop
        for _ in range(self.epochs):
            model = np.dot(X, self.W) + self.b
            y_pred = self.sigmoid(model)

            # Compute gradients
            dw = (1 / n) * np.dot(X.T, (y_pred - y))
            db = (1 / n) * np.sum(y_pred - y)

            # Update parameters
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db


    def predict(self, X):
        """
        Predicts the class labels for the given input data X.
        Returns the predicted class labels as 0 or 1.
        """
        model = np.dot(X, self.W) + self.b
        return (self.sigmoid(model) >= 0.5).astype(int)

    def score(self, X, y):
        """
        Returns the accuracy of the model by comparing predictions to true labels.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)  # Compute accuracy as the fraction of correct predictions
        return accuracy

    def save(self, file_path):
        """
        Saves the model parameters (weights) to a file.
        Args:
            file_path (str): Path to the file where model parameters will be saved.
        """
        # if self.W is None:
        #     raise ValueError("Model weights are not set. Cannot save.")
        # np.save(file_path, self.W, self.b)
        # print(f"Model weights saved to {file_path}")
        np.savez(file_path, weights=self.W, bias=self.b)

    def load(self, file_path):
        """
        Load the model parameters from a file.

        Parameters:
        file_path (str): Path to load the model parameters from
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")

        # Load the saved arrays
        loaded = np.load(file_path)
        self.W = loaded['weights']
        self.b = loaded['bias']
        print(f"Model parameters loaded from {file_path}")