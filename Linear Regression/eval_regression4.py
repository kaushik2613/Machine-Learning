# eval_regression1.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression
import pickle

# Load the Iris dataset
data = load_iris()
X = data.data  # Features: sepal length, sepal width, petal length, petal width
y = X[:, 1]  # We will predict Sepal Width

# Select features (Sepal length, Petal width)
X = X[:, [0, 3]]

# Split the data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Load the trained model parameters from file (pickle)
# with open('regression1_model.npz.npy', 'rb') as f:
#     weights = pickle.load(f)
#
# # Define the prediction function
# def predict(X, weights):
#     return X.dot(weights)
#
# # Make predictions on the test set
# y_pred = predict(X_test, weights)
#
# # Calculate the Mean Squared Error
# mse = np.mean((y_test - y_pred) ** 2)


model=LinearRegression()
model.load('regression4_model.npz')
mse = model.score(X_train,y_train)
print("Mean Squared Error test",mse)
print(f"Mean Squared Error (MSE) for Regression 1: {mse:.4f}")
