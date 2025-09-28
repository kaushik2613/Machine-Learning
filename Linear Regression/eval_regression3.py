import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression

# Load the Iris dataset
data = load_iris()
X = data.data  # Features: sepal length, sepal width, petal length, petal width
y = X[:, 2]  # We will predict petal length

# Select features (sepal width, petal width)
X = X[:, [1, 3]]

# Split the data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model=LinearRegression()
model.load('regression3_model.npz')
mse = model.score(X_train,y_train)
print("Mean Squared Error test",mse)
print(f"Mean Squared Error (MSE) for Regression 1: {mse:.4f}")
