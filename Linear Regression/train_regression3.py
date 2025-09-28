from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, stratify=y, random_state=42)

# Split the data into training and validation sets
split_idx = int(0.9 * len(X))

# Select features for this model
X_train_model = X_train[:, [1, 3]]  # Sepal width and Petal width
y_train_model = X_train[:, 2]  # Petal length

# Create and train the model
model = LinearRegression()
loss_history = model.fit(X_train_model, y_train_model, batch_size=32, max_epochs=100, patience=3)

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title('Training Loss over Time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()

# Save the model
model.save('regression3_model.npz')

# Print final weights
print("Final weights:", model.weights)
print("Final bias:", model.bias)








