from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Split the data into training and validation sets
split_idx = int(0.9 * len(X))

# Select features for this model
X_train_model = X_train[:, [0, 1]]  # Sepal length and width
y_train_model = X_train[:, 2]  # Petal length

# Train without regularization
model_no_reg = LinearRegression(batch_size=32, regularization=0.0, max_epochs=100, patience=3)
model_no_reg.fit(X_train_model, y_train_model)

# Train with L2 regularization (0.1)
model_reg = LinearRegression(batch_size=32, regularization=0.1, max_epochs=100, patience=3)
model_reg.fit(X_train_model, y_train_model)

#Display weights and biases
print("\n----- Model Parameters -----")
print("Without Regularization:")
print("  Weights:", model_no_reg.weights.flatten())
print("  Bias:", model_no_reg.bias)

print("With Regularization:")
print("  Weights:", model_reg.weights.flatten())
print("  Bias:", model_reg.bias)

# Compute and print differences
weight_diff = model_no_reg.weights - model_reg.weights
bias_diff = model_no_reg.bias - model_reg.bias

print("----- Differences (No Reg - With Reg) -----")
print("  Weight Difference:", weight_diff.flatten())
print("  Bias Difference:", bias_diff)


plt.plot(model_reg.fit(X_train_model, y_train_model), label="With Regularization", color="red")
plt.plot(model_no_reg.fit(X_train_model, y_train_model), label="Without Regularization (L2 = 0.1)", color="blue", linestyle="dashed")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title("'Training Loss over Time' (Regularized vs Non-Regularized)")
plt.legend()
plt.savefig("training_loss1.png")
plt.show()

# Save both models
model_no_reg.save("regression1_model.npz")
model_reg.save("linear_regression2_model_reg.npz")

# Print final weights
print("Final weights:", model_no_reg.weights)
print("Final bias:", model_no_reg.bias)








