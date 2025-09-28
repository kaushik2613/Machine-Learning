import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data[:, 2:4]  # Petal length and petal width
y = (data.target == 0).astype(int)  # Class 0 vs. other classes (binary classification)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and train the model
model=LogisticRegression()
model.load('train_classify1_model.npz')
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set (Petal features): {accuracy:.4f}")

# Visualize decision boundaries
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=model, legend=2)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Logistic Regression Decision Boundary (Petal Features)')
plt.show()