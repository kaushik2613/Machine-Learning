import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data  # All features (sepal length/width, petal length/width)
y = (data.target == 0).astype(int)  # Class 0 vs. other classes (binary classification)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize and train the model
clf = LogisticRegression(learning_rate=0.1)
clf.load('train_classify3_model.npz')
clf.fit(X_train_pca, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set (All features reduced to 2D): {accuracy:.4f}")

# Visualize decision boundaries
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train_pca, y_train, clf=clf, legend=2)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Logistic Regression Decision Boundary (All Features reduced to 2D)')
plt.show()