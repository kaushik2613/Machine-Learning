import numpy as np
from collections import Counter
from DecesionTree import DecisionTree

class RandomForest:
    def __init__(self, num_trees=10, min_features=1, max_features=None):
        """
        Parameters:
        - num_trees: Number of trees in the forest.
        - min_features: Minimum number of features to consider when splitting.
        - max_features: Maximum number of features to consider when splitting (default is sqrt(num_features)).
        """
        self.num_trees = num_trees
        self.min_features = min_features
        self.max_features = max_features
        self.forest = []  # List to store trained trees and their feature subsets

    class TreeWrapper:
        def __init__(self, tree, feature_subset):
            self.tree = tree
            self.feature_subset = feature_subset

    def fit(self, X, y):
        """
        Train the random forest using bootstrap sampling and random feature selection.
        
        Parameters:
        - X: Training data of shape (n_samples, n_features).
        - y: Labels of shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Set default max_features if not provided (sqrt(num_features) is common in classification tasks)
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        # Ensure min_features <= max_features <= n_features
        if self.min_features > n_features or self.max_features > n_features:
            raise ValueError("min_features and max_features must be <= the total number of features.")

        # Train each tree in the forest
        for _ in range(self.num_trees):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[bootstrap_indices]
            y_sample = y[bootstrap_indices]

            # Randomly select a subset of features for this tree
            num_selected_features = np.random.randint(self.min_features, self.max_features + 1)
            selected_features = np.random.choice(n_features, size=num_selected_features, replace=False)

            # Train a decision tree on the selected data and features
            tree = DecisionTree()
            tree.fit(X_sample[:, selected_features], y_sample)

            # Store the trained tree and its feature subset
            self.forest.append(self.TreeWrapper(tree, selected_features))

    def predict(self, X):
        """
        Predict class labels for input samples using majority voting.
        
        Parameters:
        - X: Input data of shape (n_samples, n_features).
        
        Returns:
        - Predicted labels of shape (n_samples,).
        """
        # Collect predictions from all trees in the forest
        all_tree_predictions = np.array([self._total_tree_predict(tree_wrapper, X) for tree_wrapper in self.forest])

        # Perform majority voting across all trees for each sample
        return np.apply_along_axis(self._pre_majority_vote, axis=0, arr=all_tree_predictions)
    
    def _rep_tree(self, X, y):
        # Step 1: Bootstrap sampling
        n_samples, n_features = X.shape
        bootstrap_indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        X_sample = X[bootstrap_indices]
        y_sample = y[bootstrap_indices]

        # Step 2: Random feature selection
        num_features_to_select = np.random.randint(self.min_features, n_features + 1)
        selected_features = np.random.choice(np.arange(n_features), size=num_features_to_select, replace=False)

        # Step 3: Create and train the decision tree
        tree = DecisionTree()
        tree.fit(X_sample[:, selected_features], y_sample)

        # Step 4: Return a wrapper with the trained tree and selected features
        return self.TreeWrapper(tree=tree, selected_features=selected_features)


    def _total_tree_predict(self, tree_wrapper, X):
        """
        Predict using a single decision tree and its selected feature subset.
        
        Parameters:
        - tree_wrapper: A TreeWrapper object containing the trained tree and feature subset.
        - X: Input data of shape (n_samples, n_features).
        
        Returns:
        - Predictions from the tree for all samples.
        """
        return tree_wrapper.tree.predict(X[:, tree_wrapper.feature_subset])

    def _pre_majority_vote(self, predictions):
        """
        Perform majority voting to determine the final prediction for a sample.
        
        Parameters:
        - predictions: Array of predictions from all trees for a single sample.
        
        Returns:
        - The class with the most votes.
        """
        counter = Counter(predictions)
        return counter.most_common(1)[0][0]
