import numpy as np

class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1):
        # Initialize the decision tree with hyperparameters
        self.criterion = criterion  # Impurity measure (gini, entropy, or misclassification)
        self.max_depth = max_depth  # Maximum depth of the tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split an internal node
        self.min_samples_leaf = min_samples_leaf  # Minimum samples required to be at a leaf node
        self.root = None  # Root node of the tree

    class Node:
        # Node class to represent each node in the decision tree
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature  # Feature index used for splitting
            self.threshold = threshold  # Threshold value for the split
            self.left = left  # Left child node
            self.right = right  # Right child node
            self.value = value  # Predicted class (for leaf nodes)

    def fit(self, X, y):
        # Fit the decision tree to the training data
        self.root = self._build_tree(X, y)

    def predict(self, X):
        # Predict class labels for samples in X
        return np.apply_along_axis(lambda x: self._predict_one(x, self.root), axis=1, arr=X)

    def _build_tree(self, X, y, depth=0):
        # Recursively build the decision tree
        num_samples, num_features = X.shape

        # Check if we should stop splitting
        if self._should_stop_splitting(depth, num_samples, y):
            return self.Node(value=self._majority_class(y))

        # Find the best split
        feature, threshold = self._find_best_split(X, y)
        if feature is None:
            return self.Node(value=self._majority_class(y))

        # Split the data and build left and right subtrees
        left_mask = X[:, feature] <= threshold
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return self.Node(feature, threshold, left_node, right_node)

    def _should_stop_splitting(self, depth, num_samples, y):
        # Check if we should stop splitting based on stopping criteria
        return (self.max_depth is not None and depth >= self.max_depth) or \
               (num_samples < self.min_samples_split) or \
               (len(np.unique(y)) == 1)

    def _find_best_split(self, X, y):
        # Find the best feature and threshold for splitting
        num_samples, num_features = X.shape
        best_feature, best_threshold, best_impurity = None, None, float('inf')

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                if self._is_valid_split(left_mask):
                    impurity = self._calculate_impurity(y[left_mask], y[~left_mask])
                    if impurity < best_impurity:
                        best_feature, best_threshold, best_impurity = feature, threshold, impurity

        return best_feature, best_threshold

    def _is_valid_split(self, mask):
        # Check if a split is valid based on min_samples_leaf
        left_count = np.sum(mask)
        right_count = len(mask) - left_count
        return left_count >= self.min_samples_leaf and right_count >= self.min_samples_leaf

    def _calculate_impurity(self, left_y, right_y):
        # Calculate the impurity of a split based on the chosen criterion
        impurity_functions = {
            "gini": self._gini,
            "entropy": self._entropy,
            "misclassification": self._misclassification
        }
        
        if self.criterion not in impurity_functions:
            raise ValueError("Invalid criterion. Choose from 'gini', 'entropy', or 'misclassification'.")
        
        return impurity_functions[self.criterion](left_y, right_y)

    def _gini(self, left_y, right_y):
        # Calculate Gini impurity
        return self._weighted_impurity(left_y, right_y, lambda y: 1 - np.sum((np.bincount(y) / len(y))**2))

    def _entropy(self, left_y, right_y):
        # Calculate Entropy
        return self._weighted_impurity(left_y, right_y, lambda y: -np.sum((np.bincount(y) / len(y)) * np.log2(np.bincount(y) / len(y) + 1e-9)))

    def _misclassification(self, left_y, right_y):
        # Calculate Misclassification error
        return self._weighted_impurity(left_y, right_y, lambda y: 1 - np.max(np.bincount(y)) / len(y))

    def _weighted_impurity(self, left_y, right_y, impurity_func):
        # Calculate weighted impurity of a split
        total = len(left_y) + len(right_y)
        return (len(left_y) / total) * impurity_func(left_y) + (len(right_y) / total) * impurity_func(right_y)

    def _majority_class(self, y):
        # Determine the majority class in a node
        return np.argmax(np.bincount(y))

    def _predict_one(self, sample, node):
        # Predict the class for a single sample
        while node.value is None:
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
