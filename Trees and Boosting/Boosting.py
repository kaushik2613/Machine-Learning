import numpy as np
from DecesionTree import DecisionTree

class AdaBoost:
    def __init__(self, weak_learner, num_learners=150, learning_rate=0.3):
        self.n_estimators = num_learners
        self.weak_learner = weak_learner
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape

        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples

        estimator_count = 0
        while estimator_count < self.n_estimators:
            estimator = self.weak_learner(max_depth=1)
            estimator.fit(X, y)

            predictions = estimator.predict(X)
            incorrect = predictions != y

            estimator_error = np.sum(sample_weight * incorrect) / np.sum(sample_weight)

            if 0 < estimator_error < 0.5:
                estimator_weight = self.learning_rate * np.log((1 - estimator_error) / (estimator_error + 1e-10))

                sample_weight *= np.exp(estimator_weight * incorrect)
                sample_weight /= np.sum(sample_weight)

                self.estimators.append(estimator)
                self.estimator_weights.append(estimator_weight)

            estimator_count += 1
        # Stop early if perfect accuracy is achieved (If a perfect fit is achieved before reaching this number, the predict method should stop early.)
            if np.sum(incorrect) == 0:
                print("Perfect fit achieved at iteration", estimator_count)
                break

    def predict(self, X):
        n_samples = X.shape[0]
        ensemble_predictions = np.zeros(n_samples)

        estimator_index = 0
        while estimator_index < len(self.estimators):
            weight = self.estimator_weights[estimator_index]
            estimator = self.estimators[estimator_index]
            ensemble_predictions += weight * estimator.predict(X)
            estimator_index += 1

        return np.sign(ensemble_predictions)
