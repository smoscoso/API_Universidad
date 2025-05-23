import numpy as np

class RegressionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        
    def _mse(self, y):
        return np.mean((y - np.mean(y))**2)
    
    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_mse = float('inf')
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                if np.sum(left_indices) == 0 or np.sum(~left_indices) == 0:
                    continue
                current_mse = self._mse(y[left_indices]) + self._mse(y[~left_indices])
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def fit(self, X, y, depth=0):
        if len(y) < 2 or (self.max_depth and depth == self.max_depth):
            self.value = np.mean(y)
            return
        
        self.feature, self.threshold = self._best_split(X, y)
        # Check if no valid split was found
        if self.feature is None or self.threshold is None:
            self.value = np.mean(y)
            return
        
        left_indices = X[:, self.feature] <= self.threshold
        self.left = RegressionTree(max_depth=self.max_depth)
        self.right = RegressionTree(max_depth=self.max_depth)
        self.left.fit(X[left_indices], y[left_indices], depth+1)
        self.right.fit(X[~left_indices], y[~left_indices], depth+1)
    
    def predict(self, X):
        if hasattr(self, 'value'):
            return np.array([self.value] * X.shape[0])
        left = X[:, self.feature] <= self.threshold
        return np.where(left, self.left.predict(X), self.right.predict(X))
    
    def score(self, X, y):
        predictions = self.predict(X)
        return 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))