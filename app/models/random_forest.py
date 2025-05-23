import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.sum((counts / len(y)) ** 2)
    
    def _best_split(self, X, y):
        best = {'feature': None, 'threshold': None, 'gini': float('inf')}
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                gini = (len(left) * self._gini(left) + len(right) * self._gini(right)) / len(y)
                if gini < best['gini']:
                    best = {'feature': feature, 'threshold': threshold, 'gini': gini}
        return best['feature'], best['threshold']
    
    def fit(self, X, y, depth=0):
        self.X = X.astype(float)
        self.y = y.astype(int)
        
        if (self.max_depth and depth >= self.max_depth) or len(np.unique(y)) == 1:
            self.value = np.bincount(y).argmax()
            self.is_leaf = True
            return
        
        self.feature, self.threshold = self._best_split(X, y)
        if self.feature is None:
            self.value = np.bincount(y).argmax()
            self.is_leaf = True
            return
        
        left_idx = X[:, self.feature] <= self.threshold
        self.left = DecisionTree(self.max_depth)
        self.right = DecisionTree(self.max_depth)
        self.left.fit(X[left_idx], y[left_idx], depth + 1)
        self.right.fit(X[~left_idx], y[~left_idx], depth + 1)
        self.is_leaf = False
    
    def predict(self, X):
        if self.is_leaf:
            return np.full(X.shape[0], self.value)
        
        left_idx = X[:, self.feature] <= self.threshold
        preds = np.empty(X.shape[0], dtype=int)
        preds[left_idx] = self.left.predict(X[left_idx])
        preds[~left_idx] = self.right.predict(X[~left_idx])
        return preds

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTree(self.max_depth)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
    
    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, preds)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)