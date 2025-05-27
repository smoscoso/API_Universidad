import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature = None
        self.threshold = None
        self.value = None
        self.left = None
        self.right = None
        self.is_leaf = False
        
    def _gini(self, y):
        """Calcular impureza de Gini"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _best_split(self, X, y):
        """Encontrar la mejor división"""
        if len(y) < self.min_samples_split:
            return None, None
            
        best_feature, best_threshold = None, None
        best_gini = float('inf')
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                left_gini = self._gini(y[left_indices])
                right_gini = self._gini(y[right_indices])
                
                # Gini ponderado
                n_left, n_right = np.sum(left_indices), np.sum(right_indices)
                weighted_gini = (n_left * left_gini + n_right * right_gini) / len(y)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def fit(self, X, y, depth=0):
        """Entrenar el árbol de decisión"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        
        # Condiciones de parada
        if (len(y) < self.min_samples_split or 
            (self.max_depth and depth >= self.max_depth) or
            len(np.unique(y)) == 1):
            self.value = np.bincount(y).argmax()
            self.is_leaf = True
            return
        
        # Encontrar la mejor división
        self.feature, self.threshold = self._best_split(X, y)
        
        if self.feature is None:
            self.value = np.bincount(y).argmax()
            self.is_leaf = True
            return
        
        # Crear nodos hijos
        left_indices = X[:, self.feature] <= self.threshold
        right_indices = ~left_indices
        
        self.left = DecisionTree(max_depth=self.max_depth, 
                               min_samples_split=self.min_samples_split)
        self.right = DecisionTree(max_depth=self.max_depth, 
                                min_samples_split=self.min_samples_split)
        
        self.left.fit(X[left_indices], y[left_indices], depth + 1)
        self.right.fit(X[right_indices], y[right_indices], depth + 1)
        self.is_leaf = False
    
    def predict(self, X):
        """Realizar predicciones"""
        X = np.array(X, dtype=float)
        
        if self.is_leaf:
            return np.full(X.shape[0], self.value)
        
        predictions = np.zeros(X.shape[0], dtype=int)
        left_indices = X[:, self.feature] <= self.threshold
        right_indices = ~left_indices
        
        if np.any(left_indices):
            predictions[left_indices] = self.left.predict(X[left_indices])
        if np.any(right_indices):
            predictions[right_indices] = self.right.predict(X[right_indices])
        
        return predictions

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        
    def fit(self, X, y):
        """Entrenar el bosque aleatorio"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        
        self.trees = []
        np.random.seed(42)  # Para reproducibilidad
        
        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Entrenar árbol
            tree = DecisionTree(max_depth=self.max_depth, 
                              min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
    def predict(self, X):
        """Realizar predicciones usando votación mayoritaria"""
        if not self.trees:
            raise ValueError("El modelo no ha sido entrenado")
        
        X = np.array(X, dtype=float)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Votación mayoritaria
        predictions = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            unique_votes, counts = np.unique(votes, return_counts=True)
            predictions.append(unique_votes[np.argmax(counts)])
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calcular la precisión del modelo"""
        if len(X) == 0 or len(y) == 0:
            return 0.0
        
        predictions = self.predict(X)
        return np.mean(predictions == y)
