import numpy as np

class RegressionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature = None
        self.threshold = None
        self.value = None
        self.left = None
        self.right = None
        
    def _mse(self, y):
        """Calcular el error cuadrático medio"""
        if len(y) == 0:
            return 0
        return np.var(y)
    
    def _best_split(self, X, y):
        """Encontrar la mejor división"""
        if len(y) < self.min_samples_split:
            return None, None
            
        best_feature, best_threshold = None, None
        best_mse = float('inf')
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                left_mse = self._mse(y[left_indices])
                right_mse = self._mse(y[right_indices])
                
                # MSE ponderado
                n_left, n_right = np.sum(left_indices), np.sum(right_indices)
                weighted_mse = (n_left * left_mse + n_right * right_mse) / len(y)
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def fit(self, X, y, depth=0):
        """Entrenar el árbol de regresión"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Condiciones de parada
        if (len(y) < self.min_samples_split or 
            (self.max_depth and depth >= self.max_depth) or
            len(np.unique(y)) == 1):
            self.value = np.mean(y)
            return
        
        # Encontrar la mejor división
        self.feature, self.threshold = self._best_split(X, y)
        
        if self.feature is None:
            self.value = np.mean(y)
            return
        
        # Crear nodos hijos
        left_indices = X[:, self.feature] <= self.threshold
        right_indices = ~left_indices
        
        self.left = RegressionTree(max_depth=self.max_depth, 
                                 min_samples_split=self.min_samples_split)
        self.right = RegressionTree(max_depth=self.max_depth, 
                                  min_samples_split=self.min_samples_split)
        
        self.left.fit(X[left_indices], y[left_indices], depth + 1)
        self.right.fit(X[right_indices], y[right_indices], depth + 1)
    
    def predict(self, X):
        """Realizar predicciones"""
        X = np.array(X, dtype=float)
        
        if self.value is not None:
            return np.full(X.shape[0], self.value)
        
        predictions = np.zeros(X.shape[0])
        left_indices = X[:, self.feature] <= self.threshold
        right_indices = ~left_indices
        
        if np.any(left_indices):
            predictions[left_indices] = self.left.predict(X[left_indices])
        if np.any(right_indices):
            predictions[right_indices] = self.right.predict(X[right_indices])
        
        return predictions
    
    def score(self, X, y):
        """Calcular R² score"""
        if len(y) == 0:
            return 0.0
            
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return max(0.0, 1 - (ss_res / ss_tot))
