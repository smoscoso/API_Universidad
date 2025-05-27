import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Entrenar el modelo KNN"""
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=int)
        
        # Ajustar k si es mayor que el número de muestras
        if self.k > len(self.X_train):
            self.k = len(self.X_train)
    
    def predict(self, X):
        """Realizar predicciones"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        X = np.array(X, dtype=float)
        predictions = []
        
        for x in X:
            # Calcular distancias euclidianas
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Obtener los k vecinos más cercanos
            k_indices = np.argpartition(distances, min(self.k, len(distances)-1))[:self.k]
            k_labels = self.y_train[k_indices]
            
            # Votar por la clase más común
            unique_labels, counts = np.unique(k_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calcular la precisión del modelo"""
        if len(X) == 0 or len(y) == 0:
            return 0.0
        
        predictions = self.predict(X)
        return np.mean(predictions == y)
