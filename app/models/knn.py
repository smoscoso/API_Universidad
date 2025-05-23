import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X.astype(float)
        self.y_train = y.astype(int)
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argpartition(distances, self.k)[:self.k]
            k_labels = self.y_train[k_indices]
            counts = np.bincount(k_labels)
            predictions.append(np.argmax(counts))
        return np.array(predictions)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)