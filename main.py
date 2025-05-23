from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from app.models.knn import KNN
from app.models.random_forest import RandomForest
from app.models.regression_tree import RegressionTree
from pydantic import BaseModel
from collections import defaultdict
import uvicorn

app = FastAPI()

# Cargar dataset
df = pd.read_csv('dataset.csv')

# Codificación manual
bssid_mapping = defaultdict(lambda: len(bssid_mapping))
_ = [bssid_mapping[bssid] for bssid in df['BSSID']]

location_mapping = defaultdict(lambda: len(location_mapping))
_ = [location_mapping[loc] for loc in df['Ubicación']]

# Normalizar señal
signals = df['Intensidad_señal(dBm)'].values
mean_signal = np.mean(signals)
std_signal = np.std(signals)
normalized_signals = (signals - mean_signal) / std_signal

# Preparar datos
X = np.array([[bssid_mapping[bssid], signal] 
             for bssid, signal in zip(df['BSSID'], normalized_signals)])
y = np.array([location_mapping[loc] for loc in df['Ubicación']], dtype=int)

# Entrenar modelos
models = {
    'KNN': KNN(k=5),
    'RandomForest': RandomForest(n_trees=15, max_depth=7),
    'RegressionTree': RegressionTree(max_depth=7),
}

for name, model in models.items():
    model.fit(X, y)

# Seleccionar mejor modelo
best_model_name = max(models, key=lambda name: models[name].score(X, y))
best_model = models[best_model_name]

class WiFiData(BaseModel):
    bssid: str
    signal: int

@app.post("/predecir-ubicacion")
async def predict_location(data: WiFiData):
    try:
        encoded_bssid = bssid_mapping.get(data.bssid, -1)
        if encoded_bssid == -1:
            raise ValueError("BSSID no reconocido")
            
        normalized_signal = (data.signal - mean_signal) / std_signal
        prediction_encoded = best_model.predict([[encoded_bssid, normalized_signal]])
        
        inverse_location = {v: k for k, v in location_mapping.items()}
        precision = best_model.score(X, y) * 100  # Calcula la precisión
        
        return {
            "ubicacion": inverse_location[prediction_encoded[0]],
            "precision": round(precision, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)