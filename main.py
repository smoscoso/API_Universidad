from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pydantic import BaseModel
from collections import defaultdict
import uvicorn
import logging
# Importar modelos
from knn import KNN
from random_forest import RandomForest
from regression_tree import RegressionTree

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WiFi Location Predictor", version="1.0.0")

# Agregar CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar dataset
try:
    df = pd.read_csv('dataset.csv')
    logger.info(f"Dataset cargado con {len(df)} registros")
except FileNotFoundError:
    logger.error("No se encontró el archivo dataset.csv")
    # Crear datos de ejemplo para testing
    df = pd.DataFrame({
        'BSSID': ['aa:bb:cc:dd:ee:01', 'aa:bb:cc:dd:ee:02', 'aa:bb:cc:dd:ee:03'] * 10,
        'Intensidad_señal(dBm)': [-30, -45, -60] * 10,
        'Ubicación': ['Aula 101', 'Biblioteca', 'Cafetería'] * 10
    })

# Codificación manual mejorada
bssid_mapping = {}
location_mapping = {}

# Crear mapeos únicos
unique_bssids = df['BSSID'].unique()
unique_locations = df['Ubicación'].unique()

for i, bssid in enumerate(unique_bssids):
    bssid_mapping[bssid] = i

for i, location in enumerate(unique_locations):
    location_mapping[location] = i

logger.info(f"BSSIDs únicos: {len(bssid_mapping)}")
logger.info(f"Ubicaciones únicas: {len(location_mapping)}")

# Normalizar señal
signals = df['Intensidad_señal(dBm)'].values
mean_signal = np.mean(signals)
std_signal = np.std(signals)
if std_signal == 0:
    std_signal = 1  # Evitar división por cero

normalized_signals = (signals - mean_signal) / std_signal

# Preparar datos
X = np.array([[bssid_mapping[bssid], signal] 
             for bssid, signal in zip(df['BSSID'], normalized_signals)])
y = np.array([location_mapping[loc] for loc in df['Ubicación']], dtype=int)

logger.info(f"Datos preparados: X shape {X.shape}, y shape {y.shape}")

# Entrenar modelos
models = {
    'KNN': KNN(k=min(5, len(X))),  # Asegurar que k no sea mayor que el número de muestras
    'RandomForest': RandomForest(n_trees=10, max_depth=5),
    'RegressionTree': RegressionTree(max_depth=5),
}

for name, model in models.items():
    try:
        model.fit(X, y)
        score = model.score(X, y)
        logger.info(f"Modelo {name} entrenado con score: {score:.3f}")
    except Exception as e:
        logger.error(f"Error entrenando modelo {name}: {e}")

# Seleccionar mejor modelo
try:
    best_model_name = max(models, key=lambda name: models[name].score(X, y))
    best_model = models[best_model_name]
    logger.info(f"Mejor modelo seleccionado: {best_model_name}")
except Exception as e:
    logger.error(f"Error seleccionando mejor modelo: {e}")
    best_model_name = 'KNN'
    best_model = models['KNN']

class WiFiData(BaseModel):
    bssid: str
    signal: int
    ssid: str = None  # Campo opcional para compatibilidad

class PredictionResponse(BaseModel):
    ubicacion: str
    precision: float
    modelo_usado: str
    bssid_reconocido: bool

@app.get("/")
async def root():
    return {
        "message": "WiFi Location Predictor API",
        "version": "1.0.0",
        "endpoints": ["/predecir-ubicacion", "/health"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "best_model": best_model_name,
        "dataset_size": len(df)
    }

@app.post("/predecir-ubicacion", response_model=PredictionResponse)
async def predict_location(data: WiFiData):
    try:
        logger.info(f"Predicción solicitada para BSSID: {data.bssid}, Signal: {data.signal}")
        
        # Validar entrada
        if not data.bssid or not isinstance(data.signal, int):
            raise HTTPException(status_code=400, detail="BSSID y signal son requeridos")
        
        # Verificar si el BSSID es conocido
        if data.bssid in bssid_mapping:
            encoded_bssid = bssid_mapping[data.bssid]
            bssid_reconocido = True
        else:
            # Para BSSIDs desconocidos, usar el más similar o un valor por defecto
            encoded_bssid = 0  # Usar el primer BSSID como fallback
            bssid_reconocido = False
            logger.warning(f"BSSID desconocido: {data.bssid}")
        
        # Normalizar señal
        normalized_signal = (data.signal - mean_signal) / std_signal
        
        # Realizar predicción
        prediction_input = np.array([[encoded_bssid, normalized_signal]])
        prediction_encoded = best_model.predict(prediction_input)
        
        # Convertir predicción a ubicación
        inverse_location = {v: k for k, v in location_mapping.items()}
        predicted_location = inverse_location.get(prediction_encoded[0], "Ubicación desconocida")
        
        # Calcular precisión
        precision = best_model.score(X, y) * 100
        
        # Ajustar precisión si BSSID no es reconocido
        if not bssid_reconocido:
            precision *= 0.7  # Reducir precisión para BSSIDs desconocidos
        
        response = PredictionResponse(
            ubicacion=predicted_location,
            precision=round(precision, 2),
            modelo_usado=best_model_name,
            bssid_reconocido=bssid_reconocido
        )
        
        logger.info(f"Predicción exitosa: {predicted_location} con {precision:.2f}% precisión")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
