from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import logging
from collections import defaultdict, Counter
import requests

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced WiFi Location Predictor", version="2.0.0")

# Agregar CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Importar modelos mejorados
from enhanced_models import EnhancedKNN, EnhancedRandomForest, WiFiFingerprinting

class WiFiSignal(BaseModel):
    bssid: str
    signal: int
    ssid: Optional[str] = None

class WiFiData(BaseModel):
    signals: List[WiFiSignal]  # Múltiples señales
    primary_bssid: Optional[str] = None  # BSSID principal (más fuerte)
    primary_signal: Optional[int] = None  # Para compatibilidad

class PredictionResponse(BaseModel):
    ubicacion: str
    precision: float
    confidence: float
    modelo_usado: str
    signals_used: int
    bssids_reconocidos: int
    method: str

class DatasetManager:
    def __init__(self):
        self.df = None
        self.location_mapping = {}
        self.bssid_mapping = {}
        self.bssid_weights = {}
        self.location_fingerprints = {}
        self.signal_stats = {}
        
    def load_dataset(self, url: str = None):
        """Cargar dataset desde URL o archivo local"""
        try:
            if url:
                logger.info(f"Cargando dataset desde URL: {url}")
                response = requests.get(url)
                response.raise_for_status()
                
                # Guardar temporalmente
                with open('temp_dataset.csv', 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                self.df = pd.read_csv('temp_dataset.csv')
            else:
                self.df = pd.read_csv('dataset.csv')
                
            logger.info(f"Dataset cargado con {len(self.df)} registros")
            self._process_dataset()
            return True
            
        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            # Crear datos de ejemplo para testing
            self._create_sample_data()
            return False
    
    def _create_sample_data(self):
        """Crear datos de ejemplo si no se puede cargar el dataset"""
        logger.warning("Creando datos de ejemplo")
        self.df = pd.DataFrame({
            'BSSID': ['aa:bb:cc:dd:ee:01', 'aa:bb:cc:dd:ee:02', 'aa:bb:cc:dd:ee:03'] * 20,
            'Intensidad_señal(dBm)': ['-30', '-45', '-60'] * 20,
            'Ubicación': ['Aula 101', 'Biblioteca', 'Cafetería'] * 20,
            'Etiqueta': ['Aula 101', 'Biblioteca', 'Cafetería'] * 20
        })
        self._process_dataset()
    
    def _process_dataset(self):
        """Procesar dataset y crear mapeos"""
        # Limpiar datos
        self.df = self.df.dropna(subset=['BSSID'])
        
        # Usar Ubicación o Etiqueta
        if 'Ubicación' in self.df.columns:
            location_col = 'Ubicación'
        elif 'Etiqueta' in self.df.columns:
            location_col = 'Etiqueta'
        else:
            raise ValueError("No se encontró columna de ubicación")
        
        self.df = self.df.dropna(subset=[location_col])
        
        # Convertir intensidad de señal a numérico
        if self.df['Intensidad_señal(dBm)'].dtype == 'object':
            self.df['Intensidad_señal(dBm)'] = pd.to_numeric(
                self.df['Intensidad_señal(dBm)'], errors='coerce'
            )
        
        self.df = self.df.dropna(subset=['Intensidad_señal(dBm)'])
        
        # Crear mapeos
        unique_locations = self.df[location_col].unique()
        unique_bssids = self.df['BSSID'].unique()
        
        self.location_mapping = {loc: i for i, loc in enumerate(unique_locations)}
        self.bssid_mapping = {bssid: i for i, bssid in enumerate(unique_bssids)}
        
        # Calcular pesos de BSSIDs (basado en frecuencia y especificidad)
        self._calculate_bssid_weights(location_col)
        
        # Crear fingerprints por ubicación
        self._create_location_fingerprints(location_col)
        
        # Estadísticas de señal
        signals = self.df['Intensidad_señal(dBm)'].values
        self.signal_stats = {
            'mean': np.mean(signals),
            'std': np.std(signals) if np.std(signals) > 0 else 1,
            'min': np.min(signals),
            'max': np.max(signals)
        }
        
        logger.info(f"Procesado: {len(unique_locations)} ubicaciones, {len(unique_bssids)} BSSIDs")
    
    def _calculate_bssid_weights(self, location_col):
        """Calcular pesos de BSSIDs basado en especificidad"""
        bssid_location_counts = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df.iterrows():
            bssid = row['BSSID']
            location = row[location_col]
            bssid_location_counts[bssid][location] += 1
        
        for bssid in self.bssid_mapping:
            locations = bssid_location_counts[bssid]
            total_appearances = sum(locations.values())
            
            if total_appearances == 0:
                weight = 0.1
            else:
                # Peso basado en especificidad (menos ubicaciones = mayor peso)
                num_locations = len(locations)
                specificity = 1.0 / num_locations if num_locations > 0 else 0.1
                
                # Peso basado en frecuencia
                frequency = total_appearances / len(self.df)
                
                # Combinar especificidad y frecuencia
                weight = specificity * (1 + frequency)
            
            self.bssid_weights[bssid] = weight
    
    def _create_location_fingerprints(self, location_col):
        """Crear fingerprints de señal por ubicación"""
        for location in self.location_mapping:
            location_data = self.df[self.df[location_col] == location]
            
            fingerprint = {}
            for _, row in location_data.iterrows():
                bssid = row['BSSID']
                signal = row['Intensidad_señal(dBm)']
                
                if bssid not in fingerprint:
                    fingerprint[bssid] = []
                fingerprint[bssid].append(signal)
            
            # Calcular estadísticas por BSSID
            processed_fingerprint = {}
            for bssid, signals in fingerprint.items():
                processed_fingerprint[bssid] = {
                    'mean': np.mean(signals),
                    'std': np.std(signals),
                    'count': len(signals),
                    'min': np.min(signals),
                    'max': np.max(signals)
                }
            
            self.location_fingerprints[location] = processed_fingerprint

# Inicializar dataset manager
dataset_manager = DatasetManager()

# Cargar dataset desde URL
dataset_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/dataset-TnkfBlGAoMWyWV5CZaOczRqkMgT00j.csv"
dataset_loaded = dataset_manager.load_dataset(dataset_url)

# Preparar datos para entrenamiento
def prepare_training_data():
    """Preparar datos de entrenamiento con múltiples características"""
    if dataset_manager.df is None:
        return None, None
    
    location_col = 'Ubicación' if 'Ubicación' in dataset_manager.df.columns else 'Etiqueta'
    
    # Crear características mejoradas
    features = []
    labels = []
    
    for _, row in dataset_manager.df.iterrows():
        bssid = row['BSSID']
        signal = row['Intensidad_señal(dBm)']
        location = row[location_col]
        
        if bssid in dataset_manager.bssid_mapping:
            # Características: [bssid_encoded, normalized_signal, bssid_weight]
            bssid_encoded = dataset_manager.bssid_mapping[bssid]
            normalized_signal = (signal - dataset_manager.signal_stats['mean']) / dataset_manager.signal_stats['std']
            bssid_weight = dataset_manager.bssid_weights.get(bssid, 0.1)
            
            features.append([bssid_encoded, normalized_signal, bssid_weight])
            labels.append(dataset_manager.location_mapping[location])
    
    return np.array(features), np.array(labels)

# Entrenar modelos
X, y = prepare_training_data()
models = {}

if X is not None and len(X) > 0:
    models = {
        'EnhancedKNN': EnhancedKNN(k=min(7, len(X))),
        'EnhancedRandomForest': EnhancedRandomForest(n_trees=20, max_depth=10),
        'WiFiFingerprinting': WiFiFingerprinting(dataset_manager.location_fingerprints, dataset_manager.signal_stats)
    }
    
    for name, model in models.items():
        try:
            if hasattr(model, 'fit'):
                model.fit(X, y)
            score = model.score(X, y) if hasattr(model, 'score') else 0.0
            logger.info(f"Modelo {name} entrenado con score: {score:.3f}")
        except Exception as e:
            logger.error(f"Error entrenando modelo {name}: {e}")

@app.get("/")
async def root():
    return {
        "message": "Enhanced WiFi Location Predictor API",
        "version": "2.0.0",
        "dataset_loaded": dataset_loaded,
        "models": list(models.keys()),
        "locations": len(dataset_manager.location_mapping),
        "bssids": len(dataset_manager.bssid_mapping)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "dataset_loaded": dataset_loaded,
        "models_loaded": len(models),
        "dataset_size": len(dataset_manager.df) if dataset_manager.df is not None else 0
    }

@app.post("/predecir-ubicacion", response_model=PredictionResponse)
async def predict_location(data: WiFiData):
    try:
        # Compatibilidad con formato anterior
        if not data.signals and data.primary_bssid and data.primary_signal:
            data.signals = [WiFiSignal(bssid=data.primary_bssid, signal=data.primary_signal)]
        
        if not data.signals:
            raise HTTPException(status_code=400, detail="No se proporcionaron señales WiFi")
        
        logger.info(f"Predicción solicitada con {len(data.signals)} señales")
        
        # Método 1: Fingerprinting (más preciso)
        fingerprint_result = predict_with_fingerprinting(data.signals)
        
        # Método 2: Ensemble de modelos ML
        ml_result = predict_with_ml_ensemble(data.signals)
        
        # Combinar resultados
        if fingerprint_result['confidence'] > ml_result['confidence']:
            best_result = fingerprint_result
            method = "fingerprinting"
        else:
            best_result = ml_result
            method = "ml_ensemble"
        
        # Calcular métricas adicionales
        bssids_reconocidos = sum(1 for signal in data.signals 
                               if signal.bssid in dataset_manager.bssid_mapping)
        
        return PredictionResponse(
            ubicacion=best_result['ubicacion'],
            precision=best_result['precision'],
            confidence=best_result['confidence'],
            modelo_usado=best_result.get('modelo', 'ensemble'),
            signals_used=len(data.signals),
            bssids_reconocidos=bssids_reconocidos,
            method=method
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

def predict_with_fingerprinting(signals: List[WiFiSignal]) -> Dict:
    """Predicción usando fingerprinting de señales"""
    if 'WiFiFingerprinting' not in models:
        return {'ubicacion': 'Desconocida', 'precision': 0.0, 'confidence': 0.0}
    
    try:
        fingerprinting_model = models['WiFiFingerprinting']
        
        # Preparar datos de entrada
        signal_data = {}
        for signal in signals:
            if signal.bssid in dataset_manager.bssid_mapping:
                signal_data[signal.bssid] = signal.signal
        
        if not signal_data:
            return {'ubicacion': 'Desconocida', 'precision': 0.0, 'confidence': 0.0}
        
        prediction, confidence = fingerprinting_model.predict_with_confidence(signal_data)
        
        # Convertir predicción a ubicación
        inverse_location = {v: k for k, v in dataset_manager.location_mapping.items()}
        ubicacion = inverse_location.get(prediction, "Ubicación desconocida")
        
        # Calcular precisión basada en validación del dataset
        precision = fingerprinting_model.get_accuracy() * 100
        
        return {
            'ubicacion': ubicacion,
            'precision': precision,
            'confidence': confidence * 100,
            'modelo': 'WiFiFingerprinting'
        }
        
    except Exception as e:
        logger.error(f"Error en fingerprinting: {e}")
        return {'ubicacion': 'Error', 'precision': 0.0, 'confidence': 0.0}

def predict_with_ml_ensemble(signals: List[WiFiSignal]) -> Dict:
    """Predicción usando ensemble de modelos ML"""
    if not models or len(models) == 0:
        return {'ubicacion': 'Desconocida', 'precision': 0.0, 'confidence': 0.0}
    
    try:
        predictions = []
        confidences = []
        
        for signal in signals:
            if signal.bssid in dataset_manager.bssid_mapping:
                bssid_encoded = dataset_manager.bssid_mapping[signal.bssid]
                normalized_signal = (signal.signal - dataset_manager.signal_stats['mean']) / dataset_manager.signal_stats['std']
                bssid_weight = dataset_manager.bssid_weights.get(signal.bssid, 0.1)
                
                feature_vector = np.array([[bssid_encoded, normalized_signal, bssid_weight]])
                
                # Obtener predicciones de todos los modelos
                model_predictions = []
                for name, model in models.items():
                    if hasattr(model, 'predict') and name != 'WiFiFingerprinting':
                        try:
                            pred = model.predict(feature_vector)[0]
                            model_predictions.append(pred)
                        except:
                            continue
                
                if model_predictions:
                    # Votación mayoritaria ponderada
                    weighted_pred = np.average(model_predictions, weights=[bssid_weight] * len(model_predictions))
                    predictions.append(int(round(weighted_pred)))
                    confidences.append(bssid_weight)
        
        if not predictions:
            return {'ubicacion': 'Desconocida', 'precision': 0.0, 'confidence': 0.0}
        
        # Votación final
        prediction_counts = Counter(predictions)
        final_prediction = prediction_counts.most_common(1)[0][0]
        
        # Calcular confianza
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Convertir a ubicación
        inverse_location = {v: k for k, v in dataset_manager.location_mapping.items()}
        ubicacion = inverse_location.get(final_prediction, "Ubicación desconocida")
        
        # Calcular precisión promedio de los modelos
        precision = 0.0
        model_count = 0
        for name, model in models.items():
            if hasattr(model, 'score') and name != 'WiFiFingerprinting':
                try:
                    precision += model.score(X, y)
                    model_count += 1
                except:
                    continue
        
        if model_count > 0:
            precision = (precision / model_count) * 100
        
        return {
            'ubicacion': ubicacion,
            'precision': precision,
            'confidence': confidence * 100,
            'modelo': 'ML_Ensemble'
        }
        
    except Exception as e:
        logger.error(f"Error en ML ensemble: {e}")
        return {'ubicacion': 'Error', 'precision': 0.0, 'confidence': 0.0}

# Endpoint para compatibilidad con versión anterior
@app.post("/predecir-ubicacion-simple")
async def predict_location_simple(data: dict):
    """Endpoint de compatibilidad con formato anterior"""
    try:
        bssid = data.get('bssid')
        signal = data.get('signal')
        ssid = data.get('ssid')
        
        if not bssid or signal is None:
            raise HTTPException(status_code=400, detail="BSSID y signal son requeridos")
        
        # Convertir a nuevo formato
        wifi_data = WiFiData(
            signals=[WiFiSignal(bssid=bssid, signal=int(signal), ssid=ssid)]
        )
        
        result = await predict_location(wifi_data)
        
        # Formato de respuesta compatible
        return {
            "ubicacion": result.ubicacion,
            "precision": result.precision
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
