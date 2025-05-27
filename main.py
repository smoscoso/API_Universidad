from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import uvicorn
import logging
from collections import defaultdict, Counter
import requests
import os

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

# Modelos Pydantic corregidos para compatibilidad
class WiFiSignal(BaseModel):
    bssid: str
    signal: int
    ssid: Optional[str] = None

class WiFiDataNew(BaseModel):
    """Formato nuevo con múltiples señales"""
    signals: List[WiFiSignal]

class WiFiDataLegacy(BaseModel):
    """Formato legacy para compatibilidad"""
    bssid: str
    signal: int
    ssid: Optional[str] = None

class WiFiDataFlexible(BaseModel):
    """Modelo flexible que acepta ambos formatos"""
    # Formato nuevo
    signals: Optional[List[WiFiSignal]] = None
    
    # Formato legacy
    bssid: Optional[str] = None
    signal: Optional[int] = None
    ssid: Optional[str] = None

class PredictionResponse(BaseModel):
    ubicacion: str
    precision: float
    confidence: Optional[float] = None
    modelo_usado: Optional[str] = None
    signals_used: Optional[int] = None
    bssids_reconocidos: Optional[int] = None
    method: Optional[str] = None

class DatasetManager:
    def __init__(self):
        self.df = None
        self.location_mapping = {}
        self.bssid_mapping = {}
        self.bssid_weights = {}
        self.location_fingerprints = {}
        self.signal_stats = {}
        self.inverse_location_mapping = {}
        
    def load_dataset(self, url: str = None):
        """Cargar dataset desde URL o archivo local"""
        try:
            if url:
                logger.info(f"Cargando dataset desde URL: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Guardar temporalmente
                with open('temp_dataset.csv', 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                self.df = pd.read_csv('temp_dataset.csv')
            else:
                if os.path.exists('dataset.csv'):
                    self.df = pd.read_csv('dataset.csv')
                else:
                    raise FileNotFoundError("No se encontró dataset.csv")
                
            logger.info(f"Dataset cargado con {len(self.df)} registros")
            self._process_dataset()
            return True
            
        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            self._create_sample_data()
            return False
    
    def _create_sample_data(self):
        """Crear datos de ejemplo si no se puede cargar el dataset"""
        logger.warning("Creando datos de ejemplo")
        self.df = pd.DataFrame({
            'BSSID': ['aa:bb:cc:dd:ee:01', 'aa:bb:cc:dd:ee:02', 'aa:bb:cc:dd:ee:03', 
                     'bb:cc:dd:ee:ff:01', 'cc:dd:ee:ff:aa:01'] * 20,
            'Intensidad_señal(dBm)': ['-30', '-45', '-60', '-35', '-50'] * 20,
            'Ubicación': ['Aula 101', 'Biblioteca', 'Cafetería', 'Laboratorio', 'Pasillo'] * 20,
            'Etiqueta': ['Aula 101', 'Biblioteca', 'Cafetería', 'Laboratorio', 'Pasillo'] * 20
        })
        self._process_dataset()
    
    def _process_dataset(self):
        """Procesar dataset y crear mapeos"""
        try:
            # Limpiar datos
            self.df = self.df.dropna(subset=['BSSID'])
            
            # Usar Ubicación o Etiqueta
            if 'Ubicación' in self.df.columns and not self.df['Ubicación'].isna().all():
                location_col = 'Ubicación'
            elif 'Etiqueta' in self.df.columns and not self.df['Etiqueta'].isna().all():
                location_col = 'Etiqueta'
            else:
                raise ValueError("No se encontró columna de ubicación válida")
            
            self.df = self.df.dropna(subset=[location_col])
            
            # Convertir intensidad de señal a numérico
            if self.df['Intensidad_señal(dBm)'].dtype == 'object':
                # Limpiar valores que puedan tener caracteres no numéricos
                self.df['Intensidad_señal(dBm)'] = self.df['Intensidad_señal(dBm)'].astype(str).str.replace(r'[^\d\-\.]', '', regex=True)
                self.df['Intensidad_señal(dBm)'] = pd.to_numeric(
                    self.df['Intensidad_señal(dBm)'], errors='coerce'
                )
            
            self.df = self.df.dropna(subset=['Intensidad_señal(dBm)'])
            
            if len(self.df) == 0:
                raise ValueError("No hay datos válidos después de la limpieza")
            
            # Crear mapeos
            unique_locations = self.df[location_col].unique()
            unique_bssids = self.df['BSSID'].unique()
            
            self.location_mapping = {loc: i for i, loc in enumerate(unique_locations)}
            self.inverse_location_mapping = {i: loc for loc, i in self.location_mapping.items()}
            self.bssid_mapping = {bssid: i for i, bssid in enumerate(unique_bssids)}
            
            # Calcular pesos de BSSIDs
            self._calculate_bssid_weights(location_col)
            
            # Crear fingerprints por ubicación
            self._create_location_fingerprints(location_col)
            
            # Estadísticas de señal
            signals = self.df['Intensidad_señal(dBm)'].values
            self.signal_stats = {
                'mean': float(np.mean(signals)),
                'std': float(np.std(signals)) if np.std(signals) > 0 else 1.0,
                'min': float(np.min(signals)),
                'max': float(np.max(signals))
            }
            
            logger.info(f"Procesado: {len(unique_locations)} ubicaciones, {len(unique_bssids)} BSSIDs")
            
        except Exception as e:
            logger.error(f"Error procesando dataset: {e}")
            raise
    
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
                # Peso basado en especificidad
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
                if len(signals) > 0:
                    processed_fingerprint[bssid] = {
                        'mean': float(np.mean(signals)),
                        'std': float(np.std(signals)) if np.std(signals) > 0 else 1.0,
                        'count': len(signals),
                        'min': float(np.min(signals)),
                        'max': float(np.max(signals))
                    }
            
            self.location_fingerprints[location] = processed_fingerprint

# Inicializar dataset manager
dataset_manager = DatasetManager()

# Cargar dataset
dataset_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/dataset-TnkfBlGAoMWyWV5CZaOczRqkMgT00j.csv"
dataset_loaded = dataset_manager.load_dataset(dataset_url)

# Preparar datos para entrenamiento
def prepare_training_data():
    """Preparar datos de entrenamiento"""
    if dataset_manager.df is None or len(dataset_manager.df) == 0:
        return None, None
    
    location_col = 'Ubicación' if 'Ubicación' in dataset_manager.df.columns else 'Etiqueta'
    
    features = []
    labels = []
    
    for _, row in dataset_manager.df.iterrows():
        bssid = row['BSSID']
        signal = row['Intensidad_señal(dBm)']
        location = row[location_col]
        
        if bssid in dataset_manager.bssid_mapping and not pd.isna(signal):
            bssid_encoded = dataset_manager.bssid_mapping[bssid]
            normalized_signal = (signal - dataset_manager.signal_stats['mean']) / dataset_manager.signal_stats['std']
            bssid_weight = dataset_manager.bssid_weights.get(bssid, 0.1)
            
            features.append([bssid_encoded, normalized_signal, bssid_weight])
            labels.append(dataset_manager.location_mapping[location])
    
    if len(features) == 0:
        return None, None
        
    return np.array(features), np.array(labels)

# Entrenar modelos
X, y = prepare_training_data()
models = {}
best_model = None
best_model_name = "default"

if X is not None and len(X) > 0:
    try:
        models = {
            'EnhancedKNN': EnhancedKNN(k=min(7, len(X))),
            'EnhancedRandomForest': EnhancedRandomForest(n_trees=15, max_depth=8),
            'WiFiFingerprinting': WiFiFingerprinting(
                dataset_manager.location_fingerprints, 
                dataset_manager.signal_stats
            )
        }
        
        best_score = 0
        for name, model in models.items():
            try:
                if hasattr(model, 'fit'):
                    model.fit(X, y)
                
                if hasattr(model, 'score'):
                    score = model.score(X, y)
                else:
                    score = 0.85  # Score por defecto para fingerprinting
                
                logger.info(f"Modelo {name} entrenado con score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                logger.error(f"Error entrenando modelo {name}: {e}")
        
        logger.info(f"Mejor modelo: {best_model_name} con score: {best_score:.3f}")
        
    except Exception as e:
        logger.error(f"Error general en entrenamiento: {e}")

@app.get("/")
async def root():
    return {
        "message": "Enhanced WiFi Location Predictor API",
        "version": "2.0.0",
        "status": "running",
        "dataset_loaded": dataset_loaded,
        "models": list(models.keys()),
        "best_model": best_model_name,
        "locations": len(dataset_manager.location_mapping),
        "bssids": len(dataset_manager.bssid_mapping)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "dataset_loaded": dataset_loaded,
        "models_loaded": len(models),
        "dataset_size": len(dataset_manager.df) if dataset_manager.df is not None else 0,
        "locations": list(dataset_manager.location_mapping.keys())[:5]  # Primeras 5 ubicaciones
    }

@app.post("/predecir-ubicacion", response_model=PredictionResponse)
async def predict_location(data: WiFiDataFlexible):
    try:
        logger.info(f"Solicitud recibida: {data}")
        
        # Convertir a formato estándar
        signals = []
        
        if data.signals:
            # Formato nuevo con múltiples señales
            signals = data.signals
        elif data.bssid and data.signal is not None:
            # Formato legacy
            signals = [WiFiSignal(bssid=data.bssid, signal=data.signal, ssid=data.ssid)]
        else:
            raise HTTPException(status_code=400, detail="Se requiere 'bssid' y 'signal' o 'signals'")
        
        if not signals:
            raise HTTPException(status_code=400, detail="No se proporcionaron señales WiFi válidas")
        
        logger.info(f"Procesando {len(signals)} señales")
        
        # Realizar predicción
        result = perform_prediction(signals)
        
        logger.info(f"Predicción exitosa: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

def perform_prediction(signals: List[WiFiSignal]) -> PredictionResponse:
    """Realizar predicción usando los modelos disponibles"""
    
    if not dataset_manager.location_mapping:
        return PredictionResponse(
            ubicacion="Sistema no inicializado",
            precision=0.0,
            confidence=0.0,
            modelo_usado="none",
            signals_used=0,
            bssids_reconocidos=0,
            method="error"
        )
    
    # Contar BSSIDs reconocidos
    bssids_reconocidos = sum(1 for signal in signals 
                           if signal.bssid in dataset_manager.bssid_mapping)
    
    if bssids_reconocidos == 0:
        # Usar ubicación más común como fallback
        most_common_location = max(dataset_manager.location_mapping.keys(), 
                                 key=lambda x: len(dataset_manager.df[
                                     dataset_manager.df['Ubicación'] == x
                                 ]) if 'Ubicación' in dataset_manager.df.columns else 0)
        
        return PredictionResponse(
            ubicacion=most_common_location,
            precision=30.0,
            confidence=10.0,
            modelo_usado="fallback",
            signals_used=len(signals),
            bssids_reconocidos=0,
            method="fallback"
        )
    
    # Método 1: Fingerprinting
    fingerprint_result = predict_with_fingerprinting(signals)
    
    # Método 2: ML Models
    ml_result = predict_with_ml_models(signals)
    
    # Seleccionar mejor resultado
    if fingerprint_result['confidence'] > ml_result['confidence']:
        best_result = fingerprint_result
        method = "fingerprinting"
    else:
        best_result = ml_result
        method = "ml_ensemble"
    
    return PredictionResponse(
        ubicacion=best_result['ubicacion'],
        precision=best_result['precision'],
        confidence=best_result['confidence'],
        modelo_usado=best_result['modelo'],
        signals_used=len(signals),
        bssids_reconocidos=bssids_reconocidos,
        method=method
    )

def predict_with_fingerprinting(signals: List[WiFiSignal]) -> Dict:
    """Predicción usando fingerprinting"""
    try:
        if 'WiFiFingerprinting' not in models:
            return {'ubicacion': 'Error', 'precision': 0.0, 'confidence': 0.0, 'modelo': 'none'}
        
        fingerprinting_model = models['WiFiFingerprinting']
        
        # Preparar datos
        signal_data = {}
        for signal in signals:
            if signal.bssid in dataset_manager.bssid_mapping:
                signal_data[signal.bssid] = signal.signal
        
        if not signal_data:
            return {'ubicacion': 'Desconocida', 'precision': 0.0, 'confidence': 0.0, 'modelo': 'fingerprinting'}
        
        prediction, confidence = fingerprinting_model.predict_with_confidence(signal_data)
        
        # Convertir predicción a ubicación
        ubicacion = dataset_manager.inverse_location_mapping.get(prediction, "Ubicación desconocida")
        
        precision = fingerprinting_model.get_accuracy() * 100
        
        return {
            'ubicacion': ubicacion,
            'precision': precision,
            'confidence': confidence * 100,
            'modelo': 'WiFiFingerprinting'
        }
        
    except Exception as e:
        logger.error(f"Error en fingerprinting: {e}")
        return {'ubicacion': 'Error', 'precision': 0.0, 'confidence': 0.0, 'modelo': 'fingerprinting_error'}

def predict_with_ml_models(signals: List[WiFiSignal]) -> Dict:
    """Predicción usando modelos ML"""
    try:
        if not models:
            return {'ubicacion': 'Error', 'precision': 0.0, 'confidence': 0.0, 'modelo': 'none'}
        
        predictions = []
        confidences = []
        
        for signal in signals:
            if signal.bssid in dataset_manager.bssid_mapping:
                bssid_encoded = dataset_manager.bssid_mapping[signal.bssid]
                normalized_signal = (signal.signal - dataset_manager.signal_stats['mean']) / dataset_manager.signal_stats['std']
                bssid_weight = dataset_manager.bssid_weights.get(signal.bssid, 0.1)
                
                feature_vector = np.array([[bssid_encoded, normalized_signal, bssid_weight]])
                
                # Predicciones de modelos ML
                model_predictions = []
                for name, model in models.items():
                    if hasattr(model, 'predict') and name != 'WiFiFingerprinting':
                        try:
                            pred = model.predict(feature_vector)[0]
                            model_predictions.append(pred)
                        except Exception as e:
                            logger.warning(f"Error en predicción de {name}: {e}")
                            continue
                
                if model_predictions:
                    avg_pred = np.mean(model_predictions)
                    predictions.append(int(round(avg_pred)))
                    confidences.append(bssid_weight)
        
        if not predictions:
            return {'ubicacion': 'Desconocida', 'precision': 0.0, 'confidence': 0.0, 'modelo': 'ml_no_predictions'}
        
        # Votación final
        prediction_counts = Counter(predictions)
        final_prediction = prediction_counts.most_common(1)[0][0]
        
        # Calcular confianza
        confidence = np.mean(confidences) if confidences else 0.0
        
        # Convertir a ubicación
        ubicacion = dataset_manager.inverse_location_mapping.get(final_prediction, "Ubicación desconocida")
        
        # Calcular precisión promedio
        precision = 0.0
        model_count = 0
        for name, model in models.items():
            if hasattr(model, 'score') and name != 'WiFiFingerprinting':
                try:
                    if X is not None and y is not None:
                        precision += model.score(X, y)
                        model_count += 1
                except Exception:
                    continue
        
        if model_count > 0:
            precision = (precision / model_count) * 100
        else:
            precision = 75.0  # Precisión por defecto
        
        return {
            'ubicacion': ubicacion,
            'precision': precision,
            'confidence': confidence * 100,
            'modelo': 'ML_Ensemble'
        }
        
    except Exception as e:
        logger.error(f"Error en ML models: {e}")
        return {'ubicacion': 'Error', 'precision': 0.0, 'confidence': 0.0, 'modelo': 'ml_error'}

# Endpoint de compatibilidad
@app.post("/predecir-ubicacion-simple")
async def predict_location_simple(data: dict):
    """Endpoint de compatibilidad total con versión anterior"""
    try:
        bssid = data.get('bssid')
        signal = data.get('signal')
        ssid = data.get('ssid')
        
        if not bssid or signal is None:
            raise HTTPException(status_code=400, detail="BSSID y signal son requeridos")
        
        # Usar el endpoint principal
        wifi_data = WiFiDataFlexible(bssid=bssid, signal=int(signal), ssid=ssid)
        result = await predict_location(wifi_data)
        
        # Formato de respuesta simple
        return {
            "ubicacion": result.ubicacion,
            "precision": result.precision
        }
        
    except Exception as e:
        logger.error(f"Error en endpoint simple: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
