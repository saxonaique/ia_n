import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import scipy.ndimage as ndi
from typing import Optional, Tuple, Dict, Any, Union, List
import os
import json
import time
from PIL import Image, ImageTk
import matplotlib.cm as cm
import matplotlib.colors as colors
import importlib.util
from abc import ABC, abstractmethod
import threading
import queue
from datetime import datetime

# Dependencia para audio .wav y espectrograma
from scipy.io import wavfile
from scipy.signal import spectrogram

# Intentar importar matplotlib para gráficos
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib no disponible. Los gráficos estarán deshabilitados.")

# --- Módulo 1: SensorModule (Sensorium Informacional Universal) ---
class SensorModule:
    """
    Sensorium Informacional: Módulo de percepción de la IA DIG.
    Transforma datos brutos (texto, imágenes, audio, CSV) en un campo [0, 1].
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {'target_field_size': (64, 64)}
        self.field_shape = self.config['target_field_size']

    def process(self, source: str, source_type: str) -> np.ndarray:
        """Punto de entrada principal para procesar cualquier tipo de dato."""
        print(f"[SensorModule] Recibido para procesar: tipo='{source_type}'")
        if source_type == 'text':
            return self._process_text_structured(source)
        elif source_type == 'file':
            file_ext = os.path.splitext(source)[1].lower()
            if file_ext in ['.png', '.jpg', '.jpeg']:
                return self._process_image(source)
            elif file_ext == '.wav':
                return self._process_audio_wav(source)
            elif file_ext == '.csv':
                return self._process_brainwaves_csv(source)
            else:
                print(f"[SensorModule] WARN: Extensión de archivo '{file_ext}' no soportada.")
                return np.zeros(self.field_shape)
        else:
            return np.zeros(self.field_shape)

    def _create_gaussian_blob(self, center_x, center_y, intensity, size):
        y, x = np.mgrid[0:self.field_shape[0], 0:self.field_shape[1]]
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        sigma_sq = (size / 2)**2
        return intensity * np.exp(-dist_sq / (2 * sigma_sq))

    def _process_text_structured(self, text_input: str) -> np.ndarray:
        field = np.zeros(self.field_shape, dtype=np.float32)
        words = text_input.split()
        if not words: return field
        rows, cols = self.field_shape
        np.random.seed(sum(ord(c) for c in text_input))
        for i, word in enumerate(words):
            word_hash = sum(ord(c) for c in word)
            center_x = (word_hash % cols + i * 13) % cols
            center_y = (len(word) * 17 + word_hash % rows) % rows
            intensity = min(1.0, 0.5 + len(word) / 10.0)
            size = max(5, min(rows // 4, len(word) * 2))
            field += self._create_gaussian_blob(center_x, center_y, intensity, size)
        
        ### SOLUCIÓN: Se corrige el riesgo de división por cero.
        # Se comprueba si el rango de valores del campo es mayor que una pequeña
        # tolerancia para evitar dividir por cero si el campo es plano.
        field_range = field.max() - field.min()
        if field_range > 1e-9:
            field = (field - field.min()) / field_range
        else:
            # Si el campo es plano, se asegura que sea todo ceros.
            field = np.zeros(self.field_shape, dtype=np.float32)
            
        return field.astype(np.float32)

    def _process_image(self, file_path: str) -> np.ndarray:
        try:
            img = Image.open(file_path).convert('L') # Convertir a escala de grises
            img_resized = img.resize(self.field_shape, Image.Resampling.LANCZOS)
            field = np.array(img_resized) / 255.0
            return field.astype(np.float32)
        except Exception as e:
            print(f"[SensorModule] ERROR procesando imagen: {e}")
            return np.zeros(self.field_shape)

    def _process_audio_wav(self, file_path: str) -> np.ndarray:
        try:
            sample_rate, samples = wavfile.read(file_path)
            if samples.ndim > 1: samples = samples[:, 0]
            _, _, Sxx = spectrogram(samples, sample_rate)
            log_spectrogram = np.log1p(Sxx)
            img = Image.fromarray(log_spectrogram)
            img_resized = img.resize(self.field_shape, Image.Resampling.LANCZOS)
            field = np.array(img_resized)
            if field.max() > field.min(): field = (field - field.min()) / (field.max() - field.min())
            return field.astype(np.float32)
        except Exception as e:
            print(f"[SensorModule] ERROR procesando audio: {e}")
            return np.zeros(self.field_shape)

    def _process_brainwaves_csv(self, file_path: str) -> np.ndarray:
        try:
            # Asume que el CSV es una matriz de números (tiempo en filas, canales en columnas)
            data = np.loadtxt(file_path, delimiter=',')
            img = Image.fromarray(data)
            img_resized = img.resize(self.field_shape, Image.Resampling.LANCZOS)
            field = np.array(img_resized)
            if field.max() > field.min(): field = (field - field.min()) / (field.max() - field.min())
            return field.astype(np.float32)
        except Exception as e:
            print(f"[SensorModule] ERROR procesando CSV: {e}")
            return np.zeros(self.field_shape)

# ... (El resto de los módulos CoreNucleus, MemoryModule, LearningMemory, FingerprintSystem no necesitan cambios y se omiten por brevedad) ...
class CoreNucleus:
    def __init__(self, field_shape: Tuple[int, int] = (64, 64)):
        self.field_shape = field_shape
        self.field = np.zeros(field_shape, dtype=np.float32)
        print("[CoreNucleus] INFO: Inicializado.")
    
    def receive_field(self, field: np.ndarray):
        if not isinstance(field, np.ndarray):
            self.field = np.zeros(self.field_shape, dtype=np.float32)
            return
        if field.shape != self.field_shape:
            # Corrección para evitar error si el campo de entrada es 1D
            if field.ndim == 1:
                field = np.resize(field, self.field_shape)
            else:
                field = ndi.zoom(field, [t / s for t, s in zip(self.field_shape, field.shape)])
        self.field = np.clip(field, 0, 1).astype(np.float32)
    
    def get_metrics(self) -> Dict[str, Any]:
        if self.field.size == 0:
            return {}
        hist, _ = np.histogram(self.field.flatten(), bins=10, range=(0, 1))
        probs = hist / self.field.size
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        return {
            "entropía": float(entropy),
            "varianza": float(np.var(self.field)),
            "máximo": float(np.max(self.field)),
            "simetría": float(np.mean(np.abs(self.field - np.fliplr(self.field)))),
            "active_cells": int(np.sum(self.field > 0.66)),
            "inhibited_cells": int(np.sum(self.field < 0.33)),
            "neutral_cells": int(self.field.size - np.sum(self.field > 0.66) - np.sum(self.field < 0.33))
        }
    
    def reorganize_field(self):
        self.field = np.clip(
            ndi.convolve(
                self.field,
                np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]]),
                mode='reflect'
            ),
            0, 1
        )
    
    def apply_attractor(self, attractor_field: np.ndarray, strength: float = 0.25):
        if self.field.shape != attractor_field.shape:
            if attractor_field.ndim == 1:
                attractor_field = np.resize(attractor_field, self.field_shape)
            else:
                attractor_field = ndi.zoom(
                    attractor_field,
                    [t / s for t, s in zip(self.field_shape, attractor_field.shape)]
                )
        self.field = np.clip(
            self.field * (1 - strength) + attractor_field * strength,
            0, 1
        )

class MemoryModule:
    def __init__(self):
        self.attractors: Dict[str, Dict[str, Any]] = {}
        self._initialize_sample_attractors()
    
    def _initialize_sample_attractors(self):
        print("[MemoryModule] INFO: Creando atractores.")
        x, y = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
        self.add_attractor("orden_gradiente", np.sin(x * np.pi) * np.cos(y * np.pi) * 0.5 + 0.5)
        self.add_attractor("complejidad_ajedrez", np.kron([[1, 0] * 32, [0, 1] * 32] * 32, np.ones((1, 1)))[:64, :64])
    
    def add_attractor(self, name: str, pattern: np.ndarray):
        self.attractors[name] = {'pattern': pattern.astype(np.float32), 'usage_count': 0}
    
    def find_closest_attractor(self, field: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self.attractors:
            return None
        min_dist, best_match_name = float('inf'), None
        for name, data in self.attractors.items():
            dist = np.mean((field - data['pattern']) ** 2)
            if dist < min_dist:
                min_dist, best_match_name = dist, name
        if best_match_name:
            self.attractors[best_match_name]['usage_count'] += 1
            print(f"[MemoryModule] INFO: Atractor más cercano: '{best_match_name}'")
            return self.attractors[best_match_name]
        return None

def interpretar_metrica(m: Dict[str, Any]) -> str:
    if not m:
        return "IA: Esperando métricas..."
    
    e, v, sym = m.get("entropía", 0.0), m.get("varianza", 0.0), m.get("simetría", 0.0)
    total = m.get("active_cells", 0) + m.get("inhibited_cells", 0) + m.get("neutral_cells", 0)
    active_ratio = m.get("active_cells", 0) / total if total > 0 else 0
    
    estado, reco = "EQUILIBRIO DINÁMICO", "Continuar monitorización."
    if e < 0.5 and v < 0.01:
        estado, reco = "CAMPO ESTANCADO", "Inyectar nueva entrada."
    elif e > 2.0 and v > 0.1:
        estado, reco = "SOBRECARGA INFORMACIONAL", "Aplicar atractor de orden."
    elif active_ratio > 0.5:
        estado, reco = "ALTA ACTIVIDAD", "Aplicar atractor inhibidor."
    
    return "\n".join([
        f"✅ ESTADO: {estado}",
        f"  - Entropía: {e:.4f} | Varianza: {v:.4f}",
        f"  - Simetría: {sym:.4f} | Células Activas: {active_ratio:.1%}",
        f"💡 RECOMENDACIÓN: {reco}"
    ])

class LearningMemory:
    """Sistema de memoria de aprendizaje para reconocimiento de patrones."""
    
    def __init__(self, memory_file: str = "learning_memory.json"):
        self.memory_file = memory_file
        self.patterns = {}  # Patrones aprendidos
        self.similarity_threshold = 0.85  # Umbral de similitud
        self.max_patterns = 100  # Máximo de patrones almacenados
        
        # Cargar memoria existente
        self.load_memory()
    
    def add_pattern(self, input_data: str, input_type: str, fingerprints: Dict[str, Any], 
                   user_label: str = None) -> str:
        """Agrega un nuevo patrón a la memoria de aprendizaje."""
        try:
            # Generar ID único para el patrón
            pattern_id = f"pattern_{int(time.time())}_{len(self.patterns)}"
            
            # Crear vector de características del patrón
            pattern_vector = self._extract_pattern_vector(fingerprints)
            
            # Crear entrada del patrón
            pattern_entry = {
                "id": pattern_id,
                "timestamp": time.time(),
                "input_data": input_data,
                "input_type": input_type,
                "user_label": user_label or f"Patrón_{len(self.patterns) + 1}",
                "auto_label": self._generate_auto_label(input_data, input_type),
                "fingerprints": fingerprints,
                "pattern_vector": pattern_vector,
                "usage_count": 1,
                "last_used": time.time()
            }
            
            # Agregar a la memoria
            self.patterns[pattern_id] = pattern_entry
            
            # Limpiar memoria si excede el máximo
            if len(self.patterns) > self.max_patterns:
                self._cleanup_old_patterns()
            
            # Guardar memoria
            self.save_memory()
            
            print(f"[LearningMemory] Nuevo patrón agregado: {pattern_entry['auto_label']}")
            return pattern_id
            
        except Exception as e:
            print(f"[LearningMemory] ERROR agregando patrón: {e}")
            return ""
    
    def find_similar_pattern(self, input_data: str, input_type: str, 
                           fingerprints: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Busca un patrón similar en la memoria."""
        try:
            if not self.patterns:
                return None
            
            # Extraer vector de características del input actual
            current_vector = self._extract_pattern_vector(fingerprints)
            
            best_match = None
            best_similarity = 0.0
            
            # Buscar el patrón más similar
            for pattern_id, pattern in self.patterns.items():
                similarity = self._calculate_similarity(current_vector, pattern["pattern_vector"])
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = pattern.copy()
                    best_match["similarity_score"] = similarity
            
            if best_match:
                # Actualizar contador de uso
                self.patterns[best_match["id"]]["usage_count"] += 1
                self.patterns[best_match["id"]]["last_used"] = time.time()
                self.save_memory()
                
                print(f"[LearningMemory] Patrón similar encontrado: {best_match['auto_label']} (similitud: {best_similarity:.3f})")
            
            return best_match
            
        except Exception as e:
            print(f"[LearningMemory] ERROR buscando patrón similar: {e}")
            return None
    
    def _extract_pattern_vector(self, fingerprints: Dict[str, Any]) -> List[float]:
        """Extrae un vector de características de las huellas."""
        vector = []
        
        # Características de las huellas iniciales
        if "inicial" in fingerprints and fingerprints["inicial"]:
            initial = fingerprints["inicial"]
            vector.extend([
                initial["field_stats"]["entropy"],
                initial["field_stats"]["variance"],
                initial["field_stats"]["symmetry"],
                initial["field_stats"]["mean"],
                initial["field_stats"]["std"]
            ])
        else:
            vector.extend([0.0] * 5)
        
        # Características de las huellas intermedias
        if "intermedia" in fingerprints and fingerprints["intermedia"]:
            intermedia = fingerprints["intermedia"]
            vector.extend([
                intermedia["field_stats"]["entropy"],
                intermedia["field_stats"]["variance"],
                intermedia["field_stats"]["symmetry"],
                intermedia["field_stats"]["mean"],
                intermedia["field_stats"]["std"]
            ])
        else:
            vector.extend([0.0] * 5)
        
        # Características de las huellas finales
        if "final" in fingerprints and fingerprints["final"]:
            final = fingerprints["final"]
            vector.extend([
                final["field_stats"]["entropy"],
                final["field_stats"]["variance"],
                final["field_stats"]["symmetry"],
                final["field_stats"]["mean"],
                final["field_stats"]["std"]
            ])
        else:
            vector.extend([0.0] * 5)
        
        # Características de evolución
        if len(vector) >= 15:
            # Cambios entre etapas
            vector.append(vector[5] - vector[0])  # Cambio entropía inicial-intermedia
            vector.append(vector[10] - vector[5])  # Cambio entropía intermedia-final
            vector.append(vector[6] - vector[1])   # Cambio varianza inicial-intermedia
            vector.append(vector[11] - vector[6])  # Cambio varianza intermedia-final
        
        return vector
    
    def _calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calcula la similitud entre dos vectores de características."""
        try:
            if len(vector1) != len(vector2):
                return 0.0
            
            # Normalizar vectores
            v1_norm = np.array(vector1)
            v2_norm = np.array(vector2)
            
            # Calcular similitud coseno
            dot_product = np.dot(v1_norm, v2_norm)
            norm1 = np.linalg.norm(v1_norm)
            norm2 = np.linalg.norm(v2_norm)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"[LearningMemory] ERROR calculando similitud: {e}")
            return 0.0
    
    def _generate_auto_label(self, input_data: str, input_type: str) -> str:
        """Genera una etiqueta automática basada en el tipo de entrada."""
        try:
            if input_type == "text":
                # Analizar el texto para generar etiqueta
                words = input_data.lower().split()
                key_words = [w for w in words if len(w) > 4]
                
                if key_words:
                    # Tomar las primeras 2-3 palabras clave
                    label = "_".join(key_words[:3])
                    return f"texto_{label[:20]}"
                else:
                    return f"texto_{len(input_data)}chars"
            
            elif input_type == "file":
                # Etiqueta basada en extensión del archivo
                import os
                filename = os.path.basename(input_data)
                name, ext = os.path.splitext(filename)
                return f"archivo_{ext[1:]}_{name[:10]}"
            
            else:
                return f"entrada_{input_type}_{int(time.time())}"
                
        except Exception as e:
            print(f"[LearningMemory] ERROR generando etiqueta: {e}")
            return f"patron_{int(time.time())}"
    
    def _cleanup_old_patterns(self):
        """Limpia patrones antiguos para mantener la memoria optimizada."""
        try:
            # Ordenar por último uso y frecuencia de uso
            sorted_patterns = sorted(
                self.patterns.items(),
                key=lambda x: (x[1]["last_used"], x[1]["usage_count"]),
                reverse=True
            )
            
            # Mantener solo los mejores patrones
            keep_patterns = sorted_patterns[:self.max_patterns]
            self.patterns = dict(keep_patterns)
            
            print(f"[LearningMemory] Memoria limpiada, {len(self.patterns)} patrones mantenidos")
            
        except Exception as e:
            print(f"[LearningMemory] ERROR limpiando memoria: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema de aprendizaje."""
        try:
            total_patterns = len(self.patterns)
            total_usage = sum(p["usage_count"] for p in self.patterns.values())
            
            # Patrones más usados
            top_patterns = sorted(
                self.patterns.values(),
                key=lambda x: x["usage_count"],
                reverse=True
            )[:5]
            
            return {
                "total_patterns": total_patterns,
                "total_usage": total_usage,
                "top_patterns": [
                    {
                        "label": p["auto_label"],
                        "usage_count": p["usage_count"],
                        "last_used": p["last_used"]
                    }
                    for p in top_patterns
                ],
                "memory_usage": f"{total_patterns}/{self.max_patterns}"
            }
            
        except Exception as e:
            print(f"[LearningMemory] ERROR obteniendo estadísticas: {e}")
            return {}
    
    def save_memory(self):
        """Guarda la memoria de aprendizaje en disco."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.patterns, f, indent=2, ensure_ascii=False)
            
            print(f"[LearningMemory] Memoria guardada: {self.memory_file}")
            
        except Exception as e:
            print(f"[LearningMemory] ERROR guardando memoria: {e}")
    
    def load_memory(self):
        """Carga la memoria de aprendizaje desde disco."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.patterns = json.load(f)
                
                print(f"[LearningMemory] Memoria cargada: {len(self.patterns)} patrones")
            else:
                print("[LearningMemory] No se encontró memoria existente, iniciando nueva")
                
        except Exception as e:
            print(f"[LearningMemory] ERROR cargando memoria: {e}")
            self.patterns = {}

class FingerprintSystem:
    """Sistema de huellas para capturar estados clave del campo informacional."""
    
    def __init__(self, fingerprint_dir: str = "fingerprints"):
        self.fingerprint_dir = fingerprint_dir
        self.fingerprints = {}
        self.session_id = None
        
        # Sistema de aprendizaje por reconocimiento de patrones
        self.learning_memory = LearningMemory()
        
        # Crear directorio de huellas si no existe
        os.makedirs(self.fingerprint_dir, exist_ok=True)
        
        # Inicializar huellas de la sesión actual
        self.reset_session()
    
    def reset_session(self):
        """Reinicia las huellas para una nueva sesión."""
        self.fingerprints = {
            "inicial": None,
            "intermedia": None,
            "final": None
        }
        self.session_id = f"session_{int(time.time())}"
        print(f"[FingerprintSystem] Nueva sesión iniciada: {self.session_id}")
    
    def extract_fingerprint(self, field: np.ndarray, cycle: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae una huella del campo actual."""
        try:
            # Calcular características clave del campo
            fingerprint = {
                "timestamp": time.time(),
                "cycle": cycle,
                "field_shape": field.shape,
                "field_stats": {
                    "mean": float(np.mean(field)),
                    "std": float(np.std(field)),
                    "min": float(np.min(field)),
                    "max": float(np.max(field)),
                    "entropy": metrics.get("entropía", 0.0),
                    "variance": metrics.get("varianza", 0.0),
                    "symmetry": metrics.get("simetría", 0.0)
                },
                "active_cells": metrics.get("active_cells", 0),
                "inhibited_cells": metrics.get("inhibited_cells", 0),
                "neutral_cells": metrics.get("neutral_cells", 0),
                "field_sample": self._sample_field(field)  # Muestra representativa del campo
            }
            
            return fingerprint
            
        except Exception as e:
            print(f"[FingerprintSystem] ERROR extrayendo huella: {e}")
            return {}
    
    def _sample_field(self, field: np.ndarray, sample_size: int = 16) -> List[float]:
        """Toma una muestra representativa del campo para análisis."""
        try:
            # Tomar muestras en puntos estratégicos del campo
            rows, cols = field.shape
            samples = []
            
            # Muestras en las esquinas
            samples.extend([
                float(field[0, 0]),           # Esquina superior izquierda
                float(field[0, cols-1]),      # Esquina superior derecha
                float(field[rows-1, 0]),      # Esquina inferior izquierda
                float(field[rows-1, cols-1])  # Esquina inferior derecha
            ])
            
            # Muestras en el centro
            center_row, center_col = rows // 2, cols // 2
            samples.extend([
                float(field[center_row, center_col]),           # Centro exacto
                float(field[center_row-1, center_col]),         # Centro arriba
                float(field[center_row+1, center_col]),         # Centro abajo
                float(field[center_row, center_col-1]),         # Centro izquierda
                float(field[center_row, center_col+1])          # Centro derecha
            ])
            
            # Muestras aleatorias para completar
            import random
            random.seed(int(time.time()))
            for _ in range(sample_size - len(samples)):
                r, c = random.randint(0, rows-1), random.randint(0, cols-1)
                samples.append(float(field[r, c]))
            
            return samples
            
        except Exception as e:
            print(f"[FingerprintSystem] ERROR muestreando campo: {e}")
            return [0.0] * sample_size
    
    def capture_fingerprint(self, stage: str, field: np.ndarray, cycle: int, metrics: Dict[str, Any]):
        """Captura una huella en una etapa específica."""
        if stage in self.fingerprints:
            self.fingerprints[stage] = self.extract_fingerprint(field, cycle, metrics)
            print(f"[FingerprintSystem] Huella '{stage}' capturada en ciclo {cycle}")
        else:
            print(f"[FingerprintSystem] WARN: Etapa '{stage}' no válida")
    
    def save_fingerprints(self, filename: str = None) -> str:
        """Guarda las huellas de la sesión actual."""
        if not filename:
            filename = f"fingerprints_{self.session_id}.json"
        
        filepath = os.path.join(self.fingerprint_dir, filename)
        
        try:
            # Preparar datos para guardar
            save_data = {
                "session_id": self.session_id,
                "fingerprints": self.fingerprints,
                "summary": self._generate_summary()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
           
            
            print(f"[FingerprintSystem] Huellas guardadas en: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"[FingerprintSystem] ERROR guardando huellas: {e}")
            return ""
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Genera un resumen de las huellas capturadas."""
        summary = {
            "total_fingerprints": sum(1 for f in self.fingerprints.values() if f is not None),
            "stages_captured": [stage for stage, fp in self.fingerprints.items() if fp is not None],
            "evolution_analysis": {}
        }
        
        # Análisis de evolución si tenemos múltiples huellas
        stages_with_data = [(stage, fp) for stage, fp in self.fingerprints.items() if fp is not None]
        
        if len(stages_with_data) > 1:
            # Calcular cambios entre etapas
            stages_with_data.sort(key=lambda x: x[1]['cycle'])
            
            for i in range(len(stages_with_data) - 1):
                current_stage = stages_with_data[i][0]
                next_stage = stages_with_data[i + 1][0]
                
                current_metrics = stages_with_data[i][1]['field_stats']
                next_metrics = stages_with_data[i + 1][1]['field_stats']
                
                summary["evolution_analysis"][f"{current_stage}_to_{next_stage}"] = {
                    "entropy_change": next_metrics['entropy'] - current_metrics['entropy'],
                    "variance_change": next_metrics['variance'] - current_metrics['variance'],
                    "symmetry_change": next_metrics['symmetry'] - current_metrics['symmetry'],
                    "cycles_between": stages_with_data[i + 1][1]['cycle'] - stages_with_data[i][1]['cycle']
                }
        
        return summary
    
    def load_fingerprints(self, filepath: str) -> bool:
        """Carga huellas desde un archivo."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.fingerprints = data.get('fingerprints', {})
            self.session_id = data.get('session_id', f"loaded_{int(time.time())}")
            
            print(f"[FingerprintSystem] Huellas cargadas desde: {filepath}")
            return True
            
        except Exception as e:
            print(f"[FingerprintSystem] ERROR cargando huellas: {e}")
            return False
    
    def get_fingerprint_info(self) -> Dict[str, Any]:
        """Retorna información sobre las huellas actuales."""
        return {
            "session_id": self.session_id,
            "fingerprints": {stage: fp is not None for stage, fp in self.fingerprints.items()},
            "summary": self._generate_summary()
        }

class Metamodulo:
    def __init__(self):
        print("[Metamodulo] INFO: Inicializando...")
        self.sensor_module = SensorModule()
        self.core_nucleus = CoreNucleus()
        self.memory_module = MemoryModule()
        self.current_cycle = 0
        
        # Sistema de huellas 3 puntos
        self.fingerprint_system = FingerprintSystem()
        self.max_iterations = 2000
        self.fingerprint_interval = self.max_iterations // 2
    
    def receive_input(self, source: str, source_type: str):
        """Procesa una entrada y la carga en el núcleo. AHORA SEPARADO PARA THREADING."""
        field = self.sensor_module.process(source, source_type)
        self.core_nucleus.receive_field(field)
        print("[Metamodulo] INFO: Entrada procesada y campo recibido por el núcleo.")
    
    def process_step(self) -> Dict[str, Any]:
        self.current_cycle += 1
        print(f"\n--- [Metamodulo] CICLO {self.current_cycle} ---")
        metrics_before = self.core_nucleus.get_metrics()
        decision = self.make_decision(metrics_before)
        print(f"[Metamodulo] Decisión: {decision.upper()}")
        
        # Capturar huella inicial
        if self.current_cycle == 1:
            self.fingerprint_system.capture_fingerprint(
                "inicial", 
                self.core_nucleus.field, 
                self.current_cycle, 
                metrics_before
            )
        
        # Capturar huella intermedia
        elif self.current_cycle == self.fingerprint_interval:
            self.fingerprint_system.capture_fingerprint(
                "intermedia", 
                self.core_nucleus.field, 
                self.current_cycle, 
                metrics_before
            )
        
        # Capturar huella final (último ciclo o cuando se estabilice)
        elif self.current_cycle >= self.max_iterations:
            self.fingerprint_system.capture_fingerprint(
                "final", 
                self.core_nucleus.field, 
                self.current_cycle, 
                metrics_before
            )
        
        if decision == 'exploit':
            attractor_data = self.memory_module.find_closest_attractor(self.core_nucleus.field)
            if attractor_data:
                self.core_nucleus.apply_attractor(attractor_data['pattern'], strength=0.3)
                print("[Metamodulo] Acción: Atractor aplicado.")
        
        self.core_nucleus.reorganize_field()
        print("[Metamodulo] Acción: Campo reorganizado.")
        metrics_after = self.core_nucleus.get_metrics()
        
        return {
            'cycle': self.current_cycle,
            'field': self.core_nucleus.field,
            'metrics': metrics_after,
            'decision': decision,
            'interpretation': interpretar_metrica(metrics_after)
        }
    
    def make_decision(self, metrics: Dict[str, Any]) -> str:
        entropy, variance = metrics.get("entropía", 0.0), metrics.get("varianza", 0.0)
        if entropy > 2.0 or variance > 0.1:
            return "exploit"
        elif entropy < 0.5 and variance < 0.01:
            return "explore"
        else:
            return "stabilize"
    
    def save_session_fingerprints(self, filename: str = None) -> str:
        """Guarda las huellas de la sesión actual."""
        return self.fingerprint_system.save_fingerprints(filename)
    
    def get_fingerprint_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del sistema de huellas."""
        return self.fingerprint_system.get_fingerprint_info()
    
    def learn_from_session(self, input_data: str, input_type: str, user_label: str = None) -> str:
        """Aprende de la sesión actual y la agrega a la memoria."""
        # Se asegura de que haya huellas para aprender
        if self.fingerprint_system.fingerprints and self.fingerprint_system.fingerprints["inicial"] is not None:
            pattern_id = self.fingerprint_system.learning_memory.add_pattern(
                input_data, input_type, self.fingerprint_system.fingerprints, user_label
            )
            return pattern_id
        print("[Metamodulo] WARN: No hay suficientes huellas para aprender de la sesión.")
        return ""
    
    def recognize_pattern(self, input_data: str, input_type: str) -> Optional[Dict[str, Any]]:
        """Reconoce si el input actual es similar a patrones aprendidos."""
        # Se asegura de que haya huellas para comparar
        if self.fingerprint_system.fingerprints and self.fingerprint_system.fingerprints["inicial"] is not None:
            return self.fingerprint_system.learning_memory.find_similar_pattern(
                input_data, input_type, self.fingerprint_system.fingerprints
            )
        print("[Metamodulo] WARN: No hay suficientes huellas para reconocer un patrón.")
        return None
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema de aprendizaje."""
        return self.fingerprint_system.learning_memory.get_learning_stats()

# --- Módulo 5: Sistema de Plugins ---
class DIGPlugin(ABC):
    """Clase base abstracta para todos los plugins del sistema DIG."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Retorna el nombre del plugin."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Retorna la versión del plugin."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Retorna la descripción del plugin."""
        pass
    
    @abstractmethod
    def initialize(self, metamodulo: 'Metamodulo') -> bool:
        """Inicializa el plugin con el metamodulo."""
        pass
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Procesa datos usando el plugin."""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Limpia recursos del plugin."""
        pass

class PluginManager:
    """Gestor de plugins para el sistema DIG."""
    
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, DIGPlugin] = {}
        self.loaded_plugins: Dict[str, DIGPlugin] = {}
        self.metamodulo = None
        
        # Crear directorio de plugins si no existe
        os.makedirs(self.plugin_dir, exist_ok=True)
        
        # Crear archivo __init__.py si no existe
        init_file = os.path.join(self.plugin_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Directorio de plugins del sistema DIG\n")
    
    def set_metamodulo(self, metamodulo: 'Metamodulo'):
        """Establece la referencia al metamodulo para los plugins."""
        self.metamodulo = metamodulo
    
    def discover_plugins(self) -> List[str]:
        """Descubre plugins disponibles en el directorio."""
        available_plugins = []
        
        if not os.path.exists(self.plugin_dir):
            return available_plugins
        
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                plugin_name = filename[:-3]
                available_plugins.append(plugin_name)
        
        return available_plugins
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Carga un plugin específico."""
        try:
            plugin_path = os.path.join(self.plugin_dir, f"{plugin_name}.py")
            
            if not os.path.exists(plugin_path):
                print(f"[PluginManager] ERROR: Plugin '{plugin_name}' no encontrado en {plugin_path}")
                return False
            
            # Cargar plugin dinámicamente
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec is None or spec.loader is None:
                print(f"[PluginManager] ERROR: No se pudo crear spec para '{plugin_name}'")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Verificar que el plugin implementa la interfaz correcta
            if not hasattr(module, 'create_plugin'):
                print(f"[PluginManager] ERROR: Plugin '{plugin_name}' no tiene función 'create_plugin'")
                return False
            
            plugin_instance = module.create_plugin()
            
            # Verificar que el plugin tiene los métodos requeridos
            required_methods = ['get_name', 'get_version', 'get_description', 'initialize', 'process', 'cleanup']
            missing_methods = [method for method in required_methods if not hasattr(plugin_instance, method)]
            
            if missing_methods:
                print(f"[PluginManager] ERROR: Plugin '{plugin_name}' le faltan métodos: {missing_methods}")
                return False
            
            # Inicializar plugin
            if self.metamodulo and plugin_instance.initialize(self.metamodulo):
                self.loaded_plugins[plugin_name] = plugin_instance
                print(f"[PluginManager] INFO: Plugin '{plugin_name}' cargado exitosamente")
                return True
            else:
                print(f"[PluginManager] ERROR: Plugin '{plugin_name}' falló en la inicialización")
                return False
                
        except Exception as e:
            print(f"[PluginManager] ERROR cargando plugin '{plugin_name}': {e}")
            return False
    
    def load_all_plugins(self) -> int:
        """Carga todos los plugins disponibles."""
        available_plugins = self.discover_plugins()
        loaded_count = 0
        
        print(f"[PluginManager] INFO: Descubiertos {len(available_plugins)} plugins")
        
        for plugin_name in available_plugins:
            if self.load_plugin(plugin_name):
                loaded_count += 1
        
        print(f"[PluginManager] INFO: {loaded_count}/{len(available_plugins)} plugins cargados")
        return loaded_count
    
    def get_plugin(self, plugin_name: str) -> Optional[DIGPlugin]:
        """Obtiene un plugin cargado por nombre."""
        return self.loaded_plugins.get(plugin_name)
    
    def get_loaded_plugins(self) -> Dict[str, DIGPlugin]:
        """Retorna todos los plugins cargados."""
        return self.loaded_plugins.copy()
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Descarga un plugin específico."""
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            if plugin.cleanup():
                del self.loaded_plugins[plugin_name]
                print(f"[PluginManager] INFO: Plugin '{plugin_name}' descargado")
                return True
            else:
                print(f"[PluginManager] WARN: Plugin '{plugin_name}' falló en cleanup")
        
        return False
    
    def unload_all_plugins(self):
        """Descarga todos los plugins."""
        plugin_names = list(self.loaded_plugins.keys())
        for plugin_name in plugin_names:
            self.unload_plugin(plugin_name)
    
    def execute_plugin(self, plugin_name: str, data: Any) -> Optional[Any]:
        """Ejecuta un plugin específico con datos."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            try:
                return plugin.process(data)
            except Exception as e:
                print(f"[PluginManager] ERROR ejecutando plugin '{plugin_name}': {e}")
                return None
        return None

# --- Módulo 6: DIGVisualizerApp (GUI Mejorada con Pestañas y Gráficos) ---
class DIGVisualizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧠 Sistema DIG - Organismo Informacional Avanzado")
        self.geometry("1400x900")
        
        # Configurar tema y estilo
        self.setup_theme()
        
        # Inicializar componentes
        self.metamodulo = Metamodulo()
        self.plugin_manager = PluginManager()
        self.plugin_manager.set_metamodulo(self.metamodulo)
        self.is_running = False
        self.cycle_delay = 200
        self.plugin_warning_shown = False # Para mostrar la advertencia de plugins solo una vez
        
        ### SOLUCIÓN: Variables de estado para la entrada de datos activa
        # Esto asegura que las acciones de aprender/reconocer usen los datos correctos.
        self.current_input_data: Optional[str] = None
        self.current_input_type: Optional[str] = None

        ### SOLUCIÓN: Cola para comunicación entre hilos (para procesamiento de archivos)
        self.processing_queue = queue.Queue()
        
        # Datos para gráficos
        self.metrics_history = {
            'entropía': [], 'varianza': [], 'simetría': [], 'active_cells': [], 'inhibited_cells': [], 'neutral_cells': [], 'cycles': []
        }
        self.max_history_points = 100
        
        # Tamaño del canvas
        self.canvas_size = 600
        
        # Configurar interfaz
        self.setup_ui()
        self.load_plugins()
        self.process_text()
        
        # Actualizar estados
        self.update_fingerprint_status()
        self.update_learning_status()
        
        # Centrar ventana
        self.center_window()
        
        # Iniciar actualización de gráficos
        self.update_graphs()
    
    def setup_theme(self):
        """Configura el tema y colores de la aplicación."""
        # Colores modernos
        self.colors = {
            'bg_primary': '#2E3440',      # Fondo principal oscuro
            'bg_secondary': '#3B4252',    # Fondo secundario
            'bg_accent': '#4C566A',       # Fondo de acento
            'text_primary': '#ECEFF4',    # Texto principal claro
            'text_secondary': '#D8DEE9',  # Texto secundario
            'accent_blue': '#5E81AC',     # Azul de acento
            'accent_green': '#A3BE8C',    # Verde de éxito
            'accent_red': '#BF616A',      # Rojo de error
            'accent_orange': '#D08770',   # Naranja de advertencia
            'border': '#434C5E'           # Color de bordes
        }
        
        # Configurar estilo de ttk
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores de widgets
        style.configure('TFrame', background=self.colors['bg_primary'])
        style.configure('TLabel', background=self.colors['bg_primary'], foreground=self.colors['text_primary'])
        style.configure('TButton', background=self.colors['accent_blue'], foreground=self.colors['text_primary'])
        style.map('TButton', background=[('active', self.colors['accent_blue'])])
        style.configure('TLabelframe', background=self.colors['bg_primary'], foreground=self.colors['text_primary'])
        style.configure('TLabelframe.Label', background=self.colors['bg_primary'], foreground=self.colors['text_primary'])
        style.configure('TNotebook', background=self.colors['bg_primary'], borderwidth=0)
        style.configure('TNotebook.Tab', background=self.colors['bg_secondary'], foreground=self.colors['text_secondary'], padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', self.colors['accent_blue'])], foreground=[('selected', self.colors['text_primary'])])
        
        
        # Estilos personalizados para botones
        style.configure('Success.TButton', background=self.colors['accent_green'], foreground=self.colors['text_primary'])
        style.map('Success.TButton', background=[('active', '#B3CC9C')])
        style.configure('Info.TButton', background=self.colors['accent_blue'], foreground=self.colors['text_primary'])
        style.map('Info.TButton', background=[('active', '#6E91BC')])
        style.configure('Warning.TButton', background=self.colors['accent_orange'], foreground=self.colors['text_primary'])
        style.map('Warning.TButton', background=[('active', '#D09780')])
        style.configure('Danger.TButton', background=self.colors['accent_red'], foreground=self.colors['text_primary'])
        style.map('Danger.TButton', background=[('active', '#CF717A')])
    
    def center_window(self):
        """Centra la ventana en la pantalla."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        # Configurar el fondo principal
        self.configure(bg=self.colors['bg_primary'])
        
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Pestaña 1: Visualización Principal
        self.main_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.main_tab, text="🎯 Visualización Principal")
        self.setup_main_tab()
        
        # Pestaña 2: Análisis y Gráficos
        self.analysis_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.analysis_tab, text="📊 Análisis y Gráficos")
        self.setup_analysis_tab()
        
        # Pestaña 3: Sistema de Aprendizaje
        self.learning_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.learning_tab, text="🧠 Aprendizaje")
        self.setup_learning_tab()
        
        # Pestaña 4: Plugins y Herramientas
        self.tools_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tools_tab, text="🔌 Herramientas")
        self.setup_tools_tab()
        
        # Pestaña 5: Configuración
        self.config_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.config_tab, text="⚙️ Configuración")
        self.setup_config_tab()
        
        # Barra de estado en la parte inferior
        self.setup_status_bar()

    def setup_main_tab(self):
        """Configura la pestaña principal con el canvas y controles básicos."""
        # Frame principal con grid para mejor distribución
        main_frame = ttk.Frame(self.main_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar grid weights para distribución flexible
        main_frame.grid_columnconfigure(0, weight=3)  # Canvas (más ancho)
        main_frame.grid_columnconfigure(1, weight=1)  # Controles (más estrecho)
        main_frame.grid_columnconfigure(2, weight=0)  # Scrollbar (ancho fijo)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Frame del canvas (lado izquierdo) - más grande
        canvas_frame = ttk.LabelFrame(main_frame, text="🎯 Campo Informacional", padding="10")
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Canvas para visualización del campo
        self.field_canvas = tk.Canvas(
            canvas_frame, 
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg=self.colors['bg_secondary'],
            highlightthickness=2,
            highlightbackground=self.colors['border']
        )
        self.field_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Frame de controles (lado derecho) - CON SCROLL VERTICAL
        controls_canvas = tk.Canvas(main_frame, bg=self.colors['bg_primary'], highlightthickness=0)
        controls_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=controls_canvas.yview)
        controls_frame = ttk.Frame(controls_canvas)
        
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_canvas.grid(row=0, column=1, sticky="nsew")
        controls_scrollbar.grid(row=0, column=2, sticky="ns")
        controls_canvas.create_window((0, 0), window=controls_frame, anchor="nw")
        
        def configure_scroll(event):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))
        
        controls_frame.bind("<Configure>", configure_scroll)
        
        def on_mousewheel(event):
            controls_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        controls_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # --- Panel de Entrada de Datos ---
        input_frame = ttk.LabelFrame(controls_frame, text="📥 Entrada de Datos", padding="10")
        input_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(input_frame, text="Texto:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.text_input = tk.Text(
            input_frame, height=4, width=40, font=("Consolas", 9),
            bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary'], selectbackground=self.colors['accent_blue'],
            relief="flat", padx=8, pady=8
        )
        self.text_input.insert("1.0", "La gravedad emerge de un balance dinámico entre desorden (S) y orden (I).")
        self.text_input.pack(fill="x", pady=(0, 8))
        
        input_buttons_frame = ttk.Frame(input_frame)
        input_buttons_frame.pack(fill="x")
        
        self.process_text_button = ttk.Button(
            input_buttons_frame, text="🔄 Procesar Texto", command=self.process_text, style='Info.TButton'
        )
        self.process_text_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        self.process_file_button = ttk.Button(
            input_buttons_frame, text="📁 Cargar Archivo", command=self.load_file, style='Info.TButton'
        )
        self.process_file_button.pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        self.loaded_file_label = ttk.Label(
            input_frame, text="📄 Archivo: Ninguno", font=("Helvetica", 8, "italic"),
            foreground=self.colors['text_secondary']
        )
        self.loaded_file_label.pack(anchor=tk.W, pady=(8, 0))

        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # --- Panel de Simulación ---
        simulation_frame = ttk.LabelFrame(controls_frame, text="⚡ Control de Simulación", padding="10")
        simulation_frame.pack(fill="x", pady=(0, 15))
        
        speed_frame = ttk.Frame(simulation_frame)
        speed_frame.pack(fill="x", pady=(0, 8))
        
        ttk.Label(speed_frame, text="Velocidad:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="200")
        speed_spinbox = ttk.Spinbox(
            speed_frame, from_=50, to=2000, textvariable=self.speed_var, width=8,
            command=self.update_simulation_speed
        )
        speed_spinbox.pack(side=tk.RIGHT)
        ttk.Label(speed_frame, text="ms").pack(side=tk.RIGHT, padx=(0, 5))
        
        sim_buttons_frame = ttk.Frame(simulation_frame)
        sim_buttons_frame.pack(fill="x", pady=(0, 8))
        
        self.run_button = ttk.Button(
            sim_buttons_frame, text="▶ Iniciar", command=self.toggle_simulation, style='Success.TButton'
        )
        self.run_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        self.step_button = ttk.Button(
            sim_buttons_frame, text="⏭ Paso", command=self.run_single_step, style='Info.TButton'
        )
        self.step_button.pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        self.status_label = ttk.Label(
            simulation_frame, text="⏸ Estado: En espera", font=("Helvetica", 9, "italic"),
            foreground=self.colors['text_secondary']
        )
        self.status_label.pack(anchor=tk.W)
        
        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # --- Panel de Interpretación ---
        interpretation_frame = ttk.LabelFrame(controls_frame, text="🧠 Interpretación de la IA", padding="10")
        interpretation_frame.pack(fill="x", pady=(0, 15))
        
        metrics_frame = ttk.Frame(interpretation_frame)
        metrics_frame.pack(fill="x", pady=(0, 8))
        
        cycle_frame = ttk.Frame(metrics_frame)
        cycle_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(cycle_frame, text="🔄 Ciclo:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        self.cycle_label = ttk.Label(cycle_frame, text="0", font=("Helvetica", 9, "bold"), foreground=self.colors['accent_green'])
        self.cycle_label.pack(side=tk.RIGHT)
        
        decision_frame = ttk.Frame(metrics_frame)
        decision_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(decision_frame, text="🎯 Decisión:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        self.decision_label = ttk.Label(decision_frame, text="N/A", font=("Helvetica", 9, "bold"), foreground=self.colors['accent_blue'])
        self.decision_label.pack(side=tk.RIGHT)
        
        ### SOLUCIÓN: Los paneles de Huellas y Aprendizaje se eliminaron de esta
        ### pestaña para evitar redundancia, ya que tienen sus propias pestañas.
    
    def setup_analysis_tab(self):
        """Configura la pestaña de análisis y gráficos."""
        # Frame principal
        main_frame = ttk.Frame(self.analysis_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar grid
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # Gráfico de métricas en tiempo real
        metrics_frame = ttk.LabelFrame(main_frame, text="📊 Métricas en Tiempo Real", padding="10")
        metrics_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))
        
        if MATPLOTLIB_AVAILABLE:
            # Crear figura de matplotlib
            self.metrics_fig = Figure(figsize=(6, 4), dpi=100, facecolor=self.colors['bg_secondary'])
            self.metrics_ax = self.metrics_fig.add_subplot(111)
            self.metrics_ax.set_facecolor(self.colors['bg_secondary'])
            
            # Configurar colores del gráfico
            self.metrics_ax.tick_params(colors=self.colors['text_primary'])
            self.metrics_ax.spines['bottom'].set_color(self.colors['text_primary'])
            self.metrics_ax.spines['top'].set_color(self.colors['text_primary'])
            self.metrics_ax.spines['left'].set_color(self.colors['text_primary'])
            self.metrics_ax.spines['right'].set_color(self.colors['text_primary'])
            
            # Crear canvas de tkinter
            self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, metrics_frame)
            self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Inicializar gráfico
            self.update_metrics_graph()
            
            # Botón de prueba para métricas
            ttk.Button(metrics_frame, text="🧪 Probar Métricas", 
                      command=self.test_metrics, style='Info.TButton').pack(pady=(5, 0))
        else:
            # Fallback si matplotlib no está disponible
            ttk.Label(metrics_frame, text="⚠️ Matplotlib no disponible\nLos gráficos están deshabilitados",
                     font=("Helvetica", 12), foreground=self.colors['accent_orange']).pack(expand=True)
        
        # Gráfico de evolución del campo
        evolution_frame = ttk.LabelFrame(main_frame, text="🔄 Evolución del Campo", padding="10")
        evolution_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=(0, 5))
        
        if MATPLOTLIB_AVAILABLE:
            # Crear figura para evolución
            self.evolution_fig = Figure(figsize=(6, 4), dpi=100, facecolor=self.colors['bg_secondary'])
            self.evolution_ax = self.evolution_fig.add_subplot(111)
            self.evolution_ax.set_facecolor(self.colors['bg_secondary'])
            
            # Configurar colores
            self.evolution_ax.tick_params(colors=self.colors['text_primary'])
            self.evolution_ax.spines['bottom'].set_color(self.colors['text_primary'])
            self.evolution_ax.spines['top'].set_color(self.colors['text_primary'])
            self.evolution_ax.spines['left'].set_color(self.colors['text_primary'])
            self.evolution_ax.spines['right'].set_color(self.colors['text_primary'])
            
            # Crear canvas
            self.evolution_canvas = FigureCanvasTkAgg(self.evolution_fig, evolution_frame)
            self.evolution_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Inicializar gráfico de evolución
            self.update_evolution_graph()
        else:
            ttk.Label(evolution_frame, text="⚠️ Matplotlib no disponible\nLos gráficos están deshabilitados",
                     font=("Helvetica", 12), foreground=self.colors['accent_orange']).pack(expand=True)
        
        # Panel de estadísticas avanzadas
        stats_frame = ttk.LabelFrame(main_frame, text="📈 Estadísticas Avanzadas", padding="10")
        stats_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(5, 0))
        
        # Crear frame con scroll para estadísticas
        stats_canvas = tk.Canvas(stats_frame, bg=self.colors['bg_primary'], highlightthickness=0)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=stats_canvas.yview)
        stats_inner_frame = ttk.Frame(stats_canvas)
        
        stats_canvas.configure(yscrollcommand=stats_scrollbar.set)
        stats_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        stats_canvas.create_window((0, 0), window=stats_inner_frame, anchor="nw")
        
        def configure_stats_scroll(event):
            stats_canvas.configure(scrollregion=stats_canvas.bbox("all"))
        
        stats_inner_frame.bind("<Configure>", configure_stats_scroll)
        
        # Estadísticas detalladas
        self.setup_detailed_stats(stats_inner_frame)
    
    def setup_detailed_stats(self, parent_frame):
        """Configura las estadísticas detalladas."""
        # Métricas principales
        main_metrics_frame = ttk.Frame(parent_frame)
        main_metrics_frame.pack(fill="x", pady=(0, 15))
        
        # Fila 1: Métricas básicas
        row1 = ttk.Frame(main_metrics_frame)
        row1.pack(fill="x", pady=(0, 10))
        
        # Entropía
        entropy_frame = ttk.LabelFrame(row1, text="Entropía", padding="5")
        entropy_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        self.entropy_value_label = ttk.Label(entropy_frame, text="0.000", font=("Helvetica", 12, "bold"))
        self.entropy_value_label.pack()
        
        # Varianza
        variance_frame = ttk.LabelFrame(row1, text="Varianza", padding="5")
        variance_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        self.variance_value_label = ttk.Label(variance_frame, text="0.000", font=("Helvetica", 12, "bold"))
        self.variance_value_label.pack()
        
        # Simetría
        symmetry_frame = ttk.LabelFrame(row1, text="Simetría", padding="5")
        symmetry_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        self.symmetry_value_label = ttk.Label(symmetry_frame, text="0.000", font=("Helvetica", 12, "bold"))
        self.symmetry_value_label.pack()
        
        # Fila 2: Distribución de células
        row2 = ttk.Frame(main_metrics_frame)
        row2.pack(fill="x", pady=(0, 10))
        
        # Células activas
        active_frame = ttk.LabelFrame(row2, text="Células Activas", padding="5")
        active_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        self.active_cells_label = ttk.Label(active_frame, text="0", font=("Helvetica", 12, "bold"))
        self.active_cells_label.pack()
        
        # Células inhibidas
        inhibited_frame = ttk.LabelFrame(row2, text="Células Inhibidas", padding="5")
        inhibited_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        self.inhibited_cells_label = ttk.Label(inhibited_frame, text="0", font=("Helvetica", 12, "bold"))
        self.inhibited_cells_label.pack()
        
        # Células neutrales
        neutral_frame = ttk.LabelFrame(row2, text="Células Neutrales", padding="5")
        neutral_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        self.neutral_cells_label = ttk.Label(neutral_frame, text="0", font=("Helvetica", 12, "bold"))
        self.neutral_cells_label.pack()
        
        # Botones de exportación
        export_frame = ttk.Frame(parent_frame)
        export_frame.pack(fill="x", pady=(15, 0))
        
        ttk.Button(
            export_frame, text="📊 Exportar Métricas (CSV)", 
            command=self.export_metrics_csv, style='Info.TButton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            export_frame, text="🖼️ Exportar Campo (PNG)", 
            command=self.export_field_image, style='Info.TButton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            export_frame, text="📈 Exportar Gráficos (PNG)", 
            command=self.export_graphs, style='Info.TButton'
        ).pack(side=tk.LEFT)
    
    def setup_learning_tab(self):
        """Configura la pestaña de sistema de aprendizaje."""
        # Frame principal
        main_frame = ttk.Frame(self.learning_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar grid
        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Panel izquierdo: Sistema de huellas
        fingerprint_frame = ttk.LabelFrame(main_frame, text="🔄 Sistema de Huellas", padding="15")
        fingerprint_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Estado de las huellas
        ttk.Label(fingerprint_frame, text="📊 Estado:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(0, 10))
        self.fingerprint_status_label = ttk.Label(
            fingerprint_frame, text="⏳ Inicializando...", font=("Helvetica", 9),
            foreground=self.colors['text_secondary']
        )
        self.fingerprint_status_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Botones de huellas
        fingerprint_buttons_frame = ttk.Frame(fingerprint_frame)
        fingerprint_buttons_frame.pack(fill="x", pady=(0, 15))
        
        self.save_fingerprints_button = ttk.Button(
            fingerprint_buttons_frame, text="💾 Guardar Huellas", 
            command=self.save_fingerprints, style='Info.TButton'
        )
        self.save_fingerprints_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        self.load_fingerprints_button = ttk.Button(
            fingerprint_buttons_frame, text="📂 Cargar Huellas", 
            command=self.load_fingerprints, style='Info.TButton'
        )
        self.load_fingerprints_button.pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        # Información de la sesión actual
        session_frame = ttk.LabelFrame(fingerprint_frame, text="📋 Sesión Actual", padding="10")
        session_frame.pack(fill="x", pady=(0, 15))
        
        # Lista de huellas capturadas
        ttk.Label(session_frame, text="Huellas capturadas:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.fingerprints_listbox = tk.Listbox(
            session_frame, height=6, font=("Consolas", 8),
            bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
            selectbackground=self.colors['accent_blue'], relief="flat", borderwidth=0
        )
        self.fingerprints_listbox.pack(fill="x", pady=(0, 10))
        
        # Botón para limpiar sesión
        ttk.Button(
            session_frame, text="🗑️ Limpiar Sesión", 
            command=self.clear_fingerprint_session, style='Danger.TButton'
        ).pack(fill="x")
        
        # Panel derecho: Sistema de aprendizaje
        learning_frame = ttk.LabelFrame(main_frame, text="🧠 Sistema de Aprendizaje", padding="15")
        learning_frame.grid(row=0, column=1, sticky="nsew")
        
        # Estado del aprendizaje
        ttk.Label(learning_frame, text="📊 Estado:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(0, 10))
        self.learning_status_label = ttk.Label(
            learning_frame, text="⏳ Inicializando...", font=("Helvetica", 9),
            foreground=self.colors['text_secondary']
        )
        self.learning_status_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Campo para etiqueta personalizada
        label_frame = ttk.Frame(learning_frame)
        label_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(label_frame, text="🏷️ Etiqueta:", font=("Helvetica", 9)).pack(anchor=tk.W, pady=(0, 5))
        self.custom_label_var = tk.StringVar(value="")
        self.custom_label_entry = ttk.Entry(
            label_frame, textvariable=self.custom_label_var, font=("Helvetica", 9)
        )
        self.custom_label_entry.pack(fill="x")
        
        # Botones de aprendizaje
        learning_buttons_frame = ttk.Frame(learning_frame)
        learning_buttons_frame.pack(fill="x", pady=(0, 15))
        
        self.learn_button = ttk.Button(
            learning_buttons_frame, text="🎓 Aprender", 
            command=self.learn_from_current_session, style='Success.TButton'
        )
        self.learn_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        self.recognize_button = ttk.Button(
            learning_buttons_frame, text="🔍 Reconocer", 
            command=self.recognize_current_pattern, style='Info.TButton'
        )
        self.recognize_button.pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        # Estadísticas de aprendizaje
        stats_frame = ttk.LabelFrame(learning_frame, text="📈 Estadísticas", padding="10")
        stats_frame.pack(fill="x", pady=(0, 15))
        
        # Información de patrones aprendidos
        ttk.Label(stats_frame, text="Patrones aprendidos:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.patterns_info_label = ttk.Label(
            stats_frame, text="0 patrones", font=("Helvetica", 9),
            foreground=self.colors['text_secondary']
        )
        self.patterns_info_label.pack(anchor=tk.W)
        
        # Botón para ver estadísticas detalladas
        ttk.Button(
            stats_frame, text="📊 Ver Estadísticas Detalladas", 
            command=self.show_learning_stats, style='Info.TButton'
        ).pack(fill="x", pady=(10, 0))
        
        # Panel inferior: Logs de aprendizaje
        logs_frame = ttk.LabelFrame(main_frame, text="📝 Logs de Aprendizaje", padding="10")
        logs_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        
        # Área de logs
        self.learning_logs_text = tk.Text(
            logs_frame, height=8, font=("Consolas", 8), wrap=tk.WORD,
            bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
            relief="flat", padx=8, pady=8, state=tk.DISABLED
        )
        
        logs_scrollbar = ttk.Scrollbar(logs_frame, orient="vertical", command=self.learning_logs_text.yview)
        self.learning_logs_text.configure(yscrollcommand=logs_scrollbar.set)
        
        self.learning_logs_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        logs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        
        # Botón para limpiar logs
        ttk.Button(
            logs_frame, text="🗑️ Limpiar Logs", 
            command=self.clear_learning_logs, style='Danger.TButton'
        ).pack(anchor=tk.E, pady=(5, 0))
    
    def setup_tools_tab(self):
        """Configura la pestaña de herramientas y plugins."""
        # Frame principal
        main_frame = ttk.Frame(self.tools_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar grid
        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Panel izquierdo: Sistema de Plugins
        plugins_frame = ttk.LabelFrame(main_frame, text="🔌 Sistema de Plugins", padding="15")
        plugins_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Botón para recargar plugins
        ttk.Button(
            plugins_frame, text="🔄 Recargar Plugins", 
            command=self.reload_plugins, style='Info.TButton'
        ).pack(fill="x", pady=(0, 15))
        
        # Lista de plugins cargados
        ttk.Label(plugins_frame, text="📦 Plugins Activos:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        plugins_list_frame = ttk.Frame(plugins_frame)
        plugins_list_frame.pack(fill="x", expand=True, pady=(0, 15))
        
        self.plugins_listbox = tk.Listbox(
            plugins_list_frame, height=8, font=("Consolas", 9),
            bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
            selectbackground=self.colors['accent_blue'], relief="flat", borderwidth=0
        )
        
        plugins_scrollbar = ttk.Scrollbar(plugins_list_frame, orient="vertical", command=self.plugins_listbox.yview)
        self.plugins_listbox.configure(yscrollcommand=plugins_scrollbar.set)

        self.plugins_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        plugins_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        
        # Botones de control de plugins
        plugins_buttons_frame = ttk.Frame(plugins_frame)
        plugins_buttons_frame.pack(fill="x")
        
        self.execute_plugin_button = ttk.Button(
            plugins_buttons_frame, text="▶ Ejecutar Plugin", 
            command=self.execute_selected_plugin, style='Success.TButton'
        )
        self.execute_plugin_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        ttk.Button(
            plugins_buttons_frame, text="ℹ️ Info Plugin", 
            command=self.show_plugin_info, style='Info.TButton'
        ).pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        # Panel derecho: Herramientas Avanzadas
        tools_frame = ttk.LabelFrame(main_frame, text="🛠️ Herramientas Avanzadas", padding="15")
        tools_frame.grid(row=0, column=1, sticky="nsew")
        
        # Herramientas de análisis
        analysis_tools_frame = ttk.LabelFrame(tools_frame, text="📊 Análisis", padding="10")
        analysis_tools_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Button(
            analysis_tools_frame, text="🔍 Análisis de Patrones", 
            command=self.analyze_patterns, style='Info.TButton'
        ).pack(fill="x", pady=(0, 5))
        
        ttk.Button(
            analysis_tools_frame, text="📈 Comparación de Sesiones", 
            command=self.compare_sessions, style='Info.TButton'
        ).pack(fill="x", pady=(0, 5))
        
        ttk.Button(
            analysis_tools_frame, text="🎯 Detección de Anomalías", 
            command=self.detect_anomalies, style='Info.TButton'
        ).pack(fill="x")
        
        # Herramientas de exportación
        export_tools_frame = ttk.LabelFrame(tools_frame, text="📤 Exportación", padding="10")
        export_tools_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Button(
            export_tools_frame, text="📊 Reporte Completo (PDF)", 
            command=self.export_complete_report, style='Info.TButton'
        ).pack(fill="x", pady=(0, 5))
        
        ttk.Button(
            export_tools_frame, text="🎬 Video de Evolución", 
            command=self.export_evolution_video, style='Info.TButton'
        ).pack(fill="x", pady=(0, 5))
        
        ttk.Button(
            export_tools_frame, text="📁 Backup Completo", 
            command=self.create_backup, style='Info.TButton'
        ).pack(fill="x")
        
        # Herramientas de mantenimiento
        maintenance_frame = ttk.LabelFrame(tools_frame, text="🔧 Mantenimiento", padding="10")
        maintenance_frame.pack(fill="x")
        
        ttk.Button(
            maintenance_frame, text="🧹 Limpiar Datos Temporales", 
            command=self.cleanup_temp_data, style='Warning.TButton'
        ).pack(fill="x", pady=(0, 5))
        
        ttk.Button(
            maintenance_frame, text="📊 Optimizar Memoria", 
            command=self.optimize_memory, style='Warning.TButton'
        ).pack(fill="x", pady=(0, 5))
        
        ttk.Button(
            maintenance_frame, text="🔄 Reiniciar Sistema", 
            command=self.restart_system, style='Danger.TButton'
        ).pack(fill="x")
        
        # Panel inferior: Logs del Sistema
        logs_frame = ttk.LabelFrame(main_frame, text="📝 Logs del Sistema", padding="10")
        logs_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        
        # Área de logs del sistema
        self.system_logs_text = tk.Text(
            logs_frame, height=6, font=("Consolas", 8), wrap=tk.WORD,
            bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
            relief="flat", padx=8, pady=8, state=tk.DISABLED
        )
        system_logs_scrollbar = ttk.Scrollbar(logs_frame, orient="vertical", command=self.system_logs_text.yview)
        self.system_logs_text.configure(yscrollcommand=system_logs_scrollbar.set)
        
        self.system_logs_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        system_logs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botones de control de logs
        logs_buttons_frame = ttk.Frame(logs_frame)
        logs_buttons_frame.pack(fill="x", pady=(5, 0))
        
        ttk.Button(
            logs_buttons_frame, text="🗑️ Limpiar Logs", 
            command=self.clear_system_logs, style='Danger.TButton'
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            logs_buttons_frame, text="💾 Guardar Logs", 
            command=self.save_system_logs, style='Info.TButton'
        ).pack(side=tk.RIGHT)
    
    def setup_config_tab(self):
        """Configura la pestaña de configuración del sistema."""
        # Frame principal
        main_frame = ttk.Frame(self.config_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar grid
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Panel izquierdo: Configuración General
        general_frame = ttk.LabelFrame(main_frame, text="⚙️ Configuración General", padding="15")
        general_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Tamaño del campo
        field_size_frame = ttk.Frame(general_frame)
        field_size_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(field_size_frame, text="Tamaño del campo:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        size_frame = ttk.Frame(field_size_frame)
        size_frame.pack(fill="x")
        
        ttk.Label(size_frame, text="Ancho:").pack(side=tk.LEFT)
        self.field_width_var = tk.StringVar(value="64")
        width_spinbox = ttk.Spinbox(size_frame, from_=32, to=128, textvariable=self.field_width_var, width=8)
        width_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(size_frame, text="Alto:").pack(side=tk.LEFT)
        self.field_height_var = tk.StringVar(value="64")
        height_spinbox = ttk.Spinbox(size_frame, from_=32, to=128, textvariable=self.field_height_var, width=8)
        height_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Configuración de simulación
        sim_config_frame = ttk.Frame(general_frame)
        sim_config_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(sim_config_frame, text="Configuración de simulación:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Máximo de iteraciones
        max_iter_frame = ttk.Frame(sim_config_frame)
        max_iter_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(max_iter_frame, text="Máximo de iteraciones:").pack(side=tk.LEFT)
        self.max_iterations_var = tk.StringVar(value="2000")
        max_iter_spinbox = ttk.Spinbox(max_iter_frame, from_=100, to=10000, textvariable=self.max_iterations_var, width=10)
        max_iter_spinbox.pack(side=tk.RIGHT)
        
        # Intervalo de huellas
        fingerprint_interval_frame = ttk.Frame(sim_config_frame)
        fingerprint_interval_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(fingerprint_interval_frame, text="Intervalo de huellas:").pack(side=tk.LEFT)
        self.fingerprint_interval_var = tk.StringVar(value="1000")
        interval_spinbox = ttk.Spinbox(fingerprint_interval_frame, from_=100, to=5000, textvariable=self.fingerprint_interval_var, width=10)
        interval_spinbox.pack(side=tk.RIGHT)
        
        # Configuración de memoria
        memory_config_frame = ttk.Frame(general_frame)
        memory_config_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(memory_config_frame, text="Configuración de memoria:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Máximo de patrones
        max_patterns_frame = ttk.Frame(memory_config_frame)
        max_patterns_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(max_patterns_frame, text="Máximo de patrones:").pack(side=tk.LEFT)
        self.max_patterns_var = tk.StringVar(value="100")
        max_patterns_spinbox = ttk.Spinbox(max_patterns_frame, from_=10, to=1000, textvariable=self.max_patterns_var, width=10)
        max_patterns_spinbox.pack(side=tk.RIGHT)
        
        # Umbral de similitud
        similarity_threshold_frame = ttk.Frame(memory_config_frame)
        similarity_threshold_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(similarity_threshold_frame, text="Umbral de similitud:").pack(side=tk.LEFT)
        self.similarity_threshold_var = tk.StringVar(value="0.85")
        threshold_spinbox = ttk.Spinbox(similarity_threshold_frame, from_=0.1, to=1.0, increment=0.05, textvariable=self.similarity_threshold_var, width=10)
        threshold_spinbox.pack(side=tk.RIGHT)
        
        # Botones de configuración
        config_buttons_frame = ttk.Frame(general_frame)
        config_buttons_frame.pack(fill="x")
        
        ttk.Button(
            config_buttons_frame, text="💾 Guardar Configuración", 
            command=self.save_configuration, style='Success.TButton'
        ).pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        ttk.Button(
            config_buttons_frame, text="🔄 Aplicar Cambios", 
            command=self.apply_configuration, style='Info.TButton'
        ).pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        # Panel derecho: Configuración Avanzada
        advanced_frame = ttk.LabelFrame(main_frame, text="🔧 Configuración Avanzada", padding="15")
        advanced_frame.grid(row=0, column=1, sticky="nsew")
        
        # Configuración de plugins
        plugins_config_frame = ttk.Frame(advanced_frame)
        plugins_config_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(plugins_config_frame, text="Configuración de plugins:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Directorio de plugins
        plugin_dir_frame = ttk.Frame(plugins_config_frame)
        plugin_dir_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(plugin_dir_frame, text="Directorio:").pack(side=tk.LEFT)
        self.plugin_dir_var = tk.StringVar(value="plugins")
        plugin_dir_entry = ttk.Entry(plugin_dir_frame, textvariable=self.plugin_dir_var)
        plugin_dir_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 5))
        ttk.Button(plugin_dir_frame, text="📁", command=self.browse_plugin_dir, width=3).pack(side=tk.RIGHT)
        
        # Auto-carga de plugins
        auto_load_frame = ttk.Frame(plugins_config_frame)
        auto_load_frame.pack(fill="x", pady=(0, 5))
        self.auto_load_plugins_var = tk.BooleanVar(value=True)
        auto_load_check = ttk.Checkbutton(auto_load_frame, text="Auto-cargar plugins al iniciar", 
                                        variable=self.auto_load_plugins_var)
        auto_load_check.pack(anchor=tk.W)
        
        # Configuración de logs
        logs_config_frame = ttk.Frame(advanced_frame)
        logs_config_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(logs_config_frame, text="Configuración de logs:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Nivel de log
        log_level_frame = ttk.Frame(logs_config_frame)
        log_level_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(log_level_frame, text="Nivel:").pack(side=tk.LEFT)
        self.log_level_var = tk.StringVar(value="INFO")
        log_level_combo = ttk.Combobox(log_level_frame, textvariable=self.log_level_var, 
                                      values=["DEBUG", "INFO", "WARNING", "ERROR"], state="readonly", width=10)
        log_level_combo.pack(side=tk.RIGHT)
        
        # Directorio de logs
        log_dir_frame = ttk.Frame(logs_config_frame)
        log_dir_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(log_dir_frame, text="Directorio:").pack(side=tk.LEFT)
        self.log_dir_var = tk.StringVar(value="logs")
        log_dir_entry = ttk.Entry(log_dir_frame, textvariable=self.log_dir_var)
        log_dir_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 5))
        ttk.Button(log_dir_frame, text="📁", command=self.browse_log_dir, width=3).pack(side=tk.RIGHT)
        
        # Configuración de tema
        theme_config_frame = ttk.Frame(advanced_frame)
        theme_config_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(theme_config_frame, text="Configuración de tema:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Tema de color
        theme_frame = ttk.Frame(theme_config_frame)
        theme_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(theme_frame, text="Tema:").pack(side=tk.LEFT)
        self.theme_var = tk.StringVar(value="Oscuro")
        theme_combo = ttk.Combobox(theme_frame, textvariable=self.theme_var, 
                                  values=["Oscuro", "Claro", "Automático"], state="readonly", width=10)
        theme_combo.pack(side=tk.RIGHT)
        
        # Botones de configuración avanzada
        advanced_buttons_frame = ttk.Frame(advanced_frame)
        advanced_buttons_frame.pack(fill="x")
        
        ttk.Button(
            advanced_buttons_frame, text="🔄 Restaurar Predeterminados", 
            command=self.restore_defaults, style='Warning.TButton'
        ).pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        ttk.Button(
            advanced_buttons_frame, text="📤 Exportar Configuración", 
            command=self.export_configuration, style='Info.TButton'
        ).pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        # Panel inferior: Información del Sistema
        system_info_frame = ttk.LabelFrame(main_frame, text="ℹ️ Información del Sistema", padding="15")
        system_info_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        
        # Información del sistema
        info_text = f"""
        🖥️ Sistema: {os.name}
        🐍 Python: {os.sys.version.split()[0]}
        📁 Directorio de trabajo: {os.getcwd()}
        🧠 Módulos cargados: {len(self.metamodulo.__dict__)}
        🔌 Plugins disponibles: {len(self.plugin_manager.discover_plugins())}
        """
        
        self.system_info_label = ttk.Label(system_info_frame, text=info_text, font=("Consolas", 9), 
                                         foreground=self.colors['text_secondary'])
        self.system_info_label.pack(anchor=tk.W)
    
    def setup_status_bar(self):
        """Configura la barra de estado en la parte inferior."""
        # Frame de estado
        status_frame = ttk.Frame(self)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Separador
        ttk.Separator(status_frame, orient="horizontal").pack(fill="x")
        
        # Frame de información de estado
        info_frame = ttk.Frame(status_frame)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        # Estado del sistema
        self.system_status_label = ttk.Label(
            info_frame, text="✅ Sistema: Listo", font=("Helvetica", 8),
            foreground=self.colors['accent_green']
        )
        self.system_status_label.pack(side=tk.LEFT)
        
        # Información de memoria
        self.memory_info_label = ttk.Label(
            info_frame, text="💾 Memoria: 0 patrones", font=("Helvetica", 8),
            foreground=self.colors['text_secondary']
        )
        self.memory_info_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Información de plugins
        self.plugins_info_label = ttk.Label(
            info_frame, text="🔌 Plugins: 0 cargados", font=("Helvetica", 8),
            foreground=self.colors['text_secondary']
        )
        self.plugins_info_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Información de tiempo
        self.time_info_label = ttk.Label(
            info_frame, text="🕐 Iniciado: Ahora", font=("Helvetica", 8),
            foreground=self.colors['text_secondary']
        )
        self.time_info_label.pack(side=tk.RIGHT)
        
        # Iniciar actualización de estado
        self.update_status_bar()
    
    def update_graphs(self):
        """Actualiza los gráficos en tiempo real."""
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'metrics_ax'):
            try:
                # Actualizar gráfico de métricas
                self.update_metrics_graph()
                # Actualizar gráfico de evolución
                self.update_evolution_graph()
            except Exception as e:
                print(f"Error actualizando gráficos: {e}")
        
        # Programar próxima actualización
        self.after(1000, self.update_graphs)  # Actualizar cada segundo
    
    def update_metrics_graph(self):
        """Actualiza el gráfico de métricas en tiempo real."""
        if not hasattr(self, 'metrics_ax') or not self.metrics_history['cycles']:
            return
        
        try:
            self.metrics_ax.clear()
            
            # Configurar colores
            self.metrics_ax.set_facecolor(self.colors['bg_secondary'])
            self.metrics_ax.tick_params(colors=self.colors['text_primary'], labelcolor=self.colors['text_secondary'])
            self.metrics_ax.spines['bottom'].set_color(self.colors['border'])
            self.metrics_ax.spines['top'].set_color(self.colors['border'])
            self.metrics_ax.spines['left'].set_color(self.colors['border'])
            self.metrics_ax.spines['right'].set_color(self.colors['border'])
            
            # Graficar métricas
            cycles = self.metrics_history['cycles']
            if len(cycles) > 1:
                self.metrics_ax.plot(cycles, self.metrics_history['entropía'], 
                                   label='Entropía', color='#5E81AC', linewidth=2)
                self.metrics_ax.plot(cycles, self.metrics_history['varianza'], 
                                   label='Varianza', color='#A3BE8C', linewidth=2)
                self.metrics_ax.plot(cycles, self.metrics_history['simetría'], 
                                   label='Simetría', color='#D08770', linewidth=2)
                
                self.metrics_ax.set_xlabel('Ciclo', color=self.colors['text_secondary'])
                self.metrics_ax.set_ylabel('Valor', color=self.colors['text_secondary'])
                self.metrics_ax.set_title('Métricas en Tiempo Real', color=self.colors['text_primary'])
                legend = self.metrics_ax.legend(facecolor=self.colors['bg_accent'], 
                                     edgecolor=self.colors['border'], labelcolor=self.colors['text_primary'])
                for text in legend.get_texts():
                    text.set_color(self.colors['text_primary'])
                self.metrics_ax.grid(True, alpha=0.3, color=self.colors['border'])
            
            self.metrics_canvas.draw()
            
        except Exception as e:
            print(f"Error actualizando gráfico de métricas: {e}")
    
    def update_evolution_graph(self):
        """Actualiza el gráfico de evolución del campo."""
        if not hasattr(self, 'evolution_ax') or not self.metrics_history['cycles']:
            return
        
        try:
            self.evolution_ax.clear()
            
            # Configurar colores
            self.evolution_ax.set_facecolor(self.colors['bg_secondary'])
            self.evolution_ax.tick_params(colors=self.colors['text_primary'], labelcolor=self.colors['text_secondary'])
            self.evolution_ax.spines['bottom'].set_color(self.colors['border'])
            self.evolution_ax.spines['top'].set_color(self.colors['border'])
            self.evolution_ax.spines['left'].set_color(self.colors['border'])
            self.evolution_ax.spines['right'].set_color(self.colors['border'])
            
            # Graficar evolución de células
            cycles = self.metrics_history['cycles']
            if len(cycles) > 1:
                self.evolution_ax.plot(cycles, self.metrics_history['active_cells'], 
                                     label='Activas', color='#A3BE8C', linewidth=2)
                self.evolution_ax.plot(cycles, self.metrics_history['inhibited_cells'], 
                                     label='Inhibidas', color='#BF616A', linewidth=2)
                self.evolution_ax.plot(cycles, self.metrics_history['neutral_cells'], 
                                     label='Neutrales', color='#D8DEE9', linewidth=2)
                
                self.evolution_ax.set_xlabel('Ciclo', color=self.colors['text_secondary'])
                self.evolution_ax.set_ylabel('Número de Células', color=self.colors['text_secondary'])
                self.evolution_ax.set_title('Evolución del Campo', color=self.colors['text_primary'])
                legend = self.evolution_ax.legend(facecolor=self.colors['bg_accent'], 
                                       edgecolor=self.colors['border'])
                for text in legend.get_texts():
                    text.set_color(self.colors['text_primary'])
                self.evolution_ax.grid(True, alpha=0.3, color=self.colors['border'])
            
            self.evolution_canvas.draw()
            
        except Exception as e:
            print(f"Error actualizando gráfico de evolución: {e}")
    
    def update_status_bar(self):
        """Actualiza la barra de estado."""
        try:
            # Actualizar información de memoria
            learning_stats = self.metamodulo.get_learning_stats()
            total_patterns = learning_stats.get('total_patterns', 0)
            self.memory_info_label.config(text=f"💾 Memoria: {total_patterns} patrones")
            
            # Actualizar información de plugins
            loaded_plugins = len(self.plugin_manager.get_loaded_plugins())
            self.plugins_info_label.config(text=f"🔌 Plugins: {loaded_plugins} cargados")
            
            # Actualizar tiempo
            current_time = datetime.now().strftime("%H:%M:%S")
            self.time_info_label.config(text=f"🕐 Última actualización: {current_time}")
            
        except Exception as e:
            print(f"Error actualizando barra de estado: {e}")
        
        # Programar próxima actualización
        self.after(5000, self.update_status_bar)  # Actualizar cada 5 segundos

    def update_field_canvas(self, field: np.ndarray):
        if field.size == 0: return
        norm = colors.Normalize(vmin=np.min(field), vmax=np.max(field))
        colormap = cm.get_cmap('plasma')
        
        rgba_field = colormap(norm(field))
        img_data = (rgba_field[:, :, :3] * 255).astype(np.uint8)
        
        img = Image.fromarray(img_data)
        
        img_resized = img.resize((self.canvas_size, self.canvas_size), Image.Resampling.NEAREST)
        self.photo_image = ImageTk.PhotoImage(image=img_resized)
        self.field_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

    def update_display(self, summary: Dict[str, Any]):
        self.update_field_canvas(summary.get('field'))
        
        # Actualizar métricas en tiempo real
        cycle = summary.get('cycle', 0)
        decision = summary.get('decision', 'N/A').upper()
        metrics = summary.get('metrics', {})
        
        self.cycle_label.config(text=str(cycle))
        self.decision_label.config(text=decision)
        
        # Actualizar estado con iconos
        status_icons = {
            'exploit': '🔍',
            'explore': '🔎',
            'stabilize': '⚖️'
        }
        icon = status_icons.get(decision.lower(), '❓')
        self.status_label.config(text=f"{icon} Ciclo {cycle} | Decisión: {decision}")
        
        # Actualizar historial de métricas para gráficos
        self.update_metrics_history(cycle, metrics)
        
        # Actualizar estadísticas detalladas si están disponibles
        self.update_detailed_stats_display(metrics)
        
        # Actualizar lista de huellas si están disponibles
        self.update_fingerprints_list()
    
    def update_metrics_history(self, cycle: int, metrics: Dict[str, Any]):
        """Actualiza el historial de métricas para los gráficos."""
        try:
            # Agregar ciclo actual
            self.metrics_history['cycles'].append(cycle)
            
            # Agregar métricas
            self.metrics_history['entropía'].append(metrics.get('entropía', 0.0))
            self.metrics_history['varianza'].append(metrics.get('varianza', 0.0))
            self.metrics_history['simetría'].append(metrics.get('simetría', 0.0))
            self.metrics_history['active_cells'].append(metrics.get('active_cells', 0))
            self.metrics_history['inhibited_cells'].append(metrics.get('inhibited_cells', 0))
            self.metrics_history['neutral_cells'].append(metrics.get('neutral_cells', 0))
            
            # Limitar el historial al máximo de puntos
            if len(self.metrics_history['cycles']) > self.max_history_points:
                for key in self.metrics_history:
                    self.metrics_history[key] = self.metrics_history[key][-self.max_history_points:]
                    
        except Exception as e:
            print(f"Error actualizando historial de métricas: {e}")
    
    def update_detailed_stats_display(self, metrics: Dict[str, Any]):
        """Actualiza la visualización de estadísticas detalladas."""
        try:
            if hasattr(self, 'entropy_value_label'):
                self.entropy_value_label.config(text=f"{metrics.get('entropía', 0.0):.3f}")
            if hasattr(self, 'variance_value_label'):
                self.variance_value_label.config(text=f"{metrics.get('varianza', 0.0):.3f}")
            if hasattr(self, 'symmetry_value_label'):
                self.symmetry_value_label.config(text=f"{metrics.get('simetría', 0.0):.3f}")
            if hasattr(self, 'active_cells_label'):
                self.active_cells_label.config(text=str(metrics.get('active_cells', 0)))
            if hasattr(self, 'inhibited_cells_label'):
                self.inhibited_cells_label.config(text=str(metrics.get('inhibited_cells', 0)))
            if hasattr(self, 'neutral_cells_label'):
                self.neutral_cells_label.config(text=str(metrics.get('neutral_cells', 0)))
                
        except Exception as e:
            print(f"Error actualizando estadísticas detalladas: {e}")
    
    def update_fingerprints_list(self):
        """Actualiza la lista de huellas capturadas."""
        try:
            if hasattr(self, 'fingerprints_listbox'):
                self.fingerprints_listbox.delete(0, tk.END)
                
                status_info = self.metamodulo.get_fingerprint_status()
                fingerprints = status_info.get('fingerprints', {})
                
                for stage, captured in fingerprints.items():
                    if captured:
                        status_icon = "✅"
                        status_text = "Capturada"
                    else:
                        status_icon = "⏳"
                        status_text = "Pendiente"
                    
                    display_text = f"{status_icon} {stage.title()}: {status_text}"
                    self.fingerprints_listbox.insert(tk.END, display_text)
                    
        except Exception as e:
            print(f"Error actualizando lista de huellas: {e}")
    
    def export_metrics_csv(self):
        """Exporta las métricas a un archivo CSV."""
        try:
            if not self.metrics_history['cycles']:
                messagebox.showwarning("Advertencia", "No hay métricas para exportar")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Exportar Métricas a CSV"
            )
            
            if filename:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    # Encabezados
                    writer.writerow(['Ciclo', 'Entropía', 'Varianza', 'Simetría', 'Células Activas', 'Células Inhibidas', 'Células Neutrales'])
                    
                    # Datos
                    for i in range(len(self.metrics_history['cycles'])):
                        writer.writerow([
                            self.metrics_history['cycles'][i],
                            self.metrics_history['entropía'][i],
                            self.metrics_history['varianza'][i],
                            self.metrics_history['simetría'][i],
                            self.metrics_history['active_cells'][i],
                            self.metrics_history['inhibited_cells'][i],
                            self.metrics_history['neutral_cells'][i]
                        ])
                
                messagebox.showinfo("Éxito", f"Métricas exportadas a {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exportando métricas: {e}")
    
    def export_field_image(self):
        """Exporta el campo actual como imagen PNG."""
        try:
            if not hasattr(self.metamodulo, 'core_nucleus') or self.metamodulo.core_nucleus.field.size == 0:
                messagebox.showwarning("Advertencia", "No hay campo para exportar")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Exportar Campo como Imagen"
            )
            
            if filename:
                field = self.metamodulo.core_nucleus.field
                # Normalizar a 0-255
                field_normalized = ((field - field.min()) / (field.max() - field.min()) * 255).astype(np.uint8)
                img = Image.fromarray(field_normalized, mode='L')
                img.save(filename)
                
                messagebox.showinfo("Éxito", f"Campo exportado a {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exportando campo: {e}")
    
    def export_graphs(self):
        """Exporta los gráficos actuales como imágenes PNG."""
        try:
            if not MATPLOTLIB_AVAILABLE:
                messagebox.showwarning("Advertencia", "Matplotlib no está disponible")
                return
            
            directory = filedialog.askdirectory(title="Seleccionar directorio para exportar gráficos")
            
            if directory:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Exportar gráfico de métricas
                if hasattr(self, 'metrics_fig'):
                    metrics_filename = os.path.join(directory, f"metricas_{timestamp}.png")
                    self.metrics_fig.savefig(metrics_filename, dpi=300, bbox_inches='tight')
                
                # Exportar gráfico de evolución
                if hasattr(self, 'evolution_fig'):
                    evolution_filename = os.path.join(directory, f"evolucion_{timestamp}.png")
                    self.evolution_fig.savefig(evolution_filename, dpi=300, bbox_inches='tight')
                
                messagebox.showinfo("Éxito", f"Gráficos exportados a {directory}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exportando gráficos: {e}")
    
    def add_learning_log(self, message: str):
        """Agrega un mensaje al log de aprendizaje."""
        try:
            if hasattr(self, 'learning_logs_text'):
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] {message}\n"
                
                self.learning_logs_text.config(state=tk.NORMAL)
                self.learning_logs_text.insert(tk.END, log_entry)
                self.learning_logs_text.see(tk.END)
                self.learning_logs_text.config(state=tk.DISABLED)
                
        except Exception as e:
            print(f"Error agregando log de aprendizaje: {e}")
    
    def add_system_log(self, message: str):
        """Agrega un mensaje al log del sistema."""
        try:
            if hasattr(self, 'system_logs_text'):
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] {message}\n"
                
                self.system_logs_text.config(state=tk.NORMAL)
                self.system_logs_text.insert(tk.END, log_entry)
                self.system_logs_text.see(tk.END)
                self.system_logs_text.config(state=tk.DISABLED)
                
        except Exception as e:
            print(f"Error agregando log del sistema: {e}")
    
    def clear_learning_logs(self):
        """Limpia los logs de aprendizaje."""
        try:
            if hasattr(self, 'learning_logs_text'):
                self.learning_logs_text.config(state=tk.NORMAL)
                self.learning_logs_text.delete(1.0, tk.END)
                self.learning_logs_text.config(state=tk.DISABLED)
                self.add_learning_log("Logs limpiados")
                
        except Exception as e:
            print(f"Error limpiando logs de aprendizaje: {e}")
    
    def clear_system_logs(self):
        """Limpia los logs del sistema."""
        try:
            if hasattr(self, 'system_logs_text'):
                self.system_logs_text.config(state=tk.NORMAL)
                self.system_logs_text.delete(1.0, tk.END)
                self.system_logs_text.config(state=tk.DISABLED)
                self.add_system_log("Logs del sistema limpiados")
                
        except Exception as e:
            print(f"Error limpiando logs del sistema: {e}")
    
    def clear_fingerprint_session(self):
        """Limpia la sesión actual de huellas."""
        try:
            if hasattr(self, 'metamodulo') and hasattr(self.metamodulo, 'fingerprint_system'):
                self.metamodulo.fingerprint_system.reset_session()
                self.update_fingerprint_status()
                self.update_fingerprints_list()
                self.add_learning_log("Sesión de huellas limpiada")
                
        except Exception as e:
            print(f"Error limpiando sesión de huellas: {e}")
    
    def show_learning_stats(self):
        """Muestra estadísticas detalladas del aprendizaje."""
        try:
            stats = self.metamodulo.get_learning_stats()
            
            stats_text = f"""
            📊 Estadísticas de Aprendizaje
            
            Total de patrones: {stats.get('total_patterns', 0)}
            Total de usos: {stats.get('total_usage', 0)}
            Uso de memoria: {stats.get('memory_usage', '0/100')}
            
            🏆 Patrones más usados:
            """
            
            top_patterns = stats.get('top_patterns', [])
            if not top_patterns:
                stats_text += "\n  (No hay patrones suficientemente usados)"
            else:
                for pattern in top_patterns[:5]:
                    last_used = datetime.fromtimestamp(pattern['last_used']).strftime("%Y-%m-%d %H:%M")
                    stats_text += f"\n• {pattern['label']}: {pattern['usage_count']} usos (último: {last_used})"
            
            messagebox.showinfo("Estadísticas de Aprendizaje", stats_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error obteniendo estadísticas: {e}")
    
    def show_plugin_info(self):
        """Muestra información del plugin seleccionado."""
        try:
            selection = self.plugins_listbox.curselection()
            if not selection:
                messagebox.showwarning("Advertencia", "Selecciona un plugin para ver su información")
                return
            
            plugin_index = selection[0]
            loaded_plugins = list(self.plugin_manager.get_loaded_plugins().keys())
            
            if plugin_index < len(loaded_plugins):
                plugin_name = loaded_plugins[plugin_index]
                plugin = self.plugin_manager.get_plugin(plugin_name)
                
                if plugin:
                    info_text = f"""
                    🔌 Información del Plugin: {plugin_name}
                    
                    Nombre: {plugin.get_name()}
                    Versión: {plugin.get_version()}
                    Descripción: {plugin.get_description()}
                    """
                    
                    messagebox.showinfo(f"Info Plugin: {plugin_name}", info_text)
                else:
                    messagebox.showerror("Error", f"No se pudo obtener información del plugin {plugin_name}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error mostrando información del plugin: {e}")
    
    # --- Métodos para funcionalidades futuras (placeholders) ---
    def analyze_patterns(self):
        # TODO: Implementar el análisis de patrones.
        messagebox.showinfo("Funcionalidad Futura", "El análisis de patrones estará disponible en futuras versiones.")
    
    def compare_sessions(self):
        # TODO: Implementar la comparación de sesiones.
        messagebox.showinfo("Funcionalidad Futura", "La comparación de sesiones estará disponible en futuras versiones.")
    
    def detect_anomalies(self):
        # TODO: Implementar la detección de anomalías.
        messagebox.showinfo("Funcionalidad Futura", "La detección de anomalías estará disponible en futuras versiones.")
    
    def export_complete_report(self):
        # TODO: Implementar la exportación de reportes completos.
        messagebox.showinfo("Funcionalidad Futura", "La exportación de reportes completos estará disponible en futuras versiones.")
    
    def export_evolution_video(self):
        # TODO: Implementar la exportación de videos.
        messagebox.showinfo("Funcionalidad Futura", "La exportación de videos estará disponible en futuras versiones.")
    
    def create_backup(self):
        # TODO: Implementar la creación de backups.
        messagebox.showinfo("Funcionalidad Futura", "La creación de backups estará disponible en futuras versiones.")
    
    def cleanup_temp_data(self):
        # TODO: Implementar la limpieza de datos temporales.
        messagebox.showinfo("Funcionalidad Futura", "La limpieza de datos temporales estará disponible en futuras versiones.")
    
    def optimize_memory(self):
        # TODO: Implementar la optimización de memoria.
        messagebox.showinfo("Funcionalidad Futura", "La optimización de memoria estará disponible en futuras versiones.")
    
    def restart_system(self):
        # TODO: Implementar el reinicio del sistema.
        messagebox.showinfo("Funcionalidad Futura", "El reinicio del sistema estará disponible en futuras versiones.")
    
    def save_configuration(self):
        # TODO: Implementar el guardado de configuración.
        messagebox.showinfo("Funcionalidad Futura", "El guardado de configuración estará disponible en futuras versiones.")
    
    def apply_configuration(self):
        # TODO: Implementar la aplicación de configuración.
        messagebox.showinfo("Funcionalidad Futura", "La aplicación de configuración estará disponible en futuras versiones.")
    
    def browse_plugin_dir(self):
        # TODO: Implementar la exploración de directorios.
        messagebox.showinfo("Funcionalidad Futura", "La exploración de directorios estará disponible en futuras versiones.")
    
    def browse_log_dir(self):
        # TODO: Implementar la exploración de directorios.
        messagebox.showinfo("Funcionalidad Futura", "La exploración de directorios estará disponible en futuras versiones.")
    
    def restore_defaults(self):
        # TODO: Implementar la restauración de configuración.
        messagebox.showinfo("Funcionalidad Futura", "La restauración de configuración estará disponible en futuras versiones.")
    
    def export_configuration(self):
        # TODO: Implementar la exportación de configuración.
        messagebox.showinfo("Funcionalidad Futura", "La exportación de configuración estará disponible en futuras versiones.")
    
    def save_system_logs(self):
        # TODO: Implementar el guardado de logs del sistema.
        messagebox.showinfo("Funcionalidad Futura", "El guardado de logs del sistema estará disponible en futuras versiones.")
    
    def test_metrics(self):
        """Método de prueba para verificar que las métricas funcionen."""
        try:
            # Generar datos de prueba
            test_cycle = len(self.metrics_history['cycles']) + 1
            test_metrics = {
                'entropía': 1.5 + 0.5 * np.sin(test_cycle * 0.1),
                'varianza': 0.3 + 0.2 * np.cos(test_cycle * 0.15),
                'simetría': 0.8 + 0.2 * np.sin(test_cycle * 0.2),
                'active_cells': int(100 + 50 * np.sin(test_cycle * 0.1)),
                'inhibited_cells': int(50 + 25 * np.cos(test_cycle * 0.15)),
                'neutral_cells': int(200 - 75 * np.sin(test_cycle * 0.2))
            }
            
            # Actualizar historial con datos de prueba
            self.update_metrics_history(test_cycle, test_metrics)
            messagebox.showinfo("Prueba de Métricas", f"Métricas de prueba agregadas para el ciclo {test_cycle}")
            
        except Exception as e:
            messagebox.showerror("Error en Prueba", f"Error al probar métricas: {e}")

    def stop_simulation(self):
        self.is_running = False
        self.run_button.config(text="▶ Iniciar")
        self.status_label.config(text="⏸ Estado: En espera")

    def process_text(self):
        self.stop_simulation()
        input_text = self.text_input.get("1.0", tk.END).strip()
        if not input_text: 
            self.status_label.config(text="Error: Entrada de texto vacía."); return
        
        ### SOLUCIÓN: Actualizar el estado de la entrada activa
        self.current_input_data = input_text
        self.current_input_type = 'text'

        self.metamodulo.receive_input(self.current_input_data, self.current_input_type)
        self.update_field_canvas(self.metamodulo.core_nucleus.field)
        self.status_label.config(text="✅ Estado: Texto procesado. Listo para simular.")
        self.loaded_file_label.config(text="📄 Archivo: Ninguno (usando texto)")
        
        self.update_fingerprint_status()

    ### SOLUCIÓN: Métodos para manejar el procesamiento de archivos en un hilo separado
    def _process_file_thread(self, file_path: str):
        """Esta función se ejecuta en un hilo separado para no bloquear la GUI."""
        try:
            field = self.metamodulo.sensor_module.process(file_path, 'file')
            # Poner el resultado y el path original en la cola para la GUI
            self.processing_queue.put({'field': field, 'path': file_path})
        except Exception as e:
            # En caso de error, poner el error en la cola
            self.processing_queue.put({'error': str(e)})

    def _check_processing_queue(self):
        """Comprueba la cola de procesamiento y actualiza la GUI si hay un resultado."""
        try:
            result = self.processing_queue.get_nowait()
            if 'error' in result:
                messagebox.showerror("Error de Procesamiento", f"No se pudo procesar el archivo:\n{result['error']}")
                self.status_label.config(text="❌ Error procesando archivo.")
            else:
                field = result['field']
                file_path = result['path']

                ### SOLUCIÓN: Actualizar el estado de la entrada activa
                self.current_input_data = file_path
                self.current_input_type = 'file'

                self.metamodulo.core_nucleus.receive_field(field)
                self.update_field_canvas(self.metamodulo.core_nucleus.field)
                self.status_label.config(text="✅ Estado: Archivo procesado. Listo para simular.")
                self.loaded_file_label.config(text=f"📄 Archivo: {os.path.basename(file_path)}")
                self.update_fingerprint_status()
                
        except queue.Empty:
            # Si no hay nada, volver a comprobar más tarde
            self.after(100, self._check_processing_queue)

    def load_file(self):
        self.stop_simulation()
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[("Todos los soportados", ".png .jpg .jpeg .wav .csv"),
                       ("Imágenes", ".png .jpg .jpeg"),
                       ("Audio WAV", ".wav"),
                       ("Datos CSV", ".csv")]
        )
        if not file_path: return

        # Desactivar botones mientras se procesa
        self.process_file_button.config(state=tk.DISABLED)
        self.process_text_button.config(state=tk.DISABLED)
        self.status_label.config(text=f"⏳ Procesando {os.path.basename(file_path)}...")

        # Iniciar el procesamiento en el hilo y luego revisar la cola
        thread = threading.Thread(target=self._process_file_thread, args=(file_path,), daemon=True)
        thread.start()
        self.after(100, self._check_processing_queue)
        
        # Reactivar botones
        self.process_file_button.config(state=tk.NORMAL)
        self.process_text_button.config(state=tk.NORMAL)


    def run_single_step(self):
        try:
            summary = self.metamodulo.process_step()
            self.update_display(summary)
        except Exception as e:
            self.status_label.config(text=f"Error en ciclo: {e}"); self.stop_simulation()

    def toggle_simulation(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.run_button.config(text="⏹ Detener")
            self.status_label.config(text="▶ Estado: Ejecutando simulación...")
            self.run_simulation_loop()
        else:
            self.run_button.config(text="▶ Iniciar")
            self.status_label.config(text="⏸ Estado: Simulación pausada")

    def run_simulation_loop(self):
        if self.is_running:
            self.run_single_step()
            self.after(self.cycle_delay, self.run_simulation_loop)
    
    def update_simulation_speed(self):
        """Actualiza la velocidad de simulación."""
        try:
            self.cycle_delay = int(self.speed_var.get())
            if self.is_running:
                self.status_label.config(text=f"⚡ Velocidad actualizada: {self.cycle_delay}ms")
        except ValueError:
            self.speed_var.set("200")
            self.cycle_delay = 200
    
    def load_plugins(self):
        """Carga todos los plugins disponibles."""
        ### SOLUCIÓN: Advertencia de seguridad sobre plugins
        if not self.plugin_warning_shown:
            messagebox.showwarning("Advertencia de Seguridad",
                "Está a punto de cargar plugins. Los plugins son código ejecutable "
                "que se encuentra en el directorio 'plugins'.\n\n"
                "NO ejecute plugins de fuentes no confiables, ya que podrían dañar su sistema.")
            self.plugin_warning_shown = True

        try:
            loaded_count = self.plugin_manager.load_all_plugins()
            self.update_plugins_display()
            self.status_label.config(text=f"Estado: {loaded_count} plugins cargados")
            self.add_system_log(f"{loaded_count} plugins cargados exitosamente")
            
        except Exception as e:
            self.status_label.config(text=f"Error cargando plugins: {e}")
            self.add_system_log(f"Error cargando plugins: {e}")
    
    def reload_plugins(self):
        """Recarga todos los plugins."""
        try:
            self.plugin_manager.unload_all_plugins()
            self.load_plugins() # Reutiliza load_plugins que ya tiene la advertencia
            self.add_system_log(f"Plugins recargados.")
            
        except Exception as e:
            self.status_label.config(text=f"Error recargando plugins: {e}")
            self.add_system_log(f"Error recargando plugins: {e}")
    
    def update_plugins_display(self):
        """Actualiza la lista de plugins en la interfaz."""
        self.plugins_listbox.delete(0, tk.END)
        loaded_plugins = self.plugin_manager.get_loaded_plugins()
        
        if not loaded_plugins:
            self.plugins_listbox.insert(tk.END, " (No hay plugins cargados)")
            return

        for plugin_name, plugin in loaded_plugins.items():
            display_text = f"{plugin_name} v{plugin.get_version()}"
            self.plugins_listbox.insert(tk.END, display_text)
    
    def execute_selected_plugin(self):
        """Ejecuta el plugin seleccionado en la lista."""
        selection = self.plugins_listbox.curselection()
        if not selection:
            self.status_label.config(text="Estado: Selecciona un plugin para ejecutar")
            return
        
        plugin_index = selection[0]
        loaded_plugins = list(self.plugin_manager.get_loaded_plugins().keys())
        
        if plugin_index < len(loaded_plugins):
            plugin_name = loaded_plugins[plugin_index]
            try:
                # Ejecutar plugin con el campo actual
                result = self.plugin_manager.execute_plugin(
                    plugin_name, 
                    self.metamodulo.core_nucleus.field
                )
                
                if result is not None:
                    # Aplicar resultado del plugin al campo
                    if isinstance(result, np.ndarray):
                        self.metamodulo.core_nucleus.receive_field(result)
                        self.update_field_canvas(result)
                        self.status_label.config(text=f"Plugin '{plugin_name}' ejecutado exitosamente")
                        self.add_system_log(f"Plugin '{plugin_name}' ejecutado exitosamente")
                    else:
                        self.status_label.config(text=f"Plugin '{plugin_name}' retornó: {type(result).__name__}")
                        self.add_system_log(f"Plugin '{plugin_name}' retornó: {type(result).__name__}")
                else:
                    self.status_label.config(text=f"Plugin '{plugin_name}' no retornó resultado")
                    self.add_system_log(f"Plugin '{plugin_name}' no retornó resultado")
                    
            except Exception as e:
                self.status_label.config(text=f"Error ejecutando plugin '{plugin_name}': {e}")
                self.add_system_log(f"Error ejecutando plugin '{plugin_name}': {e}")
    
    def save_fingerprints(self):
        """Guarda las huellas de la sesión actual."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Guardar Huellas de Sesión"
            )
            
            if filename:
                filepath = self.metamodulo.save_session_fingerprints(filename)
                if filepath:
                    self.status_label.config(text=f"✅ Huellas guardadas en: {os.path.basename(filepath)}")
                    self.update_fingerprint_status()
                    self.add_learning_log(f"Huellas guardadas en: {os.path.basename(filepath)}")
                else:
                    self.status_label.config(text="❌ Error guardando huellas")
                    self.add_learning_log("Error guardando huellas")
                    
        except Exception as e:
            self.status_label.config(text=f"Error guardando huellas: {e}")
            self.add_learning_log(f"Error guardando huellas: {e}")
    
    def load_fingerprints(self):
        """Carga huellas desde un archivo."""
        try:
            filepath = filedialog.askopenfilename(
                title="Cargar Huellas de Sesión",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filepath:
                    if self.metamodulo.fingerprint_system.load_fingerprints(filepath):
                        self.status_label.config(text=f"✅ Huellas cargadas desde: {os.path.basename(filepath)}")
                        self.update_fingerprint_status()
                        self.add_learning_log(f"Huellas cargadas desde: {os.path.basename(filepath)}")
                    else:
                        self.status_label.config(text="❌ Error cargando huellas")
                        self.add_learning_log("Error cargando huellas")
                    
        except Exception as e:
            self.status_label.config(text=f"Error cargando huellas: {e}")
            self.add_learning_log(f"Error cargando huellas: {e}")
    
    def update_fingerprint_status(self):
        """Actualiza el estado del sistema de huellas en la interfaz."""
        try:
            status_info = self.metamodulo.get_fingerprint_status()
            
            total = status_info.get('summary', {}).get('total_fingerprints', 0)
            
            if total == 0:
                status_text = "⏳ Esperando captura..."
            elif total == 1:
                status_text = f"📸 1 huella capturada"
            elif total == 2:
                status_text = f"📸 2 huellas capturadas"
            elif total == 3:
                status_text = f"✅ 3 huellas completas"
            else:
                status_text = f"📸 {total} huellas capturadas"
            
            self.fingerprint_status_label.config(text=status_text)
            
        except Exception as e:
            self.fingerprint_status_label.config(text="❌ Error actualizando estado")
    
    def learn_from_current_session(self):
        """Aprende de la sesión actual y la agrega a la memoria."""
        ### SOLUCIÓN: Usa las variables de estado para garantizar la consistencia de los datos.
        if self.current_input_data is None or self.current_input_type is None:
            messagebox.showerror("Error", "No hay una entrada de datos activa para aprender.\nProcesa un texto o carga un archivo primero.")
            return

        try:
            # Obtener etiqueta personalizada si existe
            user_label = self.custom_label_var.get().strip()
            
            # Aprender de la sesión usando los datos de estado
            pattern_id = self.metamodulo.learn_from_session(
                self.current_input_data, self.current_input_type, user_label
            )
            
            if pattern_id:
                self.status_label.config(text=f"✅ Patrón aprendido: {pattern_id}")
                self.custom_label_var.set("")  # Limpiar campo de etiqueta
                self.update_learning_status()
                
                self.add_learning_log(f"Patrón aprendido: {pattern_id} (tipo: {self.current_input_type})")
                
                if hasattr(self, 'patterns_info_label'):
                    learning_stats = self.metamodulo.get_learning_stats()
                    total_patterns = learning_stats.get('total_patterns', 0)
                    self.patterns_info_label.config(text=f"{total_patterns} patrones")
            else:
                messagebox.showwarning("Aprender", "No se pudo aprender el patrón.\nAsegúrate de que la simulación haya capturado al menos una huella.")
                self.status_label.config(text="❌ Error aprendiendo patrón")
                self.add_learning_log("Error aprendiendo patrón: no había huellas suficientes.")
                
        except Exception as e:
            self.status_label.config(text=f"Error aprendiendo: {e}")
            self.add_learning_log(f"Error aprendiendo: {e}")
    
    def recognize_current_pattern(self):
        """Reconoce si el patrón actual es similar a patrones aprendidos."""
        ### SOLUCIÓN: Usa las variables de estado para garantizar la consistencia de los datos.
        if self.current_input_data is None or self.current_input_type is None:
            messagebox.showerror("Error", "No hay una entrada de datos activa para reconocer.\nProcesa un texto o carga un archivo primero.")
            return
            
        try:
            # Intentar reconocer el patrón
            recognized_pattern = self.metamodulo.recognize_pattern(
                self.current_input_data, self.current_input_type
            )
            
            if recognized_pattern:
                label = recognized_pattern.get('auto_label', 'Desconocido')
                similarity = recognized_pattern.get('similarity_score', 0.0)
                usage_count = recognized_pattern.get('usage_count', 0)
                
                msg = f"🔍 Patrón reconocido: {label}\nSimilitud: {similarity:.2%}, Usado: {usage_count} veces"
                self.status_label.config(text=msg.replace("\n", " | "))
                self.learning_status_label.config(text=f"✅ Reconocido: {label} ({similarity:.1%})")
                self.add_learning_log(f"Patrón reconocido: {label} (similitud: {similarity:.1%})")
                messagebox.showinfo("Reconocimiento de Patrón", msg)
            else:
                self.status_label.config(text="❓ Patrón no reconocido - es nuevo")
                self.learning_status_label.config(text="❓ Patrón no reconocido")
                self.add_learning_log("Patrón no reconocido - es nuevo o no hay huellas.")
                messagebox.showinfo("Reconocimiento de Patrón", "No se encontró un patrón similar en la memoria.\n(Asegúrate de que se hayan capturado huellas para la entrada actual).")

        except Exception as e:
            self.status_label.config(text=f"Error reconociendo patrón: {e}")
            self.add_learning_log(f"Error reconociendo patrón: {e}")
    
    def update_learning_status(self):
        """Actualiza el estado del sistema de aprendizaje en la interfaz."""
        try:
            learning_stats = self.metamodulo.get_learning_stats()
            
            total_patterns = learning_stats.get('total_patterns', 0)
            memory_usage = learning_stats.get('memory_usage', '0/100')
            
            if total_patterns == 0:
                status_text = "⏳ No hay patrones aprendidos"
            else:
                status_text = f"📚 {total_patterns} patrones aprendidos ({memory_usage})"
            
            self.learning_status_label.config(text=status_text)
            
        except Exception as e:
            self.learning_status_label.config(text="❌ Error actualizando estado")

if __name__ == "__main__":
    print("Iniciando la aplicación de visualización del Sistema DIG...")
    app = DIGVisualizerApp()
    app.mainloop()