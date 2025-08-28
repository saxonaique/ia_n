import tkinter as tk
from tkinter import ttk, filedialog
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

# Dependencia para audio .wav y espectrograma
from scipy.io import wavfile
from scipy.signal import spectrogram

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
        if field.max() > 0: field = (field - field.min()) / (field.max() - field.min())
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

# ... (El resto de los módulos CoreNucleus, MemoryModule, IA_Interpreter y Metamodulo no necesitan cambios) ...
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

# --- Módulo 4: Sistema de Huellas 3 Puntos ---
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
        field = self.sensor_module.process(source, source_type)
        self.core_nucleus.receive_field(field)
        print("[Metamodulo] INFO: Entrada procesada.")
    
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
        if self.fingerprint_system.fingerprints:
            pattern_id = self.fingerprint_system.learning_memory.add_pattern(
                input_data, input_type, self.fingerprint_system.fingerprints, user_label
            )
            return pattern_id
        return ""
    
    def recognize_pattern(self, input_data: str, input_type: str) -> Optional[Dict[str, Any]]:
        """Reconoce si el input actual es similar a patrones aprendidos."""
        if self.fingerprint_system.fingerprints:
            return self.fingerprint_system.learning_memory.find_similar_pattern(
                input_data, input_type, self.fingerprint_system.fingerprints
            )
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

# --- Módulo 6: DIGVisualizerApp (GUI con Carga de Archivos) ---
class DIGVisualizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧠 Sistema DIG - Organismo Informacional")
        self.geometry("1200x1000")  # Aumentar altura para mostrar todos los paneles
        
        # Configurar tema y estilo
        self.setup_theme()
        
        # Inicializar componentes
        self.metamodulo = Metamodulo()
        self.plugin_manager = PluginManager()
        self.plugin_manager.set_metamodulo(self.metamodulo)
        self.is_running = False
        self.cycle_delay = 200  # Velocidad de simulación en ms
        
        # Configurar interfaz
        self.setup_ui()
        self.load_plugins()
        self.process_text()  # Cargar texto inicial
        
        # Actualizar estados de sistemas
        self.update_fingerprint_status()
        self.update_learning_status()
        
        # Centrar ventana
        self.center_window()
    
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
        style.configure('TLabelframe', background=self.colors['bg_primary'], foreground=self.colors['text_primary'])
        style.configure('TLabelframe.Label', background=self.colors['bg_primary'], foreground=self.colors['text_primary'])
        
        # Estilos personalizados para botones
        style.configure('Success.TButton', background=self.colors['accent_green'])
        style.configure('Info.TButton', background=self.colors['accent_blue'])
        style.configure('Warning.TButton', background=self.colors['accent_orange'])
        style.configure('Danger.TButton', background=self.colors['accent_red'])
        style.configure('Accent.TButton', background=self.colors['accent_blue'])
    
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
        
        # Frame principal con grid para mejor distribución
        main_frame = ttk.Frame(self, padding="15")
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
        self.canvas_size = 512
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
        # Crear un canvas con scrollbar para los controles
        controls_canvas = tk.Canvas(main_frame, bg=self.colors['bg_primary'], highlightthickness=0)
        controls_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=controls_canvas.yview)
        controls_frame = ttk.Frame(controls_canvas)
        
        # Configurar el scroll
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        
        # Posicionar canvas y scrollbar
        controls_canvas.grid(row=0, column=1, sticky="nsew")
        controls_scrollbar.grid(row=0, column=2, sticky="ns")
        
        # Configurar el frame interno para que se expanda
        controls_canvas.create_window((0, 0), window=controls_frame, anchor="nw")
        
        # Configurar el scroll cuando el frame cambie de tamaño
        def configure_scroll(event):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))
        
        controls_frame.bind("<Configure>", configure_scroll)
        
        # Permitir scroll con la rueda del ratón
        def on_mousewheel(event):
            controls_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        controls_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # --- Panel de Entrada de Datos ---
        input_frame = ttk.LabelFrame(controls_frame, text="📥 Entrada de Datos", padding="10")
        input_frame.pack(fill="x", pady=(0, 15))
        
        # Entrada de texto
        ttk.Label(input_frame, text="Texto:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.text_input = tk.Text(
            input_frame, 
            height=4, 
            width=40, 
            font=("Consolas", 9),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary'],
            selectbackground=self.colors['accent_blue'],
            relief="flat",
            padx=8,
            pady=8
        )
        self.text_input.insert("1.0", "La gravedad emerge de un balance dinámico entre desorden (S) y orden (I).")
        self.text_input.pack(fill="x", pady=(0, 8))
        
        # Botones de entrada
        input_buttons_frame = ttk.Frame(input_frame)
        input_buttons_frame.pack(fill="x")
        
        self.process_text_button = ttk.Button(
            input_buttons_frame, 
            text="🔄 Procesar Texto", 
            command=self.process_text,
            style='Accent.TButton'
        )
        self.process_text_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        self.process_file_button = ttk.Button(
            input_buttons_frame, 
            text="📁 Cargar Archivo", 
            command=self.load_file,
            style='Accent.TButton'
        )
        self.process_file_button.pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        # Información del archivo cargado
        self.loaded_file_label = ttk.Label(
            input_frame, 
            text="📄 Archivo: Ninguno", 
            font=("Helvetica", 8, "italic"),
            foreground=self.colors['text_secondary']
        )
        self.loaded_file_label.pack(anchor=tk.W, pady=(8, 0))

        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # --- Panel de Simulación ---
        simulation_frame = ttk.LabelFrame(controls_frame, text="⚡ Control de Simulación", padding="10")
        simulation_frame.pack(fill="x", pady=(0, 15))
        
        # Control de velocidad
        speed_frame = ttk.Frame(simulation_frame)
        speed_frame.pack(fill="x", pady=(0, 8))
        
        ttk.Label(speed_frame, text="Velocidad:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="200")
        speed_spinbox = ttk.Spinbox(
            speed_frame, 
            from_=50, 
            to=2000, 
            textvariable=self.speed_var, 
            width=8,
            command=self.update_simulation_speed
        )
        speed_spinbox.pack(side=tk.RIGHT)
        ttk.Label(speed_frame, text="ms").pack(side=tk.RIGHT, padx=(0, 5))
        
        # Botones de simulación
        sim_buttons_frame = ttk.Frame(simulation_frame)
        sim_buttons_frame.pack(fill="x", pady=(0, 8))
        
        self.run_button = ttk.Button(
            sim_buttons_frame, 
            text="▶ Iniciar", 
            command=self.toggle_simulation,
            style='Success.TButton'
        )
        self.run_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        self.step_button = ttk.Button(
            sim_buttons_frame, 
            text="⏭ Paso", 
            command=self.run_single_step,
            style='Info.TButton'
        )
        self.step_button.pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        # Estado de la simulación
        self.status_label = ttk.Label(
            simulation_frame, 
            text="⏸ Estado: En espera", 
            font=("Helvetica", 9, "italic"),
            foreground=self.colors['text_secondary']
        )
        self.status_label.pack(anchor=tk.W)
        
        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # --- Panel de Interpretación ---
        interpretation_frame = ttk.LabelFrame(controls_frame, text="🧠 Interpretación de la IA", padding="10")
        interpretation_frame.pack(fill="x", pady=(0, 15))
        
        # Métricas en tiempo real
        metrics_frame = ttk.Frame(interpretation_frame)
        metrics_frame.pack(fill="x", pady=(0, 8))
        
        # Contador de ciclos
        cycle_frame = ttk.Frame(metrics_frame)
        cycle_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(cycle_frame, text="🔄 Ciclo:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        self.cycle_label = ttk.Label(cycle_frame, text="0", font=("Helvetica", 9, "bold"), foreground=self.colors['accent_green'])
        self.cycle_label.pack(side=tk.RIGHT)
        
        # Decisión actual
        decision_frame = ttk.Frame(metrics_frame)
        decision_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(decision_frame, text="🎯 Decisión:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        self.decision_label = ttk.Label(decision_frame, text="N/A", font=("Helvetica", 9, "bold"), foreground=self.colors['accent_blue'])
        self.decision_label.pack(side=tk.RIGHT)
        
        # Texto de interpretación
        ttk.Label(interpretation_frame, text="Análisis:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(8, 5))
        self.interpretation_text = tk.Text(
            interpretation_frame, 
            height=10, 
            width=40, 
            state=tk.DISABLED, 
            font=("Consolas", 8), 
            wrap=tk.WORD,
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            relief="flat",
            padx=8,
            pady=8
        )
        self.interpretation_text.pack(fill="x")
        
        # --- Panel de Plugins ---
        plugins_frame = ttk.LabelFrame(controls_frame, text="🔌 Sistema de Plugins", padding="10")
        plugins_frame.pack(fill="x", pady=(0, 15))
        
        # Botón para recargar plugins
        ttk.Button(
            plugins_frame, 
            text="🔄 Recargar Plugins", 
            command=self.reload_plugins,
            style='Info.TButton'
        ).pack(fill="x", pady=(0, 8))
        
        # Lista de plugins cargados
        ttk.Label(plugins_frame, text="📦 Plugins Activos:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.plugins_listbox = tk.Listbox(
            plugins_frame, 
            height=4, 
            font=("Consolas", 8),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            selectbackground=self.colors['accent_blue'],
            relief="flat",
            borderwidth=0
        )
        self.plugins_listbox.pack(fill="x", pady=(0, 8))
        
        # Botón para ejecutar plugin seleccionado
        ttk.Button(
            plugins_frame, 
            text="▶ Ejecutar Plugin", 
            command=self.execute_selected_plugin,
            style='Success.TButton'
        ).pack(fill="x")
        
        # --- Panel de Sistema de Huellas ---
        fingerprint_frame = ttk.LabelFrame(controls_frame, text="🔄 Sistema de Huellas", padding="10")
        fingerprint_frame.pack(fill="x", pady=(0, 15))
        
        # Estado de las huellas
        ttk.Label(fingerprint_frame, text="📊 Estado:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.fingerprint_status_label = ttk.Label(
            fingerprint_frame, 
            text="⏳ Inicializando...", 
            font=("Helvetica", 8),
            foreground=self.colors['text_secondary']
        )
        self.fingerprint_status_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Botones de huellas
        fingerprint_buttons_frame = ttk.Frame(fingerprint_frame)
        fingerprint_buttons_frame.pack(fill="x")
        
        self.save_fingerprints_button = ttk.Button(
            fingerprint_buttons_frame, 
            text="💾 Guardar Huellas", 
            command=self.save_fingerprints,
            style='Info.TButton'
        )
        self.save_fingerprints_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        self.load_fingerprints_button = ttk.Button(
            fingerprint_buttons_frame, 
            text="📂 Cargar Huellas", 
            command=self.load_fingerprints,
            style='Info.TButton'
        )
        self.load_fingerprints_button.pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        # --- Panel de Sistema de Aprendizaje ---
        learning_frame = ttk.LabelFrame(controls_frame, text="🧠 Sistema de Aprendizaje", padding="10")
        learning_frame.pack(fill="x", pady=(0, 15))
        
        # Estado del aprendizaje
        ttk.Label(learning_frame, text="📊 Estado:", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.learning_status_label = ttk.Label(
            learning_frame, 
            text="⏳ Inicializando...", 
            font=("Helvetica", 8),
            foreground=self.colors['text_secondary']
        )
        self.learning_status_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Campo para etiqueta personalizada
        label_frame = ttk.Frame(learning_frame)
        label_frame.pack(fill="x", pady=(0, 8))
        
        ttk.Label(label_frame, text="🏷️ Etiqueta:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.custom_label_var = tk.StringVar(value="")
        self.custom_label_entry = ttk.Entry(
            label_frame, 
            textvariable=self.custom_label_var,
            width=20,
            font=("Helvetica", 8)
        )
        self.custom_label_entry.pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))
        
        # Botones de aprendizaje
        learning_buttons_frame = ttk.Frame(learning_frame)
        learning_buttons_frame.pack(fill="x")
        
        self.learn_button = ttk.Button(
            learning_buttons_frame, 
            text="🎓 Aprender", 
            command=self.learn_from_current_session,
            style='Success.TButton'
        )
        self.learn_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        
        self.recognize_button = ttk.Button(
            learning_buttons_frame, 
            text="🔍 Reconocer", 
            command=self.recognize_current_pattern,
            style='Info.TButton'
        )
        self.recognize_button.pack(side=tk.RIGHT, fill="x", expand=True, padx=(5, 0))

    def update_field_canvas(self, field: np.ndarray):
        if field.size == 0: return
        norm = colors.Normalize(vmin=np.min(field), vmax=np.max(field))
        # Usar la API de matplotlib compatible con todas las versiones
        try:
            # Intentar usar la API antigua (más compatible)
            colormap = cm.get_cmap('plasma')
        except AttributeError:
            # Fallback para versiones más nuevas
            import matplotlib.pyplot as plt
            colormap = plt.get_cmap('plasma')
        
        rgba_field = colormap(norm(field))
        img_data = (rgba_field[:, :, :3] * 255).astype(np.uint8)
        
        # Usar la nueva API de Pillow para evitar deprecación
        try:
            # Intentar usar la nueva API primero (sin parámetro mode)
            img = Image.fromarray(img_data)
        except (TypeError, ValueError):
            # Fallback para versiones más antiguas de Pillow
            img = Image.fromarray(img_data, 'RGB')
        
        img_resized = img.resize((self.canvas_size, self.canvas_size), Image.Resampling.NEAREST)
        self.photo_image = ImageTk.PhotoImage(image=img_resized)
        self.field_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

    def update_display(self, summary: Dict[str, Any]):
        self.update_field_canvas(summary.get('field'))
        
        # Actualizar métricas en tiempo real
        cycle = summary.get('cycle', 0)
        decision = summary.get('decision', 'N/A').upper()
        
        self.cycle_label.config(text=str(cycle))
        self.decision_label.config(text=decision)
        
        # Actualizar texto de interpretación
        self.interpretation_text.config(state=tk.NORMAL)
        self.interpretation_text.delete(1.0, tk.END)
        self.interpretation_text.insert(tk.END, summary.get('interpretation', ''))
        self.interpretation_text.config(state=tk.DISABLED)
        
        # Actualizar estado con iconos
        status_icons = {
            'exploit': '🔍',
            'explore': '🔎',
            'stabilize': '⚖️'
        }
        icon = status_icons.get(decision.lower(), '❓')
        self.status_label.config(text=f"{icon} Ciclo {cycle} | Decisión: {decision}")

    def stop_simulation(self):
        self.is_running = False
        self.run_button.config(text="▶ Iniciar")
        self.status_label.config(text="⏸ Estado: En espera")

    def process_text(self):
        self.stop_simulation()
        input_text = self.text_input.get("1.0", tk.END).strip()
        if not input_text: self.status_label.config(text="Error: Entrada vacía."); return
        self.metamodulo.receive_input(input_text, 'text')
        self.update_field_canvas(self.metamodulo.core_nucleus.field)
        self.status_label.config(text="✅ Estado: Texto procesado. Listo para simular.")
        self.loaded_file_label.config(text="📄 Archivo: Ninguno (usando texto)")
        
        # Actualizar estado de huellas
        self.update_fingerprint_status()

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
        self.metamodulo.receive_input(file_path, 'file')
        self.update_field_canvas(self.metamodulo.core_nucleus.field)
        self.status_label.config(text="✅ Estado: Archivo procesado. Listo para simular.")
        self.loaded_file_label.config(text=f"📄 Archivo: {os.path.basename(file_path)}")
        
        # Actualizar estado de huellas
        self.update_fingerprint_status()

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
        try:
            loaded_count = self.plugin_manager.load_all_plugins()
            self.update_plugins_display()
            self.status_label.config(text=f"Estado: {loaded_count} plugins cargados")
        except Exception as e:
            self.status_label.config(text=f"Error cargando plugins: {e}")
    
    def reload_plugins(self):
        """Recarga todos los plugins."""
        try:
            self.plugin_manager.unload_all_plugins()
            loaded_count = self.plugin_manager.load_all_plugins()
            self.update_plugins_display()
            self.status_label.config(text=f"Estado: {loaded_count} plugins recargados")
        except Exception as e:
            self.status_label.config(text=f"Error recargando plugins: {e}")
    
    def update_plugins_display(self):
        """Actualiza la lista de plugins en la interfaz."""
        self.plugins_listbox.delete(0, tk.END)
        loaded_plugins = self.plugin_manager.get_loaded_plugins()
        
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
                    else:
                        self.status_label.config(text=f"Plugin '{plugin_name}' retornó: {type(result).__name__}")
                else:
                    self.status_label.config(text=f"Plugin '{plugin_name}' no retornó resultado")
                    
            except Exception as e:
                self.status_label.config(text=f"Error ejecutando plugin '{plugin_name}': {e}")
    
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
                else:
                    self.status_label.config(text="❌ Error guardando huellas")
                    
        except Exception as e:
            self.status_label.config(text=f"Error guardando huellas: {e}")
    
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
                else:
                    self.status_label.config(text="❌ Error cargando huellas")
                    
        except Exception as e:
            self.status_label.config(text=f"Error cargando huellas: {e}")
    
    def update_fingerprint_status(self):
        """Actualiza el estado del sistema de huellas en la interfaz."""
        try:
            status_info = self.metamodulo.get_fingerprint_status()
            
            # Crear texto descriptivo del estado
            captured = status_info.get('fingerprints', {})
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
        try:
            # Obtener el texto actual o información del archivo
            if hasattr(self, 'text_input') and self.text_input.get("1.0", tk.END).strip():
                input_data = self.text_input.get("1.0", tk.END).strip()
                input_type = "text"
            else:
                # Buscar información del archivo cargado
                file_label = self.loaded_file_label.cget("text")
                if "Archivo:" in file_label and "Ninguno" not in file_label:
                    input_data = file_label.split("Archivo: ")[1]
                    input_type = "file"
                else:
                    self.status_label.config(text="❌ No hay datos para aprender")
                    return
            
            # Obtener etiqueta personalizada si existe
            user_label = self.custom_label_var.get().strip()
            
            # Aprender de la sesión
            pattern_id = self.metamodulo.learn_from_session(input_data, input_type, user_label)
            
            if pattern_id:
                self.status_label.config(text=f"✅ Patrón aprendido: {pattern_id}")
                self.custom_label_var.set("")  # Limpiar campo de etiqueta
                self.update_learning_status()
            else:
                self.status_label.config(text="❌ Error aprendiendo patrón")
                
        except Exception as e:
            self.status_label.config(text=f"Error aprendiendo: {e}")
    
    def recognize_current_pattern(self):
        """Reconoce si el patrón actual es similar a patrones aprendidos."""
        try:
            # Obtener el texto actual o información del archivo
            if hasattr(self, 'text_input') and self.text_input.get("1.0", tk.END).strip():
                input_data = self.text_input.get("1.0", tk.END).strip()
                input_type = "text"
            else:
                # Buscar información del archivo cargado
                file_label = self.loaded_file_label.cget("text")
                if "Archivo:" in file_label and "Ninguno" not in file_label:
                    input_data = file_label.split("Archivo: ")[1]
                    input_type = "file"
                else:
                    self.status_label.config(text="❌ No hay datos para reconocer")
                    return
            
            # Intentar reconocer el patrón
            recognized_pattern = self.metamodulo.recognize_pattern(input_data, input_type)
            
            if recognized_pattern:
                label = recognized_pattern.get('auto_label', 'Desconocido')
                similarity = recognized_pattern.get('similarity_score', 0.0)
                usage_count = recognized_pattern.get('usage_count', 0)
                
                self.status_label.config(
                    text=f"🔍 Patrón reconocido: {label} (similitud: {similarity:.3f}, usado: {usage_count}x)"
                )
                
                # Mostrar información en el panel de aprendizaje
                self.learning_status_label.config(
                    text=f"✅ Reconocido: {label} ({similarity:.1%})"
                )
            else:
                self.status_label.config(text="❓ Patrón no reconocido - es nuevo")
                self.learning_status_label.config(text="❓ Patrón no reconocido")
                
        except Exception as e:
            self.status_label.config(text=f"Error reconociendo patrón: {e}")
    
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