import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json
import time
import os

@dataclass
class AtractorInfo:
    """Estructura para almacenar información sobre un atractor."""
    id: str
    field: np.ndarray
    frequency: int = 1
    last_accessed: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el atractor a un diccionario serializable."""
        return {
            'id': self.id,
            'field': self.field.tolist(),
            'frequency': self.frequency,
            'last_accessed': self.last_accessed,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AtractorInfo':
        """Crea un AtractorInfo a partir de un diccionario."""
        atractor = cls(
            id=data['id'],
            field=np.array(data['field']),
            frequency=data['frequency'],
            last_accessed=data['last_accessed'],
            metadata=data['metadata']
        )
        return atractor

class MemoriaAtractores:
    """
    Memoria de Atractores: Módulo de almacenamiento y recuperación de patrones
    
    Almacena configuraciones estables del campo informacional (atractores) y permite
    su recuperación basada en similitud con el estado actual.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa la Memoria de Atractores.
        
        Args:
            config: Configuración para el almacenamiento y recuperación de atractores
        """
        # Configuración por defecto
        default_config = {
            'similarity_threshold': 0.8,  # Umbral de similitud para considerar coincidencia
            'max_attractors': 1000,  # Número máximo de atractores a almacenar
            'prune_frequency': 100,  # Frecuencia de limpieza de atractores poco usados
            'access_counter': 0,  # Contador de accesos para limpieza periódica
            'min_frequency': 3,  # Frecuencia mínima para retener un atractor
            'max_age_days': 30,  # Edad máxima en días antes de considerar eliminar un atractor
            'auto_save': True,  # Guardar automáticamente en disco
            'storage_path': 'memory_data.json'  # Ruta para guardar/recuperar la memoria
        }
        
        # Inicializar con configuración por defecto
        self.attractors: Dict[str, AtractorInfo] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.config = default_config
        
        # Actualizar con la configuración proporcionada
        if config:
            self.config.update(config)
        
        # Cargar atractores desde disco si existen
        self._load_from_disk()
    
    def store_attractor(self, field: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Almacena un nuevo atractor o actualiza uno existente.
        
        Args:
            field: Campo informacional a almacenar
            metadata: Metadatos adicionales para el atractor
            
        Returns:
            str: ID único del atractor almacenado
        """
        # Generar un identificador único basado en el contenido del campo
        field_bytes = field.tobytes()
        field_hash = hashlib.sha256(field_bytes).hexdigest()
        
        if field_hash in self.attractors:
            # Actualizar atractor existente
            self.attractors[field_hash].frequency += 1
            self.attractors[field_hash].last_accessed = time.time()
            if metadata:
                self.attractors[field_hash].metadata.update(metadata)
        else:
            # Crear nuevo atractor
            if len(self.attractors) >= self.config['max_attractors']:
                self._prune_attractors()
                
            self.attractors[field_hash] = AtractorInfo(
                id=field_hash,
                field=field.copy(),
                frequency=1,
                last_accessed=time.time(),
                metadata=metadata or {}
            )
        
        # Limpieza periódica
        self.config['access_counter'] += 1
        if self.config['access_counter'] >= self.config['prune_frequency']:
            self._prune_attractors()
            self.config['access_counter'] = 0
        
        # Guardar en disco si está habilitado
        if self.config['auto_save']:
            self.save_to_disk()
            
        return field_hash
    
    def find_similar(self, field: np.ndarray, threshold: Optional[float] = None) -> Tuple[Optional[np.ndarray], float]:
        """
        Encuentra el atractor más similar al campo dado.
        
        Args:
            field: Campo informacional a comparar
            threshold: Umbral de similitud mínimo (0-1). Si es None, usa el valor por defecto.
            
        Returns:
            tuple: (atractor más similar, puntuación de similitud) o (None, 0.0) si no se encuentra ninguno
        """
        if not self.attractors:
            return None, 0.0
            
        threshold = threshold or self.config['similarity_threshold']
        best_match = None
        best_score = 0.0
        
        for atractor_id, atractor in self.attractors.items():
            # Calcular similitud (usando correlación cruzada normalizada)
            similarity = self._calculate_similarity(field, atractor.field)
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = atractor.field
                
                # Actualizar frecuencia y último acceso
                atractor.frequency += 1
                atractor.last_accessed = time.time()
        
        return best_match, best_score
    
    def find_top_k_similar(self, field: np.ndarray, k: int = 5, min_similarity: float = 0.5) -> List[Tuple[np.ndarray, float]]:
        """
        Encuentra los k atractores más similares al campo dado.
        
        Args:
            field: Campo informacional a comparar
            k: Número de atractores a devolver
            min_similarity: Similitud mínima requerida
            
        Returns:
            list: Lista de tuplas (atractor, similitud) ordenadas por similitud descendente
        """
        if not self.attractors:
            return []
            
        similarities = []
        
        for atractor in self.attractors.values():
            similarity = self._calculate_similarity(field, atractor.field)
            if similarity >= min_similarity:
                similarities.append((atractor.field, similarity))
                
                # Actualizar frecuencia y último acceso
                atractor.frequency += 1
                atractor.last_accessed = time.time()
        
        # Ordenar por similitud descendente y devolver los k primeros
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _calculate_similarity(self, field1: np.ndarray, field2: np.ndarray) -> float:
        """
        Calcula la similitud entre dos campos informacionales.
        
        Args:
            field1: Primer campo
            field2: Segundo campo
            
        Returns:
            float: Puntuación de similitud normalizada (0-1)
        """
        # Verificar si los campos tienen el mismo tamaño
        if field1.shape != field2.shape:
            # Redimensionar el campo más grande para que coincida con el más pequeño
            min_shape = (min(field1.shape[0], field2.shape[0]), 
                        min(field1.shape[1], field2.shape[1]))
            field1 = field1[:min_shape[0], :min_shape[1]]
            field2 = field2[:min_shape[0], :min_shape[1]]
        
        # Usar correlación cruzada normalizada como medida de similitud
        cross_corr = np.correlate(field1.flatten(), field2.flatten())
        norm1 = np.linalg.norm(field1)
        norm2 = np.linalg.norm(field2)
        
        # Evitar división por cero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = cross_corr / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _prune_attractors(self) -> None:
        """Elimina atractores poco usados o antiguos para mantener el tamaño bajo control."""
        current_time = time.time()
        to_remove = []
        
        for atractor_id, atractor in self.attractors.items():
            # Verificar frecuencia y antigüedad
            age_days = (current_time - atractor.last_accessed) / (24 * 3600)
            
            if (atractor.frequency < self.config['min_frequency'] and 
                age_days > self.config['max_age_days']):
                to_remove.append(atractor_id)
        
        # Eliminar atractores seleccionados
        for atractor_id in to_remove:
            del self.attractors[atractor_id]
        
        # Si aún hay demasiados atractores, eliminar los menos frecuentes
        if len(self.attractors) > self.config['max_attractors']:
            sorted_attractors = sorted(
                self.attractors.items(),
                key=lambda x: (x[1].frequency, x[1].last_accessed)
            )
            for atractor_id, _ in sorted_attractors[:len(self.attractors) - self.config['max_attractors']]:
                del self.attractors[atractor_id]
    
    def save_to_disk(self, path: Optional[str] = None) -> bool:
        """
        Guarda los atractores en disco.
        
        Args:
            path: Ruta del archivo. Si es None, usa la ruta configurada.
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        path = path or self.config['storage_path']
        try:
            data = {
                'config': self.config,
                'attractors': {k: v.to_dict() for k, v in self.attractors.items()}
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, default=str, indent=2)
            return True
        except Exception as e:
            print(f"Error al guardar la memoria: {e}")
            return False
    
    def _load_from_disk(self, path: Optional[str] = None) -> bool:
        """
        Carga los atractores desde disco.
        
        Args:
            path: Ruta del archivo. Si es None, usa la ruta configurada.
            
        Returns:
            bool: True si se cargó correctamente, False en caso contrario
        """
        try:
            # Usar la ruta proporcionada o la de configuración, o una por defecto
            path = path or self.config.get('storage_path', 'memory_data.json')
            if not os.path.exists(path):
                return False
                
            with open(path, 'r') as f:
                data = json.load(f)
                
            self.attractors = {}
            for attractor_id, attractor_data in data.get('attractors', {}).items():
                self.attractors[attractor_id] = AtractorInfo(
                    id=attractor_data['id'],
                    field=np.array(attractor_data['field']),
                    frequency=attractor_data.get('frequency', 1),
                    last_accessed=attractor_data.get('last_accessed', time.time()),
                    metadata=attractor_data.get('metadata', {})
                )
                
            # Actualizar configuración manteniendo los valores por defecto
            if 'config' in data:
                self.config.update(data['config'])
                
            return True
            
        except Exception as e:
            print(f"Error al cargar atractores desde disco: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Devuelve estadísticas sobre la memoria de atractores.
        
        Returns:
            dict: Estadísticas de la memoria
        """
        if not self.attractors:
            return {
                'total_attractors': 0,
                'avg_frequency': 0,
                'oldest_access': None,
                'newest_access': None
            }
        
        frequencies = [a.frequency for a in self.attractors.values()]
        last_accesses = [a.last_accessed for a in self.attractors.values()]
        
        return {
            'total_attractors': len(self.attractors),
            'avg_frequency': float(np.mean(frequencies)),
            'max_frequency': int(max(frequencies, default=0)),
            'min_frequency': int(min(frequencies, default=0)),
            'oldest_access': time.ctime(min(last_accesses, default=0)),
            'newest_access': time.ctime(max(last_accesses, default=0))
        }

# Alias para compatibilidad con código existente
MemoryModule = MemoriaAtractores
