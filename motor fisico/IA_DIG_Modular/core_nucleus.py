import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import Counter

class CoreNucleus:
    """
    Núcleo Entrópico: Módulo central de procesamiento de la IA DIG
    
    Responsable del análisis de entropía y reorganización del campo informacional
    hacia estados de equilibrio.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el Núcleo Entrópico.
        
        Args:
            config: Configuración para el cálculo de entropía y equilibrio
        """
        self.field: Optional[np.ndarray] = None
        self.entropy: Optional[float] = None
        self.history: list = []
        self.config = config or {
            'entropy_window': 3,  # Tamaño de la ventana para cálculo de entropía local
            'equilibrium_threshold': 0.1,  # Umbral para considerar equilibrio
            'max_history': 10,  # Máximo de estados históricos a mantener
            'min_entropy_change': 1e-5,  # Cambio mínimo para considerar significativo
            'reorganization_iterations': 15, # <-- CAMBIO: Más iteraciones para la convergencia
            'reorganization_alpha': 0.15, # <-- CAMBIO: Mayor influencia de vecinos
            'reorganization_noise_factor': 0.03, # <-- CAMBIO: Menos ruido para fomentar estabilidad
            'reorganization_stability_threshold': 0.1 # <-- NUEVO: Umbral para celdas a estabilizar
        }
    
    def receive_field(self, field: np.ndarray) -> None:
        """
        Recibe el campo informacional para su procesamiento.
        
        Args:
            field: Campo informacional ternario (-1, 0, 1)
        """
        if not isinstance(field, np.ndarray):
            raise ValueError("El campo debe ser un array de NumPy")
        if len(field.shape) != 2: # Asegurarse de que el campo sea 2D
             raise ValueError(f"El campo debe ser 2D, se recibió un campo con forma {field.shape}")
            
        self.field = field.copy()
        self._update_history()
    
    def compute_entropy(self, window_size: Optional[int] = None) -> Tuple[float, np.ndarray]:
        """
        Calcula la entropía del campo informacional.
        
        Args:
            window_size: Tamaño de la ventana para cálculo de entropía local.
                        Si es None, usa el valor de la configuración.
                        
        Returns:
            tuple: (entropía global, mapa de entropía local)
        """
        if self.field is None:
            raise ValueError("No se ha recibido ningún campo. Use receive_field() primero.")
        
        window_size = window_size or self.config['entropy_window']
        half_window = window_size // 2
        
        # Rellenar bordes para manejar los límites
        padded = np.pad(self.field, half_window, mode='edge')
        
        # Inicializar mapa de entropía
        entropy_map = np.zeros_like(self.field, dtype=float)
        
        # Calcular entropía local en ventanas deslizantes
        for i in range(self.field.shape[0]):
            for j in range(self.field.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                
                if window.size == 0: 
                    entropy_map[i, j] = 0.0
                    continue

                counts = Counter(window.flatten())
                total = sum(counts.values())
                entropy = 0.0
                for count in counts.values():
                    p = count / total
                    entropy -= p * np.log2(p) if p > 0 else 0
                entropy_map[i, j] = entropy
        
        global_entropy = float(np.mean(entropy_map))
        self.entropy = global_entropy
        
        return global_entropy, entropy_map
    
    def get_variance(self) -> float:
        """
        Calcula la varianza del campo informacional.
        
        Returns:
            float: Varianza del campo
        """
        if self.field is None:
            return 0.0 # Valor por defecto si no hay campo para evitar errores
        return float(np.var(self.field))

    def get_max_value(self) -> float:
        """
        Calcula el valor máximo absoluto del campo informacional.
        
        Returns:
            float: Valor máximo absoluto del campo
        """
        if self.field is None:
            return 0.0 # Valor por defecto si no hay campo para evitar errores
        return float(np.max(np.abs(self.field)))


    def is_equilibrium(self, threshold: Optional[float] = None) -> bool:
        """
        Determina si el sistema ha alcanzado un estado de equilibrio.
        
        Args:
            threshold: Umbral de entropía para considerar equilibrio.
                     Si es None, usa el valor de la configuración.
                     
        Returns:
            bool: True si el sistema está en equilibrio, False en caso contrario
        """
        if self.entropy is None:
            self.compute_entropy()
            
        threshold = threshold or self.config['equilibrium_threshold']
        return self.entropy < threshold
    
    def reorganize_field(self, memory_reference: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reorganiza el campo informacional para reducir la entropía, fomentando patrones.
        
        Args:
            memory_reference: Campo de referencia de la memoria de atractores (opcional)
            
        Returns:
            np.ndarray: Campo reorganizado
        """
        if self.field is None:
            raise ValueError("No hay campo para reorganizar.")
            
        current_field_float = self.field.copy().astype(float) 
        rows, cols = self.field.shape
        
        num_iterations = self.config['reorganization_iterations']
        alpha = self.config['reorganization_alpha']
        noise_factor = self.config['reorganization_noise_factor']
        stability_threshold = self.config['reorganization_stability_threshold']

        for _ in range(num_iterations):
            next_field_float = current_field_float.copy()

            # Aplicar difusión o influencia de vecinos con un sesgo hacia la estabilidad
            for i in range(rows):
                for j in range(cols):
                    neighborhood = current_field_float[
                        max(0, i-1):min(rows, i+2),
                        max(0, j-1):min(cols, j+2)
                    ]
                    
                    if neighborhood.size > 0:
                        avg_neighbor = np.mean(neighborhood)
                        
                        # Calcula la "tensión" local (qué tan diferente es la celda del promedio)
                        local_tension = np.abs(current_field_float[i, j] - avg_neighbor)
                        
                        # Si la tensión local es baja (ya es similar a los vecinos), tender a estabilizarse
                        # Si es alta, permitir más cambio (exploración/reorganización)
                        if local_tension < stability_threshold:
                            # Estabilizar: Moverse más hacia el promedio
                            next_field_float[i, j] += (avg_neighbor - next_field_float[i, j]) * alpha * 1.5 # Más influencia del promedio
                        else:
                            # Reorganizar: Influencia del promedio más ruido
                            next_field_float[i, j] += (avg_neighbor - next_field_float[i, j]) * alpha + \
                                                      np.random.randn() * noise_factor

            current_field_float = next_field_float
            current_field_float = np.clip(current_field_float, -1.0, 1.0) 

            # Si hay una referencia de memoria, aplicarla gradualmente en cada iteración
            if memory_reference is not None and memory_reference.shape == self.field.shape:
                beta = 0.1 # Factor de influencia de la memoria por iteración
                current_field_float = (1 - beta) * current_field_float + beta * memory_reference.astype(float)
                current_field_float = np.clip(current_field_float, -1.0, 1.0) 

        # Finalmente, convertir el campo de nuevo a ternario después de las iteraciones
        # <-- CAMBIO CLAVE AQUÍ: Ajustar umbrales para que el estado neutro sea más "grande"
        new_field_ternary = np.where(current_field_float > 0.2, 1, # Umbral más alto para ser '1'
                                     np.where(current_field_float < -0.2, -1, 0)).astype(np.int8) # Umbral más bajo para ser '-1'

        self.field = new_field_ternary
        self._update_history()
        
        return new_field_ternary
    
    def get_entropy_gradient(self) -> np.ndarray:
        """
        Calcula el gradiente de entropía del campo.
        
        Returns:
            np.ndarray: Gradiente de entropía
        """
        if self.field is None:
            return np.zeros((64, 64)) 
            
        grad_y, grad_x = np.gradient(self.field.astype(float))
        return np.sqrt(grad_x**2 + grad_y**2)
    
    def _update_history(self) -> None:
        """Actualiza el historial de estados del campo."""
        if self.field is not None:
            self.history.append(self.field.copy())
            if len(self.history) > self.config['max_history']:
                self.history.pop(0)
    
    def get_entropy_change_rate(self) -> float:
        """
        Calcula la tasa de cambio de entropía a lo largo del tiempo.
        
        Returns:
            float: Tasa de cambio de entropía por paso de tiempo
        """
        if len(self.history) < 2:
            return 0.0
            
        entropies = []
        original_field_backup = self.field.copy()
        for state in self.history:
            self.field = state
            entropy, _ = self.compute_entropy()
            entropies.append(entropy)
        self.field = original_field_backup
            
        if len(entropies) < 2:
            return 0.0
            
        changes = [entropies[i+1] - entropies[i] for i in range(len(entropies)-1)]
        return float(np.mean(changes))


