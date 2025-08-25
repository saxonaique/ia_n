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
            'min_entropy_change': 1e-5  # Cambio mínimo para considerar significativo
        }
    
    def receive_field(self, field: np.ndarray) -> None:
        """
        Recibe el campo informacional para su procesamiento.
        
        Args:
            field: Campo informacional ternario (-1, 0, 1)
        """
        if not isinstance(field, np.ndarray):
            raise ValueError("El campo debe ser un array de NumPy")
            
        self.field = field.copy()
        self._update_history()
    
    def compute_entropy(self, window_size: Optional[int] = None) -> Tuple[float, np.ndarray]:
        """
        Calcula la entropía del campo informacional con un enfoque optimizado.
        
        Esta implementación usa un enfoque vectorizado para el cálculo de entropía local
        ponderada por varianza, mejorando significativamente el rendimiento.
        
        Args:
            window_size: Tamaño de la ventana para cálculo de entropía local.
                        Si es None, usa el valor de la configuración.
                        
        Returns:
            tuple: (entropía global, mapa de entropía local)
        """
        if self.field is None:
            raise ValueError("No hay campo para calcular la entropía.")
            
        window = window_size or self.config['entropy_window']
        half_window = window // 2
        height, width = self.field.shape
        
        # Convertir a float32 para mayor eficiencia de memoria
        field_float = self.field.astype(np.float32)
        
        # Rellenar los bordes con reflexión
        padded_field = np.pad(field_float, half_window, mode='reflect')
        
        # Precalcular ventanas deslizantes para el cálculo de varianza
        shape = (height, width, window, window)
        strides = (padded_field.strides[0], padded_field.strides[1], 
                  padded_field.strides[0], padded_field.strides[1])
        
        # Usar as_strided para crear una vista de las ventanas deslizantes
        from numpy.lib.stride_tricks import as_strided
        windows = as_strided(padded_field, shape=shape, strides=strides)
        
        # Calcular varianza local de manera vectorizada
        local_variance = np.var(windows, axis=(2, 3))
        
        # Normalizar varianza para usar como pesos (evitando división por cero)
        max_var = np.max(local_variance)
        if max_var > 1e-10:
            local_variance = local_variance / max_var
        
        # Calcular entropía local de manera vectorizada
        # Discretizar los valores en bins para el cálculo de entropía
        bins = np.linspace(-1, 1, 21)  # 20 bins entre -1 y 1
        digitized = np.digitize(windows, bins) - 1  # Índices de bins
        
        # Calcular histogramas normalizados (probabilidades) para cada ventana
        probs = np.zeros((height, width, len(bins)-1), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                hist, _ = np.histogram(digitized[i,j], bins=len(bins)-1, range=(0, len(bins)-2))
                probs[i,j] = (hist + 1e-10) / (window*window + 1e-10*len(bins))
        
        # Calcular entropía de Shannon
        entropy_map = -np.sum(probs * np.log2(probs + 1e-10), axis=2)
        
        # Aplicar ponderación por varianza local
        entropy_map = entropy_map * (1.0 + local_variance)
        
        # Asegurar que la entropía esté en [0, log2(3)]
        max_entropy = np.log2(3)
        entropy_map = np.clip(entropy_map, 0, max_entropy)
        
        # Calcular entropía global como la media ponderada
        if np.sum(local_variance) > 0:
            global_entropy = np.sum(entropy_map * local_variance) / np.sum(local_variance)
        else:
            global_entropy = np.mean(entropy_map)
        
        # Suavizado final del mapa de entropía
        if window > 1:
            try:
                from scipy.ndimage import gaussian_filter
                entropy_map = gaussian_filter(entropy_map, sigma=0.7)
            except ImportError:
                # Filtro de media simple si scipy no está disponible
                kernel = np.ones((3, 3), dtype=np.float32) / 9.0
                entropy_map = np.pad(entropy_map, 1, mode='edge')
                entropy_map = np.array([
                    np.sum(kernel * entropy_map[i:i+3, j:j+3])
                    for i in range(height) for j in range(width)
                ]).reshape(height, width)
        
        self.entropy = float(global_entropy)
        return float(global_entropy), entropy_map
    
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
    
    def reorganize_field(self, memory_reference: Optional[np.ndarray] = None, num_iterations: int = 3) -> np.ndarray:
        """
        Reorganiza el campo informacional para reducir la entropía usando un enfoque mejorado
        que combina reglas de autómatas celulares con influencia de memoria y patrones emergentes.
        
        Args:
            memory_reference: Campo de referencia de la memoria de atractores (opcional)
            num_iterations: Número de iteraciones de reorganización a realizar (1-10)
            
        Returns:
            np.ndarray: Campo reorganizado
            
        Raises:
            ValueError: Si no hay campo para reorganizar o los parámetros son inválidos
        """
        if self.field is None:
            raise ValueError("No hay campo para reorganizar.")
            
        # Validar parámetros
        if not isinstance(num_iterations, int) or num_iterations < 1 or num_iterations > 10:
            num_iterations = 3
            
        if memory_reference is not None and not isinstance(memory_reference, np.ndarray):
            raise ValueError("La referencia de memoria debe ser un array de NumPy o None")
        
        height, width = self.field.shape
        
        # Inicializar el campo actual con valores flotantes para mayor precisión
        current_field = self.field.astype(np.float32)
        
        # Preparar la influencia de la memoria si está disponible
        if memory_reference is not None and memory_reference.shape == (height, width):
            # Suavizar la referencia de memoria para una transición más natural
            try:
                from scipy.ndimage import gaussian_filter
                memory_influence = gaussian_filter(memory_reference.astype(np.float32), sigma=0.7)
            except ImportError:
                memory_influence = memory_reference.astype(np.float32)
        else:
            memory_influence = None
        
        # Preparar kernel para operaciones de convolución
        kernel = np.array([[0.5, 1.0, 0.5],
                          [1.0, 0.0, 1.0],
                          [0.5, 1.0, 0.5]], dtype=np.float32)
        kernel = kernel / np.sum(kernel)  # Normalizar
        
        # Bucle principal de iteraciones
        for iteration in range(num_iterations):
            # Crear una copia del campo para la nueva iteración
            new_field = np.zeros_like(current_field)
            
            # Aplicar una convolución 2D para calcular la influencia de los vecinos
            padded = np.pad(current_field, 1, mode='wrap')  # Condiciones de contorno periódicas
            
            # Calcular la influencia de los vecinos usando convolución
            neighbor_influence = np.zeros_like(current_field)
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue  # Saltar la celda central
                    neighbor_influence += padded[i:i+height, j:j+width] * kernel[i,j]
            
            # Aplicar reglas de transición mejoradas
            for i in range(height):
                for j in range(width):
                    center = current_field[i,j]
                    influence = neighbor_influence[i,j]
                    
                    # Aplicar reglas basadas en el estado actual y la influencia de los vecinos
                    if center > 0.33:  # Celda activa
                        # Sobrevive con 2-3 vecinos activos, muere por soledad o sobrepoblación
                        if 1.5 <= influence <= 3.0:
                            new_field[i,j] = 1.0  # Sobrevive
                        else:
                            new_field[i,j] = 0.0  # Muere
                            
                    elif center < -0.33:  # Celda inhibida
                        # Se mantiene con 2-3 vecinos inhibidos, de lo contrario se disipa
                        if -3.0 <= influence <= -1.5:
                            new_field[i,j] = -1.0  # Se mantiene
                        else:
                            new_field[i,j] = 0.0  # Se disipa
                            
                    else:  # Celda neutra
                        # Nacimiento con exactamente 3 vecinos activos o inhibidos
                        if abs(influence - 1.0) < 0.1:  # ~3 vecinos activos
                            new_field[i,j] = 1.0  # Se activa
                        elif abs(influence + 1.0) < 0.1:  # ~3 vecinos inhibidos
                            new_field[i,j] = -1.0  # Se inhibe
                        else:
                            new_field[i,j] = 0.0  # Permanece neutra
            
            # Aplicar influencia de la memoria (si está disponible)
            if memory_influence is not None and iteration < num_iterations - 2:
                # La influencia de la memoria disminuye con las iteraciones
                memory_strength = 0.3 * (1 - (iteration / (num_iterations - 1)))
                new_field = (1 - memory_strength) * new_field + memory_strength * memory_influence
            
            # Aplicar ruido controlado (excepto en la última iteración)
            if iteration < num_iterations - 1:
                noise = np.random.normal(0, 0.05, size=current_field.shape)
                new_field = np.clip(new_field + noise, -1, 1)
            
            current_field = new_field
        
        # Redondear a valores ternarios con histeresis para mayor estabilidad
        new_field = np.zeros_like(current_field, dtype=np.int8)
        new_field[current_field > 0.4] = 1
        new_field[current_field < -0.4] = -1
        
        # Actualizar el campo y su historia
        self.field = new_field
        self._update_history()
        
        return self.field
    
    def get_entropy_gradient(self) -> np.ndarray:
        """
        Calcula el gradiente de entropía del campo.
        
        Returns:
            np.ndarray: Gradiente de entropía
        """
        if self.field is None:
            raise ValueError("No hay campo para calcular el gradiente.")
            
        # Calcular gradiente usando diferencias finitas
        grad_y, grad_x = np.gradient(self.field.astype(float))
        return np.sqrt(grad_x**2 + grad_y**2)
    
    def _update_history(self) -> None:
        """Actualiza el historial de estados del campo."""
        if self.field is not None:
            self.history.append(self.field.copy())
            # Mantener solo los últimos N estados
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
            
        # Calcular entropías para los últimos N estados
        entropies = []
        for state in self.history:
            self.field = state
            entropy, _ = self.compute_entropy()
            entropies.append(entropy)
            
        # Calcular tasa de cambio promedio
        if len(entropies) < 2:
            return 0.0
            
        changes = [entropies[i+1] - entropies[i] for i in range(len(entropies)-1)]
        return float(np.mean(changes))
