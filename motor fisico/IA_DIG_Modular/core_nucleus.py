import numpy as np
import scipy.ndimage as ndi
from typing import Optional, Tuple, Dict, Any, List # <-- CORRECCIÓN: Añadido List
from collections import Counter

class CoreNucleus:
    """
    Núcleo Entrópico: Módulo central de procesamiento de la IA DIG
    
    Responsable del análisis de entropía y reorganización del campo informacional
    hacia estados de equilibrio utilizando un campo de valores en [0, 1].
    """
    
    def __init__(self, field_shape: Tuple[int, int] = (64, 64)):
        """
        Inicializa el Núcleo Entrópico.

        Args:
            field_shape: La forma (alto, ancho) del campo informacional.
        """
        self.field_shape = field_shape
        self.field = np.zeros(field_shape, dtype=np.float32) # Usar float32 para coherencia
        self.entropy = 0.0
        self.varianza = 0.0
        self.max_val = 0.0
        self.log_history = [] # Para registrar estancamiento


    def receive_field(self, field: np.ndarray) -> None:
        """
        Recibe el campo informacional para su procesamiento.
        Asegura que el campo esté en el rango [0, 1].
        
        Args:
            field: Campo informacional con valores en el rango [0, 1].
        """
        if not isinstance(field, np.ndarray):
            raise ValueError("El campo debe ser un array de NumPy")
        if field.shape != self.field_shape:
             # Redimensionar el campo entrante si su forma no coincide con la esperada
            # Esto puede distorsionar, pero evita errores si el sensorium produce un tamaño diferente
            field = np.resize(field, self.field_shape)
        
        self.field = np.clip(field, 0, 1).astype(np.float32) # Asegurarse de que esté en [0, 1] y sea float

    def calculate_entropy(self) -> float:
        """
        Calcula la entropía del campo informacional.
        Asume que los valores del campo están en [0, 1].
        """
        # Crear un histograma de los valores del campo
        counts, _ = np.histogram(self.field, bins=256, range=(0, 1))
        # Filtrar los conteos que son cero
        counts = counts[counts > 0]
        # Calcular las probabilidades
        probabilities = counts / counts.sum()
        # Calcular la entropía de Shannon
        self.entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(self.entropy)

    def calculate_variance(self) -> float:
        """Calcula la varianza del campo informacional."""
        self.varianza = np.var(self.field)
        return float(self.varianza)

    def calculate_max(self) -> float:
        """Calcula el valor máximo del campo informacional."""
        self.max_val = np.max(self.field)
        return float(self.max_val)

    def reorganize_field(self, applied_attractors: List[str] = None) -> np.ndarray:
        """
        Reorganiza el campo informacional utilizando convolución y ruido.
        
        Args:
            applied_attractors: Lista de nombres de atractores que fueron decididos por Metamodulo.
        Returns:
            np.ndarray: Campo reorganizado en el rango [0, 1].
        """
        if applied_attractors is None:
            applied_attractors = []

        initial_field_for_reorg = self.field.copy() # Estado inicial para detección de estancamiento
        
        # Kernel de convolución simple (suavizado/difusión)
        kernel = np.array([[0.05, 0.1, 0.05],
                           [0.1,  0.4, 0.1],
                           [0.05, 0.1, 0.05]])
        
        reorganized = ndi.convolve(self.field, kernel, mode='reflect')

        # Influencias de atractores: se aplica ruido moderado si se decidió una intervención.
        # Este es un placeholder simple para la influencia de atractores en esta versión simplificada.
        if applied_attractors: # Si al menos un atractor fue activado
            noise_strength = 0.15 
            noise = np.random.normal(0, noise_strength, self.field_shape)
            reorganized += noise

        # Asegurar que los valores permanezcan en el rango [0, 1]
        reorganized = np.clip(reorganized, 0, 1)
        
        # Detección de estancamiento (comparación con el estado antes de la reorganización)
        if np.array_equal(reorganized, initial_field_for_reorg):
            self.log_history.append(f"[Núcleo] El campo no cambió significativamente tras la reorganización. Posible estancamiento.")
            print(f"[Núcleo] El campo no cambió significativamente tras la reorganización. Posible estancamiento.") # DEBUG

        self.field = reorganized
        return self.field
    
    def get_entropy_gradient(self) -> np.ndarray:
        """
        Calcula el gradiente de entropía del campo.
        Utiliza el campo actual en [0,1].
        
        Returns:
            np.ndarray: Gradiente de entropía.
        """
        if self.field is None or self.field.size == 0:
            return np.zeros(self.field_shape) # Devuelve un campo de ceros si no hay campo

        grad_y, grad_x = np.gradient(self.field)
        return np.sqrt(grad_x**2 + grad_y**2)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Calcula y devuelve las métricas actuales del campo.
        
        Returns:
            Dict: Diccionario con entropía, varianza y máximo del campo.
        """
        return {
            "entropía": self.calculate_entropy(),
            "varianza": self.calculate_variance(),
            "máximo": self.calculate_max(),
        }
























