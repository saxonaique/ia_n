"""
Plugin Detector de Bordes para el Sistema DIG
Implementa detección de bordes usando operadores Sobel
"""

import numpy as np
from typing import Any

# Clase base para plugins (definida localmente para evitar dependencias circulares)
class DIGPlugin:
    """Clase base abstracta para todos los plugins del sistema DIG."""
    
    def get_name(self) -> str:
        """Retorna el nombre del plugin."""
        pass
    
    def get_version(self) -> str:
        """Retorna la versión del plugin."""
        pass
    
    def get_description(self) -> str:
        """Retorna la descripción del plugin."""
        pass
    
    def initialize(self, metamodulo) -> bool:
        """Inicializa el plugin con el metamodulo."""
        pass
    
    def process(self, data) -> Any:
        """Procesa datos usando el plugin."""
        pass
    
    def cleanup(self) -> bool:
        """Limpia recursos del plugin."""
        pass

class EdgeDetectorPlugin(DIGPlugin):
    """Plugin que detecta bordes en campos de entrada usando operadores Sobel."""
    
    def __init__(self):
        self.name = "EdgeDetector"
        self.version = "1.0.0"
        self.description = "Detecta bordes usando operadores Sobel"
        self.metamodulo = None
        self.threshold = 0.1
    
    def get_name(self) -> str:
        return self.name
    
    def get_version(self) -> str:
        return self.version
    
    def get_description(self) -> str:
        return self.description
    
    def initialize(self, metamodulo) -> bool:
        """Inicializa el plugin con el metamodulo."""
        try:
            self.metamodulo = metamodulo
            print(f"[{self.name}] Plugin inicializado exitosamente")
            return True
        except Exception as e:
            print(f"[{self.name}] ERROR en inicialización: {e}")
            return False
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """Procesa el campo de entrada detectando bordes."""
        try:
            if not isinstance(data, np.ndarray):
                print(f"[{self.name}] ERROR: Entrada debe ser numpy array")
                return data
            
            # Aplicar detección de bordes
            edge_field = self._detect_edges(data)
            
            # Normalizar resultado
            if edge_field.max() > 0:
                edge_field = (edge_field - edge_field.min()) / (edge_field.max() - edge_field.min())
            
            print(f"[{self.name}] Bordes detectados exitosamente")
            return edge_field.astype(np.float32)
            
        except Exception as e:
            print(f"[{self.name}] ERROR procesando datos: {e}")
            return data
    
    def cleanup(self) -> bool:
        """Limpia recursos del plugin."""
        try:
            self.metamodulo = None
            print(f"[{self.name}] Plugin limpiado exitosamente")
            return True
        except Exception as e:
            print(f"[{self.name}] ERROR en cleanup: {e}")
            return False
    
    def _detect_edges(self, field: np.ndarray) -> np.ndarray:
        """Detecta bordes usando operadores Sobel."""
        # Operadores Sobel
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Aplicar convolución
        grad_x = self._convolve2d(field, sobel_x)
        grad_y = self._convolve2d(field, sobel_y)
        
        # Calcular magnitud del gradiente
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Aplicar umbral
        edges = np.where(magnitude > self.threshold, magnitude, 0)
        
        return edges
    
    def _convolve2d(self, field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Aplica convolución 2D."""
        field_height, field_width = field.shape
        kernel_height, kernel_width = kernel.shape
        
        # Padding
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        padded_field = np.pad(
            field, 
            ((pad_height, pad_height), (pad_width, pad_width)), 
            mode='edge'
        )
        
        result = np.zeros_like(field)
        
        for i in range(field_height):
            for j in range(field_width):
                result[i, j] = np.sum(
                    padded_field[i:i + kernel_height, j:j + kernel_width] * kernel
                )
        
        return result

def create_plugin() -> DIGPlugin:
    """Función requerida para crear instancia del plugin."""
    return EdgeDetectorPlugin()
