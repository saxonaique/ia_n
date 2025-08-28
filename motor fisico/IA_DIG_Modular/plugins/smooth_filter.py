"""
Plugin de Filtro de Suavizado para el Sistema DIG
Implementa un filtro gaussiano para suavizar campos de entrada
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

class SmoothFilterPlugin(DIGPlugin):
    """Plugin que aplica un filtro de suavizado gaussiano a los campos de entrada."""
    
    def __init__(self):
        self.name = "SmoothFilter"
        self.version = "1.0.0"
        self.description = "Aplica filtro gaussiano de suavizado a campos de entrada"
        self.metamodulo = None
        self.kernel_size = 5
        self.sigma = 1.0
    
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
        """Procesa el campo de entrada aplicando suavizado gaussiano."""
        try:
            if not isinstance(data, np.ndarray):
                print(f"[{self.name}] ERROR: Entrada debe ser numpy array")
                return data
            
            # Crear kernel gaussiano
            kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)
            
            # Aplicar convolución 2D
            smoothed_field = self._convolve2d(data, kernel)
            
            # Normalizar resultado
            if smoothed_field.max() > 0:
                smoothed_field = (smoothed_field - smoothed_field.min()) / (smoothed_field.max() - smoothed_field.min())
            
            print(f"[{self.name}] Campo suavizado exitosamente")
            return smoothed_field.astype(np.float32)
            
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
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Crea un kernel gaussiano 2D."""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.sum()
    
    def _convolve2d(self, field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Aplica convolución 2D manualmente."""
        field_height, field_width = field.shape
        kernel_height, kernel_width = kernel.shape
        
        # Padding para mantener dimensiones
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        padded_field = np.pad(field, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
        result = np.zeros_like(field)
        
        for i in range(field_height):
            for j in range(field_width):
                result[i, j] = np.sum(
                    padded_field[i:i + kernel_height, j:j + kernel_width] * kernel
                )
        
        return result

def create_plugin() -> DIGPlugin:
    """Función requerida para crear instancia del plugin."""
    return SmoothFilterPlugin()
