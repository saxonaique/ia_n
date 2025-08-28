# Sistema de Plugins DIG

Este directorio contiene plugins para el Sistema de Inteligencia Artificial Digital (DIG).

## Plugins Disponibles

### 1. SmoothFilter (smooth_filter.py)
- **Función**: Aplica un filtro gaussiano de suavizado a los campos de entrada
- **Versión**: 1.0.0
- **Uso**: Suaviza campos ruidosos o con alta variabilidad

### 2. EdgeDetector (edge_detector.py)
- **Función**: Detecta bordes en campos de entrada usando operadores Sobel
- **Versión**: 1.0.0
- **Uso**: Identifica transiciones y estructuras en los campos

## Cómo Crear un Nuevo Plugin

Para crear un nuevo plugin, sigue estos pasos:

1. **Crear archivo**: Crea un archivo `.py` en este directorio
2. **Heredar de DIGPlugin**: Tu clase debe heredar de `DIGPlugin`
3. **Implementar métodos abstractos**:
   - `get_name()`: Retorna el nombre del plugin
   - `get_version()`: Retorna la versión
   - `get_description()`: Retorna descripción
   - `initialize(metamodulo)`: Inicializa el plugin
   - `process(data)`: Procesa los datos
   - `cleanup()`: Limpia recursos

4. **Función create_plugin()**: Debe existir una función que retorne una instancia del plugin

### Ejemplo de Estructura:

```python
from ia_dig_organismo import DIGPlugin

class MiPlugin(DIGPlugin):
    def __init__(self):
        self.name = "MiPlugin"
        self.version = "1.0.0"
        self.description = "Descripción de mi plugin"
    
    def get_name(self) -> str:
        return self.name
    
    def get_version(self) -> str:
        return self.version
    
    def get_description(self) -> str:
        return self.description
    
    def initialize(self, metamodulo) -> bool:
        # Inicialización
        return True
    
    def process(self, data) -> Any:
        # Procesamiento de datos
        return processed_data
    
    def cleanup(self) -> bool:
        # Limpieza
        return True

def create_plugin() -> DIGPlugin:
    return MiPlugin()
```

## Uso en la Aplicación

1. Los plugins se cargan automáticamente al iniciar la aplicación
2. Usa el botón "🔄 Recargar Plugins" para recargar plugins
3. Selecciona un plugin de la lista y haz clic en "▶ Ejecutar Plugin"
4. El resultado del plugin se aplica al campo actual

## Notas Importantes

- Los plugins deben ser compatibles con numpy arrays
- El método `process()` debe retornar un numpy array del mismo tamaño
- Maneja errores apropiadamente en todos los métodos
- Los plugins se ejecutan en el contexto del metamodulo actual
