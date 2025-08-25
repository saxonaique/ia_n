# DIG AI System

Un sistema de inteligencia artificial basado en dinámicas de campo informacional y atractores de memoria.

## Visión General

El sistema DIG (Dynamic Information Gradient) es una arquitectura de IA que modela el procesamiento de información a través de un campo dinámico que evoluciona en el tiempo, influenciado por entradas sensoriales, memoria de patrones y reglas de reorganización local.

## Características Principales

- **Procesamiento de Campo Dinámico**: Modela la información como un campo que evoluciona según reglas inspiradas en autómatas celulares.
- **Memoria de Atractores**: Almacena y recupera patrones estables para guiar la reorganización del campo.
- **Análisis de Entropía**: Mide la complejidad y el orden del sistema en tiempo real.
- **Interpretación Cualitativa**: Proporciona análisis legibles por humanos del estado del sistema.
- **Visualización Integrada**: Muestra la evolución del campo y métricas clave.

## Mejoras Recientes

### 1. Cálculo de Entropía Optimizado
- Implementación vectorizada para mejor rendimiento
- Cálculo ponderado por varianza local
- Suavizado adaptativo del mapa de entropía

### 2. Reorganización de Campo Mejorada
- Reglas de evolución celular más estables
- Integración mejorada con memoria de atractores
- Manejo de condiciones de contorno mejorado

### 3. Análisis de Métricas Avanzado
- Seguimiento de distribución de estados (activos/inhibidos/neutrales)
- Cálculo de simetría de patrones
- Análisis de tendencias temporales

### 4. Integración de Memoria
- Búsqueda jerárquica de atractores similares
- Combinación ponderada de múltiples atractores
- Actualización adaptativa de la memoria basada en éxito

## Estructura del Proyecto

- `core_nucleus.py`: Núcleo del sistema, maneja el campo informacional y su evolución
- `metamodulo.py`: Orquestador principal que coordina los módulos
- `sensor_module.py`: Procesamiento de entradas sensoriales
- `memory_module.py`: Almacenamiento y recuperación de atractores
- `evolution_processor.py`: Generación de nuevos patrones a través de evolución
- `action_module.py`: Generación de salidas basadas en el estado del sistema
- `ia_interpreter.py`: Análisis cualitativo del estado del sistema
- `visualizer.py`: Visualización del campo y métricas
- `main.py`: Punto de entrada de la aplicación

## Requisitos

- Python 3.8+
- NumPy
- SciPy (opcional, para mejor rendimiento)
- Matplotlib (para visualización)

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd IA_DIG_Modular
   ```

2. Crear un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para iniciar el sistema con la interfaz de visualización:

```bash
python main.py
```

## Configuración

El sistema se puede configurar mediante el diccionario de configuración en `metamodulo.py`. Los parámetros clave incluyen:

- Umbrales de entropía para la toma de decisiones
- Tamaño y forma del campo informacional
- Parámetros de la memoria de atractores
- Configuración del procesador evolutivo

## Contribución

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios propuestos antes de hacer un pull request.

## Licencia

[Incluir información de licencia aquí]

---

Desarrollado por [Tu Nombre/Organización]
