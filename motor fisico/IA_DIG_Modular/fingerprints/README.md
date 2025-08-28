# 🔄 Sistema de Huellas 3 Puntos - Motor N DIG

Este directorio contiene las huellas capturadas por el Sistema de Inteligencia Artificial Digital (DIG) durante las simulaciones.

## 🎯 ¿Qué son las Huellas?

Las **huellas** son instantáneas del estado del campo informacional en momentos clave de la evolución:

### 📸 **3 Etapas de Captura:**

1. **🟢 INICIAL** → Estado 0 del campo (ciclo 1)
2. **🟡 INTERMEDIA** → Estado al 50% de las iteraciones (ciclo 1000)
3. **🔴 FINAL** → Estado final o cuando se estabiliza (ciclo 2000+)

## 🔍 **Información Capturada en Cada Huella:**

- **Timestamp** de captura
- **Ciclo** de simulación
- **Forma del campo** (dimensiones)
- **Estadísticas del campo:**
  - Media, desviación estándar
  - Mínimo, máximo
  - Entropía, varianza, simetría
- **Células activas/inhibidas/neutrales**
- **Muestra representativa** del campo (16 puntos estratégicos)

## 📁 **Estructura de Archivos:**

```
fingerprints/
├── README.md                           # Esta documentación
├── fingerprints_session_XXXXXXXX.json  # Huellas de sesión específica
└── ...
```

## 🚀 **Cómo Funciona:**

### **Captura Automática:**
- **Ciclo 1**: Se captura la huella INICIAL
- **Ciclo 1000**: Se captura la huella INTERMEDIA  
- **Ciclo 2000+**: Se captura la huella FINAL

### **Análisis de Evolución:**
El sistema calcula automáticamente:
- Cambios en entropía entre etapas
- Evolución de la varianza
- Transformaciones de simetría
- Ciclos entre capturas

## 💾 **Formato de Archivo JSON:**

```json
{
  "session_id": "session_1234567890",
  "timestamp": 1234567890.123,
  "fingerprints": {
    "inicial": { ... },
    "intermedia": { ... },
    "final": { ... }
  },
  "summary": {
    "total_fingerprints": 3,
    "stages_captured": ["inicial", "intermedia", "final"],
    "evolution_analysis": {
      "inicial_to_intermedia": { ... },
      "intermedia_to_final": { ... }
    }
  }
}
```

## 🎮 **Uso en la Interfaz:**

1. **Ver Estado**: El panel muestra cuántas huellas se han capturado
2. **Guardar**: Botón "💾 Guardar Huellas" para exportar la sesión
3. **Cargar**: Botón "📂 Cargar Huellas" para importar sesiones anteriores
4. **Monitoreo**: Estado en tiempo real del sistema de huellas

## 🔬 **Casos de Uso:**

### **Investigación:**
- Comparar evolución de diferentes entradas
- Analizar patrones de estabilización
- Estudiar transiciones de fase

### **Debugging:**
- Identificar cuándo el sistema se desestabiliza
- Rastrear cambios en métricas clave
- Verificar comportamiento esperado

### **Documentación:**
- Registrar experimentos específicos
- Compartir resultados con otros investigadores
- Crear base de datos de comportamientos

## ⚙️ **Configuración:**

- **Máximo de iteraciones**: 2000 ciclos (configurable)
- **Punto intermedio**: 50% de las iteraciones
- **Muestra del campo**: 16 puntos estratégicos
- **Formato de salida**: JSON con metadatos completos

## 🎯 **Beneficios:**

✅ **Eficiencia**: Solo 3 capturas por sesión  
✅ **Completitud**: Información estadística completa  
✅ **Análisis**: Comparación automática entre etapas  
✅ **Persistencia**: Guardado en formato estándar JSON  
✅ **Integración**: Totalmente integrado con el Motor N DIG  

---

*El Sistema de Huellas 3 Puntos transforma tu Motor N en un laboratorio de investigación con capacidad de documentación automática y análisis evolutivo.* 🚀
