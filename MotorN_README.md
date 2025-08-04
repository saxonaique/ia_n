# MotorN – Simulador del Campo Informacional DIG

Este módulo implementa un motor dinámico en Python para simular la evolución de un campo informacional según los principios de la **Teoría DIG (Dualidad Información-Gravedad)**. Incluye dos implementaciones:

1. **`MotorN`**: Implementación básica con difusión y entropía.
2. **`MotorNDIGExtendido`**: Versión extendida con memoria persistente y efectos DIG mejorados.

## 📦 Clase: `MotorN`

```python
MotorN(dim=50, gamma=0.05, lambda_=0.02)
```

### Parámetros de inicialización:
- `dim`: Dimensión del campo cuadrado (NxN).
- `gamma`: Coeficiente de difusión informacional (∇²ρ).
- `lambda_`: Coeficiente de reorganización entrópica (∂S).

---

## 📦 Clase: `MotorNDIGExtendido`

```python
MotorNDIGExtendido(dim=100, gamma=0.1, lambda_=0.05, mu=0.1, kappa=0.3)
```

### Características adicionales:
- **Memoria persistente** que conserva información entre reinicios
- **Efectos DIG mejorados** con parámetros ajustables
- **Panel de control** para ajuste en tiempo real de parámetros

### Parámetros adicionales:
- `mu`: Tasa de actualización de la memoria (0-1)
- `kappa`: Fuerza del feedback de la memoria (0-1)

## 🎛️ Panel de Control

La interfaz gráfica incluye un panel de control con pestañas:

1. **Controles principales**:
   - Activar/desactivar motor extendido
   - Reiniciar campo (con o sin memoria)
   - Control de animación
   - Guardar/cargar configuraciones

2. **Parámetros**:
   - Deslizadores para ajustar en tiempo real:
     - Difusión (γ)
     - Entropía (λ)
     - Memoria (μ)
     - Feedback (κ)
     - Tamaño de la rejilla

## 🔧 Métodos disponibles

### `evolucionar()`
Avanza el campo ρ un paso en el tiempo según la ecuación:

\[
\rho_{t+1} = \rho_t + \gamma 
\nabla^2 
\rho - \lambda \partial S + \kappa (\rho_{mem} - \rho)
\]

- La difusión se calcula con un **Laplaciano discreto**.
- La reorganización por entropía local se modela como la desviación estándar local.
- El término de feedback de memoria solo está presente en `MotorNDIGExtendido`.

---

### `inyectar(x, y, intensidad=1.0)`
Permite **alterar un punto** del campo en las coordenadas (x, y) añadiendo información.

- Las coordenadas se mapean automáticamente a la rejilla actual.
- La inyección se mantiene acotada entre 0 y 1.
- Compatible con interacción del ratón.

---

### `reiniciar(reiniciar_memoria=False)`
Reinicia el campo a un estado aleatorio inicial.

- En `MotorNDIGExtendido`, puede preservar la memoria si `reiniciar_memoria=False`.

---

### `obtener_rho()`
Devuelve la **matriz ρ** actual (numpy array de shape (dim, dim)).

## 🐛 Correcciones recientes

1. **Alineación del puntero**:
   - Corregido el mapeo de coordenadas entre el canvas y la rejilla del motor.
   - Ahora el dibujo aparece exactamente donde se hace clic.

2. **Conversión de dimensiones**:
   - Asegurado que el parámetro `dim` siempre sea entero.
   - Prevención de errores al cambiar el tamaño de la rejilla.

3. **Persistencia de memoria**:
   - La memoria ahora persiste incluso cuando se reinicia el campo visual.
   - Mejor control sobre la interacción entre memoria y campo actual.

---

### `obtener_entropia_global()`
Calcula y devuelve el valor medio de la entropía sigma local (entropía estructural del campo).

---

## ⚙️ Implementación matemática

### Laplaciano discreto:
Aproximación clásica en mallas finitas:

```python
-4 * campo + np.roll(campo, 1, axis=0) + np.roll(campo, -1, axis=0) + ...
```

### Entropía local (σ):
Calculada como desviación estándar local usando convolución 3x3:

```python
kernel = np.ones((3, 3)) / 9
mean = convolve2d(campo, kernel, mode="same")
varianza = convolve2d((campo - mean)**2, kernel)
```


---

## 🖥️ Interfaz gráfica Tkinter

El proyecto incluye una interfaz visual interactiva basada en Tkinter (`motor_interfaz_n.py`) que permite:

- **Visualización en tiempo real** del campo informacional.
- **Evolucionar** el campo paso a paso (`Evolucionar`) o de forma continua (`Iniciar animación` / `Pausar animación`).
- **Reiniciar** el campo a un estado aleatorio (`Reiniciar`).
- **Inyectar información** en cualquier punto con un clic o "pintar" manteniendo pulsado el botón izquierdo del ratón sobre el canvas.
- **Controlar la velocidad** de la animación con un slider (milisegundos entre pasos).
- **Guardar campo**: almacena el estado actual del campo en la carpeta `datos_campo` como archivo `.npy`.
- **Cargar campo**: recupera un estado guardado previamente desde la carpeta `datos_campo`.
- **Visualización de la entropía global** en tiempo real.

### Controles disponibles

| Botón/Control           | Función                                                                 |
|-------------------------|------------------------------------------------------------------------|
| Evolucionar             | Avanza el campo un paso temporal                                        |
| Reiniciar               | Reinicia el campo a un estado aleatorio                                 |
| Iniciar/Pausar animación| Comienza o detiene la evolución automática del campo                    |
| Guardar campo           | Guarda el estado actual como `.npy` en `datos_campo`                    |
| Cargar campo            | Carga un archivo `.npy` guardado (si coincide la dimensión del campo)   |
| Slider velocidad        | Controla el tiempo entre pasos de animación (ms)                        |
| Canvas                  | Inyecta información con clic o "pinta" manteniendo pulsado el botón     |
| Entropía                | Muestra el valor medio de la entropía global del campo                  |

---

- Simulación de perturbaciones informacionales (ondas, impulsos).
- Medición de entropía en sistemas complejos.
- Bases para algoritmos de IA adaptativa con campo autoorganizado.
- Visualización en tiempo real (Tkinter, Pygame, WebGL).
- Inyección de patrones para terapia informacional (ansiedad, depresión, etc).

---

© 2025 José María Pérez Rodríguez  
Versión inicial DIG – Campo Informacional en Python
