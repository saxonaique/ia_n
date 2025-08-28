import tkinter as tk
from tkinter import ttk
import numpy as np
import scipy.ndimage as ndi
from typing import Optional, Tuple, Dict, Any, List, Union
from collections import deque
import time
import os
from PIL import Image, ImageTk

# --- Módulo 1: SensorModule (Sensorium Informacional) ---
# CORREGIDO: Eliminada la dependencia de 'librosa' y el procesamiento de audio.
class SensorModule:
    """
    Sensorium Informacional: Módulo de percepción de la IA DIG.
    Transforma datos brutos (texto, imágenes) en el campo informacional [0, 1].
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'target_field_size': (64, 64),
            'text': {'max_length': 256, 'embedding_dim': 128}
        }
        self.field_shape = self.config['target_field_size']

    def process_input(self, raw_input: Any, input_type: str = 'text') -> np.ndarray:
        print(f"[SensorModule] Procesando entrada de tipo: '{input_type}'")
        if input_type == 'text':
            return self._process_text(raw_input)
        elif input_type == 'image':
            return self._process_image(raw_input)
        else:
            print(f"[SensorModule] WARN: Tipo '{input_type}' no soportado, generando campo por defecto.")
            return np.random.rand(*self.field_shape)

    def _process_text(self, text_input: str) -> np.ndarray:
        """Procesa texto para generar un campo informacional en [0, 1]."""
        cfg = self.config['text']
        char_values = np.array([ord(c) for c in text_input.ljust(cfg['max_length'])[:cfg['max_length']]])
        embedding = np.zeros(cfg['embedding_dim'])
        for i, val in enumerate(char_values):
            embedding[i % cfg['embedding_dim']] += val
        
        # Normalizar embedding a [0, 1]
        if embedding.max() > embedding.min():
            embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min())
        
        field_flat = np.resize(embedding, self.field_shape[0] * self.field_shape[1])
        return field_flat.reshape(self.field_shape).astype(np.float32)

    def _process_image(self, image_source: Union[str, np.ndarray]) -> np.ndarray:
        """Procesa una imagen para generar un campo informacional en [0, 1]."""
        try:
            if isinstance(image_source, str):
                img = Image.open(image_source).convert('L')
            elif isinstance(image_source, np.ndarray):
                img = Image.fromarray(image_source).convert('L')
            else:
                raise ValueError("Formato de imagen no soportado.")
            img = img.resize(self.field_shape)
            return (np.array(img) / 255.0).astype(np.float32)
        except Exception as e:
            print(f"[SensorModule] ERROR: No se pudo procesar la imagen: {e}")
            return np.random.rand(*self.field_shape)

# --- Módulo 2: CoreNucleus (Núcleo Entrópico) ---
class CoreNucleus:
    """
    Núcleo Entrópico: Módulo central que gestiona el campo informacional [0, 1].
    Calcula métricas y aplica transformaciones.
    """
    def __init__(self, field_shape: Tuple[int, int] = (64, 64)):
        self.field_shape = field_shape
        self.field = np.zeros(field_shape, dtype=np.float32)
        print("[CoreNucleus] INFO: Núcleo Entrópico inicializado.")

    def receive_field(self, field: np.ndarray) -> None:
        if not isinstance(field, np.ndarray):
            print(f"[CoreNucleus] ERROR: El campo debe ser un array de NumPy. Se recibió {type(field)}")
            self.field = np.zeros(self.field_shape, dtype=np.float32)
            return
        if field.shape != self.field_shape:
             print(f"[CoreNucleus] WARN: Forma del campo ({field.shape}) no coincide. Redimensionando a {self.field_shape}.")
             zoom_factors = [t / s for t, s in zip(self.field_shape, field.shape)]
             field = ndi.zoom(field, zoom_factors)
        self.field = np.clip(field, 0, 1).astype(np.float32)

    def get_metrics(self) -> Dict[str, Any]:
        if self.field.size == 0: return {}
        
        hist, _ = np.histogram(self.field.flatten(), bins=10, range=(0, 1))
        probs = hist / self.field.size
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs + 1e-9))

        return {
            "entropía": float(entropy),
            "varianza": float(np.var(self.field)),
            "máximo": float(np.max(self.field)),
            "simetría": float(np.mean(np.abs(self.field - np.fliplr(self.field)))),
            "active_cells": int(np.sum(self.field > 0.66)),
            "inhibited_cells": int(np.sum(self.field < 0.33)),
            "neutral_cells": int(self.field.size - np.sum(self.field > 0.66) - np.sum(self.field < 0.33))
        }

    def reorganize_field(self) -> None:
        """Aplica una reorganización suave (difusión) al campo."""
        kernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
        self.field = ndi.convolve(self.field, kernel, mode='reflect')
        self.field = np.clip(self.field, 0, 1)

    def apply_attractor(self, attractor_field: np.ndarray, strength: float = 0.25) -> None:
        """Aplica un atractor al campo, moviéndolo hacia el patrón del atractor."""
        if self.field.shape != attractor_field.shape:
            print("[CoreNucleus] WARN: El atractor no coincide con la forma del campo. Redimensionando.")
            zoom_factors = [t / s for t, s in zip(self.field_shape, attractor_field.shape)]
            attractor_field = ndi.zoom(attractor_field, zoom_factors)
        
        self.field = self.field * (1 - strength) + attractor_field * strength
        self.field = np.clip(self.field, 0, 1)

# --- Módulo 3: MemoryModule (Memoria de Atractores) ---
class MemoryModule:
    """
    Memoria de Atractores: Almacena y recupera patrones de equilibrio (atractores).
    """
    def __init__(self):
        self.attractors: Dict[str, Dict[str, Any]] = {}
        self._initialize_sample_attractors()

    def _initialize_sample_attractors(self):
        print("[MemoryModule] INFO: Creando atractores de ejemplo...")
        x = np.linspace(0, 1, 64)
        y = np.linspace(0, 1, 64)
        xx, yy = np.meshgrid(x, y)
        self.add_attractor("orden_gradiente", np.sin(xx * np.pi) * np.cos(yy * np.pi) * 0.5 + 0.5)
        
        checkerboard = np.kron([[1, 0] * 32, [0, 1] * 32] * 32, np.ones((1, 1)))
        self.add_attractor("complejidad_ajedrez", checkerboard[:64, :64])

    def add_attractor(self, name: str, pattern: np.ndarray):
        self.attractors[name] = {'pattern': pattern.astype(np.float32), 'usage_count': 0}

    def find_closest_attractor(self, field: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self.attractors: return None
        
        min_dist = float('inf')
        best_match_name = None
        for name, data in self.attractors.items():
            dist = np.mean((field - data['pattern'])**2)
            if dist < min_dist:
                min_dist = dist
                best_match_name = name
        
        if best_match_name:
            self.attractors[best_match_name]['usage_count'] += 1
            print(f"[MemoryModule] INFO: Atractor más cercano encontrado: '{best_match_name}' (Distancia: {min_dist:.4f})")
            return self.attractors[best_match_name]
        return None

# --- Módulo 4: IA_Interpreter ---
def interpretar_metrica(m: Dict[str, Any]) -> str:
    """Interpreta las métricas para dar un estado cualitativo y recomendaciones."""
    if not m:
        return "IA: Esperando métricas para interpretar..."
    
    e = m.get("entropía", 0.0)
    v = m.get("varianza", 0.0)
    sym = m.get("simetría", 0.0)
    active = m.get("active_cells", 0)
    total_cells = m.get("active_cells", 0) + m.get("inhibited_cells", 0) + m.get("neutral_cells", 0)
    active_ratio = active / total_cells if total_cells > 0 else 0

    estado_sistema = "EQUILIBRIO DINÁMICO"
    recomendacion = "Continuar con reorganización y monitorización."

    if e < 0.5 and v < 0.01:
        estado_sistema = "CAMPO ESTANCADO / MUERTO"
        recomendacion = "Inyectar ruido o una entrada nueva para reactivar."
    elif e > 2.0 and v > 0.1:
        estado_sistema = "SOBRECARGA INFORMACIONAL"
        recomendacion = "Aplicar un atractor de orden para estabilizar el campo."
    elif active_ratio > 0.5:
        estado_sistema = "ALTA ACTIVIDAD"
        recomendacion = "Considerar aplicar un atractor inhibidor o suavizado."

    resumen = [
        f"✅ ESTADO: {estado_sistema}",
        f"  - Entropía: {e:.4f} | Varianza: {v:.4f}",
        f"  - Simetría (error): {sym:.4f} | Células Activas: {active_ratio:.1%}",
        f"💡 RECOMENDACIÓN: {recomendacion}"
    ]
    return "\n".join(resumen)

# --- Módulo 5: Metamodulo (Meta-Capa / Orquestador) ---
class Metamodulo:
    def __init__(self):
        print("[Metamodulo] INFO: Inicializando Meta-Capa y submódulos...")
        self.sensor_module = SensorModule()
        self.core_nucleus = CoreNucleus()
        self.memory_module = MemoryModule()
        self.current_cycle = 0

    def receive_input(self, raw_input: Any, input_type: str = 'text'):
        field = self.sensor_module.process_input(raw_input, input_type)
        self.core_nucleus.receive_field(field)
        print("[Metamodulo] INFO: Entrada procesada y campo recibido por el Núcleo.")

    def process_step(self) -> Dict[str, Any]:
        self.current_cycle += 1
        print(f"\n--- [Metamodulo] INICIO CICLO {self.current_cycle} ---")

        metrics_before = self.core_nucleus.get_metrics()
        decision = self.make_decision(metrics_before)
        print(f"[Metamodulo] Decisión: {decision.upper()}")

        if decision == 'exploit':
            attractor_data = self.memory_module.find_closest_attractor(self.core_nucleus.field)
            if attractor_data:
                self.core_nucleus.apply_attractor(attractor_data['pattern'], strength=0.3)
                print("[Metamodulo] Acción: Atractor aplicado.")
        
        self.core_nucleus.reorganize_field()
        print("[Metamodulo] Acción: Campo reorganizado.")
        
        metrics_after = self.core_nucleus.get_metrics()
        interpretation = interpretar_metrica(metrics_after)

        return {
            'cycle': self.current_cycle,
            'field': self.core_nucleus.field,
            'metrics': metrics_after,
            'decision': decision,
            'interpretation': interpretation
        }

    def make_decision(self, metrics: Dict[str, Any]) -> str:
        """Decide la estrategia general para el ciclo actual."""
        entropy = metrics.get("entropía", 0.0)
        variance = metrics.get("varianza", 0.0)

        if entropy > 2.0 or variance > 0.1:
            return "exploit"
        elif entropy < 0.5 and variance < 0.01:
            return "explore"
        else:
            return "stabilize"

# --- Módulo 6: DIGVisualizerApp (GUI) ---
class DIGVisualizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visualizador del Organismo Informacional DIG")
        self.geometry("1100x850")
        
        self.metamodulo = Metamodulo()
        self.is_running = False
        
        self.setup_ui()
        self.process_text()

    def setup_ui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=2)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas_size = 512
        self.field_canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.field_canvas.pack(fill=tk.BOTH, expand=True)
        
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        ttk.Label(controls_frame, text="Entrada de Texto:", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))
        self.text_input = tk.Text(controls_frame, height=5, width=40, font=("Courier", 10))
        self.text_input.insert("1.0", "El universo es una singularidad holográfica en equilibrio dinámico.")
        self.text_input.pack(pady=5)
        
        self.process_button = ttk.Button(controls_frame, text="Procesar Texto e Iniciar", command=self.process_text)
        self.process_button.pack(pady=5)
        
        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)
        
        self.run_button = ttk.Button(controls_frame, text="Ejecutar Simulación", command=self.toggle_simulation)
        self.run_button.pack(pady=5)
        
        self.status_label = ttk.Label(controls_frame, text="Estado: En espera", font=("Helvetica", 10, "italic"))
        self.status_label.pack(pady=5)

        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)
        
        interpretation_frame = ttk.Frame(controls_frame)
        interpretation_frame.pack(pady=5, fill="x")
        ttk.Label(interpretation_frame, text="Interpretación de la IA:", font=("Helvetica", 12, "bold")).pack(pady=(0, 5))
        self.interpretation_text = tk.Text(interpretation_frame, height=12, width=40, state=tk.DISABLED, font=("Courier", 10), wrap=tk.WORD)
        self.interpretation_text.pack()

    def update_field_canvas(self, field: np.ndarray):
        """Actualiza el canvas con una imagen generada desde el campo [0, 1]."""
        if field.size == 0: return
        img_data = (np.clip(field, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(img_data, 'L')
        img_resized = img.resize((self.canvas_size, self.canvas_size), Image.Resampling.NEAREST)
        self.photo_image = ImageTk.PhotoImage(image=img_resized)
        self.field_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

    def update_display(self, summary: Dict[str, Any]):
        self.update_field_canvas(summary.get('field'))
        
        self.interpretation_text.config(state=tk.NORMAL)
        self.interpretation_text.delete(1.0, tk.END)
        self.interpretation_text.insert(tk.END, summary.get('interpretation', ''))
        self.interpretation_text.config(state=tk.DISABLED)
        
        self.status_label.config(text=f"Ciclo {summary.get('cycle')} | Decisión: {summary.get('decision', 'N/A').upper()}")
        
    def process_text(self):
        self.is_running = False
        self.run_button.config(text="Ejecutar Simulación")
        input_text = self.text_input.get("1.0", tk.END).strip()
        if not input_text:
            self.status_label.config(text="Error: El texto de entrada no puede estar vacío.")
            return

        self.metamodulo.receive_input(input_text, 'text')
        self.update_field_canvas(self.metamodulo.core_nucleus.field)
        self.status_label.config(text="Estado: Texto procesado. Listo para simular.")

    def run_single_step(self):
        try:
            summary = self.metamodulo.process_step()
            self.update_display(summary)
        except Exception as e:
            self.status_label.config(text=f"Error en el ciclo: {e}")
            self.is_running = False
            self.run_button.config(text="Ejecutar Simulación")

    def toggle_simulation(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.run_button.config(text="Detener Simulación")
            self.run_simulation_loop()
        else:
            self.run_button.config(text="Ejecutar Simulación")

    def run_simulation_loop(self):
        if self.is_running:
            self.run_single_step()
            self.after(200, self.run_simulation_loop)

# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    try:
        print("Iniciando la aplicación de visualización del Sistema DIG...")
        app = DIGVisualizerApp()
        app.mainloop()
    except Exception as e:
        print(f"Error fatal al iniciar la aplicación: {e}")
