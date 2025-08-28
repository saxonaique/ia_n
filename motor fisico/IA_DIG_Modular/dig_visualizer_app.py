# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import numpy as np
import scipy.ndimage as ndi
from typing import Optional, Tuple, Dict, Any, List
from collections import deque
import json
import time
import os
import sys

# --- Módulo: SensorModule (Sensorium Informacional) ---
class SensorModule:
    """
    Sensorium Informacional: Módulo de percepción de la IA DIG
    
    Responsable de la captura y transformación de datos brutos en el campo informacional [0, 1].
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'text': {
                'target_field_size': (64, 64),
            }
        }
        self.field_shape = self.config['text']['target_field_size']
    
    def process_input(self, raw_input: Any, input_type: str) -> np.ndarray:
        """
        Procesa una entrada cruda (solo texto por ahora) y la convierte en un campo numérico.
        """
        print(f"[SensorModule] Procesando entrada de tipo: '{input_type}'")
        if input_type == 'text':
            # Una forma simple de generar un campo a partir de un string
            # Usar la longitud del texto para un valor inicial
            value = len(raw_input) / 100.0
            field = np.full(self.field_shape, value, dtype=np.float32)
            return field
        else:
            raise ValueError("Tipo de entrada no soportado.")

# --- Módulo: CoreNucleus (Núcleo Entrópico) ---
class CoreNucleus:
    """
    Núcleo Entrópico: Módulo central de procesamiento de la IA DIG
    
    Responsable del análisis de entropía y reorganización del campo informacional
    hacia estados de equilibrio utilizando un campo de valores en [0, 1].
    """
    def __init__(self, field_shape: Tuple[int, int] = (64, 64)):
        self.field_shape = field_shape
        self.field = np.zeros(field_shape, dtype=np.float32)
        self.entropy = 0.0
        self.varianza = 0.0
        self.max_val = 0.0
        self.log_history = []
        print("[CoreNucleus] [INFO] CoreNucleus inicializado.")

    def receive_field(self, field: np.ndarray) -> None:
        if not isinstance(field, np.ndarray):
            raise ValueError("El campo debe ser un array de NumPy")
        if field.shape != self.field_shape:
            zoom_factors = [target_size / current_size for target_size, current_size in zip(self.field_shape, field.shape)]
            field = ndi.zoom(field, zoom_factors, order=1)
        field = np.clip(field, 0, 1)
        self.field = field

    def reorganize_field(self) -> np.ndarray:
        if self.field is None or self.field.size == 0:
            return np.zeros(self.field_shape)
        initial_field_for_reorg = self.field.copy()
        reorganized = ndi.gaussian_filter(self.field, sigma=1.0)
        reorganized = np.where(reorganized > 0.5, reorganized + (reorganized - 0.5) * 0.5, reorganized + (reorganized - 0.5) * 0.5)
        reorganized = np.clip(reorganized, 0, 1)
        if np.array_equal(reorganized, initial_field_for_reorg):
            self.log_history.append(f"[Núcleo] El campo no cambió significativamente tras la reorganización. Posible estancamiento.")
            print(f"[Núcleo] El campo no cambió significativamente tras la reorganización. Posible estancamiento.")
        self.field = reorganized
        return self.field
    
    def get_entropy_gradient(self) -> np.ndarray:
        if self.field is None or self.field.size == 0:
            return np.zeros(self.field_shape)
        grad_y, grad_x = np.gradient(self.field)
        return np.sqrt(grad_x**2 + grad_y**2)

    def calculate_entropy(self) -> float:
        if self.field is None or self.field.size == 0:
            return 0.0
        p = np.clip(self.field, 1e-9, 1 - 1e-9)
        entropy = -np.sum(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return entropy

    def calculate_variance(self) -> float:
        if self.field is None or self.field.size == 0:
            return 0.0
        return float(np.var(self.field))

    def calculate_max(self) -> float:
        if self.field is None or self.field.size == 0:
            return 0.0
        return float(np.max(self.field))

    def get_metrics(self) -> Dict[str, Any]:
        entropy_val = self.calculate_entropy()
        variance_val = self.calculate_variance()
        max_val = self.calculate_max()

        return {
            "entropía": entropy_val,
            "varianza": variance_val,
            "máximo": max_val
        }

    def apply_attractor(self, attractor_field: np.ndarray) -> None:
        if self.field is not None and attractor_field is not None and self.field.shape == attractor_field.shape:
            self.field = (self.field + attractor_field) / 2.0
            self.field = np.clip(self.field, 0, 1)

# --- Módulo: MemoryModule (Memoria de Atractores) ---
class MemoryModule:
    """
    Simulación de un módulo de memoria.
    """
    def __init__(self):
        # Esta es solo una simulación, la lógica real es más compleja
        self.attractors = {}
    
    def find_similar(self, field: np.ndarray) -> List:
        return []

# --- Módulo: ActionModule (Módulo de Acción) ---
class ModuloAccion:
    """
    Simulación de un módulo de acción.
    """
    def __init__(self):
        pass

    def take_action(self, field: np.ndarray, decision: Dict):
        print(f"[ActionModule] Tomando acción basada en la decisión: {decision.get('decision_type', 'N/A')}")

# --- Módulo: MetaLayer (Metamódulo) ---
class Metamodulo:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'entropy_window': 10,
            'decision_thresholds': {
                'high_entropy': 0.8,
                'low_entropy': 0.2,
                'stability': 0.1,
                'confidence': 0.7
            }
        }
        self.entropy_history: deque[float] = deque(maxlen=self.config['entropy_window'])
        self.decision_history: List[Dict[str, Any]] = []
        self.state_history: List[Dict[str, Any]] = []
        self.last_decision: Optional[Dict[str, Any]] = None
        self.last_decision_time: float = time.time()
        self.initialize_modules()
        
    def initialize_modules(self):
        print("[Metamodulo] Inicializando submódulos...")
        self.sensor_module = SensorModule()
        self.core_nucleus = CoreNucleus()
        self.memory_module = MemoryModule()
        self.action_module = ModuloAccion()
        print("[Metamodulo] Submódulos inicializados correctamente.")
        
    def receive_input(self, raw_input: Any, input_type: str = 'text'):
        try:
            field = self.sensor_module.process_input(raw_input, input_type)
            self.core_nucleus.receive_field(field)
            print("[Metamodulo] Entrada procesada y enviada al núcleo.")
        except Exception as e:
            print(f"[Metamodulo] Error al procesar entrada: {e}", file=sys.stderr)

    def process_step(self) -> Dict[str, Any]:
        try:
            self.core_nucleus.reorganize_field()
            metrics = self.core_nucleus.get_metrics()
            decision = self.make_decision(metrics)
            
            if decision.get('decision_type') == 'exploit':
                similar_attractors = self.memory_module.find_similar(self.core_nucleus.field)
                if similar_attractors:
                    attractor = similar_attractors[0]
                    self.core_nucleus.apply_attractor(attractor.field)
            
            self.action_module.take_action(self.core_nucleus.field, decision)
            
            return {
                'field': self.core_nucleus.field,
                'metrics': metrics,
                'decision': decision.get('reasoning', 'N/A')
            }
        except Exception as e:
            print(f"[Metamodulo] Error en el paso de procesamiento: {e}", file=sys.stderr)
            raise

    def make_decision(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        e = metrics.get("entropía", 0.0)
        v = metrics.get("varianza", 0.0)
        decision_type = "continue"
        if e > self.config['decision_thresholds']['high_entropy']:
            decision_type = "explore"
        elif v < self.config['decision_thresholds']['low_entropy']:
            decision_type = "reset"
        
        return {'decision_type': decision_type, 'reasoning': f"Decisión: {decision_type}"}

# --- Módulo: IA Interpreter ---
def interpretar_metrica(m: Dict[str, Any]) -> str:
    e = float(m.get("entropía", 0.0))
    v = float(m.get("varianza", 0.0))
    
    estado_sistema = "Desconocido"
    if e > 10.0:
        estado_sistema = "Caótico"
    elif e < 1.0:
        estado_sistema = "Estable"
    else:
        estado_sistema = "Dinámico"

    interpretacion = [
        f"🔍 ESTADO ACTUAL:",
        f"- Estado del sistema: {estado_sistema}",
        f"- Entropía: {e:.3f}",
        f"- Varianza: {v:.3f}",
    ]
    
    return "\n".join(interpretacion)

# --- Módulo: DIG Visualizer App ---
class DIGVisualizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visualizador del Sistema DIG")
        self.geometry("1100x850")
        
        self.metamodulo = Metamodulo()
        self.is_running = False
        self.current_step = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=2)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        field_shape = self.metamodulo.core_nucleus.field_shape
        self.canvas_size = 500
        self.cell_size = self.canvas_size // max(field_shape)
        
        self.field_canvas = tk.Canvas(self.canvas_frame, 
                                     width=self.canvas_size, 
                                     height=self.canvas_size, 
                                     bg="black", 
                                     relief="flat",
                                     highlightthickness=0)
        self.field_canvas.pack(fill=tk.BOTH, expand=True)
        
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        ttk.Label(controls_frame, text="Entrada de Texto:", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))
        self.text_input = tk.Text(controls_frame, height=5, width=40, font=("Courier", 10))
        self.text_input.pack(pady=5)
        
        self.process_button = ttk.Button(controls_frame, text="Procesar Texto", command=self.process_text)
        self.process_button.pack(pady=5)
        
        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)
        
        self.step_button = ttk.Button(controls_frame, text="Siguiente Paso", command=self.run_single_step)
        self.step_button.pack(pady=5)
        
        self.run_button = ttk.Button(controls_frame, text="Ejecutar Simulación (Auto)", command=self.toggle_simulation)
        self.run_button.pack(pady=5)
        
        self.status_label = ttk.Label(controls_frame, text="Estado: En espera", font=("Helvetica", 10, "italic"))
        self.status_label.pack(pady=5)

        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)

        metrics_frame = ttk.Frame(controls_frame)
        metrics_frame.pack(pady=5, fill="x")
        ttk.Label(metrics_frame, text="Métricas del Campo:", font=("Helvetica", 12, "bold")).pack(pady=(0, 5))
        self.entropy_label = ttk.Label(metrics_frame, text="Entropía: N/A")
        self.entropy_label.pack(anchor="w")
        self.variance_label = ttk.Label(metrics_frame, text="Varianza: N/A")
        self.variance_label.pack(anchor="w")
        self.max_label = ttk.Label(metrics_frame, text="Máximo: N/A")
        self.max_label.pack(anchor="w")
        
        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)

        interpretation_frame = ttk.Frame(controls_frame)
        interpretation_frame.pack(pady=5, fill="x")
        ttk.Label(interpretation_frame, text="Interpretación de la IA:", font=("Helvetica", 12, "bold")).pack(pady=(0, 5))
        self.interpretation_text = tk.Text(interpretation_frame, height=8, width=40, state=tk.DISABLED, font=("Courier", 10))
        self.interpretation_text.pack()

    def update_field_canvas(self, field: np.ndarray):
        self.field_canvas.delete("all")
        rows, cols = field.shape
        norm_field = np.clip(field, 0, 1)

        for y in range(rows):
            for x in range(cols):
                val = norm_field[y, x]
                color_val = int(val * 255)
                color = f'#{color_val:02x}{color_val:02x}{color_val:02x}'
                
                x1, y1 = x * self.cell_size, y * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.field_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def update_metrics_display(self, metrics: Dict[str, Any], decision: str, interpretation_text: str):
        self.entropy_label.config(text=f"Entropía: {metrics.get('entropía', 'N/A'):.3f}")
        self.variance_label.config(text=f"Varianza: {metrics.get('varianza', 'N/A'):.3f}")
        self.max_label.config(text=f"Máximo: {metrics.get('máximo', 'N/A'):.3f}")
        
        self.interpretation_text.config(state=tk.NORMAL)
        self.interpretation_text.delete(1.0, tk.END)
        self.interpretation_text.insert(tk.END, interpretation_text)
        self.interpretation_text.config(state=tk.DISABLED)
        
        self.status_label.config(text=f"Paso {self.current_step} | Decisión: {decision}")
        
    def process_text(self):
        input_text = self.text_input.get("1.0", tk.END).strip()
        if not input_text:
            self.status_label.config(text="Estado: Escriba algo para procesar.")
            return

        try:
            self.metamodulo.receive_input(input_text, 'text')
            self.status_label.config(text="Estado: Texto enviado al sistema. Listo para procesar.")
            self.current_step = 0
            self.update_field_canvas(self.metamodulo.core_nucleus.field)
        except Exception as e:
            self.status_label.config(text=f"Error al procesar: {e}")

    def run_single_step(self):
        try:
            result = self.metamodulo.process_step()
            self.current_step += 1
            
            field = result.get('field', self.metamodulo.core_nucleus.field)
            self.update_field_canvas(field)
            
            metrics = result.get('metrics', {})
            decision = result.get('decision', 'N/A')
            
            interpretation_text = interpretar_metrica(metrics)
            
            self.update_metrics_display(metrics, decision, interpretation_text)
        except Exception as e:
            self.status_label.config(text=f"Error en el paso: {e}")
            self.is_running = False
            self.run_button.config(text="Ejecutar Simulación (Auto)")

    def toggle_simulation(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.run_button.config(text="Detener Simulación")
            self.run_simulation_loop()
        else:
            self.run_button.config(text="Ejecutar Simulación (Auto)")

    def run_simulation_loop(self):
        if self.is_running:
            self.run_single_step()
            self.after(100, self.run_simulation_loop)

# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    try:
        print("Iniciando la aplicación de visualización del Sistema DIG...")
        app = DIGVisualizerApp()
        app.mainloop()
    except Exception as e:
        print(f"Error fatal al iniciar la aplicación: {e}", file=sys.stderr)

