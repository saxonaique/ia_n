import tkinter as tk
from tkinter import ttk
from typing import Dict, Any
import numpy as np

# AÑADIDO: Importación del sistema y sus módulos desde el directorio principal.
# Esto asegura que la aplicación pueda encontrar los módulos necesarios.
import sys
import os
# Corrección de la ruta de importación para que sea más robusta
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)


from metamodulo import Metamodulo
from core_nucleus import CoreNucleus
from memory_module import MemoryModule
from sensor_module import SensorModule
from action_module import ModuloAccion
from ia_interpreter import interpretar_metrica

# CLASE METAMODULO CORREGIDA
class Metamodulo(Metamodulo):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.initialize_modules()

    def initialize_modules(self):
        """Inicializa todos los módulos del sistema DIG."""
        print("[Metamodulo] Inicializando submódulos...")
        self.sensor_module = SensorModule()
        self.core_nucleus = CoreNucleus()
        self.memory_module = MemoryModule()
        self.action_module = ModuloAccion()
        print("[Metamodulo] Submódulos inicializados correctamente.")
        
    def receive_input(self, raw_input: Any, input_type: str = 'text'):
        """
        Recibe una entrada y la procesa a través del sistema.
        """
        # 1. Procesamiento de entrada
        field = self.sensor_module.process_input(raw_input, input_type)
        
        # 2. Envío del campo al núcleo
        self.core_nucleus.receive_field(field)

    def process_step(self) -> Dict[str, Any]:
        """Ejecuta un único paso de procesamiento del sistema."""
        # 1. El núcleo reorganiza el campo
        self.core_nucleus.reorganize_field()
        
        # 2. El metamódulo obtiene métricas
        metrics = self.core_nucleus.get_metrics()
        
        # 3. El metamódulo toma una decisión
        decision = self.make_decision(metrics)
        
        # 4. El metamódulo le dice al núcleo que aplique los atractores
        # Lógica simplificada: aplicar atractores basados en la decisión
        if decision.get('decision_type') == 'exploit':
            # Simular la consulta y aplicación de un atractor de la memoria
            similar_attractors = self.memory_module.find_similar(self.core_nucleus.field)
            if similar_attractors:
                attractor = similar_attractors[0]
                self.core_nucleus.apply_attractor(attractor.field)

        # 5. El módulo de acción toma el campo y produce una salida (simulada)
        self.action_module.take_action(self.core_nucleus.field, decision)
        
        # Devolver el estado actual para la visualización
        return {
            'field': self.core_nucleus.field,
            'metrics': metrics,
            'decision': decision.get('reasoning', 'N/A')
        }

# Asegúrate de que las otras clases necesarias estén importadas y disponibles
class DIGVisualizerApp(tk.Tk):
    """
    Aplicación de visualización del sistema DIG.
    
    Proporciona una interfaz gráfica para interactuar con el sistema
    y visualizar su estado interno.
    """
    def __init__(self):
        super().__init__()
        self.title("Visualizador del Sistema DIG")
        self.geometry("1100x850")
        
        # Inicializar el Metamodulo y sus dependencias
        # Se inicializa la versión corregida de Metamodulo
        self.metamodulo = Metamodulo()
        
        self.is_running = False
        self.current_step = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configura los elementos de la interfaz gráfica."""
        # --- Marco Principal ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Sección de Canvas de Visualización ---
        self.canvas_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=2)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Crear un lienzo para dibujar el campo
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
        
        # --- Sección de Controles y Métricas ---
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Controles de entrada de texto
        ttk.Label(controls_frame, text="Entrada de Texto:", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))
        self.text_input = tk.Text(controls_frame, height=5, width=40, font=("Courier", 10))
        self.text_input.pack(pady=5)
        
        self.process_button = ttk.Button(controls_frame, text="Procesar Texto", command=self.process_text)
        self.process_button.pack(pady=5)
        
        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # Controles de simulación
        self.step_button = ttk.Button(controls_frame, text="Siguiente Paso", command=self.run_single_step)
        self.step_button.pack(pady=5)
        
        self.run_button = ttk.Button(controls_frame, text="Ejecutar Simulación (Auto)", command=self.toggle_simulation)
        self.run_button.pack(pady=5)
        
        self.status_label = ttk.Label(controls_frame, text="Estado: En espera", font=("Helvetica", 10, "italic"))
        self.status_label.pack(pady=5)

        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)

        # Marco para las métricas
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

        # Marco para la interpretación de la IA
        interpretation_frame = ttk.Frame(controls_frame)
        interpretation_frame.pack(pady=5, fill="x")
        ttk.Label(interpretation_frame, text="Interpretación de la IA:", font=("Helvetica", 12, "bold")).pack(pady=(0, 5))
        self.interpretation_text = tk.Text(interpretation_frame, height=8, width=40, state=tk.DISABLED, font=("Courier", 10))
        self.interpretation_text.pack()

    def update_field_canvas(self, field: np.ndarray):
        """Dibuja el campo en el lienzo."""
        self.field_canvas.delete("all")
        rows, cols = field.shape
        # Normalizar el campo para el color (asegurar que esté en el rango [0, 1])
        norm_field = np.clip(field, 0, 1)

        for y in range(rows):
            for x in range(cols):
                val = norm_field[y, x]
                # Convertir valor a color en una escala de grises
                color_val = int(val * 255)
                color = f'#{color_val:02x}{color_val:02x}{color_val:02x}'
                
                x1, y1 = x * self.cell_size, y * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.field_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def update_metrics_display(self, metrics: Dict[str, Any], decision: str, interpretation_text: str):
        """Actualiza las etiquetas de métricas e interpretación."""
        self.entropy_label.config(text=f"Entropía: {metrics.get('entropía', 'N/A'):.3f}")
        self.variance_label.config(text=f"Varianza: {metrics.get('varianza', 'N/A'):.3f}")
        self.max_label.config(text=f"Máximo: {metrics.get('máximo', 'N/A'):.3f}")
        
        self.interpretation_text.config(state=tk.NORMAL)
        self.interpretation_text.delete(1.0, tk.END)
        self.interpretation_text.insert(tk.END, interpretation_text)
        self.interpretation_text.config(state=tk.DISABLED)
        
        self.status_label.config(text=f"Paso {self.current_step} | Decisión: {decision}")
        
    def process_text(self):
        """Maneja el evento del botón de procesamiento de texto."""
        input_text = self.text_input.get("1.0", tk.END).strip()
        if not input_text:
            self.status_label.config(text="Estado: Escriba algo para procesar.")
            return

        try:
            self.metamodulo.receive_input(input_text, 'text')
            self.status_label.config(text="Estado: Texto enviado al sistema. Listo para procesar.")
            self.current_step = 0 # Reiniciar contador de pasos
            self.update_field_canvas(self.metamodulo.core_nucleus.field)
        except Exception as e:
            self.status_label.config(text=f"Error al procesar: {e}")

    def run_single_step(self):
        """Ejecuta un solo paso de simulación."""
        try:
            result = self.metamodulo.process_step()
            self.current_step += 1
            
            field = result.get('field', self.metamodulo.core_nucleus.field)
            self.update_field_canvas(field)
            
            metrics = result.get('metrics', {})
            decision = result.get('decision', 'N/A')
            
            # Obtener la interpretación completa del ia_interpreter
            interpretation_text = interpretar_metrica(metrics)
            
            self.update_metrics_display(metrics, decision, interpretation_text)
        except Exception as e:
            self.status_label.config(text=f"Error en el paso: {e}")

    def toggle_simulation(self):
        """Alterna entre correr y detener la simulación automática."""
        self.is_running = not self.is_running
        if self.is_running:
            self.run_button.config(text="Detener Simulación")
            self.run_simulation_loop()
        else:
            self.run_button.config(text="Ejecutar Simulación (Auto)")

    def run_simulation_loop(self):
        """Bucle de simulación automática."""
        if self.is_running:
            self.run_single_step()
            # Llama a sí mismo de nuevo después de 100ms
            self.after(100, self.run_simulation_loop)

# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    print("Iniciando la aplicación de visualización del Sistema DIG...")
    app = DIGVisualizerApp()
    app.mainloop()

