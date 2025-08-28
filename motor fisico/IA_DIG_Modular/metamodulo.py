# -*- coding: utf-8 -*-
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional, List, Tuple
from queue import Queue, Empty
import threading

# --- Implementaciones de ejemplo de los módulos para hacer el código ejecutable ---
# Si ya tienes estas clases en archivos separados, puedes eliminarlas
# y corregir las importaciones en su lugar.

class SensorModule:
    """
    Módulo de ejemplo para procesar la entrada.
    Convierte un texto en un campo numérico.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.field_shape = self.config.get('field_shape', (64, 64))
        self._log(f"Iniciando SensorModule con la configuración: {self.config}", level='DEBUG')

    def process_input(self, raw_input: Any, input_type: str) -> np.ndarray:
        """
        Procesa una entrada cruda y la convierte en un campo numérico.
        """
        self._log(f"Procesando entrada de tipo: '{input_type}' con forma de campo: {self.field_shape}")
        
        if not isinstance(raw_input, str):
            self._log(f"Entrada no es un string, convirtiendo a string.", level='WARNING')
            raw_input = str(raw_input)
            
        value = len(raw_input) / 100.0
        # Siempre crea un campo 2D del tamaño correcto
        field = np.full(self.field_shape, value, dtype=np.float32)
        return field

    def _log(self, message: str, level: str = 'INFO') -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [SensorModule] [{level}] {message}")

class CoreNucleus:
    """
    Núcleo de ejemplo para gestionar el campo de la IA.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        field_shape = self.config.get('field_shape', (64, 64))
        if not isinstance(field_shape, (tuple, list)) or len(field_shape) != 2 or not all(isinstance(x, int) for x in field_shape):
            self._log("La forma del campo en la configuración es inválida. Usando la forma por defecto (64, 64).", level='WARNING')
            field_shape = (64, 64)

        self.field = np.zeros(field_shape, dtype=np.float32)
        self.field_history = []
        self.hist_capacity = 10
        self._log("CoreNucleus inicializado.", level='INFO')

    def _log(self, message: str, level: str = 'INFO') -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [CoreNucleus] [{level}] {message}")

    def receive_field(self, new_field: np.ndarray) -> None:
        """
        Recibe un nuevo campo y lo valida antes de almacenarlo.
        """
        if not isinstance(new_field, np.ndarray) or new_field.ndim != 2:
            self._log(f"Campo recibido con forma incorrecta: {new_field.shape}. Esperado: {self.field.shape}. Creando un campo de ceros.", level='ERROR')
            self.field = np.zeros(self.field.shape, dtype=np.float32)
        elif new_field.shape != self.field.shape:
             self._log(f"Tamaño de campo recibido ({new_field.shape}) no coincide con el tamaño esperado ({self.field.shape}). Intentando redimensionar.", level='WARNING')
             try:
                 self.field = new_field.reshape(self.field.shape).copy()
             except Exception as e:
                 self._log(f"Error al redimensionar el campo: {e}. Creando un nuevo campo de ceros.", level='ERROR')
                 self.field = np.zeros(self.field.shape, dtype=np.float32)
        else:
            self.field = new_field.copy()

        self.field_history.append(self.field.copy())
        if len(self.field_history) > self.hist_capacity:
            self.field_history.pop(0)

    def get_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de ejemplo para el campo actual.
        """
        if self.field.size == 0 or self.field.ndim != 2:
            self._log("El campo no es una matriz válida o está vacío para calcular métricas. Devolviendo valores predeterminados.", level='ERROR')
            return {'varianza': 0.0, 'entropia': 0.0, 'maximo': 0.0}
        
        try:
            variance = np.var(self.field)
            
            # Aplanamos el campo para el cálculo de entropía
            flattened_field = self.field.flatten()
            
            # Se asegura de que la matriz no esté vacía antes de bincount
            if flattened_field.size > 0:
                bins = np.digitize(flattened_field, bins=10)
                counts = np.bincount(bins)
                probabilities = counts / counts.sum()
                shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            else:
                self._log("La matriz aplanada está vacía. Entropía no calculable.", level='WARNING')
                shannon_entropy = 0.0
                
            max_val = np.max(self.field)
            
            return {'varianza': float(variance), 'entropia': float(shannon_entropy), 'maximo': float(max_val)}
        except Exception as e:
            self._log(f"Error general al calcular métricas: {e}. Devolviendo valores predeterminados.", level='ERROR')
            return {'varianza': 0.0, 'entropia': 0.0, 'maximo': 0.0}

    def reorganize_field(self, applied_attractors: List[Any]) -> np.ndarray:
        """
        Simula la reorganización del campo.
        """
        return self.field * 0.9

class MemoriaAtractores:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    def find_closest_attractor(self, field_input: np.ndarray) -> Optional[np.ndarray]:
        return np.full_like(field_input, 0.5)
    def save(self) -> None:
        pass

class EvolutionProcessor:
    def __init__(self, memoria: MemoriaAtractores):
        self.memoria = memoria
    def evolve_memory(self) -> None:
        pass

class ModuloAccion:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    def apply_attractor(self, attractor: Any) -> None:
        pass

def interpretar_metrica(metrics: Dict[str, float]) -> str:
    if metrics['entropia'] > 1.5:
        return 'caotico'
    elif metrics['varianza'] < 0.01:
        return 'ordenado'
    else:
        return 'equilibrado'


# --- Visualizador Tkinter Mejorado ---
class Visualizador(tk.Tk):
    def __init__(self, message_queue: Queue):
        super().__init__()
        self.message_queue = message_queue
        self.title("Visualizador IA DIG")
        self.geometry("800x600")
        self.configure(bg='#282c34')
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('.', background='#282c34', foreground='#abb2bf')
        self.style.configure('TFrame', background='#282c34')
        self.style.configure('TLabel', background='#282c34', foreground='#abb2bf', font=('Inter', 10))

        self._create_widgets()
        self.after(100, self._process_messages)
        print("[Visualizador] Visualizador Tkinter iniciado.")
    
    def _create_widgets(self):
        # Frame principal con padding
        main_frame = ttk.Frame(self, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Panel para el resumen ejecutivo
        summary_frame = ttk.LabelFrame(main_frame, text="✅ RESUMEN EJECUTIVO ✅", padding="10 10 10 10", style='TFrame')
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(summary_frame, text="Estado: Esperando datos...", font=('Inter', 12, 'bold'))
        self.status_label.pack(fill=tk.X, pady=(0, 5))
        
        self.decision_label = ttk.Label(summary_frame, text="Decisión: N/A", font=('Inter', 11))
        self.decision_label.pack(fill=tk.X)
        
        self.recommendation_label = ttk.Label(summary_frame, text="Recomendación: N/A", font=('Inter', 11))
        self.recommendation_label.pack(fill=tk.X)

        # Panel para las métricas detalladas
        metrics_frame = ttk.LabelFrame(main_frame, text="📊 MÉTRICAS DETALLADAS 📊", padding="10 10 10 10", style='TFrame')
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.entropy_label = ttk.Label(metrics_frame, text="Entropía: N/A")
        self.entropy_label.pack(fill=tk.X)
        
        self.variance_label = ttk.Label(metrics_frame, text="Varianza: N/A")
        self.variance_label.pack(fill=tk.X)

        self.max_label = ttk.Label(metrics_frame, text="Máximo: N/A")
        self.max_label.pack(fill=tk.X)

        # Panel para los registros
        log_frame = ttk.LabelFrame(main_frame, text="📜 REGISTROS DEL SISTEMA 📜", padding="10 10 10 10", style='TFrame')
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, bg='#1e1e1e', fg='#f8f8f2', font=('Consolas', 9), relief=tk.FLAT)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # Tags para colores de logs
        self.log_text.tag_config('INFO', foreground='#61afef')
        self.log_text.tag_config('WARNING', foreground='#e5c07b')
        self.log_text.tag_config('ERROR', foreground='#e06c75')
        self.log_text.tag_config('DEBUG', foreground='#98c379')

    def _process_messages(self):
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get(block=False)
                self.update_ui(message)
            except Empty:
                pass
        self.after(100, self._process_messages)

    def update_ui(self, message: Dict[str, Any]):
        if not isinstance(message, dict) or message.get('status') == 'success' or message.get('status') == 'error':
            # Ignora mensajes de estado simples para evitar errores
            print(f"[Visualizador] Ignorando mensaje de estado simple: {message}")
            return
            
        summary = message.get('ia_interpretation', 'IA: Esperando datos...')
        
        # Actualizar logs
        self.log_text.insert(tk.END, f"\n--- Inicio Ciclo {message.get('cycle')} ---\n", 'INFO')
        self.log_text.insert(tk.END, message.get('full_log', '') + "\n")
        self.log_text.see(tk.END)

        # Extraer y actualizar el resumen ejecutivo
        status_line = "Estado: N/A"
        decision_line = "Decisión actual: N/A"
        recommendation_line = "Recomendación: N/A"
        
        lines = summary.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('✅ ESTADO:'):
                status_line = line.replace('✅ ESTADO:', 'Estado:')
            elif 'Decisión actual:' in line:
                decision_line = line.strip()
            elif '💡 RECOMENDACIÓN:' in line:
                recommendation_line = line.strip()
                
        self.status_label.config(text=status_line, foreground=self._get_status_color(status_line))
        self.decision_label.config(text=decision_line)
        self.recommendation_label.config(text=recommendation_line)

        # Extraer y actualizar métricas detalladas
        metrics = message.get('reorganized_field_metrics', {})
        self.entropy_label.config(text=f"Entropía: {metrics.get('entropia', 'N/A'):.4f}")
        self.variance_label.config(text=f"Varianza: {metrics.get('varianza', 'N/A'):.4f}")
        self.max_label.config(text=f"Máximo: {metrics.get('maximo', 'N/A'):.4f}")
    
    def _get_status_color(self, status: str) -> str:
        if 'EQUILIBRIO' in status or 'ORDERED' in status:
            return '#98c379' # Verde
        elif 'CAOTICO' in status or 'CHAOTIC' in status:
            return '#e06c75' # Rojo
        return '#61afef' # Azul por defecto

# --- Código del Metamodulo ---

class Metamodulo:
    """
    Metamódulo: Orquestador principal de la IA DIG.
    Coordina la interacción entre todos los módulos, monitorea el estado
    y toma decisiones estratégicas.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, message_queue: Optional[Queue] = None):
        self.config = config or self._default_config()
        self.message_queue = message_queue if message_queue else Queue()

        self.sensorium = SensorModule(self.config.get('sensor_module'))
        self.core_nucleus = CoreNucleus(self.config.get('core_nucleus'))
        self.memoria = MemoriaAtractores(self.config.get('memory_module'))
        self.evolution_processor = EvolutionProcessor(self.memoria) 
        self.action_module = ModuloAccion(self.config.get('action_module'))

        self.current_cycle = 0
        self.last_global_decision = None
        self.log_history = [] 
        
    def _default_config(self) -> Dict[str, Any]:
        default_field_shape = (64, 64)
        return {
            'sensor_module': {'field_shape': default_field_shape},
            'core_nucleus': {'field_shape': default_field_shape},
            'memory_module': {},
            'action_module': {}
        }

    def _log(self, message: str, level: str = 'INFO') -> None:
        log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}"
        self.log_history.append(log_entry)
        print(log_entry)

    def run_dig_system(self, input_sources: List[Tuple[Any, str]], max_cycles: Optional[int] = None) -> None:
        if not input_sources:
            self._log("No se proporcionaron fuentes de entrada para la ejecución.", level='WARNING')
            return

        self._log("Iniciando ejecución del sistema DIG...")
        num_inputs = len(input_sources)
        
        for i in range(max_cycles if max_cycles else num_inputs):
            raw_input, input_type = input_sources[i % num_inputs]
            self._log(f"\n--- Ejecutando ciclo {self.current_cycle + 1} con entrada '{input_type}' ---")
            cycle_result = self.process_cycle(raw_input, input_type)
            
            if cycle_result.get('error'):
                self._log(f"Ejecución detenida debido a error en ciclo {self.current_cycle}. Error: {cycle_result.get('error')}", level='ERROR')
                break

        self._log("Ejecución del sistema DIG finalizada.")

    def process_cycle(self, raw_input: Any, input_type: str) -> Dict[str, Any]:
        try:
            field_input = self.sensorium.process_input(raw_input, input_type)
            self.core_nucleus.receive_field(field_input)
            metrics = self.core_nucleus.get_metrics()
            self._log(f"Métricas del campo: {metrics}")

            decision = interpretar_metrica(metrics)
            self._log(f"El sistema ha tomado la decisión: '{decision}'")
            
            if decision in ["caotico", "equilibrado"]:
                atractor = self.memoria.find_closest_attractor(field_input)
                if atractor is not None:
                    self.action_module.apply_attractor(atractor)
                    reorganized_field = self.core_nucleus.reorganize_field(applied_attractors=[atractor])
                    self.core_nucleus.receive_field(reorganized_field)
                else:
                    self._log("No se encontró un atractor cercano. El campo no se reorganizó.", level='WARNING')

            self.evolution_processor.evolve_memory()
            self.memoria.save()
            
            self.current_cycle += 1
            
            # Simulación de un resumen de IA para la UI
            ia_interpretation = f"""
✅ ESTADO: EQUILIBRIO DINÁMICO: El sistema mantiene un balance saludable entre orden y actividad.

📊 MÉTRICAS DETALLADAS:
- Entropía: {metrics.get('entropia', 0.0):.4f}
- Varianza: {metrics.get('varianza', 0.0):.4f}
- Simetría: N/A
- Composición: N/A
- Ratios: N/A

🔍 ANÁLISIS DE COMPOSICIÓN:

🎯 DECISIÓN DEL SISTEMA:
- Decisión actual: {decision.upper()}
- Atractores aplicados: N/A

💡 RECOMENDACIÓN: Se recomienda continuar con REORGANIZACIÓN LOCAL CON MEMORIA.

📌 RESUMEN EJECUTIVO:
Estado: EQUILIBRIO DINÁMICO
Entropía: ALTA
Simetría: BAJA
Balance: NEUTRO
            """.strip()

            cycle_summary = {
                'cycle': self.current_cycle,
                'input_type': input_type,
                'reorganized_field_metrics': metrics,
                'metamodule_decision': decision,
                'ia_interpretation': ia_interpretation,
                'full_log': "\n".join(self.log_history)
            }
            
            self.message_queue.put(cycle_summary)
            
            return {'status': 'success'}

        except Exception as e:
            self._log(f"Error durante el ciclo de procesamiento: {e}", level='ERROR')
            # En caso de un error, todavía enviamos un mensaje a la UI
            self.message_queue.put({'status': 'error', 'error': str(e), 'full_log': "\n".join(self.log_history)})
            return {'status': 'error', 'error': str(e)}

# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    print("Iniciando la ejecución del Metamodulo principal...")
    
    # Cola para comunicación entre hilos
    message_queue = Queue()
    
    # Crear una instancia del Metamodulo
    metamodulo_instance = Metamodulo(message_queue=message_queue)

    # Preparar algunas entradas de ejemplo
    example_inputs = [
        ("Este es un texto de prueba para el sistema DIG.", "text"),
        ("Otro texto con datos repetidos, bbb y ccc.", "text"),
        ("Un campo mas equilibrado.", "text"),
        ("Datos caoticos para el test de entropia.", "text"),
        ("Una entrada mas controlada, con orden.", "text"),
    ]

    # Crear e iniciar el visualizador en el hilo principal
    visualizer_app = Visualizador(message_queue)
    
    # Ejecutar el sistema DIG en un hilo separado
    dig_thread = threading.Thread(target=metamodulo_instance.run_dig_system, 
                                  args=(example_inputs, 5), 
                                  daemon=True)
    dig_thread.start()

    # Iniciar el bucle de la interfaz gráfica
    visualizer_app.mainloop()









