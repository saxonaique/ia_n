import tkinter as tk
import numpy as np
from tkinter import ttk
from PIL import Image, ImageTk

# Importar el Metamodulo y la función ia_interpreter
# Asegúrate de que todos los otros módulos (sensor_module, core_nucleus, etc.)
# estén en la misma carpeta o en el PYTHONPATH.
from metamodulo import Metamodulo
from ia_interpreter import interpretar_metrica

class VisualizadorDIG(tk.Tk):
    def __init__(self, metamodulo_instance: Metamodulo):
        super().__init__()
        self.title("DIG Visualizer")
        self.geometry("800x600") # Aumentado el tamaño para mejor visualización

        self.metamodulo = metamodulo_instance
        self.current_text_input = "Inicia el sistema DIG con este texto de ejemplo." # Texto por defecto

        self.label_imagen = tk.Label(self, bd=2, relief="groove") # Añadido borde para el campo
        self.label_imagen.pack(pady=10)

        # Marco para las métricas y la interpretación
        metrics_frame = ttk.Frame(self)
        metrics_frame.pack(pady=5)

        self.etiqueta_metrica = ttk.Label(metrics_frame, text="Métrica actual: N/A", font=("Arial", 12))
        self.etiqueta_metrica.pack(side=tk.LEFT, padx=10)

        self.etiqueta_interpretacion = ttk.Label(metrics_frame, text="IA: N/A", font=("Arial", 12, "italic"))
        self.etiqueta_interpretacion.pack(side=tk.RIGHT, padx=10)

        # Campo de entrada de texto
        self.text_input_label = ttk.Label(self, text="Entrada de texto para el Sensorium:", font=("Arial", 10))
        self.text_input_label.pack(pady=(10, 2))
        self.text_input_entry = tk.Text(self, height=3, width=70, bd=2, relief="sunken", font=("Arial", 10))
        self.text_input_entry.insert(tk.END, self.current_text_input)
        self.text_input_entry.pack(pady=(0, 10))
        self.text_input_entry.bind("<KeyRelease>", self._update_current_text_input)


        self.boton_start = ttk.Button(self, text="Iniciar/Actualizar Ciclo DIG", command=self.actualizar)
        self.boton_start.pack(pady=10)
        
        # Etiqueta para la decisión del Metamodulo
        self.etiqueta_decision_metamodulo = ttk.Label(self, text="Decisión Metamódulo: N/A", font=("Arial", 12, "bold"))
        self.etiqueta_decision_metamodulo.pack(pady=5)

        self.after_id = None
        # Iniciar el primer ciclo al arrancar
        self.actualizar()

    def _update_current_text_input(self, event=None):
        """Actualiza la variable con el contenido del widget Text."""
        self.current_text_input = self.text_input_entry.get("1.0", tk.END).strip()

    def normalizar_y_convertir(self, campo: np.ndarray) -> ImageTk.PhotoImage:
        """
        Normaliza un campo numpy a rango [0, 255] y lo convierte a PhotoImage para Tkinter.
        """
        if campo is None or campo.size == 0:
            # Crear una imagen en blanco o un mensaje si no hay campo
            img_data = np.zeros((64, 64), dtype=np.uint8)
            img = Image.fromarray(img_data)
        else:
            # Normalizar el campo para mapear valores a una escala de color
            # Aquí asumimos que el campo tiene valores en [-1, 1]
            campo_normalizado = ((campo + 1) / 2 * 255).astype(np.uint8) # Mapea [-1, 1] a [0, 255]

            # Si el campo no es 2D (debería serlo ahora), intentar remodelar
            if len(campo_normalizado.shape) == 1:
                size = int(np.sqrt(campo_normalizado.size))
                if size * size == campo_normalizado.size:
                    campo_normalizado = campo_normalizado.reshape((size, size))
                else:
                    # Rellenar y remodelar si no es un cuadrado perfecto
                    padded_length = size * size
                    padded_field = np.pad(campo_normalizado, (0, padded_length - campo_normalizado.size), 'constant', constant_values=0)
                    campo_normalizado = padded_field.reshape((size, size))
                    
            # Redimensionar la imagen para que sea visible en la GUI si no es 64x64
            display_size = (256, 256) # Tamaño de visualización en la GUI
            img_raw = Image.fromarray(campo_normalizado, mode='L') # 'L' para escala de grises
            img = img_raw.resize(display_size, Image.Resampling.NEAREST) # NEAREST para ver píxeles claros
        
        return ImageTk.PhotoImage(image=img)

    def actualizar(self):
        """
        Ejecuta un ciclo del sistema DIG a través del Metamodulo y actualiza la GUI.
        """
        # Obtener el texto actual del widget de entrada
        input_text_for_cycle = self.current_text_input

        # Ejecutar un ciclo del Metamodulo
        cycle_summary = self.metamodulo.process_cycle(raw_input=input_text_for_cycle, input_type="text")

        # --- DEBUG PRINTS START (Visualizador) ---
        print(f"[DEBUG Visualizador] Recibido cycle_summary completo: {cycle_summary}")
        # --- DEBUG PRINTS END (Visualizador) ---

        # Extraer datos del resumen del ciclo
        final_field = self.metamodulo.core_nucleus.field # El campo final después del procesamiento
        metamodule_decision = cycle_summary.get('metamodule_decision', 'N/A')
        metrics = cycle_summary.get('reorganized_field_metrics', {})
        
        # --- DEBUG PRINTS START (Visualizador) ---
        print(f"[DEBUG Visualizador] Extraído metrics diccionario: {metrics}")
        # --- DEBUG PRINTS END (Visualizador) ---

        # Actualizar la imagen del campo
        img = self.normalizar_y_convertir(final_field)
        self.label_imagen.configure(image=img)
        self.label_imagen.image = img # Mantener una referencia para evitar que sea recolectada por el GC

        # Actualizar métricas
        entropia = metrics.get('entropía', 'N/A')
        varianza = metrics.get('varianza', 'N/A')
        maximo = metrics.get('máximo', 'N/A')

        # --- DEBUG PRINTS START (Visualizador) ---
        print(f"[DEBUG Visualizador] entropia={entropia}, varianza={varianza}, maximo={maximo}")
        print(f"[DEBUG Visualizador] Tipo de entropia={type(entropia)}")
        # --- DEBUG PRINTS END (Visualizador) ---
        
        # Formatear el texto de las métricas de forma segura
        texto_m = (
            f"Entropía: {entropia:.3f} | Var: {varianza:.3f} | Max: {maximo:.3f}"
            if isinstance(entropia, (int, float)) and isinstance(varianza, (int, float)) and isinstance(maximo, (int, float))
            else f"Entropía: {entropia} | Var: {varianza} | Max: {maximo}"
        )
        self.etiqueta_metrica.config(text=texto_m)

        # Actualizar interpretación de la IA
        # También haremos que ia_interpreter.py sea más robusto por si acaso.
        interpretacion = interpretar_metrica(metrics) if metrics and isinstance(entropia, (int, float)) else "IA: Esperando datos..."
        self.etiqueta_interpretacion.config(text=f"IA: {interpretacion}")

        # Actualizar decisión del Metamodulo
        self.etiqueta_decision_metamodulo.config(text=f"Decisión Metamódulo: {metamodule_decision.upper()}")

        # CORRECCIÓN: Formateo condicional para el print statement
        log_entropia = f"{entropia:.3f}" if isinstance(entropia, (int, float)) else str(entropia)
        print(f"[Visualizador] Ciclo {cycle_summary.get('cycle')}: Decisión={metamodule_decision}, Entropía={log_entropia}, Interpretación='{interpretacion}'")






