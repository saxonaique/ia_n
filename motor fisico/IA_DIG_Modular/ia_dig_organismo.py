import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import scipy.ndimage as ndi
from typing import Optional, Tuple, Dict, Any, Union
import os
from PIL import Image, ImageTk
import matplotlib.cm as cm
import matplotlib.colors as colors

# Dependencia para audio .wav y espectrograma
from scipy.io import wavfile
from scipy.signal import spectrogram

# --- Módulo 1: SensorModule (Sensorium Informacional Universal) ---
class SensorModule:
    """
    Sensorium Informacional: Módulo de percepción de la IA DIG.
    Transforma datos brutos (texto, imágenes, audio, CSV) en un campo [0, 1].
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {'target_field_size': (64, 64)}
        self.field_shape = self.config['target_field_size']

    def process(self, source: str, source_type: str) -> np.ndarray:
        """Punto de entrada principal para procesar cualquier tipo de dato."""
        print(f"[SensorModule] Recibido para procesar: tipo='{source_type}'")
        if source_type == 'text':
            return self._process_text_structured(source)
        elif source_type == 'file':
            file_ext = os.path.splitext(source)[1].lower()
            if file_ext in ['.png', '.jpg', '.jpeg']:
                return self._process_image(source)
            elif file_ext == '.wav':
                return self._process_audio_wav(source)
            elif file_ext == '.csv':
                return self._process_brainwaves_csv(source)
            else:
                print(f"[SensorModule] WARN: Extensión de archivo '{file_ext}' no soportada.")
                return np.zeros(self.field_shape)
        else:
            return np.zeros(self.field_shape)

    def _create_gaussian_blob(self, center_x, center_y, intensity, size):
        y, x = np.mgrid[0:self.field_shape[0], 0:self.field_shape[1]]
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        sigma_sq = (size / 2)**2
        return intensity * np.exp(-dist_sq / (2 * sigma_sq))

    def _process_text_structured(self, text_input: str) -> np.ndarray:
        field = np.zeros(self.field_shape, dtype=np.float32)
        words = text_input.split()
        if not words: return field
        rows, cols = self.field_shape
        np.random.seed(sum(ord(c) for c in text_input))
        for i, word in enumerate(words):
            word_hash = sum(ord(c) for c in word)
            center_x = (word_hash % cols + i * 13) % cols
            center_y = (len(word) * 17 + word_hash % rows) % rows
            intensity = min(1.0, 0.5 + len(word) / 10.0)
            size = max(5, min(rows // 4, len(word) * 2))
            field += self._create_gaussian_blob(center_x, center_y, intensity, size)
        if field.max() > 0: field = (field - field.min()) / (field.max() - field.min())
        return field.astype(np.float32)

    def _process_image(self, file_path: str) -> np.ndarray:
        try:
            img = Image.open(file_path).convert('L') # Convertir a escala de grises
            img_resized = img.resize(self.field_shape, Image.Resampling.LANCZOS)
            field = np.array(img_resized) / 255.0
            return field.astype(np.float32)
        except Exception as e:
            print(f"[SensorModule] ERROR procesando imagen: {e}")
            return np.zeros(self.field_shape)

    def _process_audio_wav(self, file_path: str) -> np.ndarray:
        try:
            sample_rate, samples = wavfile.read(file_path)
            if samples.ndim > 1: samples = samples[:, 0]
            _, _, Sxx = spectrogram(samples, sample_rate)
            log_spectrogram = np.log1p(Sxx)
            img = Image.fromarray(log_spectrogram)
            img_resized = img.resize(self.field_shape, Image.Resampling.LANCZOS)
            field = np.array(img_resized)
            if field.max() > field.min(): field = (field - field.min()) / (field.max() - field.min())
            return field.astype(np.float32)
        except Exception as e:
            print(f"[SensorModule] ERROR procesando audio: {e}")
            return np.zeros(self.field_shape)

    def _process_brainwaves_csv(self, file_path: str) -> np.ndarray:
        try:
            # Asume que el CSV es una matriz de números (tiempo en filas, canales en columnas)
            data = np.loadtxt(file_path, delimiter=',')
            img = Image.fromarray(data)
            img_resized = img.resize(self.field_shape, Image.Resampling.LANCZOS)
            field = np.array(img_resized)
            if field.max() > field.min(): field = (field - field.min()) / (field.max() - field.min())
            return field.astype(np.float32)
        except Exception as e:
            print(f"[SensorModule] ERROR procesando CSV: {e}")
            return np.zeros(self.field_shape)

# ... (El resto de los módulos CoreNucleus, MemoryModule, IA_Interpreter y Metamodulo no necesitan cambios) ...
class CoreNucleus:
    def __init__(self, field_shape: Tuple[int, int] = (64, 64)): self.field_shape, self.field = field_shape, np.zeros(field_shape, dtype=np.float32); print("[CoreNucleus] INFO: Inicializado.")
    def receive_field(self, field: np.ndarray):
        if not isinstance(field, np.ndarray): self.field = np.zeros(self.field_shape, dtype=np.float32); return
        if field.shape != self.field_shape: field = ndi.zoom(field, [t / s for t, s in zip(self.field_shape, field.shape)])
        self.field = np.clip(field, 0, 1).astype(np.float32)
    def get_metrics(self) -> Dict[str, Any]:
        if self.field.size==0: return {}
        hist, _=np.histogram(self.field.flatten(), bins=10, range=(0,1)); probs=hist/self.field.size; probs=probs[probs > 0]; entropy=-np.sum(probs*np.log2(probs+1e-9))
        return {"entropía": float(entropy), "varianza": float(np.var(self.field)), "máximo": float(np.max(self.field)), "simetría": float(np.mean(np.abs(self.field-np.fliplr(self.field)))), "active_cells": int(np.sum(self.field>0.66)), "inhibited_cells": int(np.sum(self.field<0.33)), "neutral_cells": int(self.field.size-np.sum(self.field>0.66)-np.sum(self.field<0.33))}
    def reorganize_field(self): self.field = np.clip(ndi.convolve(self.field, np.array([[0.05,0.1,0.05],[0.1,0.4,0.1],[0.05,0.1,0.05]]), mode='reflect'), 0, 1)
    def apply_attractor(self, attractor_field: np.ndarray, strength: float=0.25):
        if self.field.shape!=attractor_field.shape: attractor_field=ndi.zoom(attractor_field, [t/s for t,s in zip(self.field_shape, attractor_field.shape)])
        self.field = np.clip(self.field*(1-strength) + attractor_field*strength, 0, 1)

class MemoryModule:
    def __init__(self): self.attractors: Dict[str, Dict[str, Any]]={}; self._initialize_sample_attractors()
    def _initialize_sample_attractors(self):
        print("[MemoryModule] INFO: Creando atractores."); x,y=np.meshgrid(np.linspace(0,1,64),np.linspace(0,1,64)); self.add_attractor("orden_gradiente",np.sin(x*np.pi)*np.cos(y*np.pi)*0.5+0.5)
        self.add_attractor("complejidad_ajedrez", np.kron([[1,0]*32,[0,1]*32]*32,np.ones((1,1)))[:64,:64])
    def add_attractor(self, name: str, pattern: np.ndarray): self.attractors[name]={'pattern':pattern.astype(np.float32), 'usage_count':0}
    def find_closest_attractor(self, field: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self.attractors: return None
        min_dist,best_match_name=float('inf'),None
        for name,data in self.attractors.items():
            dist=np.mean((field-data['pattern'])**2)
            if dist<min_dist: min_dist,best_match_name=dist,name
        if best_match_name: self.attractors[best_match_name]['usage_count']+=1; print(f"[MemoryModule] INFO: Atractor más cercano: '{best_match_name}'"); return self.attractors[best_match_name]
        return None

def interpretar_metrica(m: Dict[str, Any]) -> str:
    if not m: return "IA: Esperando métricas..."
    e,v,sym=m.get("entropía",0.0),m.get("varianza",0.0),m.get("simetría",0.0); total=m.get("active_cells",0)+m.get("inhibited_cells",0)+m.get("neutral_cells",0); active_ratio=m.get("active_cells",0)/total if total>0 else 0
    estado,reco="EQUILIBRIO DINÁMICO","Continuar monitorización."
    if e<0.5 and v<0.01: estado,reco="CAMPO ESTANCADO","Inyectar nueva entrada."
    elif e>2.0 and v>0.1: estado,reco="SOBRECARGA INFORMACIONAL","Aplicar atractor de orden."
    elif active_ratio>0.5: estado,reco="ALTA ACTIVIDAD","Aplicar atractor inhibidor."
    return "\n".join([f"✅ ESTADO: {estado}", f"  - Entropía: {e:.4f} | Varianza: {v:.4f}", f"  - Simetría: {sym:.4f} | Células Activas: {active_ratio:.1%}", f"💡 RECOMENDACIÓN: {reco}"])

class Metamodulo:
    def __init__(self): print("[Metamodulo] INFO: Inicializando..."); self.sensor_module,self.core_nucleus,self.memory_module,self.current_cycle=SensorModule(),CoreNucleus(),MemoryModule(),0
    def receive_input(self, source: str, source_type: str): field=self.sensor_module.process(source,source_type); self.core_nucleus.receive_field(field); print("[Metamodulo] INFO: Entrada procesada.")
    def process_step(self) -> Dict[str, Any]:
        self.current_cycle+=1; print(f"\n--- [Metamodulo] CICLO {self.current_cycle} ---"); metrics_before=self.core_nucleus.get_metrics(); decision=self.make_decision(metrics_before); print(f"[Metamodulo] Decisión: {decision.upper()}")
        if decision=='exploit':
            attractor_data=self.memory_module.find_closest_attractor(self.core_nucleus.field)
            if attractor_data: self.core_nucleus.apply_attractor(attractor_data['pattern'],strength=0.3); print("[Metamodulo] Acción: Atractor aplicado.")
        self.core_nucleus.reorganize_field(); print("[Metamodulo] Acción: Campo reorganizado."); metrics_after=self.core_nucleus.get_metrics()
        return {'cycle':self.current_cycle, 'field':self.core_nucleus.field, 'metrics':metrics_after, 'decision':decision, 'interpretation':interpretar_metrica(metrics_after)}
    def make_decision(self, metrics: Dict[str, Any]) -> str:
        entropy,variance=metrics.get("entropía",0.0),metrics.get("varianza",0.0)
        if entropy>2.0 or variance>0.1: return "exploit"
        elif entropy<0.5 and variance<0.01: return "explore"
        else: return "stabilize"

# --- Módulo 6: DIGVisualizerApp (GUI con Carga de Archivos) ---
class DIGVisualizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visualizador del Organismo Informacional DIG")
        self.geometry("1100x900") # Un poco más alto
        self.metamodulo = Metamodulo()
        self.is_running = False
        self.setup_ui()
        self.process_text() # Cargar texto inicial

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
        
        # --- Controles de Entrada ---
        ttk.Label(controls_frame, text="Entrada de Texto:", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))
        self.text_input = tk.Text(controls_frame, height=5, width=45, font=("Courier", 10))
        self.text_input.insert("1.0", "La gravedad emerge de un balance dinámico entre desorden (S) y orden (I).")
        self.text_input.pack(pady=5)
        self.process_text_button = ttk.Button(controls_frame, text="Procesar Texto", command=self.process_text)
        self.process_text_button.pack(pady=5, fill='x')
        
        self.process_file_button = ttk.Button(controls_frame, text="Cargar y Procesar Archivo", command=self.load_file)
        self.process_file_button.pack(pady=5, fill='x')
        
        self.loaded_file_label = ttk.Label(controls_frame, text="Archivo: Ninguno", font=("Helvetica", 9, "italic"), wraplength=350)
        self.loaded_file_label.pack(pady=(0, 10))

        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # --- Controles de Simulación ---
        self.run_button = ttk.Button(controls_frame, text="Ejecutar Simulación", command=self.toggle_simulation)
        self.run_button.pack(pady=5, fill='x')
        self.status_label = ttk.Label(controls_frame, text="Estado: En espera", font=("Helvetica", 10, "italic"))
        self.status_label.pack(pady=5)
        
        ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # --- Panel de Interpretación ---
        interpretation_frame = ttk.Frame(controls_frame)
        interpretation_frame.pack(pady=5, fill="x")
        ttk.Label(interpretation_frame, text="Interpretación de la IA:", font=("Helvetica", 12, "bold")).pack(pady=(0, 5))
        self.interpretation_text = tk.Text(interpretation_frame, height=12, width=45, state=tk.DISABLED, font=("Courier", 10), wrap=tk.WORD)
        self.interpretation_text.pack()

    def update_field_canvas(self, field: np.ndarray):
        if field.size == 0: return
        norm = colors.Normalize(vmin=np.min(field), vmax=np.max(field))
        colormap = cm.get_cmap('plasma')
        rgba_field = colormap(norm(field))
        img_data = (rgba_field[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(img_data, 'RGB')
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

    def stop_simulation(self):
        self.is_running = False
        self.run_button.config(text="Ejecutar Simulación")

    def process_text(self):
        self.stop_simulation()
        input_text = self.text_input.get("1.0", tk.END).strip()
        if not input_text: self.status_label.config(text="Error: Entrada vacía."); return
        self.metamodulo.receive_input(input_text, 'text')
        self.update_field_canvas(self.metamodulo.core_nucleus.field)
        self.status_label.config(text="Estado: Texto procesado. Listo para simular.")
        self.loaded_file_label.config(text="Archivo: Ninguno (usando texto)")

    def load_file(self):
        self.stop_simulation()
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[("Todos los soportados", ".png .jpg .jpeg .wav .csv"),
                       ("Imágenes", ".png .jpg .jpeg"),
                       ("Audio WAV", ".wav"),
                       ("Datos CSV", ".csv")]
        )
        if not file_path: return
        self.metamodulo.receive_input(file_path, 'file')
        self.update_field_canvas(self.metamodulo.core_nucleus.field)
        self.status_label.config(text="Estado: Archivo procesado. Listo para simular.")
        self.loaded_file_label.config(text=f"Archivo: {os.path.basename(file_path)}")

    def run_single_step(self):
        try:
            summary = self.metamodulo.process_step()
            self.update_display(summary)
        except Exception as e:
            self.status_label.config(text=f"Error en ciclo: {e}"); self.stop_simulation()

    def toggle_simulation(self):
        self.is_running = not self.is_running
        self.run_button.config(text="Detener Simulación" if self.is_running else "Ejecutar Simulación")
        if self.is_running: self.run_simulation_loop()

    def run_simulation_loop(self):
        if self.is_running: self.run_single_step(); self.after(200, self.run_simulation_loop)

if __name__ == "__main__":
    print("Iniciando la aplicación de visualización del Sistema DIG...")
    app = DIGVisualizerApp()
    app.mainloop()