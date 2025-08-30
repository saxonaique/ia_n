import tkinter as tk
from tkinter import ttk

class TestApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Test Interfaz Simple")
        self.geometry("800x600")
        self.setup_ui()
    
    def setup_ui(self):
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Pestaña 1: Principal
        self.main_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.main_tab, text="🎯 Principal")
        self.setup_main_tab()
        
        # Pestaña 2: Análisis
        self.analysis_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.analysis_tab, text="📊 Análisis")
        self.setup_analysis_tab()
    
    def setup_main_tab(self):
        """Configura la pestaña principal."""
        # Frame principal
        main_frame = ttk.Frame(self.main_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar grid
        main_frame.grid_columnconfigure(0, weight=3)  # Canvas
        main_frame.grid_columnconfigure(1, weight=1)  # Controles
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Frame del canvas (izquierda)
        canvas_frame = ttk.LabelFrame(main_frame, text="🎯 Campo Informacional", padding="10")
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Canvas para visualización
        self.canvas_size = 400
        self.field_canvas = tk.Canvas(
            canvas_frame, 
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg='#3B4252',
            highlightthickness=2,
            highlightbackground='#434C5E'
        )
        self.field_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Frame de controles (derecha)
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky="nsew")
        
        # Panel de entrada
        input_frame = ttk.LabelFrame(controls_frame, text="📥 Entrada", padding="10")
        input_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(input_frame, text="Texto:").pack(anchor=tk.W, pady=(0, 5))
        self.text_input = tk.Text(input_frame, height=4, width=30)
        self.text_input.insert("1.0", "Texto de prueba")
        self.text_input.pack(fill="x")
        
        # Botón de procesar
        ttk.Button(input_frame, text="🔄 Procesar", command=self.process_text).pack(fill="x", pady=(10, 0))
        
        # Panel de simulación
        sim_frame = ttk.LabelFrame(controls_frame, text="⚡ Simulación", padding="10")
        sim_frame.pack(fill="x", pady=(0, 15))
        
        self.run_button = ttk.Button(sim_frame, text="▶ Iniciar", command=self.toggle_simulation)
        self.run_button.pack(fill="x", pady=(0, 5))
        
        self.step_button = ttk.Button(sim_frame, text="⏭ Paso", command=self.run_step)
        self.step_button.pack(fill="x")
        
        # Estado
        self.status_label = ttk.Label(sim_frame, text="⏸ En espera")
        self.status_label.pack(anchor=tk.W, pady=(10, 0))
    
    def setup_analysis_tab(self):
        """Configura la pestaña de análisis."""
        # Frame principal
        main_frame = ttk.Frame(self.analysis_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(main_frame, text="📊 Análisis y Gráficos", font=("Helvetica", 16, "bold")).pack(pady=20)
        
        # Información
        info_text = """
        Esta pestaña mostrará:
        • Gráficos en tiempo real
        • Métricas del sistema
        • Estadísticas avanzadas
        • Exportación de datos
        
        Funcionalidad en desarrollo...
        """
        
        ttk.Label(main_frame, text=info_text, font=("Helvetica", 12)).pack(pady=20)
    
    def process_text(self):
        """Procesa el texto de entrada."""
        text = self.text_input.get("1.0", tk.END).strip()
        self.status_label.config(text=f"✅ Procesado: {len(text)} caracteres")
    
    def toggle_simulation(self):
        """Alterna la simulación."""
        if self.run_button.cget("text") == "▶ Iniciar":
            self.run_button.config(text="⏹ Detener")
            self.status_label.config(text="▶ Ejecutando...")
        else:
            self.run_button.config(text="▶ Iniciar")
            self.status_label.config(text="⏸ Pausado")
    
    def run_step(self):
        """Ejecuta un paso de simulación."""
        self.status_label.config(text="⏭ Paso ejecutado")

if __name__ == "__main__":
    print("Iniciando aplicación de prueba...")
    app = TestApp()
    app.mainloop()

