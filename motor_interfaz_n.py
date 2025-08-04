import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import zoom

# Importar motores
try:
    from motor_n_dig import MotorN
except ImportError:
    # Si el módulo no está en el path, intentar importar desde el directorio actual
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from motor_n_dig import MotorN
    from motor_n_dig_extendido import MotorNDIGExtendido

# Set environment variable for matplotlib backend before any other imports
os.environ['MPLBACKEND'] = 'TkAgg'

# Now import matplotlib and force the backend
import matplotlib
matplotlib.use('TkAgg', force=True)

# Import the rest of the required modules
import numpy as np
from PIL import Image, ImageTk

# Import DIG simulator
from dig_simulator import simular_universo_dig, exportar_campo_tiempo

# Import matplotlib components after setting the backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from motor_n_dig import MotorN
from motor_n_dig_extendido import MotorNDIGExtendido

# Verify the backend is set correctly
print(f"Using matplotlib backend: {matplotlib.get_backend()}")

# Import scipy after matplotlib to avoid backend issues
from scipy.signal import convolve2d

class MotorNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motor N – Campo Informacional DIG")
        self.root.configure(bg="#111")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        self.root.resizable(True, True)

        self.canvas_size = 700
        
        # Inicializar atributos del motor
        self.motor_extendido = False
        self.motor = None
        
        # Frame principal vertical (canvas arriba, área de texto abajo)
        self.main_vertical = tk.Frame(root)
        self.main_vertical.pack(expand=True, fill=tk.BOTH)

        # Frame horizontal para canvas y controles
        self.main_frame = tk.Frame(self.main_vertical)
        self.main_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # Frame vertical para canvas y área de texto
        self.canvas_and_text = tk.Frame(self.main_frame)
        self.canvas_and_text.pack(side=tk.LEFT, padx=10, pady=10)

        # Canvas arriba
        self.canvas = tk.Canvas(self.canvas_and_text, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=0)
        self.canvas.pack()
        
        # Estilos para los botones
        style = ttk.Style()
        
        # Estilo para botón de motor extendido (azul)
        style.configure('Accent.TButton', background='#4a7abc', foreground='white')
        style.map('Accent.TButton', 
                 background=[('active', '#3a6a9c'), ('!disabled', '#4a7abc')],
                 foreground=[('!disabled', 'white')])
        
        # Estilo para botón de reinicio (naranja)
        style.configure('Warning.TButton', background='#e67e22', foreground='white')
        style.map('Warning.TButton',
                 background=[('active', '#d35400'), ('!disabled', '#e67e22')],
                 foreground=[('!disabled', 'white')])

        # Frame para gráficos
        self.graph_frame = ttk.Frame(self.canvas_and_text)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Crear figura de matplotlib
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # Gráfico de entropía
        self.ax2 = self.fig.add_subplot(212)  # Gráfico de varianza
        
        # Configurar gráficos
        self.ax1.set_title('Evolución de la Entropía')
        self.ax1.set_ylim(0, 0.5)
        self.ax1.grid(True)
        
        self.ax2.set_title('Evolución de la Varianza')
        self.ax2.set_ylim(0, 0.1)
        self.ax2.grid(True)
        
        # Inicializar líneas de los gráficos
        self.entropy_line, = self.ax1.plot([], [], 'b-')
        self.variance_line, = self.ax2.plot([], [], 'r-')
        
        # Integrar gráficos en Tkinter
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Datos para los gráficos
        self.time_steps = []
        self.entropy_values = []
        self.variance_values = []
        self.max_data_points = 100
        
        # Área de texto debajo de los gráficos
        self.text_area = tk.Text(self.canvas_and_text, height=4, width=int(self.canvas_size/8), state='disabled')
        self.text_area.pack(pady=(10, 0))

        # Frame de controles a la derecha con pestañas
        self.controls_right = ttk.Notebook(self.main_frame)
        self.controls_right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Pestaña de controles principales
        self.control_frame = ttk.Frame(self.controls_right)
        self.controls_right.add(self.control_frame, text='Controles')
        
        # Pestaña de parámetros avanzados
        self.param_frame = ttk.Frame(self.controls_right)
        self.controls_right.add(self.param_frame, text='Parámetros')
        self.crear_panel_parametros()

        # Botón para alternar entre motores (normal/extendido/sensorial)
        self.motor_mode = 0  # 0=normal, 1=extendido, 2=sensorial
        self.btn_toggle_motor = ttk.Button(
            self.control_frame, 
            text="Activar Motor Extendido", 
            command=self.toggle_motor,
            style='Accent.TButton'
        )
        self.btn_toggle_motor.pack(fill=tk.X, pady=2)
        
        # Frame para controles de texto (siempre visible)
        text_frame = ttk.LabelFrame(self.control_frame, text="Inyección de Texto")
        text_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Entrada de texto para inyección
        ttk.Label(text_frame, text="Texto a inyectar:").pack(fill=tk.X, pady=2)
        self.text_entry = ttk.Entry(text_frame)
        self.text_entry.pack(fill=tk.X, pady=2)
        self.text_entry.bind("<Return>", self.inyectar_texto)
        
        # Frame para controles sensoriales (inicialmente oculto)
        self.sensory_frame = ttk.LabelFrame(self.control_frame, text="Controles Sensoriales")
        
        # Botones para cargar/guardar diccionario (solo para modo sensorial)
        btn_frame = ttk.Frame(self.sensory_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Cargar JSON", command=self.cargar_json_sensorial).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(btn_frame, text="Guardar JSON", command=self.guardar_json_sensorial).pack(side=tk.LEFT, expand=True, padx=2)
        
        # Botón para reiniciar solo el campo (manteniendo la memoria)
        self.btn_reiniciar_campo = ttk.Button(
            self.control_frame,
            text="Reiniciar Campo (Mantener Memoria)",
            command=lambda: self.reiniciar_campo(mantener_memoria=True),
            style='Warning.TButton'
        )
        self.btn_reiniciar_campo.pack(fill=tk.X, pady=2)
        self.btn_reiniciar_campo.pack_forget()  # Ocultar inicialmente
        
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        ttk.Button(self.control_frame, text="Evolucionar", command=self.evolucionar).pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="Reiniciar", command=self.reiniciar).pack(fill=tk.X, pady=2)
        
        # Botón para ejecutar simulación DIG
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Label(self.control_frame, text="Simulador DIG:").pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="Ejecutar Simulación DIG", command=self.ejecutar_simulacion_dig).pack(fill=tk.X, pady=2)
        ttk.Label(self.control_frame, text="Entropía:").pack(fill=tk.X, pady=2)
        self.entropia_label = ttk.Label(self.control_frame, text="0.000")
        self.entropia_label.pack(fill=tk.X, pady=2)
        ttk.Label(self.control_frame, text="Velocidad (ms):").pack(fill=tk.X, pady=2)
        self.velocidad = tk.IntVar(value=100)
        self.slider = ttk.Scale(self.control_frame, from_=10, to=1000, variable=self.velocidad, orient=tk.HORIZONTAL)
        self.slider.pack(fill=tk.X, pady=2)
        self.animando = False
        self.boton_animar = ttk.Button(self.control_frame, text="Iniciar animación", command=self.toggle_animacion)
        self.boton_animar.pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="Guardar campo", command=self.guardar_campo).pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="Cargar campo", command=self.cargar_campo).pack(fill=tk.X, pady=2)


        self.canvas.bind("<Button-1>", self.inyectar)
        self.canvas.bind("<B1-Motion>", self.pintar_lapiz)

        # Inicializar con motor normal
        self.inicializar_motor(extendido=False)
        self.actualizar_vista()
        
    def crear_panel_parametros(self):
        """Crea el panel de control de parámetros"""
        # Frame para los controles deslizantes
        param_container = ttk.Frame(self.param_frame)
        param_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Variables para los parámetros
        self.param_vars = {}
        
        # Configuración de los parámetros (nombre, min, max, default, label)
        parametros = [
            ('gamma', 0.01, 1.0, 0.1, 'Difusión (γ)'),
            ('lambda_', 0.01, 0.5, 0.05, 'Entropía (λ)'),
            ('mu', 0.1, 1.0, 0.5, 'Memoria (μ)'),
            ('kappa', 0.1, 1.0, 0.3, 'Feedback (κ)'),
            ('dim', 20, 200, 100, 'Tamaño')
        ]
        
        # Crear controles para cada parámetro
        for name, min_val, max_val, default, label in parametros:
            frame = ttk.Frame(param_container)
            frame.pack(fill=tk.X, pady=2)
            
            # Etiqueta
            ttk.Label(frame, text=label, width=10).pack(side=tk.LEFT, padx=5)
            
            # Variable de control
            var = tk.DoubleVar(value=default)
            self.param_vars[name] = var
            
            # Control deslizante
            scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, 
                            orient=tk.HORIZONTAL, length=150)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Valor numérico
            val_lbl = ttk.Label(frame, text=f"{default:.2f}", width=6)
            val_lbl.pack(side=tk.LEFT, padx=5)
            
            # Actualizar etiqueta cuando cambia el valor
            var.trace_add('write', 
                lambda *args, v=var, l=val_lbl: l.config(text=f"{v.get():.2f}"))
        
        # Botón para aplicar cambios
        btn_frame = ttk.Frame(param_container)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Aplicar Parámetros", 
                  command=self.aplicar_parametros).pack(fill=tk.X)

    def aplicar_parametros(self):
        """Aplica los parámetros configurados al motor actual"""
        if not self.motor_extendido:
            messagebox.showinfo("Aviso", "Los parámetros solo se pueden ajustar en modo extendido")
            return
            
        try:
            params = {}
            for name, var in self.param_vars.items():
                value = var.get()
                # Asegurar que 'dim' sea un entero
                if name == 'dim':
                    value = int(round(value))
                params[name] = value
            
            # Actualizar parámetros del motor
            for name, value in params.items():
                if hasattr(self.motor, name):
                    setattr(self.motor, name, value)
            
            # Si cambió el tamaño, reiniciar el motor
            if 'dim' in params and params['dim'] != self.motor.dim:
                self.inicializar_motor(extendido=True)
                
            self.log_mensaje("Parámetros actualizados:")
            for name, value in params.items():
                self.log_mensaje(f"- {name}: {value:.2f}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar parámetros: {str(e)}")

    def toggle_motor(self):
        """Alterna entre los modos de motor: normal → extendido → sensorial"""
        self.motor_mode = (self.motor_mode + 1) % 3
        
        # Actualizar texto del botón
        if self.motor_mode == 0:
            self.btn_toggle_motor.config(text="Activar Motor Extendido")
            self.btn_reiniciar_campo.pack_forget()
            self.sensory_frame.pack_forget()
        elif self.motor_mode == 1:
            self.btn_toggle_motor.config(text="Activar Motor Sensorial")
            self.btn_reiniciar_campo.pack(fill=tk.X, pady=2)
            self.sensory_frame.pack_forget()
        else:  # Modo sensorial
            self.btn_toggle_motor.config(text="Volver a Motor Normal")
            self.btn_reiniciar_campo.pack(fill=tk.X, pady=2)
            self.sensory_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
            
        self.inicializar_motor(extendido=(self.motor_mode > 0), sensorial=(self.motor_mode == 2))

    
    def ejecutar_simulacion_dig(self):
        """Ejecuta la simulación DIG y la carga como estado inicial del motor"""
        try:
            self.log_mensaje("Iniciando simulación DIG...")
            
            # Usar la dimensión actual del motor para la simulación
            resolucion = self.motor.dim if hasattr(self.motor, 'dim') else 100
            
            # Ejecutar la simulación con parámetros por defecto
            campo_tiempo = simular_universo_dig(
                resolucion=resolucion,
                rho_c=1.0,
                Rs=min(20.0, resolucion/5),  # Ajustar Rs según la resolución
                k_c2=1.0,
                delta_q=0.1
            )
            
            # Normalizar el campo de tiempo al rango [0,1]
            campo_tiempo = (campo_tiempo - np.min(campo_tiempo)) / (np.max(campo_tiempo) - np.min(campo_tiempo) + 1e-8)
            
            # Cargar el resultado en el motor principal
            if hasattr(self.motor, 'rho'):
                # Si el motor ya está inicializado, actualizar su estado
                if campo_tiempo.shape == self.motor.rho.shape:
                    self.motor.rho = campo_tiempo
                    self.motor.tiempo = 0  # Reiniciar el contador de tiempo
                    self.actualizar_vista()
                    self.log_mensaje("Simulación DIG cargada. Usa 'Evolucionar' para ver la dinámica.")
                else:
                    # Si las dimensiones no coinciden, reiniciar el motor
                    self.log_mensaje("Reiniciando motor con nueva simulación...")
                    self.motor = type(self.motor)(dim=resolucion)
                    self.motor.rho = campo_tiempo
                    self.actualizar_vista()
                    self.log_mensaje("Nueva simulación DIG cargada. Usa 'Evolucionar'.")
            
        except Exception as e:
            self.log_mensaje(f"Error en la simulación DIG: {str(e)}")
    
    def mostrar_imagen_dig(self, ruta_imagen):
        """Muestra la imagen generada por la simulación DIG en el canvas"""
        try:
            # Cargar la imagen
            imagen = Image.open(ruta_imagen)
            
            # Redimensionar manteniendo la relación de aspecto
            ancho, alto = imagen.size
            nuevo_alto = int(self.canvas_size * (alto / ancho))
            imagen = imagen.resize((self.canvas_size, nuevo_alto), Image.Resampling.LANCZOS)
            
            # Mostrar en el canvas
            self.tk_image = ImageTk.PhotoImage(imagen)
            self.canvas.delete("all")  # Limpiar canvas
            self.canvas.config(width=self.canvas_size, height=nuevo_alto)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
        except Exception as e:
            self.log_mensaje(f"Error al mostrar la imagen: {str(e)}")
    
    def log_mensaje(self, msg):
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, msg + '\n')
        self.text_area.see(tk.END)
        self.text_area.config(state='disabled')
        
    def inicializar_motor(self, extendido=False, sensorial=False):
        """Inicializa el motor normal, extendido o sensorial según los parámetros"""
        if sensorial:
            self.motor = MotorNDIGExtendido(dim=100, gamma=0.1, lambda_=0.05, mu=0.1, kappa=0.08)
            # Cargar diccionario sensorial por defecto
            self.diccionario_sensorial = self.cargar_diccionario_sensorial()
            self.log_mensaje("Motor sensorial inicializado")
        elif extendido:
            self.motor = MotorNDIGExtendido(dim=100, gamma=0.1, lambda_=0.05, mu=0.1, kappa=0.08)
            self.log_mensaje("Motor extendido inicializado")
        else:
            self.motor = MotorN(dim=100)
            self.log_mensaje("Motor normal inicializado")
        self.reiniciar()
        
    def evolucionar(self):
        self.motor.evolucionar()
        self.actualizar_vista()
        self.log_mensaje(f"Evolución realizada. Entropía: {self.motor.obtener_entropia_global():.3f}")
        
    def reiniciar(self):
        """Reinicia el campo y la memoria"""
        if self.motor_extendido:
            # Para el motor extendido, reinicia el campo pero preserva la memoria
            self.motor.reiniciar(reiniciar_memoria=False)
            self.log_mensaje("Campo reiniciado (memoria preservada)")
        else:
            # Para el motor normal, simplemente reinicia
            self.motor.reiniciar()
            self.log_mensaje("Campo reiniciado")
        self.actualizar_vista()

    def reiniciar_campo(self, mantener_memoria=True):
        """Reinicia solo el campo visual, manteniendo la memoria"""
        if hasattr(self.motor, 'reiniciar') and self.motor_extendido:
            self.motor.reiniciar(reiniciar_memoria=not mantener_memoria)
            self.actualizar_vista()
            self.log_mensaje("Campo visual reiniciado" + (" (memoria preservada)" if mantener_memoria else ""))

    def inyectar(self, event):
        # Convertir coordenadas del canvas a la rejilla del motor
        x = max(0, min(int(event.x * self.motor.dim / self.canvas_size), self.motor.dim - 1))
        y = max(0, min(int(event.y * self.motor.dim / self.canvas_size), self.motor.dim - 1))
        self.motor.inyectar(x, y, intensidad=1.0)
        self.actualizar_vista()
        self.log_mensaje(f"Inyección en ({x}, {y})")

    def pintar_lapiz(self, event):
        # Convertir coordenadas del canvas a la rejilla del motor
        x = max(0, min(int(event.x * self.motor.dim / self.canvas_size), self.motor.dim - 1))
        y = max(0, min(int(event.y * self.motor.dim / self.canvas_size), self.motor.dim - 1))
        
        # Asegurarse de que las coordenadas estén dentro de los límites
        if 0 <= x < self.motor.dim and 0 <= y < self.motor.dim:
            self.motor.inyectar(x, y, intensidad=1.0)
            self.actualizar_vista()
            self.log_mensaje(f"Pintado en ({x}, {y})")
        else:
            self.log_mensaje(f"Fuera de límites: ({x}, {y})")
        
    def toggle_animacion(self):
        self.animando = not self.animando
        if self.animando:
            self.boton_animar.config(text="Pausar animación")
            self.animar()
        else:
            self.boton_animar.config(text="Iniciar animación")

    def actualizar_vista(self):
        # Limpiar el canvas
        self.canvas.delete("all")
        
        # Obtener el campo actual
        campo = self.motor.obtener_rho()
        
        # Si es el motor sensorial, mostrar los patrones de manera más visible
        if hasattr(self, 'motor_mode') and self.motor_mode == 2:  # Modo sensorial
            # Crear una imagen negra de fondo
            img = Image.new('RGB', (self.canvas_size, self.canvas_size), 'black')
            draw = ImageDraw.Draw(img)
            
            # Tamaño de celda para la visualización
            cell_size = min(self.canvas_size // self.motor.dim, 20)  # Máximo 20x20 píxeles por celda
            
            # Dibujar cada celda
            for y in range(self.motor.dim):
                for x in range(self.motor.dim):
                    # Obtener el valor normalizado (0-1)
                    valor = campo[y, x]
                    
                    # Mapear a color (verde más suave con intensidad reducida)
                    if valor > 0.01:  # Umbral mínimo para mostrar
                        # Reducir la intensidad general y el contraste
                        intensidad = int(180 * min(1.0, valor * 1.2))  # Menor contraste
                        # Usar un verde más oscuro y menos saturado
                        color = f'#{intensidad//2:02x}cc{intensidad//2:02x}'
                        
                        # Coordenadas del cuadrado
                        x1 = x * cell_size
                        y1 = y * cell_size
                        
                        # Hacer los círculos un poco más pequeños para evitar saturación visual
                        radio = max(1, int(cell_size * 0.4))  # Reducir tamaño
                        centro_x = x1 + cell_size // 2
                        centro_y = y1 + cell_size // 2
                        
                        # Dibujar un círculo con borde más sutil
                        borde = 'gray' if valor > 0.5 else 'darkgray'  # Borde más suave
                        draw.ellipse([centro_x - radio, centro_y - radio, 
                                     centro_x + radio, centro_y + radio], 
                                   fill=color, outline=borde, width=1)
            
            # Convertir a PhotoImage y mostrar en el canvas
            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        else:
            # Visualización normal para los otros modos
            escala = (campo * 255).astype(np.uint8)
            imagen = Image.fromarray(escala).resize((self.canvas_size, self.canvas_size), Image.NEAREST)
            self.tk_image = ImageTk.PhotoImage(imagen.convert("L"))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Actualizar etiqueta de entropía
        entropia = self.motor.obtener_entropia_global()
        self.entropia_label.config(text=f"{entropia:.4f}")
        
        # Actualizar gráficos
        self.actualizar_graficos()
        
    def actualizar_graficos(self):
        """Actualiza los gráficos con nuevos datos de entropía y varianza"""
        if not hasattr(self, 'time_steps'):
            return
            
        try:
            # Obtener valores actuales
            entropia = self.motor.obtener_entropia_global()
            
            # Calcular varianza manualmente para asegurar que sea diferente de la entropía
            rho = self.motor.obtener_rho()
            varianza = np.var(rho)  # Varianza de los valores del campo rho
            
            # Si es la primera vez, inicializar las listas
            if not hasattr(self, 'graph_initialized'):
                self.time_steps = []
                self.entropy_values = []
                self.variance_values = []
                self.graph_initialized = True
            
            # Agregar nuevos datos
            current_time = len(self.time_steps)
            self.time_steps.append(current_time)
            self.entropy_values.append(entropia)
            self.variance_values.append(varianza)
            
            # Limitar el número de puntos mostrados (ventana deslizante)
            if len(self.time_steps) > self.max_data_points:
                # Mover la ventana en lugar de reiniciar
                self.time_steps = self.time_steps[-self.max_data_points:]
                self.entropy_values = self.entropy_values[-self.max_data_points:]
                self.variance_values = self.variance_values[-self.max_data_points:]
                
                # Reasignar los índices de tiempo para que sean secuenciales
                self.time_steps = list(range(len(self.time_steps)))
            
            # Actualizar líneas de los gráficos
            self.entropy_line.set_data(self.time_steps, self.entropy_values)
            self.variance_line.set_data(self.time_steps, self.variance_values)
            
            # Ajustar límites de los ejes
            if self.time_steps:
                # Usar un rango fijo en el eje X basado en los últimos max_data_points
                x_min = max(0, len(self.time_steps) - self.max_data_points)
                x_max = max(self.max_data_points, len(self.time_steps))
                self.ax1.set_xlim(x_min, x_max)
                self.ax2.set_xlim(x_min, x_max)
                
                # Ajustar límites del eje Y dinámicamente
                if self.entropy_values:
                    y_min = min(self.entropy_values) * 0.95 if min(self.entropy_values) > 0 else 0
                    y_max = max(self.entropy_values) * 1.05 if max(self.entropy_values) > 0 else 0.1
                    self.ax1.set_ylim(max(0, y_min), y_max)
                
                if self.variance_values:
                    y_min = min(self.variance_values) * 0.95 if min(self.variance_values) > 0 else 0
                    y_max = max(self.variance_values) * 1.05 if max(self.variance_values) > 0 else 0.1
                    self.ax2.set_ylim(max(0, y_min), y_max)
            
            # Redibujar los gráficos
            self.canvas_graph.draw()
            
        except Exception as e:
            # Si hay algún error, imprimirlo pero no detener la ejecución
            print(f"Error en actualizar_graficos: {str(e)}")
    
    def animar(self):
        if self.animando:
            self.evolucionar()
            self.root.after(self.velocidad.get(), self.animar)

    def cargar_diccionario_sensorial(self, ruta=None):
        """Carga el diccionario sensorial desde un archivo JSON"""
        try:
            if ruta is None:
                # Cargar diccionario por defecto
                ruta = os.path.join(os.path.dirname(__file__), 'diccionarios', 'diccionario_sensorial.json')
                
            if not os.path.exists(ruta):
                # Crear directorio si no existe
                os.makedirs(os.path.dirname(ruta), exist_ok=True)
                # Crear un diccionario de ejemplo si no existe
                diccionario_ejemplo = {
                    "A": {"patron": [[1,0],[0,0],[0,0]], "intensidad": "baja"},
                    "B": {"patron": [[1,0],[1,0],[0,0]], "intensidad": "media"},
                    "C": {"patron": [[1,1],[0,0],[0,0]], "intensidad": "alta"}
                }
                with open(ruta, 'w') as f:
                    json.dump(diccionario_ejemplo, f, indent=2)
                return diccionario_ejemplo
                
            with open(ruta, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el diccionario sensorial: {str(e)}")
            return {}
    
    def cargar_json_sensorial(self):
        """Abre un diálogo para cargar un archivo JSON de diccionario sensorial"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo de diccionario sensorial",
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")],
            initialdir=os.path.join(os.path.dirname(__file__), 'diccionarios')
        )
        if ruta:
            self.diccionario_sensorial = self.cargar_diccionario_sensorial(ruta)
            messagebox.showinfo("Éxito", f"Diccionario cargado desde {os.path.basename(ruta)}")
    
    def guardar_json_sensorial(self):
        """Guarda el diccionario sensorial actual en un archivo JSON"""
        if not hasattr(self, 'diccionario_sensorial') or not self.diccionario_sensorial:
            messagebox.showwarning("Advertencia", "No hay datos del diccionario sensorial para guardar")
            return
            
        # Asegurarse de que el directorio existe
        os.makedirs(os.path.join(os.path.dirname(__file__), 'diccionarios'), exist_ok=True)
        
        ruta = filedialog.asksaveasfilename(
            title="Guardar diccionario sensorial como...",
            defaultextension=".json",
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")],
            initialdir=os.path.join(os.path.dirname(__file__), 'diccionarios'),
            initialfile="diccionario_sensorial.json"
        )
        
        if ruta:
            try:
                with open(ruta, 'w') as f:
                    json.dump(self.diccionario_sensorial, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Éxito", f"Diccionario guardado en {ruta}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el archivo: {str(e)}")
    
    def inyectar_texto(self, event=None):
        """Inyecta el texto del campo de entrada usando el método correspondiente al modo actual"""
        if self.motor_mode == 2:  # Modo sensorial
            self.inyectar_texto_sensorial()
        else:  # Modo normal o extendido
            self.inyectar_texto_normal()
    
    def inyectar_texto_normal(self, event=None):
        """Inyecta el texto en modo normal o extendido (caracter por caracter)"""
        texto = self.text_entry.get().upper()
        if not texto:
            return
            
        # Obtener el centro del campo
        centro_x = self.motor.dim // 2
        centro_y = self.motor.dim // 2
        
        # Inyectar cada carácter como un patrón simple
        for i, letra in enumerate(texto):
            # Crear un patrón simple basado en el código ASCII del carácter
            codigo = ord(letra)
            patron = []
            
            # Convertir el código a binario y crear un patrón 3x3
            bits = [int(b) for b in f"{codigo:08b}"]
            patron = [
                [bits[0], bits[1], bits[2]],
                [bits[3], 1, bits[4]],
                [bits[5], bits[6], bits[7]]
            ]
            
            # Calcular posición para centrar el patrón
            alto = len(patron)
            ancho = len(patron[0]) if alto > 0 else 0
            
            # Posición inicial (esquina superior izquierda del patrón)
            x = centro_x - ancho // 2 + (i * (ancho + 1))  # Espaciado horizontal
            y = centro_y - alto // 2
            
            # Asegurarse de que las coordenadas estén dentro de los límites
            if x < 0 or y < 0 or x + ancho > self.motor.dim or y + alto > self.motor.dim:
                self.log_mensaje(f"No hay espacio suficiente para inyectar el texto")
                break
            
            # Inyectar el patrón
            for dy, fila in enumerate(patron):
                for dx, valor in enumerate(fila):
                    if valor > 0:  # Solo inyectar celdas activas
                        self.motor.inyectar(x + dx, y + dy, 1.0)
            
            self.log_mensaje(f"Inyectado '{letra}' en ({x},{y})")
        
        self.actualizar_vista()
        self.text_entry.delete(0, tk.END)  # Limpiar el campo de texto
    
    def inyectar_texto_sensorial(self, event=None):
        """Inyecta el texto del campo de entrada usando el diccionario sensorial"""
        if not hasattr(self, 'diccionario_sensorial') or not self.diccionario_sensorial:
            messagebox.showerror("Error", "No se ha cargado ningún diccionario sensorial")
            return
            
        texto = self.text_entry.get().upper()
        if not texto:
            return
            
        # Obtener el centro del campo
        centro_x = self.motor.dim // 2
        centro_y = self.motor.dim // 2
        
        # Inyectar cada carácter
        for i, letra in enumerate(texto):
            if letra in self.diccionario_sensorial:
                try:
                    # Obtener patrón e intensidad del diccionario
                    info = self.diccionario_sensorial[letra]
                    patron = info.get('patron', [])
                    intensidad = info.get('intensidad', 'media')
                    
                    # Mapear intensidad a valor numérico
                    intensidad_valor = {
                        'baja': 0.3,
                        'media': 0.6,
                        'alta': 1.0
                    }.get(intensidad.lower(), 0.5)
                    
                    # Calcular posición para centrar el patrón
                    if not patron or not any(patron):
                        continue
                        
                    alto = len(patron)
                    ancho = len(patron[0]) if alto > 0 else 0
                    
                    # Posición inicial (esquina superior izquierda del patrón)
                    x = centro_x - ancho // 2
                    y = centro_y - alto // 2 + (i * (alto + 1))  # Espaciado vertical entre caracteres
                    
                    # Asegurarse de que las coordenadas estén dentro de los límites
                    if x < 0 or y < 0 or x + ancho > self.motor.dim or y + alto > self.motor.dim:
                        self.log_mensaje(f"El patrón para '{letra}' no cabe en el campo")
                        continue
                    
                    # Inyectar el patrón
                    for dy, fila in enumerate(patron):
                        for dx, valor in enumerate(fila):
                            if valor > 0:  # Solo inyectar celdas activas
                                self.motor.inyectar(x + dx, y + dy, intensidad_valor)
                    
                    self.log_mensaje(f"Inyectado '{letra}' en ({x},{y})")
                except Exception as e:
                    self.log_mensaje(f"Error al inyectar '{letra}': {str(e)}")
        
        self.actualizar_vista()
        self.text_entry.delete(0, tk.END)  # Limpiar el campo de texto
    
    def cargar_campo(self):
        """Carga un campo desde un archivo JSON o imagen"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo de campo",
            filetypes=[("Archivos JSON", "*.json"), ("Archivos de imagen", "*.png *.jpg *.jpeg"), ("Todos los archivos", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        
        if not ruta:
            return
            
        try:
            if ruta.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Cargar imagen
                self.mostrar_imagen_dig(ruta)
            else:
                # Cargar JSON
                with open(ruta, 'r') as f:
                    datos = json.load(f)
                
                if 'campo' in datos:
                    # Formato de archivo de campo
                    campo = np.array(datos['campo'])
                    if campo.shape[0] != self.motor.dim or campo.shape[1] != self.motor.dim:
                        if messagebox.askyesno("Redimensionar", "El tamaño del campo no coincide. ¿Redimensionar el motor?"):
                            self.motor = type(self.motor)(dim=len(campo))
                    self.motor.rho = campo
                    self.actualizar_vista()
                    self.log_mensaje(f"Campo cargado desde {os.path.basename(ruta)}")
                else:
                    messagebox.showerror("Error", "Formato de archivo no reconocido")
                    
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {str(e)}")
            
    def guardar_campo(self):
        """Guarda el campo actual en un archivo JSON"""
        ruta = filedialog.asksaveasfilename(
            title="Guardar campo como...",
            defaultextension=".json",
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__)),
            initialfile="campo_guardado.json"
        )
        
        if ruta:
            try:
                datos = {
                    'campo': self.motor.rho.tolist(),
                    'dimension': self.motor.dim,
                    'tipo_motor': 'sensorial' if hasattr(self, 'diccionario_sensorial') else 'extendido' if self.motor_mode == 1 else 'normal'
                }
                with open(ruta, 'w') as f:
                    json.dump(datos, f, indent=2)
                messagebox.showinfo("Éxito", f"Campo guardado en {ruta}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el archivo: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MotorNApp(root)
    root.mainloop()





