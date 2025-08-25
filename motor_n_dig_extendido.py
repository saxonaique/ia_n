
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from scipy.signal import convolve2d
import json
import os

class MotorNDIGExtendido:
    def __init__(self, dim=50, gamma=0.1, lambda_=0.05, mu=0.5, kappa=0.3):
        """Inicializa el motor extendido con memoria persistente.
        
        Args:
            dim: Dimensión de la cuadrícula
            gamma: Coeficiente de difusión
            lambda_: Coeficiente de entropía
            mu: Tasa de actualización de la memoria (aumentado para mejor efecto)
            kappa: Fuerza de retroalimentación de la memoria (aumentado para mejor efecto)
        """
        self.dim = dim
        self.gamma = gamma
        self.lambda_ = lambda_
        self.mu = mu
        self.kappa = kappa
        
        # Campo principal
        self.rho = np.random.rand(dim, dim) * 0.1
        
        # Memoria persistente
        self.rho_mem = np.zeros((dim, dim))
        
        self.tiempo = 0

    def laplaciano(self, campo):
        return (
            -4 * campo
            + np.roll(campo, 1, axis=0)
            + np.roll(campo, -1, axis=0)
            + np.roll(campo, 1, axis=1)
            + np.roll(campo, -1, axis=1)
        )

    def entropia_local(self, campo):
        kernel = np.ones((3, 3)) / 9
        mean = convolve2d(campo, kernel, mode="same", boundary="wrap")
        varianza = convolve2d((campo - mean) ** 2, kernel, mode="same", boundary="wrap")
        return np.sqrt(varianza)

    def evolucionar(self):
        # Términos de evolución extendidos con memoria
        difusion = self.gamma * self.laplaciano(self.rho)
        entropia = self.lambda_ * self.entropia_local(self.rho)
        
        # Retroalimentación de la memoria (DIG effect)
        feedback_memoria = self.kappa * (self.rho_mem - self.rho)
        
        # Actualización del campo principal
        self.rho += difusion - entropia + feedback_memoria
        self.rho = np.clip(self.rho, 0, 1)
        
        # Actualización de la memoria (persistente)
        self.rho_mem += self.mu * (self.rho - self.rho_mem)
        
        self.tiempo += 1

    def reiniciar(self, reiniciar_memoria=False):
        """Reinicia el campo a un estado aleatorio inicial
        
        Args:
            reiniciar_memoria: Si es True, también reinicia la memoria persistente
        """
        self.rho = np.random.rand(self.dim, self.dim) * 0.1
        if reiniciar_memoria:
            self.rho_mem = np.zeros((self.dim, self.dim))
        self.tiempo = 0

    def inyectar(self, x, y, intensidad=1.0):
        """Inyecta información en una posición del campo y su memoria
        
        Args:
            x, y: Coordenadas donde inyectar
            intensidad: Cantidad a inyectar (ahora afecta tanto al campo como a la memoria)
        """
        try:
            # Asegurarse de que x e y sean enteros y estén dentro de los límites
            x_int = int(round(x))
            y_int = int(round(y))
            
            if 0 <= x_int < self.dim and 0 <= y_int < self.dim:
                # Inyectamos en el campo actual
                self.rho[y_int, x_int] = min(1.0, self.rho[y_int, x_int] + intensidad)
                # También actualizamos la memoria para reforzar el patrón
                self.rho_mem[y_int, x_int] = min(1.0, self.rho_mem[y_int, x_int] + intensidad * 0.5)
                return True
            return False
        except (ValueError, TypeError, IndexError) as e:
            # Manejar cualquier error de tipo o índice
            print(f"Error al inyectar en ({x}, {y}): {e}")
            return False

    def obtener_rho(self):
        return self.rho

    def obtener_entropia_global(self):
        return np.mean(self.entropia_local(self.rho))
        
    def inyectar_forma(self, coords, intensidad='media'):
        """Inyecta una forma en el campo del motor.
        
        Args:
            coords: Lista de coordenadas [x, y] o lista de listas [[x1,y1], [x2,y2], ...]
            intensidad: Intensidad de la inyección ('baja', 'media', 'alta')
        """
        escala = {"baja": 0.5, "media": 1.0, "alta": 1.5, "moderada": 0.8}.get(intensidad, 1.0)
        
        # Asegurarse de que coords sea una lista de listas
        if not coords or not isinstance(coords[0], (list, tuple)):
            return
            
        for punto in coords:
            if len(punto) >= 2:  # Asegurar que tenemos al menos x e y
                x, y = int(punto[0]), int(punto[1])
                # Asegurarse de que las coordenadas estén dentro de los límites
                if 0 <= x < self.dim and 0 <= y < self.dim:
                    # Usar el método inyectar existente para inyectar en las coordenadas
                    self.inyectar(x, y, escala)
                    
                    # También inyectar en las celdas adyacentes para mejor visibilidad
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.dim and 0 <= ny < self.dim:
                                self.inyectar(nx, ny, escala * 0.3)

class MotorNApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Motor N - DIG Extendido")
        
        # Crear carpeta de diccionarios si no existe
        self.diccionarios_dir = "diccionarios"
        os.makedirs(self.diccionarios_dir, exist_ok=True)

        # Frame principal
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame para controles superiores
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Frame para controles de archivo
        self.file_frame = tk.LabelFrame(self.control_frame, text="Archivos", padx=5, pady=5)
        self.file_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Botones de archivo
        self.btn_cargar = tk.Button(
            self.file_frame,
            text="Cargar JSON",
            command=self.cargar_json,
            bg="#2196F3",  # Azul
            fg="white",
            padx=10,
            pady=5
        )
        self.btn_cargar.pack(side=tk.LEFT, padx=2)

        self.btn_guardar = tk.Button(
            self.file_frame,
            text="Guardar JSON",
            command=self.guardar_json,
            bg="#FF9800",  # Naranja
            fg="white",
            padx=10,
            pady=5
        )
        self.btn_guardar.pack(side=tk.LEFT, padx=2)

        # Frame para controles de motor
        self.motor_frame = tk.LabelFrame(self.control_frame, text="Motor", padx=5, pady=5)
        self.motor_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Botones de control del motor
        self.btn_sensorial = tk.Button(
            self.motor_frame, 
            text="Motor Sensorial",
            command=self.activar_modo_sensorial,
            bg="#4CAF50",  # Verde
            fg="white",
            padx=10,
            pady=5
        )
        self.btn_sensorial.pack(side=tk.LEFT, padx=2)

        self.btn_reiniciar = tk.Button(
            self.motor_frame,
            text="Reiniciar",
            command=self.reiniciar_motor,
            bg="#f44336",  # Rojo
            fg="white",
            padx=10,
            pady=5
        )
        self.btn_reiniciar.pack(side=tk.LEFT, padx=2)

        # Frame para entrada de texto
        self.input_frame = tk.Frame(self.main_frame)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_entrada = tk.Label(self.input_frame, text="Inyectar texto:")
        self.lbl_entrada.pack(side=tk.LEFT, padx=5)

        self.entrada_texto = tk.Entry(self.input_frame)
        self.entrada_texto.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.entrada_texto.bind("<Return>", self.inyectar_texto)

        self.btn_inyectar = tk.Button(
            self.input_frame,
            text="Inyectar",
            command=lambda: self.inyectar_texto(None),
            bg="#9C27B0",  # Púrpura
            fg="white"
        )
        self.btn_inyectar.pack(side=tk.LEFT, padx=5)

        # Inicializar el motor con parámetros por defecto
        self.motor = MotorNDIGExtendido(dim=100, gamma=0.1, lambda_=0.05, mu=0.1, kappa=0.08)
        self.pixel_size = 8  # Aumentado para mejor visualización
        
        # Calcular dimensiones del canvas basadas en el tamaño del campo y el tamaño de píxel
        canvas_width = self.motor.dim * self.pixel_size
        canvas_height = self.motor.dim * self.pixel_size
        
        # Canvas para visualización con tamaño fijo basado en las dimensiones del campo
        self.canvas = tk.Canvas(
            self.main_frame, 
            width=canvas_width, 
            height=canvas_height,
            bg="black",
            highlightthickness=0  # Elimina el borde del canvas
        )
        self.canvas.pack(padx=5, pady=5)
        self.diccionario_letras = self.cargar_diccionario_por_defecto()

        self.canvas.bind("<Button-1>", self.inyectar_click)
        self.actualizar()
        
    def activar_modo_sensorial(self):
        """Activa el modo sensorial con parámetros específicos"""
        try:
            # Cargar el diccionario sensorial
            with open("diccionario_sensorial.json", "r", encoding="utf-8") as f:
                datos = json.load(f)
            
            # Actualizar parámetros del motor
            self.motor.gamma = 0.15  # Mayor difusión para patrones más suaves
            self.motor.lambda_ = 0.03  # Menor entropía para mantener patrones
            self.motor.mu = 0.7  # Mayor persistencia de memoria
            self.motor.kappa = 0.4  # Mayor retroalimentación
            
            # Reiniciar el motor para aplicar cambios
            self.motor.reiniciar(reiniciar_memoria=True)
            
            print("Modo sensorial activado con parámetros:")
            print(f"Gamma (difusión): {self.motor.gamma}")
            print(f"Lambda (entropía): {self.motor.lambda_}")
            print(f"Mu (memoria): {self.motor.mu}")
            print(f"Kappa (retroalimentación): {self.motor.kappa}")
            
        except Exception as e:
            print(f"Error al cargar parámetros sensoriales: {e}")
    
    def cargar_diccionario_por_defecto(self):
        """Carga el diccionario por defecto desde el archivo"""
        try:
            with open("diccionario_sensorial.json", "r", encoding="utf-8") as f:
                return json.load(f).get("alfabeto_sensorial", [])
        except Exception as e:
            print(f"Error al cargar diccionario por defecto: {e}")
            return []

    def cargar_json(self):
        """Abre un diálogo para cargar un archivo JSON"""
        try:
            filepath = filedialog.askopenfilename(
                initialdir=self.diccionarios_dir,
                title="Seleccionar archivo JSON",
                filetypes=[("Archivos JSON", "*.json")]
            )
            if filepath:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Aquí puedes procesar los datos cargados según sea necesario
                    messagebox.showinfo("Éxito", f"Archivo cargado: {os.path.basename(filepath)}")
                    return data
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el archivo: {e}")
        return None

    def guardar_json(self):
        """Guarda el estado actual en un archivo JSON"""
        try:
            filepath = filedialog.asksaveasfilename(
                initialdir=self.diccionarios_dir,
                defaultextension=".json",
                filetypes=[("Archivos JSON", "*.json")],
                initialfile=f"estado_{len(os.listdir(self.diccionarios_dir)) + 1}.json"
            )
            if filepath:
                estado = {
                    'dim': self.motor.dim,
                    'gamma': self.motor.gamma,
                    'lambda': self.motor.lambda_,
                    'mu': self.motor.mu,
                    'kappa': self.motor.kappa,
                    'estado_motor': self.motor.rho.tolist()
                }
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(estado, f, indent=2)
                messagebox.showinfo("Éxito", f"Estado guardado en: {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar el archivo: {e}")

    def inyectar_texto(self, event):
        """Inyecta el texto ingresado en el motor"""
        texto = self.entrada_texto.get().upper()
        if not texto:
            return
            
        # Limpiar el campo de entrada
        self.entrada_texto.delete(0, tk.END)
        
        # Inyectar cada carácter
        for letra in texto:
            # Buscar la letra en el diccionario
            for item in self.diccionario_letras:
                if item.get('letra') == letra:
                    forma = item.get('forma_braille', [])
                    intensidad = item.get('accion', 'media')
                    # Aplicar un pequeño desplazamiento para cada letra
                    desplazamiento = len(forma) * 2 if forma else 5
                    forma_desplazada = [[x + desplazamiento, y + 5] for x, y in forma]
                    self.motor.inyectar_forma(forma_desplazada, intensidad)
                    break

    def reiniciar_motor(self):
        """Reinicia el motor a los valores por defecto"""
        self.motor.reiniciar(reiniciar_memoria=True)
        messagebox.showinfo("Reinicio", "Motor reiniciado a valores por defecto")

    def inyectar_click(self, event):
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size
        self.motor.inyectar(x, y, intensidad=1.0)  # Changed 'cantidad' to 'intensidad'

    def actualizar(self):
        self.motor.evolucionar()
        self.dibujar_campo()
        self.master.after(50, self.actualizar)

    def dibujar_campo(self):
        self.canvas.delete("all")
        for i in range(self.motor.dim):
            for j in range(self.motor.dim):
                valor = self.motor.rho[i, j]
                color = self.valor_a_color(valor)
                x1 = j * self.pixel_size
                y1 = i * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def valor_a_color(self, valor):
        tono = int(valor * 255)
        return f"#{tono:02x}{tono:02x}{tono:02x}"

if __name__ == "__main__":
    root = tk.Tk()
    app = MotorNApp(root)
    root.mainloop()
