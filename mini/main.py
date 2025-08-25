import tkinter as tk
import json
import os
from motor_core import MotorN

class Aplicacion:
    def __init__(self, root):
        # Guardar referencia a la ventana principal
        self.root = root
        
        # Configuración de la ventana principal
        self.root.title("Visualizador Braille")
        self.root.geometry("1200x800")  # Ventana más ancha
        self.root.configure(bg='#f0f0f0')
        
        # Frame principal con padding
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame para el canvas (arriba) y controles (abajo)
        content_frame = tk.Frame(main_frame, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame para el canvas
        canvas_frame = tk.Frame(content_frame, bg='#f0f0f0')
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas para dibujar los puntos Braille
        self.canvas = tk.Canvas(canvas_frame, width=800, height=500, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame para los controles (abajo)
        control_frame = tk.Frame(content_frame, bg='#f0f0f0', height=200)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        control_frame.pack_propagate(False)  # Evita que el frame se encoja
        
        # Título de controles
        tk.Label(control_frame, text="Controles", font=('Arial', 12, 'bold'), 
                bg='#e0e0e0').pack(pady=(10, 5), padx=5)
                
        # Frame para entrada de texto
        input_frame = tk.Frame(control_frame, bg='#e0e0e0')
        input_frame.pack(pady=5, padx=5, fill=tk.X)
        
        tk.Label(input_frame, text="Letra:", bg='#e0e0e0').pack(side=tk.LEFT)
        self.entrada_letra = tk.Entry(input_frame, width=5)
        self.entrada_letra.pack(side=tk.LEFT, padx=5)
        self.entrada_letra.bind('<Return>', lambda e: self.inyectar())
        
        btn_frame = tk.Frame(control_frame, bg='#e0e0e0')
        btn_frame.pack(pady=5, fill=tk.X)
        
        tk.Button(btn_frame, text="Inyectar", command=self.inyectar).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Limpiar", command=self.limpiar).pack(side=tk.LEFT, padx=2)
        
        # Frame para métricas (horizontal)
        metrics_frame = tk.LabelFrame(control_frame, text="Métricas", bg='#e0e0e0', padx=5, pady=5)
        metrics_frame.pack(pady=10, padx=5, fill=tk.BOTH, expand=True)
        
        self.metricas = {
            'letra_actual': tk.StringVar(value="-"),
            'puntos_activos': tk.StringVar(value="0"),
            'intensidad': tk.StringVar(value="-"),
            'energia_total': tk.StringVar(value="0.00"),
            'entropia_espacial': tk.StringVar(value="0.00"),
            'entropia_desc': tk.StringVar(value="-")
        }
        
        # Configuración de estilos
        label_width = 15
        value_width = 10
        padx_value = 5
        
        # Función para crear una celda de métrica
        def add_metric_cell(parent, label, var, row, col, colspan=1):
            # Frame para agrupar label y valor
            cell_frame = tk.Frame(parent, bg='#e0e0e0')
            cell_frame.grid(row=row, column=col, columnspan=colspan, padx=5, pady=2, sticky='nsew')
            
            # Label de la métrica
            tk.Label(cell_frame, text=label + ":", bg='#e0e0e0', anchor='w', 
                    width=label_width).pack(fill=tk.X)
            
            # Valor de la métrica
            value_label = tk.Label(cell_frame, textvariable=var, bg='white', relief=tk.SUNKEN,
                                 anchor='w', padx=padx_value, width=value_width)
            value_label.pack(fill=tk.X, ipady=2)
            
            # Ajustar el ancho del valor al contenido
            value_label.update_idletasks()
            value_label.config(width=value_width)
        
        # Configurar grid
        for i in range(3):  # 3 columnas
            metrics_frame.columnconfigure(i, weight=1, uniform='metrics')
        
        # Fila 1: Métricas básicas
        add_metric_cell(metrics_frame, "Letra actual", self.metricas['letra_actual'], 0, 0)
        add_metric_cell(metrics_frame, "Puntos activos", self.metricas['puntos_activos'], 0, 1)
        add_metric_cell(metrics_frame, "Intensidad", self.metricas['intensidad'], 0, 2)
        
        # Fila 2: Métricas avanzadas
        add_metric_cell(metrics_frame, "Energía total", self.metricas['energia_total'], 1, 0)
        add_metric_cell(metrics_frame, "Entropía", self.metricas['entropia_espacial'], 1, 1)
        add_metric_cell(metrics_frame, "Nivel entropía", self.metricas['entropia_desc'], 1, 2)
        
        # Ajustar peso de las filas
        metrics_frame.rowconfigure(0, weight=1)
        metrics_frame.rowconfigure(1, weight=1)
        
        # Frame para botones de prueba
        test_frame = tk.LabelFrame(control_frame, text="Pruebas Rápidas", bg='#e0e0e0', padx=5, pady=5)
        test_frame.pack(pady=10, padx=5, fill=tk.X)
        
        # Botones de prueba
        test_buttons = [
            ("Abecedario", self.test_abecedario),
            ("Números", self.test_numeros),
            ("Puntuación", self.test_puntuacion)
        ]
        
        for text, command in test_buttons:
            tk.Button(test_frame, text=text, command=command, width=15).pack(pady=2)
        
        self.motor = MotorN(self.canvas)

        # Diccionario de letras en Braille
        # Cada letra se define por sus puntos en una cuadrícula 2x3
        # Las coordenadas son relativas y se escalarán según la posición
        self.letras = [
            # Letras básicas (a-j)
            {"letra": "a", "puntos": [1], "accion": "media"},
            {"letra": "b", "puntos": [1, 2], "accion": "media"},
            {"letra": "c", "puntos": [1, 4], "accion": "media"},
            {"letra": "d", "puntos": [1, 4, 5], "accion": "media"},
            {"letra": "e", "puntos": [1, 5], "accion": "media"},
            {"letra": "f", "puntos": [1, 2, 4], "accion": "media"},
            {"letra": "g", "puntos": [1, 2, 4, 5], "accion": "media"},
            {"letra": "h", "puntos": [1, 2, 5], "accion": "media"},
            {"letra": "i", "puntos": [2, 4], "accion": "media"},
            {"letra": "j", "puntos": [2, 4, 5], "accion": "media"},
            
            # Letras k-t (a-j + punto 3)
            {"letra": "k", "puntos": [1, 3], "accion": "media"},
            {"letra": "l", "puntos": [1, 2, 3], "accion": "media"},
            {"letra": "m", "puntos": [1, 3, 4], "accion": "media"},
            {"letra": "n", "puntos": [1, 3, 4, 5], "accion": "media"},
            {"letra": "o", "puntos": [1, 3, 5], "accion": "media"},
            {"letra": "p", "puntos": [1, 2, 3, 4], "accion": "media"},
            {"letra": "q", "puntos": [1, 2, 3, 4, 5], "accion": "media"},
            {"letra": "r", "puntos": [1, 2, 3, 5], "accion": "media"},
            {"letra": "s", "puntos": [2, 3, 4], "accion": "media"},
            {"letra": "t", "puntos": [2, 3, 4, 5], "accion": "media"},
            
            # Letras u-z (a-j + puntos 3 y 6)
            {"letra": "u", "puntos": [1, 3, 6], "accion": "media"},
            {"letra": "v", "puntos": [1, 2, 3, 6], "accion": "media"},
            {"letra": "w", "puntos": [2, 4, 5, 6], "accion": "media"},  # w es especial
            {"letra": "x", "puntos": [1, 3, 4, 6], "accion": "media"},
            {"letra": "y", "puntos": [1, 3, 4, 5, 6], "accion": "media"},
            {"letra": "z", "puntos": [1, 3, 5, 6], "accion": "media"},
            
            # Números (precedidos por el indicador numérico ⠼)
            {"letra": "1", "puntos": [1], "accion": "baja"},
            {"letra": "2", "puntos": [1, 2], "accion": "baja"},
            {"letra": "3", "puntos": [1, 4], "accion": "baja"},
            {"letra": "4", "puntos": [1, 4, 5], "accion": "baja"},
            {"letra": "5", "puntos": [1, 5], "accion": "baja"},
            {"letra": "6", "puntos": [1, 2, 4], "accion": "baja"},
            {"letra": "7", "puntos": [1, 2, 4, 5], "accion": "baja"},
            {"letra": "8", "puntos": [1, 2, 5], "accion": "baja"},
            {"letra": "9", "puntos": [2, 4], "accion": "baja"},
            {"letra": "0", "puntos": [2, 4, 5], "accion": "baja"},
            
            # Signos de puntuación
            {"letra": ".", "puntos": [2, 5, 6], "accion": "baja"},
            {"letra": ",", "puntos": [2], "accion": "baja"},
            {"letra": ";", "puntos": [2, 3], "accion": "baja"},
            {"letra": ":", "puntos": [2, 5], "accion": "baja"},
            {"letra": "!", "puntos": [2, 3, 5], "accion": "baja"},
            {"letra": "?", "puntos": [2, 3, 6], "accion": "baja"},
            {"letra": " ", "puntos": [], "accion": "baja"}  # Espacio en blanco
        ]
        print(f"Diccionario cargado con {len(self.letras)} caracteres.")
        
        # Crear la interfaz de usuario
        self.crear_interfaz()
    
    def cargar_diccionario(self):
        try:
            # Buscar el archivo en varios lugares comunes
            posibles_rutas = [
                "diccionario_sensorial.json",
                "diccionarios/diccionario_sensorial.json",
                "../diccionario_sensorial.json",
                "../diccionarios/diccionario_sensorial.json"
            ]
            
            for ruta in posibles_rutas:
                if os.path.exists(ruta):
                    with open(ruta, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Verificar si el JSON tiene la estructura esperada
                        if isinstance(data, dict) and 'alfabeto_sensorial' in data:
                            return data['alfabeto_sensorial']
                        elif isinstance(data, list):
                            return data
                        else:
                            print(f"Formato de archivo no reconocido en {ruta}")
                            return []
            
            print("No se encontró el archivo de diccionario en ninguna ubicación estándar.")
            return []
            
        except Exception as e:
            print(f"Error al cargar el diccionario: {e}")
            return []

    def crear_interfaz(self):
        # Interfaz para inyectar letras
        frame = tk.Frame(self.root)
        frame.pack(pady=5)

        tk.Label(frame, text="Letra a inyectar:").pack(side=tk.LEFT)
        self.entrada_letra = tk.Entry(frame, width=5)
        self.entrada_letra.pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Inyectar", command=self.inyectar).pack(side=tk.LEFT)

    def actualizar_metricas(self, letra, puntos, intensidad):
        """Actualiza las métricas mostradas en la interfaz"""
        # Actualizar métricas básicas
        self.metricas['letra_actual'].set(letra.upper())
        self.metricas['puntos_activos'].set(str(len(puntos)))
        self.metricas['intensidad'].set(intensidad.capitalize())
        
        # Obtener métricas avanzadas del motor
        metricas_avanzadas = self.motor.obtener_metricas()
        
        # Actualizar métricas avanzadas
        self.metricas['energia_total'].set(f"{metricas_avanzadas['energia_total']:.2f}")
        self.metricas['entropia_espacial'].set(f"{metricas_avanzadas['entropia_espacial']:.2f}")
        self.metricas['entropia_desc'].set(metricas_avanzadas['entropia_desc'])
        
        # Actualizar color de fondo basado en la entropía
        self.actualizar_color_fondo(metricas_avanzadas['entropia_desc'])
    
    def actualizar_color_fondo(self, nivel_entropia):
        """Actualiza el color de fondo del canvas según el nivel de entropía"""
        colores = {
            'baja': '#e6f7e6',  # Verde claro
            'moderada': '#fff7e6',  # Amarillo claro
            'alta': '#ffe6e6'   # Rojo claro
        }
        color = colores.get(nivel_entropia, '#ffffff')  # Blanco por defecto
        self.canvas.config(bg=color)
    
    def limpiar(self):
        """Limpia el canvas y las métricas"""
        self.motor.campo.limpiar()
        self.metricas['letra_actual'].set("-")
        self.metricas['puntos_activos'].set("0")
        self.metricas['intensidad'].set("-")
        self.metricas['energia_total'].set("0.00")
        self.metricas['entropia_espacial'].set("0.00")
        self.metricas['entropia_desc'].set("-")
        self.entrada_letra.delete(0, tk.END)
        # Restaurar color de fondo a negro
        self.canvas.config(bg='black')
    
    def inyectar(self, letra=None):
        """Inyecta una letra en el canvas"""
        if letra is None:
            letra = self.entrada_letra.get().strip().lower()
        
        if not letra:
            return
            
        print(f"Intentando inyectar letra: '{letra}'")
        
        # Buscar la letra en el diccionario
        for item in self.letras:
            if item['letra'] == letra:
                print(f"Letra encontrada: '{letra}'. Puntos: {item['puntos']}, Intensidad: {item['accion']}")
                
                # Convertir puntos Braille a coordenadas
                coords = self.braille_a_coordenadas(item['puntos'])
                print(f"Coordenadas generadas: {coords}")
                
                # Inyectar los puntos en el canvas
                self.motor.inyectar_letra(coords, item['accion'])
                
                # Actualizar métricas
                self.actualizar_metricas(
                    letra=letra,
                    puntos=item['puntos'],
                    intensidad=item['accion']
                )
                break
        else:
            print(f"Letra '{letra}' no encontrada en el diccionario.")
        
        self.entrada_letra.delete(0, tk.END)  # Limpiar el campo de entrada
    
    # Funciones de prueba
    def test_abecedario(self):
        """Prueba el abecedario completo"""
        self.limpiar()
        letras = [item['letra'] for item in self.letras if item['letra'].isalpha()]
        self.ejecutar_secuencia(letras)
    
    def test_numeros(self):
        """Prueba los números del 0 al 9"""
        self.limpiar()
        numeros = [str(i) for i in range(10)]
        self.ejecutar_secuencia(numeros)
    
    def test_puntuacion(self):
        """Prueba los signos de puntuación"""
        self.limpiar()
        signos = ['.', ',', ';', ':', '!', '?']
        self.ejecutar_secuencia(signos)
    
    def ejecutar_secuencia(self, secuencia, delay=1000):
        """Ejecuta una secuencia de letras con un retraso entre ellas"""
        self.secuencia_actual = secuencia.copy()
        self.indice_secuencia = 0
        self.root.after(100, self._siguiente_en_secuencia, delay)
    
    def _siguiente_en_secuencia(self, delay):
        if self.indice_secuencia < len(self.secuencia_actual):
            letra = self.secuencia_actual[self.indice_secuencia]
            self.inyectar(letra)
            self.indice_secuencia += 1
            self.root.after(delay, self._siguiente_en_secuencia, delay)
    
    def braille_a_coordenadas(self, puntos):
        """
        Convierte puntos Braille (1-6) a coordenadas en el canvas.
        
        La disposición de los puntos en Braille es:
        1 • • 4
        2 • • 5
        3 • • 6
        """
        # Tamaño y posición de la celda Braille
        cell_size = 60
        margin = 100
        
        # Mapeo de puntos a posiciones (x, y) relativas
        punto_a_pos = {
            1: (0, 0),  # Arriba izquierda
            2: (0, 1),  # Medio izquierda
            3: (0, 2),  # Abajo izquierda
            4: (1, 0),  # Arriba derecha
            5: (1, 1),  # Medio derecha
            6: (1, 2)   # Abajo derecha
        }
        
        # Convertir cada punto a coordenadas
        coords = []
        for punto in puntos:
            if punto in punto_a_pos:
                x_rel, y_rel = punto_a_pos[punto]
                # Calcular posición absoluta en el canvas
                x = margin + (x_rel * cell_size * 1.5)
                y = margin + (y_rel * cell_size * 1.5)
                coords.append([int(x), int(y)])
        
        return coords

if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacion(root)
    root.mainloop()
