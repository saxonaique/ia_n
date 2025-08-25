import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import json
import os
import traceback
import random
import time
from datetime import datetime

# --- MODELO DE NODO Y CAMPO ---

class Nodo:
    def __init__(self, x, y, S=1.0, I=1.0, color="lime"):
        self.x = x
        self.y = y
        self.S = S # Entropía
        self.I = I # Información
        # rho es ahora una propiedad derivada para fines visuales/de fuerza combinada
        self.rho = self.S + self.I 
        self.color = color # Color inicial (será sobreescrito por la función de color por R)
        # Velocidad inicial más lenta
        self.vx = random.uniform(-1, 1) * 0.5
        self.vy = random.uniform(-1, 1) * 0.5
        # Fase aleatoria para animación (pulso visual)
        self.animation_phase = random.uniform(0, 6.28)
        self.canvas_id = None  # Para almacenar la referencia al objeto en el canvas

    @property
    def ratio_realidad(self):
        """Calcula el Ratio de Realidad (I/S) para el nodo."""
        # Evitar división por cero o por valores muy pequeños
        return self.I / max(0.01, self.S)

def get_color_from_R(R_val):
    """
    Genera un color hexadecimal basado en el Ratio de Realidad (R).
    Azul (baja R/alta S) -> Verde (R~1/equilibrio) -> Amarillo (alta R/alta I)
    """
    # Rango de R para la interpolación de color
    R_min = 0.2  # Corresponde a S=2.5, I=0.5 (o similar)
    R_max = 5.0  # Corresponde a S=0.5, I=2.5 (o similar)
    R_mid = 1.0  # Equilibrio S=I

    # Asegurar que R_val esté dentro del rango esperado para la interpolación
    R_val = max(R_min, min(R_max, R_val))

    if R_val <= R_mid: # Transición de Azul (S-dominante) a Verde (equilibrado)
        # Escalar R_val desde [R_min, R_mid] a [0, 1]
        t = (R_val - R_min) / (R_mid - R_min)
        r = int(0 * (1 - t) + 0 * t)       # Azul (0,0,255) -> Verde (0,255,0)
        g = int(0 * (1 - t) + 255 * t)
        b = int(255 * (1 - t) + 0 * t)
    else: # Transición de Verde (equilibrado) a Amarillo (I-dominante)
        # Escalar R_val desde [R_mid, R_max] a [0, 1]
        t = (R_val - R_mid) / (R_max - R_mid)
        r = int(0 * (1 - t) + 255 * t)      # Verde (0,255,0) -> Amarillo (255,255,0)
        g = int(255 * (1 - t) + 255 * t)    # El componente verde se mantiene
        b = int(0 * (1 - t) + 0 * t)
    
    return f'#{r:02x}{g:02x}{b:02x}'


class Campo:
    def __init__(self, ancho=800, alto=600):
        self.nodos = []
        self.ancho = ancho
        self.alto = alto
        self.animacion_activa = True  # Iniciar activa por defecto
        self.factor_velocidad = 1.0  # Factor de velocidad (1.0 = velocidad normal)
        self._ultimo_tiempo = time.time()  # Para cálculo de delta time

    def limpiar(self):
        """Limpia todos los nodos del campo."""
        self.nodos.clear()

    def inyectar_patron(self, coords, entropias, informaciones):
        """
        Inyecta un patrón de nodos en el campo con S e I.

        Args:
            coords: Lista de tuplas (x, y) en píxeles para la posición de cada nodo.
            entropias: Puede ser un valor único (float) o una lista de valores para S.
            informaciones: Puede ser un valor único (float) o una lista de valores para I.
        """
        self.limpiar()
        # Asegurarse de que S e I sean listas para iterar
        if not isinstance(entropias, (list, tuple)):
            entropias = [entropias] * len(coords)
        if not isinstance(informaciones, (list, tuple)):
            informaciones = [informaciones] * len(coords)
            
        for (x, y), S_val, I_val in zip(coords, entropias, informaciones):
            # El color inicial no importa mucho, se sobrescribirá por el ratio en el dibujo
            self.nodos.append(Nodo(x, y, S=S_val, I=I_val))

    def actualizar_posiciones(self):
        """
        Actualiza las posiciones de los nodos para la animación, aplicando rebote en los bordes,
        fuerzas de atracción/repulsión basadas en S e I, e intercambio de información.
        Retorna True si la animación está activa y hubo actualizaciones, False en caso contrario.
        """
        if not self.animacion_activa:
            return False
            
        tiempo_actual = time.time()
        delta_time = min(0.1, tiempo_actual - self._ultimo_tiempo)
        self._ultimo_tiempo = tiempo_actual
        
        velocidad_efectiva = self.factor_velocidad * delta_time * 60
        
        # --- Fase de Intercambio de Información (Influencia de S e I) ---
        new_S_vals = {nodo: nodo.S for nodo in self.nodos}
        new_I_vals = {nodo: nodo.I for nodo in self.nodos}
        
        info_exchange_distance = 250 # Rango máximo para el intercambio de información
        exchange_rate = 0.02 # Velocidad de difusión/promediado

        for i, nodo1 in enumerate(self.nodos):
            for j, nodo2 in enumerate(self.nodos):
                if i >= j:
                    continue

                dx = nodo2.x - nodo1.x
                dy = nodo2.y - nodo1.y
                distance = math.hypot(dx, dy)

                if distance < info_exchange_distance:
                    # Factor de influencia: más fuerte cuanto más cerca estén
                    influence_factor = (1 - (distance / info_exchange_distance)) 

                    # Promediar Entropía (S)
                    avg_S = (nodo1.S + nodo2.S) / 2.0
                    new_S_vals[nodo1] += (avg_S - nodo1.S) * exchange_rate * influence_factor
                    new_S_vals[nodo2] += (avg_S - nodo2.S) * exchange_rate * influence_factor

                    # Promediar Información (I)
                    avg_I = (nodo1.I + nodo2.I) / 2.0
                    new_I_vals[nodo1] += (avg_I - nodo1.I) * exchange_rate * influence_factor
                    new_I_vals[nodo2] += (avg_I - nodo2.I) * exchange_rate * influence_factor

        # Aplicar los nuevos valores de S e I, asegurando que se mantengan dentro de un rango
        for nodo in self.nodos:
            nodo.S = max(0.1, min(3.0, new_S_vals[nodo])) # S entre 0.1 y 3.0
            nodo.I = max(0.1, min(3.0, new_I_vals[nodo])) # I entre 0.1 y 3.0
            nodo.rho = nodo.S + nodo.I # Actualizar rho basado en los nuevos S e I

        # --- Cálculo de fuerzas entre nodos (usando los S e I actualizados) ---
        fuerzas = {nodo: {'fx': 0.0, 'fy': 0.0} for nodo in self.nodos}

        for i, nodo1 in enumerate(self.nodos):
            for j, nodo2 in enumerate(self.nodos):
                if i == j:
                    continue

                dx = nodo2.x - nodo1.x
                dy = nodo2.y - nodo1.y
                distance = math.hypot(dx, dy)

                if distance == 0:
                    distance = 0.1
                    dx = random.uniform(-0.1, 0.1)
                    dy = random.uniform(-0.1, 0.1)

                # Parámetros de las fuerzas
                # La fuerza de repulsión es influenciada por la suma de sus Entropías (S)
                repulsion_base_strength = 0.005  
                # La fuerza de atracción es influenciada por la suma de sus Informaciones (I)
                attraction_base_strength = 0.00005 

                min_repulsion_distance = 50 
                max_attraction_distance = 300 

                force_magnitude = 0.0
                
                # Factores de fuerza basados en S e I
                repulsion_factor = (nodo1.S + nodo2.S) / 2.0 # Promedio de entropías para la repulsión
                attraction_factor = (nodo1.I + nodo2.I) / 2.0 # Promedio de informaciones para la atracción

                if distance < min_repulsion_distance:
                    # Repulsión más fuerte si la Entropía combinada es alta
                    force_magnitude = -repulsion_base_strength * ((min_repulsion_distance / distance)**2) * repulsion_factor
                elif distance > max_attraction_distance:
                    # Atracción más fuerte si la Información combinada es alta
                    force_magnitude = attraction_base_strength * (distance - max_attraction_distance) * attraction_factor
                
                force_x = (dx / distance) * force_magnitude
                force_y = (dy / distance) * force_magnitude

                fuerzas[nodo1]['fx'] += force_x
                fuerzas[nodo1]['fy'] += force_y
                fuerzas[nodo2]['fx'] -= force_x
                fuerzas[nodo2]['fy'] -= force_y

        # --- Aplicar movimientos y amortiguación ---
        damping = 0.98 

        for nodo in self.nodos:
            nodo.vx += fuerzas[nodo]['fx'] * self.factor_velocidad * delta_time * 0.1 
            nodo.vy += fuerzas[nodo]['fy'] * self.factor_velocidad * delta_time * 0.1

            nodo.vx *= damping
            nodo.vy *= damping

            nodo.x += nodo.vx * velocidad_efectiva
            nodo.y += nodo.vy * velocidad_efectiva
            
            # Rebotar en los bordes
            if nodo.x < 0:
                nodo.x = 0
                nodo.vx = abs(nodo.vx) * 0.95
            elif nodo.x > self.ancho:
                nodo.x = self.ancho
                nodo.vx = -abs(nodo.vx) * 0.95
                
            if nodo.y < 0:
                nodo.y = 0
                nodo.vy = abs(nodo.vy) * 0.95
            elif nodo.y > self.alto:
                nodo.y = self.alto
                nodo.vy = -abs(nodo.vy) * 0.95
            
            # El tamaño visual del nodo (rho) pulsa alrededor de su S+I
            nodo.rho = (nodo.S + nodo.I) * (1 + 0.1 * math.sin(nodo.animation_phase * 0.7))
            nodo.animation_phase += delta_time * 2
            if nodo.animation_phase > 6.28:
                nodo.animation_phase -= 6.28
            
        return True

    def exportar_json(self, nombre_archivo):
        """
        Exporta los nodos actuales a un archivo JSON, incluyendo S e I.
        """
        datos = []
        for nodo in self.nodos:
            datos.append({
                "x": int(nodo.x),
                "y": int(nodo.y),
                "S": float(nodo.S),
                "I": float(nodo.I),
                "color": nodo.color # El color se guardará, pero se recalculará al cargar
            })
        
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            json.dump(datos, f, indent=2)
        
        return len(datos)

    def energia_total(self):
        """
        Calcula la energía total del sistema (suma de (S+I) de todos los nodos).
        """
        return sum(n.S + n.I for n in self.nodos)

    def entropia_espacial(self):
        """
        Calcula una medida de entropía espacial basada en la desviación estándar de las distancias entre nodos.
        """
        if len(self.nodos) < 2:
            return 0.0
            
        distancias = []
        for i in range(len(self.nodos)):
            for j in range(i + 1, len(self.nodos)):
                dx = self.nodos[i].x - self.nodos[j].x
                dy = self.nodos[i].y - self.nodos[j].y
                distancias.append(math.hypot(dx, dy))
        
        if not distancias:
            return 0.0
            
        mean = sum(distancias) / len(distancias)
        if len(distancias) > 1:
            var = sum((d - mean) ** 2 for d in distancias) / (len(distancias) - 1)
            return math.sqrt(var)  # Desviación estándar
        return 0.0

# --- MOTOR N ---
class MotorN:
    def __init__(self, canvas, lbl_status, lbl_energy, lbl_entropy):
        self.canvas = canvas
        self.lbl_status = lbl_status
        self.lbl_energy = lbl_energy
        self.lbl_entropy = lbl_entropy
        self.campo = Campo(1, 1) 
        self.nodo_seleccionado = None
        self._after_id = None
        
        self.canvas.bind("<Button-1>", self.seleccionar_nodo)
        self.canvas.bind("<B1-Motion>", self.mover_nodo)
        self.canvas.bind("<ButtonRelease-1>", self.soltar_nodo)
        
        self.iniciar_animacion()
    
    def iniciar_animacion(self):
        """Inicia (o reasegura que esté corriendo) el bucle de animación."""
        if not hasattr(self, '_after_id') or self._after_id is None:
            self._bucle_animacion()
    
    def _bucle_animacion(self):
        """Bucle principal de animación que actualiza posiciones y redibuja."""
        if not self.canvas.winfo_viewable():
            self._after_id = self.canvas.after(100, self._bucle_animacion)
            return
            
        try:
            self.campo.ancho = self.canvas.winfo_width()
            self.campo.alto = self.canvas.winfo_height()

            if self.campo.animacion_activa:
                self.campo.actualizar_posiciones()
                self.dibujar()
                
                if not hasattr(self, '_frame_count'):
                    self._frame_count = 0
                self._frame_count += 1
                if self._frame_count >= 5: 
                    self.actualizar_metricas()
                    self._frame_count = 0
            
            self._after_id = self.canvas.after(16, self._bucle_animacion)
            
        except Exception as e:
            print(f"Error en bucle de animación: {e}")
            traceback.print_exc()
            self._after_id = self.canvas.after(500, self._bucle_animacion)
    
    def pausar_animacion(self, pausar=True):
        """Pausa o reanuda la animación."""
        self.campo.animacion_activa = not pausar
    
    def seleccionar_nodo(self, event):
        """Maneja el evento de clic para seleccionar un nodo."""
        for nodo in self.campo.nodos:
            r = nodo.rho * 10 
            dx = nodo.x - event.x
            dy = nodo.y - event.y
            if dx*dx + dy*dy <= r*r:
                self.nodo_seleccionado = nodo
                self.dibujar() 
                break
    
    def mover_nodo(self, event):
        """Mueve el nodo seleccionado mientras se arrastra el ratón."""
        if self.nodo_seleccionado:
            self.nodo_seleccionado.x = max(0, min(event.x, self.canvas.winfo_width()))
            self.nodo_seleccionado.y = max(0, min(event.y, self.canvas.winfo_height()))
            self.dibujar()
    
    def soltar_nodo(self, event):
        """Maneja el evento de soltar el botón del ratón, deseleccionando el nodo."""
        self.nodo_seleccionado = None
        self.dibujar() 

    def dibujar(self):
        """Dibuja todos los nodos y sus conexiones en el canvas."""
        self.canvas.delete("all") 
        self._dibujar_conexiones()
        for nodo in self.campo.nodos:
            self._dibujar_nodo(nodo)
            if nodo == self.nodo_seleccionado:
                r_base = int(nodo.rho * 5)
                self.canvas.create_oval(
                    nodo.x - r_base - 3, nodo.y - r_base - 3,
                    nodo.x + r_base + 3, nodo.y + r_base + 3,
                    outline="white", width=2, dash=(3,3)
                )
    
    def _dibujar_conexiones(self):
        """Dibuja líneas entre nodos cercanos, con ancho y opacidad variables."""
        for i, nodo1 in enumerate(self.campo.nodos):
            for nodo2 in self.campo.nodos[i+1:]:
                dx = nodo1.x - nodo2.x
                dy = nodo1.y - nodo2.y
                distancia = math.hypot(dx, dy)
                
                if distancia < 150: 
                    ancho = max(2, int(7 * (1 - distancia/150)))
                    opacidad = 1.0 - (distancia/200)
                    if opacidad < 0.1: 
                        continue
                        
                    color_val = int(200 * opacidad) 
                    color = f'#{color_val:02x}{color_val:02x}{color_val:02x}' 

                    self.canvas.create_line(
                        nodo1.x, nodo1.y,
                        nodo2.x, nodo2.y,
                        fill=color,
                        width=ancho,
                        tags=("conexion", f"conexion_{id(nodo1)}_{id(nodo2)}")
                    )
    
    def _dibujar_nodo(self, nodo):
        """
        Dibuja un nodo individual con efectos visuales de pulso y halo.
        El color del nodo se determina por su Ratio de Realidad (I/S).
        """
        x, y = int(nodo.x), int(nodo.y)
        r = int(nodo.rho * 5)  # Tamaño base del nodo, escalado por rho (S+I)
        
        pulse_effect = 1 + 0.1 * math.sin(nodo.animation_phase)
        r_halo = int(r * (1.2 + 0.1 * pulse_effect)) 
        
        # Obtener el color del nodo basado en su Ratio de Realidad
        nodo_color = get_color_from_R(nodo.ratio_realidad)

        self.canvas.create_oval(
            x - r_halo, y - r_halo,
            x + r_halo, y + r_halo,
            fill="",
            outline=nodo_color, 
            width=1 + 0.5 * abs(math.sin(nodo.animation_phase * 0.7)) 
        )
        
        self.canvas.create_oval(
            x - r, y - r,
            x + r, y + r,
            fill=nodo_color,
            outline="white", 
            width=1
        )
        
        if r > 8:
            # Color del texto se ajusta para ser legible sobre el color del nodo
            # Simple heurística: si el color es oscuro, texto blanco; si es claro, texto negro
            # Esto es un placeholder, una lógica RGB a Luminosidad sería mejor.
            fill_color = "white" if nodo.ratio_realidad < 1.5 else "black" # Aproximado para legibilidad
            self.canvas.create_text(
                x, y,
                text=f"S:{nodo.S:.1f}\nI:{nodo.I:.1f}", # Mostrar S e I
                fill=fill_color,
                font=("Arial", max(6, r - 4)), # Tamaño de fuente adaptable
                justify=tk.CENTER
            )
    
    def actualizar_metricas(self):
        """
        Actualiza las etiquetas de energía, entropía y estado, así como el título de la ventana.
        """
        e = self.campo.energia_total()
        s = self.campo.entropia_espacial()
        
        if self.lbl_energy:
            # Ahora muestra el "Valor Total S" y "Valor Total I"
            total_S = sum(n.S for n in self.campo.nodos)
            total_I = sum(n.I for n in self.campo.nodos)
            self.lbl_energy.config(text=f"Total S: {total_S:.2f}\nTotal I: {total_I:.2f}")
            
        if self.lbl_entropy:
            self.lbl_entropy.config(text=f"Entropía Espacial: {s:.2f}")
            
        estado = ""
        color = "gray"
        if len(self.campo.nodos) == 0:
            estado = "Inactivo"
            color = "gray"
        elif self.campo.animacion_activa:
            estado = "Ejecutando"
            color = "lime"
        else:
            estado = "Pausado"
            color = "yellow"
            
        if self.lbl_status:
            self.lbl_status.config(text=f"Estado: {estado}", fg=color)
            
        try:
            root = self.canvas.winfo_toplevel()
            if root and hasattr(root, 'title'):
                root.title(f"Simulador S-I | Nodos: {len(self.campo.nodos)} | Sum S: {sum(n.S for n in self.campo.nodos):.1f} | Sum I: {sum(n.I for n in self.campo.nodos):.1f}")
        except Exception:
            pass

    def ejecutar_inyeccion(self, coords, entropias, informaciones):
        """
        Inyecta un patrón de nodos y actualiza la visualización y métricas.
        """
        self.campo.inyectar_patron(coords, entropias, informaciones)
        self.dibujar()
        self.actualizar_metricas()
    
    def exportar_patron(self):
        """
        Guarda el patrón de nodos actual en un archivo JSON, incluyendo S e I.
        """
        if not self.campo.nodos:
            messagebox.showwarning("Sin datos", "No hay nodos para exportar")
            return False
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[
                ("Archivos JSON", "*.json"),
                ("Todos los archivos", "*.*")
            ],
            title="Guardar patrón como...",
            initialfile=f"patron_{len(self.campo.nodos)}_nodos_{int(time.time())}.json"
        )
        
        if not file_path:  # El usuario canceló la operación
            return False
            
        try:
            nodos_data = [{
                "x": nodo.x,
                "y": nodo.y,
                "S": float(nodo.S),
                "I": float(nodo.I)
            } for nodo in self.campo.nodos]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(nodos_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Exportación Exitosa", f"Patrón guardado en:\n{file_path}")
            return True
            
        except PermissionError:
            messagebox.showerror("Error de Permisos", 
                              "No se pudo guardar el archivo. Verifica que tengas permisos de escritura.")
        except Exception as e:
            messagebox.showerror("Error al Exportar", 
                              f"No se pudo guardar el archivo:\n{str(e)}")
        
        return False

# --- APLICACIÓN ---
class App:
    def __init__(self, root):
        self.root = root
        root.title("Simulador de Colapso Informacional")
        
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")
        root.configure(bg="#1e1e1e")

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=5) 
        root.grid_columnconfigure(1, weight=1, minsize=300) 

        self.configurar_estilos() 

        # --- Panel lateral de controles y métricas ---
        panel = tk.Frame(root, bg="#2b2b2b")
        panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        panel.grid_rowconfigure(3, weight=1) 

        self.lbl_status = tk.Label(panel, text="Estado: –", fg="white", bg="#2b2b2b", font=("Consolas", 12))
        self.lbl_status.grid(row=0, column=0, sticky="w", pady=(0, 10), padx=5)
        
        # Cambiado para mostrar S e I totales
        self.lbl_energy = tk.Label(panel, text="Total S: –\nTotal I: –", fg="white", bg="#2b2b2b", font=("Consolas", 11), justify=tk.LEFT)
        self.lbl_energy.grid(row=1, column=0, sticky="w", pady=(0, 5), padx=5)
        
        self.lbl_entropy = tk.Label(panel, text="Entropía Espacial: –", fg="white", bg="#2b2b2b", font=("Consolas", 11))
        self.lbl_entropy.grid(row=2, column=0, sticky="w", pady=(0, 20), padx=5)

        # --- Canvas de visualización ---
        self.canvas = tk.Canvas(root, bg="black", highlightbackground="#3b3b3b", highlightthickness=2)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.motor = MotorN(
            canvas=self.canvas, 
            lbl_status=self.lbl_status, 
            lbl_energy=self.lbl_energy, 
            lbl_entropy=self.lbl_entropy
        )

        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Controles de animación
        frame_controles = ttk.LabelFrame(panel, text="Controles de Simulación", padding=10)
        frame_controles.grid(row=4, column=0, sticky="nsew", pady=(0, 10), padx=5)
        frame_controles.grid_columnconfigure(0, weight=1)
        frame_controles.grid_columnconfigure(2, weight=2)
        
        self.btn_pausa = ttk.Button(
            frame_controles, 
            text="⏸ Pausar",
            command=self.toggle_pausa,
            style="Accent.TButton"
        )
        self.btn_pausa.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(frame_controles, text="Velocidad:", background="#2b2b2b", foreground="white").grid(row=0, column=1, padx=5, sticky="w")
        self.velocidad = tk.DoubleVar(value=1.0)
        ttk.Scale(
            frame_controles, 
            from_=0.1, 
            to=3.0, 
            orient=tk.HORIZONTAL,
            variable=self.velocidad,
            command=self.actualizar_velocidad,
            length=150 
        ).grid(row=0, column=2, padx=5, sticky="ew")
        
        frame_botones = ttk.Frame(panel)
        frame_botones.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)
        for i in range(5):
            frame_botones.columnconfigure(i, weight=1)

        btns = [
            ("Cargar Patrón", self.on_load, "📂"),
            ("Nuevo Nodo", self.agregar_nodo, "➕"),
            ("Reiniciar", self.on_reset, "🔄"),
            ("Exportar", self.on_export, "💾"),
            ("Ejemplo", self.crear_archivo_ejemplo, "✨"),
        ]
        
        for i, (txt, cmd, emoji) in enumerate(btns):
            b = ttk.Button(
                frame_botones, 
                text=f"{emoji} {txt}",
                command=cmd,
                style="Accent.TButton" if txt == "Exportar" else "TButton"
            )
            b.grid(row=0, column=i, padx=2, pady=5, sticky="nsew")

        self.status_bar = ttk.Frame(root, height=25, style="Status.TFrame")
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=0)
        
        self.lbl_status_bar = ttk.Label(
            self.status_bar, 
            text="Listo | Nodos: 0 | Arrastra para seleccionar y mover nodos",
            anchor=tk.W, 
            style="Status.TLabel"
        )
        self.lbl_status_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.canvas.bind("<Motion>", self.actualizar_status_bar)
        self.canvas.bind("<Leave>", lambda e: self.actualizar_contador_nodos_en_status_bar())
        
        self.toggle_pausa()

    def on_canvas_resize(self, event):
        """
        Actualiza las dimensiones del campo de la simulación cuando el canvas cambia de tamaño.
        También recarga el patrón de ejemplo para que los nodos se distribuyan correctamente
        en el nuevo tamaño del campo.
        """
        self.motor.campo.ancho = event.width
        self.motor.campo.alto = event.height
        self.cargar_patron_ejemplo()
        self.motor.dibujar() 
        self.motor.actualizar_metricas()


    def actualizar_status_bar(self, event):
        """Actualiza la barra de estado con la posición del ratón y el contador de nodos."""
        self.lbl_status_bar.config(
            text=f"Posición: ({event.x}, {event.y}) | "
                 f"Nodos: {len(self.motor.campo.nodos)} | "
                 f"Arrastra para seleccionar y mover nodos"
        )
    
    def actualizar_contador_nodos_en_status_bar(self):
        """Actualiza solo el contador de nodos en la barra de estado (usado al salir del canvas)."""
        if hasattr(self, 'motor') and hasattr(self.motor, 'campo'):
            self.lbl_status_bar.config(
                text=f"Listo | Nodos: {len(self.motor.campo.nodos)} | "
                     f"Arrastra para seleccionar y mover nodos"
            )

    def toggle_pausa(self):
        """Alterna el estado de la animación entre pausado y ejecutándose."""
        self.motor.campo.animacion_activa = not self.motor.campo.animacion_activa
        
        if self.motor.campo.animacion_activa:
            self.btn_pausa.config(text="⏸ Pausar")
            self.lbl_status.config(text="Estado: Ejecutando", fg="lime")
            self.motor.iniciar_animacion() 
        else:
            self.btn_pausa.config(text="▶ Reanudar")
            self.lbl_status.config(text="Estado: Pausado", fg="yellow")
    
    def actualizar_velocidad(self, *args):
        """Actualiza el factor de velocidad de la simulación."""
        if hasattr(self, 'motor') and hasattr(self.motor, 'campo'):
            self.motor.campo.factor_velocidad = self.velocidad.get()
            
            velocidad_actual = round(self.velocidad.get(), 1)
            self.lbl_status_bar.config(
                text=f"Velocidad: {velocidad_actual}x | "
                     f"Nodos: {len(self.motor.campo.nodos)}"
            )
    
    def agregar_nodo(self):
        """Añade un nuevo nodo en una posición aleatoria dentro del canvas."""
        x = random.randint(50, self.motor.campo.ancho - 50)
        y = random.randint(50, self.motor.campo.alto - 50)
        # S e I se inicializan aleatoriamente
        S_val = random.uniform(0.5, 2.5) 
        I_val = random.uniform(0.5, 2.5)
        
        self.motor.campo.nodos.append(Nodo(x, y, S=S_val, I=I_val))
        self.motor.dibujar()
        self.motor.actualizar_metricas()
        self.actualizar_contador_nodos_en_status_bar() 
    
    def on_export(self):
        """Maneja la acción de exportar el patrón actual de nodos a un archivo JSON."""
        if self.motor.exportar_patron():
            self.lbl_status_bar.config(
                text=f"Patrón exportado con éxito! | Nodos: {len(self.motor.campo.nodos)}"
            )
    
    def cargar_patron_ejemplo(self):
        """Carga un patrón de nodos predefinido para demostración.
        Las coordenadas se calculan proporcionalmente al tamaño actual del campo."""
        
        if self.motor.campo.ancho == 1 and self.motor.campo.alto == 1:
            return

        # Coordenadas y propiedades S e I para el patrón de ejemplo (proporcionales)
        patron_proporcional = [
            {"x_norm": 0.15, "y_norm": 0.15, "S": 1.0, "I": 2.0}, # Alta I, R > 1 (Amarillo/Naranja)
            {"x_norm": 0.35, "y_norm": 0.25, "S": 2.0, "I": 1.0}, # Alta S, R < 1 (Azul/Púrpura)
            {"x_norm": 0.55, "y_norm": 0.10, "S": 1.5, "I": 1.5}, # Equilibrio, R ~ 1 (Verde)
            {"x_norm": 0.25, "y_norm": 0.40, "S": 0.8, "I": 2.2}, # Muy Alta I
            {"x_norm": 0.65, "y_norm": 0.30, "S": 2.2, "I": 0.8}, # Muy Alta S
            {"x_norm": 0.80, "y_norm": 0.18, "S": 1.0, "I": 1.0}, # Equilibrio
            {"x_norm": 0.20, "y_norm": 0.65, "S": 0.5, "I": 2.5}, # Extrema I (más amarillo/blanco)
            {"x_norm": 0.40, "y_norm": 0.75, "S": 2.5, "I": 0.5}, # Extrema S (más azul/morado)
            {"x_norm": 0.60, "y_norm": 0.55, "S": 1.2, "I": 1.8}, # Ligeramente I dominante
            {"x_norm": 0.85, "y_norm": 0.90, "S": 1.8, "I": 1.2}  # Ligeramente S dominante
        ]
        
        coords = []
        entropias = []
        informaciones = []

        for n in patron_proporcional:
            x = int(n['x_norm'] * self.motor.campo.ancho)
            y = int(n['y_norm'] * self.motor.campo.alto)
            coords.append((x, y))
            entropias.append(n['S'])
            informaciones.append(n['I'])
        
        self.motor.ejecutar_inyeccion(coords, entropias, informaciones)
        self.actualizar_contador_nodos_en_status_bar() 

    def on_load(self):
        """Maneja la carga de un patrón de nodos desde un archivo JSON."""
        try:
            file_path = filedialog.askopenfilename(
                title="Selecciona archivo de patrón JSON",
                filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")]
            )
            
            if not file_path:  
                return
            
            self.lbl_status.config(text=f"Cargando {os.path.basename(file_path)}...")
            self.root.update_idletasks() 
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("El archivo JSON debe contener un arreglo de nodos.")
            
            self.motor.campo.limpiar() 
            
            nodos_cargados = 0
            for i, nodo_data in enumerate(data, 1):
                try:
                    if not isinstance(nodo_data, dict):
                        print(f"Advertencia: Elemento {i} no es un diccionario válido, omitiendo...")
                        continue
                        
                    x = float(nodo_data.get('x', 0))
                    y = float(nodo_data.get('y', 0))
                    # Cargar S e I desde el JSON
                    S_val = float(nodo_data.get('S', 1.0))
                    I_val = float(nodo_data.get('I', 1.0))
                    
                    x = max(0, min(x, self.motor.campo.ancho))
                    y = max(0, min(y, self.motor.campo.alto))
                    
                    self.motor.campo.nodos.append(Nodo(x, y, S=S_val, I=I_val))
                    nodos_cargados += 1
                    
                except (ValueError, TypeError) as e:
                    print(f"Advertencia: Error al procesar nodo {i}: {e}, omitiendo...")
                    continue
            
            if nodos_cargados == 0:
                raise ValueError("No se pudo cargar ningún nodo válido del archivo seleccionado.")
            
            self.motor.dibujar()
            self.motor.actualizar_metricas()
            self.actualizar_contador_nodos_en_status_bar()
            
            nombre_archivo = os.path.basename(file_path)
            self.lbl_status.config(text=f"Cargados {nodos_cargados} nodos desde {nombre_archivo}")
            messagebox.showinfo("Carga Exitosa", f"Se cargaron {nodos_cargados} nodos desde:\n{file_path}")
            
            self.canvas.focus_set() 
            
        except json.JSONDecodeError as e:
            error_msg = f"Error en el formato JSON del archivo: {str(e)}"
            messagebox.showerror("Error de Formato", error_msg)
            self.lbl_status.config(text=error_msg)
            print(f"Error detallado: {traceback.format_exc()}")
        except Exception as e:
            error_msg = f"Error al cargar el archivo: {str(e)}"
            messagebox.showerror("Error de Carga", error_msg)
            self.lbl_status.config(text=error_msg)
            print(f"Error detallado: {traceback.format_exc()}")

    def crear_archivo_ejemplo(self):
        """Crea un archivo JSON de ejemplo en el directorio de la aplicación y lo carga."""
        try:
            # Datos de ejemplo con coordenadas proporcionales y S/I definidos
            datos_ejemplo_proporcional = [
                {"x_norm": 0.10, "y_norm": 0.10, "S": 1.0, "I": 2.0}, # Alta I
                {"x_norm": 0.30, "y_norm": 0.20, "S": 2.0, "I": 1.0}, # Alta S
                {"x_norm": 0.50, "y_norm": 0.05, "S": 1.5, "I": 1.5}, # Equilibrio
                {"x_norm": 0.70, "y_norm": 0.35, "S": 0.8, "I": 2.2}, # Muy Alta I
                {"x_norm": 0.90, "y_norm": 0.15, "S": 2.2, "I": 0.8}, # Muy Alta S
                {"x_norm": 0.15, "y_norm": 0.50, "S": 1.0, "I": 1.0}, # Equilibrio
                {"x_norm": 0.45, "y_norm": 0.60, "S": 0.5, "I": 2.5}, # Extrema I
                {"x_norm": 0.75, "y_norm": 0.80, "S": 2.5, "I": 0.5}, # Extrema S
                {"x_norm": 0.25, "y_norm": 0.90, "S": 1.2, "I": 1.8}, # Ligeramente I dominante
                {"x_norm": 0.55, "y_norm": 0.45, "S": 1.8, "I": 1.2}  # Ligeramente S dominante
            ]
            
            datos_ejemplo_abs = []
            for n_prop in datos_ejemplo_proporcional:
                x_abs = int(n_prop['x_norm'] * self.motor.campo.ancho)
                y_abs = int(n_prop['y_norm'] * self.motor.campo.alto)
                datos_ejemplo_abs.append({
                    "x": x_abs,
                    "y": y_abs,
                    "S": n_prop['S'],
                    "I": n_prop['I']
                })

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo_base = f"ejemplo_patron_{timestamp}.json"
            script_dir = os.path.dirname(os.path.abspath(__file__))
            nombre_archivo_completo = os.path.join(script_dir, nombre_archivo_base)
            
            with open(nombre_archivo_completo, 'w', encoding='utf-8') as f:
                json.dump(datos_ejemplo_abs, f, indent=2, ensure_ascii=False)
            
            self.motor.campo.limpiar()
            for nodo_data in datos_ejemplo_abs: 
                self.motor.campo.nodos.append(Nodo(
                    nodo_data['x'], 
                    nodo_data['y'], 
                    S=nodo_data['S'], 
                    I=nodo_data['I']
                ))
            
            self.motor.dibujar()
            self.motor.actualizar_metricas()
            self.actualizar_contador_nodos_en_status_bar()
            
            mensaje = f"Archivo de ejemplo creado y cargado: {nombre_archivo_base}"
            self.lbl_status.config(text=mensaje)
            messagebox.showinfo("Éxito", f"Se creó el archivo de ejemplo en:\n{nombre_archivo_completo}\n\n¡Y se cargó automáticamente!")
            
        except Exception as e:
            error_msg = f"Error al crear el archivo de ejemplo: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.lbl_status.config(text=error_msg)
            print(f"Error detallado: {traceback.format_exc()}")
    
    def on_reset(self):
        """Reinicia la simulación, limpiando todos los nodos."""
        self.motor.campo.limpiar()
        self.motor.dibujar()
        self.motor.actualizar_metricas()
        self.lbl_status.config(text="Sistema reiniciado")
        self.actualizar_contador_nodos_en_status_bar()
        
        if not self.motor.campo.animacion_activa:
            self.btn_pausa.config(text="▶ Reanudar")

    def configurar_estilos(self):
        """Configura los estilos visuales de la aplicación usando ttk.Style."""
        style = ttk.Style()
        style.theme_use('clam') 
        
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="white")
        
        style.configure("TButton", 
                        padding=5, 
                        font=("Arial", 10),
                        background="#4a4a4a", 
                        foreground="white",
                        relief="flat") 
        style.map("TButton", 
                  background=[("active", "#5a5a5a")], 
                  foreground=[("active", "white")])
                  
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=5)
        
        style.configure("TNotebook", background="#2b2b2b")
        style.configure("TNotebook.Tab", 
                        background="#3b3b3b", 
                        foreground="white", 
                        padding=[10, 5])
        style.map("TNotebook.Tab", 
                background=[("selected", "#1e1e1e")],
                foreground=[("selected", "white")])
        
        style.configure("Accent.TButton", 
                      font=("Arial", 10, "bold"),
                      foreground="white",
                      background="#0078d7", 
                      padding=6,
                      relief="raised") 
        style.map("Accent.TButton", 
                  background=[("active", "#005bb5")], 
                  foreground=[("active", "white")])
        
        style.configure("Status.TFrame", background="#1e1e1e") 
        style.configure("Status.TLabel", 
                      background="#1e1e1e",
                      foreground="#a0a0a0", 
                      font=("Consolas", 9))
        
        style.configure("TLabelFrame", background="#2b2b2b", foreground="white")
        style.configure("TLabelFrame.Label", background="#2b2b2b", foreground="white") 

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()




