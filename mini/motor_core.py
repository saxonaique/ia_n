import math
import statistics

class MotorN:
    def __init__(self, canvas):
        self.canvas = canvas
        self.campo = Campo(canvas)
    
    def inyectar_letra(self, coords, intensidad):
        self.campo.inyectar_forma(coords, intensidad)
        # No es necesario llamar a dibujar() aquí porque ya se llama en inyectar_forma()
    
    def obtener_metricas(self):
        """Obtiene las métricas del campo"""
        return self.campo.obtener_metricas()

class Nodo:
    def __init__(self, x, y, rho=1.0):
        self.x = x
        self.y = y
        self.rho = rho

class Campo:
    def __init__(self, canvas):
        self.canvas = canvas
        self.nodos = []
        self.ids = []  # IDs de los óvalos en canvas

    def limpiar(self):
        """Limpia solo los elementos del canvas, sin afectar la lista de nodos"""
        for id_oval in self.ids:
            try:
                self.canvas.delete(id_oval)
            except:
                pass  # Ignorar si el ID ya no existe
        self.ids.clear()

    def inyectar_forma(self, coords, intensidad='media'):
        """Inyecta una nueva forma en el canvas"""
        print(f"\n{'='*50}")
        print(f"Inyectando forma - Intensidad: {intensidad}")
        
        # 1. Limpiar estado anterior
        self.limpiar()  # Limpia solo el canvas
        self.nodos.clear()  # Limpia los nodos anteriores
        
        # 2. Validar coordenadas
        if not isinstance(coords, (list, tuple)):
            print(f"Error: Se esperaba una lista de coordenadas, se recibió: {type(coords)}")
            return
            
        # 3. Obtener factor de escala según intensidad
        escala = {
            "baja": 0.7, 
            "media": 1.0, 
            "alta": 1.3, 
            "moderada": 0.9
        }.get(str(intensidad).lower() if isinstance(intensidad, str) else "media", 1.0)
        
        # 4. Procesar cada punto
        puntos_procesados = 0
        for i, punto in enumerate(coords, 1):
            try:
                # Validar punto
                if not isinstance(punto, (list, tuple)) or len(punto) < 2:
                    print(f"  - Punto {i} ignorado: formato inválido")
                    continue
                    
                # Convertir a coordenadas numéricas
                x = max(20, min(380, float(punto[0])))
                y = max(20, min(380, float(punto[1])))
                
                # Crear y agregar nodo
                self.nodos.append(Nodo(x, y, rho=escala))
                puntos_procesados += 1
                
                # Mostrar información de depuración
                print(f"  - Punto {i}: ({x:.1f}, {y:.1f}) [Escala: {escala:.1f}]")
                
            except (ValueError, TypeError) as e:
                print(f"  - Error en punto {i} ({punto}): {e}")
        
        # 5. Mostrar resumen
        print(f"\nResumen: {puntos_procesados} de {len(coords)} puntos procesados")
        print(f"Nodos activos: {len(self.nodos)}")
        print(f"IDs en canvas: {len(self.ids)}")
        print(f"{'='*50}\n")
        
        # 6. Dibujar los nodos en el canvas
        if self.nodos:  # Solo dibujar si hay nodos
            self.dibujar()

    def energia_total(self):
        """Calcula la energía total sumando todos los valores rho de los nodos"""
        return sum(nodo.rho for nodo in self.nodos)

    def entropia_espacial(self):
        """Calcula la entropía espacial basada en la dispersión de los nodos"""
        if len(self.nodos) < 2:
            return 0.0

        # Calculamos distancias entre nodos
        distancias = []
        for i in range(len(self.nodos)):
            for j in range(i + 1, len(self.nodos)):
                dx = self.nodos[i].x - self.nodos[j].x
                dy = self.nodos[i].y - self.nodos[j].y
                dist = math.sqrt(dx ** 2 + dy ** 2)
                distancias.append(dist)

        if not distancias:
            return 0.0

        # Usamos la desviación estándar de las distancias como medida de entropía
        return statistics.stdev(distancias) if len(distancias) > 1 else 0.0

    def obtener_metricas(self):
        """Devuelve un diccionario con las métricas actuales"""
        energia = self.energia_total()
        entropia = self.entropia_espacial()
        
        # Clasificación cualitativa de la entropía
        if entropia > 60:
            desc_entropia = "alta"
        elif entropia > 30:
            desc_entropia = "moderada"
        else:
            desc_entropia = "baja"
            
        return {
            "energia_total": round(energia, 2),
            "entropia_espacial": round(entropia, 2),
            "entropia_desc": desc_entropia
        }

    def dibujar(self):
        """Dibuja todos los nodos actuales en el canvas"""
        # 1. Verificar si hay nodos para dibujar
        if not self.nodos:
            print("  - No hay nodos para dibujar")
            return
            
        # 2. Dibujar cada nodo
        for i, nodo in enumerate(self.nodos, 1):
            try:
                # Calcular radio basado en la intensidad (rho)
                radio_base = 8  # Radio base en píxeles
                radio = radio_base * (getattr(nodo, 'rho', 1.0))
                
                # Coordenadas del círculo
                x1, y1 = nodo.x - radio, nodo.y - radio
                x2, y2 = nodo.x + radio, nodo.y + radio
                
                # Crear el círculo en el canvas
                id_oval = self.canvas.create_oval(
                    x1, y1, x2, y2,
                    fill="#00FF00",      # Verde brillante
                    outline="#FFFFFF",    # Borde blanco
                    width=2              # Grosor del borde
                )
                
                # Guardar el ID para poder borrarlo después
                self.ids.append(id_oval)
                
            except Exception as e:
                print(f"  - Error al dibujar nodo {i}: {e}")
        
        # 3. Forzar actualización del canvas
        self.canvas.update_idletasks()
        print(f"  - Dibujados {len(self.ids)} puntos en el canvas")


