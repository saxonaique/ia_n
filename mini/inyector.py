# inyector.py

class Inyector:
    def __init__(self):
        # Diccionario básico de patrones, puedes ampliarlo
        # Cada patrón es una lista de índices de puntos a activar
        # Por ejemplo, para un campo 8x8 (índices lineales)
        self.patrones = {
            'A': [0, 2, 4, 8, 10, 12],  # ejemplo arbitrario
            'B': [1, 3, 5, 9, 11, 13],
            # Braille para la letra A: puntos 1 (índice 0)
            'braille_A': [0],
            'braille_B': [0, 1],
            # Añade aquí más patrones o contraondas
        }

    def obtener_patron(self, clave):
        return self.patrones.get(clave, [])

    def agregar_patron(self, clave, indices):
        self.patrones[clave] = indices
