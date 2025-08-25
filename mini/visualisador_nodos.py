import tkinter as tk
import json

# Cargar el diccionario
with open("diccionarios/diccionario_sensorial.json", "r", encoding="utf-8") as f:
    letras = json.load(f)["alfabeto_sensorial"]

class Nodo:
    def __init__(self, x, y, rho=1.0):
        self.x = x
        self.y = y
        self.rho = rho

class Campo:
    def __init__(self):
        self.nodos = []

    def inyectar_forma(self, coords, intensidad='media'):
        escala = {"baja": 0.5, "media": 1.0, "alta": 1.5, "moderada": 0.9}.get(intensidad, 1.0)
        for px, py in coords:
            self.nodos.append(Nodo(px * 50 + 50, py * 50 + 50, rho=escala))

def visualizar_campo():
    campo = Campo()
    for letra_data in letras:
        campo.inyectar_forma(letra_data["forma_braille"], letra_data["accion"])

    root = tk.Tk()
    root.title("Visualizador Campo Braille")
    canvas = tk.Canvas(root, width=400, height=400, bg="black")
    canvas.pack()

    for nodo in campo.nodos:
        r = nodo.rho * 10
        canvas.create_oval(nodo.x - r, nodo.y - r, nodo.x + r, nodo.y + r, fill="lime", outline="white")

    root.mainloop()

if __name__ == "__main__":
    visualizar_campo()
