import tkinter as tk
from tkinter import filedialog
import json
import random

class MotorNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motor N - Simulación Informacional")
        self.root.geometry("1280x720")
        self.canvas = tk.Canvas(root, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.menu_frame = tk.Frame(root, bg='gray20')
        self.menu_frame.place(x=0, y=0, relwidth=1, height=40)

        load_button = tk.Button(self.menu_frame, text="📥 Cargar TFC", command=self.load_tfc, bg='gray30', fg='white')
        load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.nodes = []

    def load_tfc(self):
        file_path = filedialog.askopenfilename(filetypes=[("Tarjetas Funcionales Cognitivas", "*.json")])
        if file_path:
            with open(file_path, "r") as f:
                data = json.load(f)
                self.nodes = data.get("nodos", [])
                self.draw_nodes()

    def draw_nodes(self):
        self.canvas.delete("all")
        for nodo in self.nodes:
            x = nodo.get("x", random.randint(50, 1230))
            y = nodo.get("y", random.randint(50, 670))
            color = nodo.get("color", "#00FF00")
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill=color, outline="")

if __name__ == "__main__":
    root = tk.Tk()
    app = MotorNApp(root)
    root.mainloop()