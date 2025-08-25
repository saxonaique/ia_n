
import tkinter as tk
from tkinter import filedialog, messagebox
from tfc_loader import cargar_tarjeta
from sim_engine import ejecutar_simulacion
from gpt_soo import analizar_tarjeta

class InterfazDIG(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DIG Engine v1.0 - José María")
        self.geometry("1200x700")
        self.configure(bg="#0f0f0f")
        self.tarjeta = None

        self.crear_paneles()

    def crear_paneles(self):
        self.canvas = tk.Canvas(self, bg="black", width=900, height=700)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.panel_control = tk.Frame(self, bg="#1a1a1a", width=300)
        self.panel_control.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(self.panel_control, text="DIG Engine", fg="white", bg="#1a1a1a", font=("Arial", 18)).pack(pady=10)

        btn_cargar = tk.Button(self.panel_control, text="📥 Cargar TFC", command=self.cargar_tarjeta)
        btn_cargar.pack(pady=10)

        btn_analizar = tk.Button(self.panel_control, text="🤖 GPT-Soo Análisis", command=self.analizar)
        btn_analizar.pack(pady=10)

        btn_simular = tk.Button(self.panel_control, text="🌀 Simular Colapso", command=self.simular)
        btn_simular.pack(pady=10)

        self.log = tk.Text(self.panel_control, height=20, bg="#0f0f0f", fg="lime", font=("Courier", 9))
        self.log.pack(fill=tk.BOTH, expand=True)

    def loggear(self, mensaje):
        self.log.insert(tk.END, f"{mensaje}\n")
        self.log.see(tk.END)

    def cargar_tarjeta(self):
        archivo = filedialog.askopenfilename(filetypes=[("Tarjetas TFC", "*.json")])
        if archivo:
            self.tarjeta = cargar_tarjeta(archivo)
            self.loggear(f"Tarjeta cargada: {archivo}")

    def analizar(self):
        if self.tarjeta:
            resumen = analizar_tarjeta(self.tarjeta)
            self.loggear(f"GPT-Soo: {resumen}")
        else:
            messagebox.showwarning("Atención", "Primero carga una tarjeta TFC")

    def simular(self):
        if self.tarjeta:
            resultado = ejecutar_simulacion(self.tarjeta, self.canvas)
            self.loggear(resultado)
        else:
            messagebox.showwarning("Atención", "Primero carga una tarjeta TFC")

if __name__ == "__main__":
    app = InterfazDIG()
    app.mainloop()
