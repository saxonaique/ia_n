import tkinter as tk
from field_canvas import FieldCanvas
from dig_system import DIGSystem

class DIGVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DIG Visualizer")

        self.system = DIGSystem()

        self.canvas = FieldCanvas(self.root, self.system)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.RIGHT, padx=10)

        self.entropy_label = tk.Label(self.info_frame, text="Entropía: N/A", font=("Arial", 14))
        self.entropy_label.pack(pady=5)

        self.max_label = tk.Label(self.info_frame, text="Máximo: N/A", font=("Arial", 14))
        self.max_label.pack(pady=5)

        self.var_label = tk.Label(self.info_frame, text="Varianza: N/A", font=("Arial", 14))
        self.var_label.pack(pady=5)

        self.start_button = tk.Button(self.info_frame, text="Iniciar", command=self.start_simulation)
        self.start_button.pack(pady=20)

        self.update_loop()

    def start_simulation(self):
        self.system.activate()

    def update_loop(self):
        self.system.update()
        self.canvas.draw_field()
        self.update_metrics()
        self.root.after(100, self.update_loop)

    def update_metrics(self):
        entropy = self.system.get_entropy()
        max_val = self.system.get_max()
        var = self.system.get_var()

        self.entropy_label.config(text=f"Entropía: {entropy:.4f}")
        self.max_label.config(text=f"Máximo: {max_val:.4f}")
        self.var_label.config(text=f"Varianza: {var:.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DIGVisualizerApp(root)
    root.mainloop()
