import tkinter as tk
import numpy as np

class FieldCanvas(tk.Canvas):
    def __init__(self, parent, system, cell_size=5):
        self.system = system
        self.cell_size = cell_size
        rows, cols = self.system.field.shape
        width = cols * cell_size
        height = rows * cell_size

        super().__init__(parent, width=width, height=height, bg="black")

    def draw_field(self):
        self.delete("all")
        field = self.system.field

        rows, cols = field.shape
        norm = np.max(np.abs(field)) or 1.0  # Evitar división por cero

        for y in range(rows):
            for x in range(cols):
                val = field[y, x]
                color = self.value_to_color(val, norm)
                x1, y1 = x * self.cell_size, y * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def value_to_color(self, value, norm):
        scale = int((value + norm) / (2 * norm) * 255)
        scale = max(0, min(255, scale))
        return f'#{scale:02x}{scale:02x}{scale:02x}'
