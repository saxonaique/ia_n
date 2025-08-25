import random

class GalaxySimulator:
    def __init__(self, canvas):
        self.canvas = canvas
        self.objects = []

    def load_simulation(self, data):
        self.clear_canvas()
        for obj in data.get("galaxy", []):
            x, y = obj.get("x", 100), obj.get("y", 100)
            size = obj.get("size", 5)
            color = obj.get("color", "white")
            self.draw_star(x, y, size, color)

    def draw_star(self, x, y, size, color):
        star = self.canvas.create_oval(x-size, y-size, x+size, y+size, fill=color)
        self.objects.append(star)

    def clear_canvas(self):
        for obj in self.objects:
            self.canvas.delete(obj)
        self.objects.clear()