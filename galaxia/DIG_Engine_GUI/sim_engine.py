
import random

def ejecutar_simulacion(tarjeta, canvas):
    canvas.delete("all")
    tema = tarjeta.get("tema", "galaxia")
    puntos = random.randint(150, 250)

    for _ in range(puntos):
        x, y = random.randint(50, 850), random.randint(50, 650)
        color = random.choice(["#00ffcc", "#ffaa00", "#ff0066"])
        canvas.create_oval(x, y, x+4, y+4, fill=color, outline=color)

    return f"Simulación '{tema}' completada con {puntos} nodos visuales."
