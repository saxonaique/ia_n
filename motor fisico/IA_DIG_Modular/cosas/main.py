import tkinter as tk
from metamodulo import Metamodulo
from main_visualizer import VisualizadorDIG

# Este será el punto de entrada principal para tu IA DIG.
# En lugar de ejecutar metamodulo.py o main_visualizer.py directamente,
# ejecutarás este archivo.

if __name__ == "__main__":
    print("Iniciando el Sistema DIG AI completo...")

    # 1. Crear la instancia del Metamodulo
    # El Metamodulo se encargará de inicializar todos los submódulos (Sensorium, CoreNucleus, etc.)
    metamodulo_instance = Metamodulo()
    print("Metamodulo y todos los submódulos inicializados.")

    # 2. Crear la interfaz gráfica del Visualizador, pasándole la instancia del Metamodulo
    app = VisualizadorDIG(metamodulo_instance)
    print("Visualizador Tkinter iniciado.")

    # 3. Iniciar el bucle principal de Tkinter
    # La interfaz gráfica ahora interactuará con el Metamodulo
    app.mainloop()

    print("Aplicación DIG AI finalizada.")

