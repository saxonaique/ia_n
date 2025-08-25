import tkinter as tk
from galaxy_sim import GalaxySimulator
from tfc_loader import TFCCardLoader
from gpt_soo import GPTSooDecoder

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Motor N - Simulación Galáctica Informacional")
        self.geometry("1280x720")
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Inicializa módulos
        self.simulator = GalaxySimulator(self.canvas)
        self.loader = TFCCardLoader()
        self.decoder = GPTSooDecoder()

        self.bind("<KeyPress>", self.key_control)

    def key_control(self, event):
        if event.char == 'c':
            self.simulator.clear_canvas()
        elif event.char == 'l':
            tfc = self.loader.load_card()
            if tfc:
                decoded = self.decoder.decode(tfc)
                self.simulator.load_simulation(decoded)

if __name__ == "__main__":
    app = App()
    app.mainloop()