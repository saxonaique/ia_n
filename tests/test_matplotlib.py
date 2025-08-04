import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class SimplePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Matplotlib Test")
        
        # Create a frame for the plot
        frame = ttk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a figure and axis
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create some data
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        
        # Plot the data
        self.ax.plot(x, y)
        self.ax.set_title("Simple Sin Wave")
        
        # Create the canvas and add it to the frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add a quit button
        quit_button = ttk.Button(root, text="Quit", command=root.quit)
        quit_button.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimplePlotApp(root)
    root.mainloop()
