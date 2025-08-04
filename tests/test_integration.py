import os
os.environ['MPLBACKEND'] = 'TkAgg'  # Must be set before importing matplotlib

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Integration Test")
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a figure and axis
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create some sample data
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        
        # Plot the data
        self.line, = self.ax.plot(x, y, 'b-')
        self.ax.set_title('Sin Wave Test')
        self.ax.grid(True)
        
        # Create the canvas and add it to the frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add a button to update the plot
        self.update_btn = ttk.Button(self.main_frame, text="Update Plot", command=self.update_plot)
        self.update_btn.pack(pady=10)
        
        # Add a quit button
        self.quit_btn = ttk.Button(self.main_frame, text="Quit", command=root.quit)
        self.quit_btn.pack(pady=5)
        
        self.phase = 0
        
    def update_plot(self):
        # Update the plot with new data
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x + self.phase)
        self.phase += 0.1
        
        # Update the line data
        self.line.set_ydata(y)
        self.ax.set_ylim(-1.1, 1.1)
        
        # Redraw the canvas
        self.canvas.draw()
        print("Plot updated successfully")

if __name__ == "__main__":
    root = tk.Tk()
    app = TestApp(root)
    print("Starting application...")
    root.mainloop()
