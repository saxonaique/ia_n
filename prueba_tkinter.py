import tkinter as tk

# Crear la ventana principal
root = tk.Tk()
root.title("Prueba de Tkinter")

# Configurar tamaño y posición
ancho = 400
alto = 300
x = (root.winfo_screenwidth() - ancho) // 2
y = (root.winfo_screenheight() - alto) // 2
root.geometry(f"{ancho}x{alto}+{x}+{y}")

# Agregar un mensaje
mensaje = tk.Label(root, 
                  text="¡Tkinter está funcionando correctamente!\n\nVersión: " + str(root.tk.call('info', 'patchlevel')),
                  font=('Arial', 12),
                  pady=50)
mensaje.pack(expand=True)

# Iniciar el bucle principal
root.mainloop()
