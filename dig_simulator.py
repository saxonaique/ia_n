import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def simular_universo_dig(resolucion=200, rho_c=1.0, Rs=20.0, k_c2=1.0, delta_q=0.1, num_frames=60):
    """
    Simula la evolución del campo DIG a lo largo del tiempo.
    
    Args:
        resolucion: Tamaño de la cuadrícula (resolucion x resolucion)
        rho_c: Densidad central
        Rs: Radio característico
        k_c2: Constante de proporcionalidad
        delta_q: Parámetro cuántico
        num_frames: Número de frames a generar
        
    Returns:
        Un array 3D con la evolución temporal del campo
    """
    # Generar la cuadrícula inicial
    x = np.linspace(-Rs, Rs, resolucion)
    y = np.linspace(-Rs, Rs, resolucion)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Condición inicial: perfil gaussiano
    rho = rho_c * np.exp(- (R**2) / (Rs**2))
    
    # Almacenar todos los frames
    frames = []
    
    for _ in range(num_frames):
        # Calcular la derivada espacial
        drho_dr = (-2 * R / (Rs**2 + 1e-8)) * rho
        
        # Término cuántico
        delta_quantum = delta_q * rho**2
        
        # Evolución temporal (ecuación diferencial)
        drho_dt = k_c2 * (drho_dr / (rho + delta_quantum + 1e-8))
        
        # Actualizar el campo (método de Euler explícito)
        dt = 0.1  # Paso de tiempo
        rho = rho + dt * drho_dt
        
        # Añadir el frame actual
        frames.append(rho.copy())
    
    # Convertir a array 3D (time, y, x)
    return np.array(frames)

def exportar_campo_tiempo(campo, ruta="recursos/campo_dig_temp.png"):
    normalizado = (255 * (campo - np.min(campo)) / (np.ptp(campo))).astype(np.uint8)
    img = Image.fromarray(normalizado)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    img.save(ruta)
    return ruta
