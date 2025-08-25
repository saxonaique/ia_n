import pygame
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# --- Configuración de Pygame ---
pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de G_I en Tiempo Real")
font = pygame.font.SysFont("Arial", 18)
clock = pygame.time.Clock()

# --- Simulación de datos ---
np.random.seed(42)
N = 10  # Número de galaxias
G_I_true = 500  # Valor "real" constante
sigma_obs = 5  # Error observacional

# Datos simulados
G_I_obs = np.random.normal(G_I_true, sigma_obs, size=N)

# Configuración de la figura de Matplotlib
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
canvas = FigureCanvasAgg(fig)

# Variables para almacenar el progreso
mu_samples = []
sigma_samples = []

# Callback para capturar las muestras
def trace_callback(trace, draw):
    mu_samples.append(trace['mu'])
    sigma_samples.append(trace['sigma'])
    
    # Actualizar gráficos cada 10 muestras para mejor rendimiento
    if len(mu_samples) % 10 == 0:
        update_plots()
    
    # Verificar eventos de Pygame para mantener la ventana responsiva
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
    return True

def update_plots():
    # Limpiar los ejes
    ax1.clear()
    ax2.clear()
    
    # Gráfico de la evolución de mu
    ax1.plot(mu_samples, 'b-', alpha=0.5)
    ax1.axhline(y=G_I_true, color='r', linestyle='--', label='Valor real')
    ax1.set_title('Evolución de la estimación de μ (G_I)')
    ax1.set_xlabel('Número de muestra')
    ax1.set_ylabel('Valor de μ')
    ax1.legend()
    
    # Histograma de las muestras de mu
    ax2.hist(mu_samples, bins=30, density=True, alpha=0.7, color='blue')
    ax2.axvline(x=G_I_true, color='r', linestyle='--')
    ax2.set_title('Distribución de μ')
    ax2.set_xlabel('Valor de μ')
    ax2.set_ylabel('Densidad')
    
    # Ajustar el layout y renderizar
    plt.tight_layout()
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    
    # Convertir a superficie de Pygame
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.fill((30, 30, 30))
    screen.blit(surf, (0, 0))
    
    # Mostrar información adicional
    info_text = [
        f"Muestras: {len(mu_samples)}",
        f"μ actual: {mu_samples[-1]:.2f}",
        f"σ actual: {sigma_samples[-1]:.2f}",
        f"Valor real de G_I: {G_I_true}"
    ]
    
    y_offset = HEIGHT - 100
    for i, text in enumerate(info_text):
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(text_surface, (20, y_offset + i * 25))
    
    pygame.display.flip()

# --- Modelo PyMC ---
with pm.Model() as model:
    # Priors
    mu = pm.Normal("mu", mu=400, sigma=100)
    sigma = pm.HalfNormal("sigma", sigma=20)
    
    # Likelihood
    GI_true = pm.Normal("GI_true", mu=mu, sigma=sigma, shape=N)
    obs = pm.Normal("obs", mu=GI_true, sigma=sigma_obs, observed=G_I_obs)
    
    # Muestreo con callback
    trace = pm.sample(
        draws=1000,
        tune=1000,
        target_accept=0.9,
        progressbar=False,
        step=pm.NUTS(),
        callback=trace_callback
    )

# Bucle principal de Pygame
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    clock.tick(30)

pygame.quit()
plt.close()
