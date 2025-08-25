import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# --- Simulación de datos ---
np.random.seed(42)
N = 10  # Número de galaxias
G_I_true = 500  # Valor "real" constante
sigma_obs = 5  # Error observacional

# Datos simulados
G_I_obs = np.random.normal(G_I_true, sigma_obs, size=N)

# Configuración de la figura
plt.ion()  # Modo interactivo
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Variables para almacenar el progreso
mu_samples = []
sigma_samples = []

def update_plots():
    # Limpiar los ejes
    ax1.clear()
    ax2.clear()
    
    # Gráfico de la evolución de mu
    if len(mu_samples) > 0:
        ax1.plot(mu_samples, 'b-', alpha=0.5)
        ax1.axhline(y=G_I_true, color='r', linestyle='--', label='Valor real')
        ax1.set_title('Evolución de la estimación de μ (G_I)')
        ax1.set_xlabel('Número de muestra')
        ax1.set_ylabel('Valor de μ')
        ax1.legend()
    
    # Histograma de las muestras de mu
    if len(mu_samples) > 10:  # Esperar a tener suficientes muestras
        ax2.hist(mu_samples, bins=30, density=True, alpha=0.7, color='blue')
        ax2.axvline(x=G_I_true, color='r', linestyle='--')
        ax2.set_title('Distribución de μ')
        ax2.set_xlabel('Valor de μ')
        ax2.set_ylabel('Densidad')
    
    # Ajustar el layout
    plt.tight_layout()
    plt.pause(0.01)  # Pausa breve para actualizar la figura

# Callback para capturar las muestras
def trace_callback(trace, draw):
    # Obtener el último valor de las cadenas
    if hasattr(trace, 'posterior'):  # Para PyMC >= 4.0
        mu_val = trace.posterior['mu'].values.ravel()[-1]
        sigma_val = trace.posterior['sigma'].values.ravel()[-1]
    else:  # Para versiones anteriores
        mu_val = trace['mu'][-1]
        sigma_val = trace['sigma'][-1]
        
    mu_samples.append(mu_val)
    sigma_samples.append(sigma_val)
    
    # Actualizar gráficos cada 10 muestras para mejor rendimiento
    if len(mu_samples) % 10 == 0:
        update_plots()
    
    return True

# --- Modelo PyMC ---
with pm.Model() as model:
    # Priors
    mu = pm.Normal("mu", mu=400, sigma=100)
    sigma = pm.HalfNormal("sigma", sigma=20)
    
    # Likelihood
    GI_true = pm.Normal("GI_true", mu=mu, sigma=sigma, shape=N)
    obs = pm.Normal("obs", mu=GI_true, sigma=sigma_obs, observed=G_I_obs)
    
    # Muestreo en dos pasos
    # 1. Primero hacer un muestreo corto para inicializar
    with model:
        trace = pm.sample(
            draws=100,
            tune=100,
            target_accept=0.9,
            progressbar=True
        )
    
    # 2. Ahora sí, el muestreo principal con el callback
    with model:
        trace = pm.sample(
            draws=1000,
            tune=1000,
            target_accept=0.9,
            progressbar=True,
            initvals={'mu': trace.posterior['mu'].mean().item(),
                     'sigma': trace.posterior['sigma'].mean().item()},
            callback=trace_callback
        )

# Mantener la figura abierta
plt.ioff()
plt.show()
