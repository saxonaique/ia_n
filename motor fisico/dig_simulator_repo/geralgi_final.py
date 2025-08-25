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
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# --- Modelo PyMC ---
with pm.Model() as model:
    # Priors
    mu = pm.Normal("mu", mu=400, sigma=100)
    sigma = pm.HalfNormal("sigma", sigma=20)
    
    # Likelihood
    GI_true = pm.Normal("GI_true", mu=mu, sigma=sigma, shape=N)
    obs = pm.Normal("obs", mu=GI_true, sigma=sigma_obs, observed=G_I_obs)
    
    # Muestreo
    trace = pm.sample(
        draws=1000,
        tune=1000,
        target_accept=0.9,
        progressbar=True,
        return_inferencedata=True
    )

# Procesar resultados
df_summary = az.summary(trace, var_names=["mu", "sigma"])
mu_samples = trace.posterior['mu'].values.ravel()
sigma_samples = trace.posterior['sigma'].values.ravel()

# Crear gráficos
ax1.plot(mu_samples, 'b-', alpha=0.5)
ax1.axhline(y=G_I_true, color='r', linestyle='--', label='Valor real')
ax1.set_title('Evolución de la estimación de μ (G_I)')
ax1.set_xlabel('Número de muestra')
ax1.set_ylabel('Valor de μ')
ax1.legend()

ax2.hist(mu_samples, bins=30, density=True, alpha=0.7, color='blue')
ax2.axvline(x=G_I_true, color='r', linestyle='--')
ax2.set_title('Distribución de μ')
ax2.set_xlabel('Valor de μ')
ax2.set_ylabel('Densidad')

plt.tight_layout()

# Mostrar resultados
print("\nResumen de resultados:")
print(df_summary)
print(f"\nValor real de G_I: {G_I_true}")

# Mantener la figura abierta
plt.ioff()
plt.show()
