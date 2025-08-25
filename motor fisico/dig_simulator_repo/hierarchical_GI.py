#!/usr/bin/env python3
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np

# Cargar datos
df = pd.read_csv('results.csv')

# Reemplazar errores cero con un valor pequeño (1e-6)
GI = df['G_I'].values
GI_err = np.where(df['G_I_err'] == 0, 1e-6, df['G_I_err'].values)

with pm.Model() as model:
    # Hiperparámetros
    mu = pm.Normal('mu', mu=GI[GI > 0].mean(), sigma=50)
    sigma = pm.HalfNormal('sigma', sigma=20)
    
    # Efectos aleatorios
    GI_true = pm.Normal('GI_true', mu=mu, sigma=sigma, shape=len(GI))
    
    # Verosimilitud
    obs = pm.Normal('obs', 
                   mu=GI_true, 
                   sigma=GI_err, 
                   observed=GI)
    
    # Muestreo
    idata = pm.sample(
        draws=2000,
        tune=1000,
        target_accept=0.9,
        return_inferencedata=True,
        random_seed=42,
        progressbar=True
    )
print(az.summary(idata, var_names=['mu','sigma'], hdi_prob=0.95))
idata.to_netcdf('hierarchical_GI_trace.nc')
