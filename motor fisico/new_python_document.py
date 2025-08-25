import pandas as pd
import numpy as np
import os

# --------------------------
# 1. Función para calcular GI
# --------------------------
def calcular_gi(masa, radio, entropia, informacion):
    G = 6.67430e-11  # Constante de gravitación universal
    if entropia + informacion == 0:
        return 0  # Evita división por cero
    gi = (G * masa / radio**2) * (informacion / (entropia + 1e-5))
    return gi

# --------------------------
# 2. Crear datos de galaxias si no existen
# --------------------------
csv_entrada = "galaxias_reales.csv"

if not os.path.exists(csv_entrada):
    print("⚠️ No se encontró el archivo. Generando datos simulados.")
    num = 100
    df = pd.DataFrame({
        'nombre': [f"Galaxia_{i+1}" for i in range(num)],
        'masa': np.random.uniform(1e10, 1e12, num),
        'radio': np.random.uniform(1e3, 1e5, num),
        'entropia': np.random.uniform(1.0, 10.0, num),
        'informacion': np.random.uniform(10.0, 100.0, num)
    })
else:
    print(f"✅ Cargando datos desde: {csv_entrada}")
    df = pd.read_csv(csv_entrada)

# --------------------------
# 3. Cálculo de GI
# --------------------------
df['gi'] = df.apply(lambda row: calcular_gi(row['masa'], row['radio'], row['entropia'], row['informacion']), axis=1)

# --------------------------
# 4. Mostrar resumen
# --------------------------
print("\nResumen de datos:")
print(f"Número de galaxias: {len(df)}")
print(f"Constante GI promedio: {df['gi'].mean():.2f}")
print(f"Masa promedio: {df['masa'].mean():.2e} masas solares")
print(f"Entropía promedio: {df['entropia'].mean():.2f}")
print(f"Información promedio: {df['informacion'].mean():.2f}")

# --------------------------
# 5. Guardar resultados
# --------------------------
csv_salida = "resultado_galaxias.csv"
df.to_csv(csv_salida, index=False)
print(f"\n📁 Resultados guardados en: {csv_salida}")
