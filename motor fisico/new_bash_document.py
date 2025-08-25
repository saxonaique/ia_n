import pandas as pd
import numpy as np
from simulador_gi import calcular_gi  # Asume que ya tienes esta función en un módulo aparte

# Paso 1: Cargar catálogo local

def cargar_catalogo_local(ruta_csv="galaxias_reales.csv"):
    df = pd.read_csv(ruta_csv)
    print(f"Se cargaron {len(df)} galaxias reales.")
    return df

# Paso 2: Formatear columnas

def formatear_para_simulacion(df):
    df = df.rename(columns={
        "mass": "masa",
        "entropy": "entropia",
        "information": "informacion",
        "name": "nombre"
    })
    df["masa"] = df["masa"].fillna(df["masa"].mean())
    df["entropia"] = df["entropia"].fillna(5.0)
    df["informacion"] = df["informacion"].fillna(50.0)
    return df[["nombre", "masa", "entropia", "informacion"]]

# Paso 3: Ejecutar simulación GI

def ejecutar_simulacion(df):
    resultados = []
    for _, fila in df.iterrows():
        gi = calcular_gi(fila["masa"], fila["entropia"], fila["informacion"])
        resultados.append({
            "nombre": fila["nombre"],
            "masa": fila["masa"],
            "entropia": fila["entropia"],
            "informacion": fila["informacion"],
            "gi": gi
        })
    return pd.DataFrame(resultados)

# Paso 4: Guardar resultados

def guardar_resultados(df, salida="resultados_simulacion.csv"):
    df.to_csv(salida, index=False)
    print(f"Resultados guardados en {salida}")

# Paso 5: Ejecutar todo en cadena

def main():
    df = cargar_catalogo_local()
    df = formatear_para_simulacion(df)
    df_resultados = ejecutar_simulacion(df)
    guardar_resultados(df_resultados)

if __name__ == "__main__":
    main()
