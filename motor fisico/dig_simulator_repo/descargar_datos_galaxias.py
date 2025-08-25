import pandas as pd
import numpy as np

def generar_datos_ejemplo(n=100):
    """Genera datos de ejemplo para simulación"""
    nombres = [f'Galaxia-{i+1}' for i in range(n)]
    masas = np.random.uniform(1e10, 1e12, n)
    entropias = np.random.uniform(0.1, 10.0, n)
    informacion = np.random.uniform(1.0, 100.0, n)
    
    return pd.DataFrame({
        'nombre': nombres,
        'masa': masas,
        'entropia': entropias,
        'informacion': informacion
    })

def cargar_catalogo(ruta_csv=None):
    """Intenta cargar datos de un archivo CSV o genera datos de ejemplo"""
    if ruta_csv:
        try:
            df = pd.read_csv(ruta_csv)
            print(f"Datos cargados de {ruta_csv}")
            return df
        except FileNotFoundError:
            print(f"Archivo {ruta_csv} no encontrado. Generando datos de ejemplo...")
    
    # Generar datos de ejemplo si no se proporciona archivo o no se encuentra
    df = generar_datos_ejemplo()
    print("Generados datos de ejemplo para 100 galaxias")
    return df

def formatear_para_simulacion(df):
    """Asegura que el DataFrame tenga el formato correcto"""
    # Renombrar columnas si es necesario
    mapeo_columnas = {
        'mass': 'masa',
        'entropy': 'entropia',
        'information': 'informacion',
        'name': 'nombre'
    }
    
    df = df.rename(columns={k: v for k, v in mapeo_columnas.items() if k in df.columns})
    
    # Asegurar que existan las columnas necesarias
    if 'nombre' not in df.columns:
        df['nombre'] = [f'Galaxia-{i+1}' for i in range(len(df))]
    
    # Rellenar valores faltantes
    if 'masa' in df.columns:
        df['masa'] = df['masa'].fillna(df['masa'].mean())
    else:
        df['masa'] = np.random.uniform(1e10, 1e12, len(df))
    
    if 'entropia' not in df.columns:
        df['entropia'] = np.random.uniform(0.1, 10.0, len(df))
    else:
        df['entropia'] = df['entropia'].fillna(5.0)
    
    if 'informacion' not in df.columns:
        df['informacion'] = np.random.uniform(1.0, 100.0, len(df))
    else:
        df['informacion'] = df['informacion'].fillna(50.0)
    
    return df[['nombre', 'masa', 'entropia', 'informacion']]

def guardar_csv_preparado(df, salida="galaxias_preparadas.csv"):
    """Guarda el DataFrame en un archivo CSV"""
    df.to_csv(salida, index=False)
    print(f"\nDatos guardados en {salida}")
    print("\nResumen de datos:")
    print(f"Número de galaxias: {len(df)}")
    print(f"Masa promedio: {df['masa'].mean():.2e} masas solares")
    print(f"Entropía promedio: {df['entropia'].mean():.2f}")
    print(f"Información promedio: {df['informacion'].mean():.2f}")

if __name__ == "__main__":
    import sys
    
    # Verificar si se proporcionó un archivo como argumento
    archivo_entrada = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Cargar o generar datos
    df = cargar_catalogo(archivo_entrada)
    
    # Formatear datos
    df = formatear_para_simulacion(df)
    
    # Guardar resultados
    guardar_csv_preparado(df)
