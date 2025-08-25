import pandas as pd
import numpy as np
import os
import argparse
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# --------------------------
# 1. Configuración y constantes
# --------------------------
@dataclass
class Config:
    """Configuración de la simulación"""
    num_galaxias: int = 100
    input_file: str = "galaxias_reales.csv"
    output_file: str = "resultado_galaxias.csv"
    show_plots: bool = False
    verbose: bool = False

# --------------------------
# 2. Funciones de utilidad
# --------------------------
def parse_args() -> Config:
    """Parsea los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Simulador de Constante GI para Galaxias')
    parser.add_argument('-i', '--input', default='galaxias_reales.csv',
                      help='Archivo CSV de entrada con datos de galaxias')
    parser.add_argument('-o', '--output', default='resultado_galaxias.csv',
                      help='Archivo CSV de salida para los resultados')
    parser.add_argument('-n', '--num-galaxias', type=int, default=100,
                      help='Número de galaxias a generar (si no hay archivo de entrada)')
    parser.add_argument('--show-plots', action='store_true',
                      help='Mostrar gráficos (requiere matplotlib)')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Mostrar información detallada')
    
    args = parser.parse_args()
    return Config(
        num_galaxias=args.num_galaxias,
        input_file=args.input,
        output_file=args.output,
        show_plots=args.show_plots,
        verbose=args.verbose
    )

def print_header():
    """Muestra el encabezado del programa"""
    print("\n" + "="*70)
    print("SIMULADOR DE CONSTANTE GI PARA GALAXIAS".center(70))
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(70))
    print("="*70 + "\n")

def print_section(title: str):
    """Muestra un título de sección"""
    print(f"\n{' ' + title + ' ':-^70}")

# --------------------------
# 3. Cálculos principales
# --------------------------

def calcular_gi(masa: float, radio: float, entropia: float, informacion: float) -> float:
    """
    Calcula la Constante GI (Gravitación Informacional)
    
    Args:
        masa: Masa de la galaxia en masas solares
        radio: Radio de la galaxia en años luz
        entropia: Valor de entropía
        informacion: Valor de información
        
    Returns:
        Valor de la constante GI escalado para mejor legibilidad
    """
    G = 6.67430e-11  # Constante de gravitación universal
    if entropia + informacion == 0 or radio == 0:
        return 0.0
    
    # Cálculo base de GI
    gi_base = (G * masa / (radio ** 2)) * (informacion / (entropia + 1e-5))
    
    # Escalar el resultado para mejor legibilidad
    return gi_base * 1e10

def cargar_datos(config: Config) -> pd.DataFrame:
    """
    Carga los datos de galaxias desde un archivo CSV o genera datos simulados.
    
    Args:
        config: Configuración de la simulación
        
    Returns:
        DataFrame con los datos de las galaxias
    """
    # Columnas requeridas y sus rangos por defecto
    columnas = {
        'masa': (1e10, 1e12),  # masas solares
        'radio': (1e3, 1e5),    # años luz
        'entropia': (1.0, 10.0),
        'informacion': (10.0, 100.0)
    }
    
    if not os.path.exists(config.input_file):
        if config.verbose:
            print(f"⚠️ No se encontró el archivo '{config.input_file}'. Generando datos simulados.")
        
        data = {'nombre': [f"Galaxia_{i+1}" for i in range(config.num_galaxias)]}
        
        for col, (min_val, max_val) in columnas.items():
            data[col] = np.random.uniform(min_val, max_val, config.num_galaxias)
            
        return pd.DataFrame(data)
    
    else:
        if config.verbose:
            print(f"✅ Cargando datos desde: {config.input_file}")
            
        df = pd.read_csv(config.input_file)
        
        # Verificar columnas faltantes
        for col in columnas:
            if col not in df.columns:
                min_val, max_val = columnas[col]
                if config.verbose:
                    print(f"⚠️ No se encontró la columna '{col}'. Generando valores aleatorios.")
                df[col] = np.random.uniform(min_val, max_val, len(df))
        
        return df

def analizar_datos(df: pd.DataFrame, config: Config) -> Dict:
    """
    Realiza el análisis de los datos y cálculos de GI.
    
    Args:
        df: DataFrame con los datos de las galaxias
        config: Configuración de la simulación
        
    Returns:
        Diccionario con los resultados del análisis
    """
    if config.verbose:
        print_section("PROCESANDO DATOS")
        print(f"📊 Analizando {len(df)} galaxias...")
    
    # Calcular GI para cada galaxia
    df['gi'] = df.apply(
        lambda row: calcular_gi(
            row['masa'], 
            row['radio'], 
            row['entropia'], 
            row['informacion']
        ), 
        axis=1
    )
    
    # Ordenar por GI descendente
    df = df.sort_values('gi', ascending=False).reset_index(drop=True)
    
    # Calcular estadísticas
    stats = {
        'total_galaxias': len(df),
        'gi_min': df['gi'].min(),
        'gi_max': df['gi'].max(),
        'gi_mean': df['gi'].mean(),
        'gi_std': df['gi'].std(),
        'top_galaxias': df.head(3).to_dict('records'),
        'bottom_galaxias': df.tail(3).to_dict('records')
    }
    
    return df, stats

def mostrar_resultados(stats: Dict, df: pd.DataFrame, config: Config):
    """Muestra los resultados del análisis"""
    print_section("RESULTADOS")
    
    # Estadísticas generales
    print("📊 ESTADÍSTICAS GENERALES")
    print(f"• Total de galaxias analizadas: {stats['total_galaxias']:,}")
    print(f"• Rango de GI: {stats['gi_min']:,.2f} - {stats['gi_max']:,.2f}")
    print(f"• GI promedio: {stats['gi_mean']:,.2f} ± {stats['gi_std']:,.2f} (desv. estándar)")
    
    # Top 3 galaxias
    print("\n🏆 TOP 3 GALAXIAS (MAYOR GI)")
    for i, gal in enumerate(stats['top_galaxias'], 1):
        print(f"{i}. {gal['nombre']}:")
        print(f"   • GI: {gal['gi']:,.2f}")
        print(f"   • Masa: {gal['masa']:,.2e} masas solares")
        print(f"   • Radio: {gal['radio']:,.2f} años luz")
        print(f"   • Entropía: {gal['entropia']:.2f}")
        print(f"   • Información: {gal['informacion']:.2f}")
    
    # Otras estadísticas
    print("\n📈 DISTRIBUCIÓN DE GI")
    print(f"• Galaxias con GI > 1,000: {len(df[df['gi'] > 1000]):,}")
    print(f"• Galaxias con GI > 10,000: {len(df[df['gi'] > 10000]):,}")
    print(f"• Galaxias con GI > 100,000: {len(df[df['gi'] > 100000]):,}")

def guardar_resultados(df: pd.DataFrame, config: Config) -> bool:
    """Guarda los resultados en un archivo CSV"""
    try:
        df.to_csv(config.output_file, index=False, float_format='%.6f')
        if config.verbose:
            print(f"\n💾 Resultados guardados en: {config.output_file}")
            print(f"   - Total de registros: {len(df)}")
            print(f"   - Columnas: {', '.join(df.columns)}")
        return True
    except Exception as e:
        print(f"\n❌ Error al guardar los resultados: {e}")
        return False

def main():
    """Función principal del programa"""
    # Configuración
    config = parse_args()
    
    if config.verbose:
        print_header()
    
    try:
        # Cargar o generar datos
        df = cargar_datos(config)
        
        # Realizar análisis
        df, stats = analizar_datos(df, config)
        
        # Mostrar resultados
        mostrar_resultados(stats, df, config)
        
        # Guardar resultados
        if guardar_resultados(df, config):
            print("\n✨ Análisis completado exitosamente! ✨")
            
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        if not config.verbose:
            print("Ejecuta con -v para ver más detalles.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


