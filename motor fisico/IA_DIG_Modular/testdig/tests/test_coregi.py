# tests/test_coregi.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from core_nucleus import CoreNucleus  # Tu archivo core_nucleus.py

def test_coregi(iterations: int = 20, field_shape=(64, 64)):
    """
    Prueba del CoreNucleus con Gi estabilizado.
    Genera campos aleatorios y calcula métricas por iteración.
    """
    core = CoreNucleus(field_shape=field_shape)
    
    for i in range(1, iterations + 1):
        # Campo aleatorio en [0, 1]
        random_field = np.random.rand(*field_shape)
        core.receive_field(random_field)

        # Calcular métricas
        metrics = core.get_metrics()
        if hasattr(core, 'compute_Gi'):
            Gi_val = core.compute_Gi(metrics)  # Gi estabilizado
        else:
            Gi_val = 0.0  # Si no tienes Gi implementado todavía

        # Mostrar resultados por iteración
        print(f"Iter {i:02d} | "
              f"S={metrics['entropía']:.3f} | "
              f"V={metrics['varianza']:.3f} | "
              f"M={metrics['máximo']:.3f} | "
              f"Gi={Gi_val:.3f}")

if __name__ == "__main__":
    test_coregi(iterations=50)
