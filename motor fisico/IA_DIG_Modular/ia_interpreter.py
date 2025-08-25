# ia_interpreter.py

from typing import Dict, Any
import numpy as np # Necesario para np.float64

def interpretar_metrica(m: Dict[str, Any]) -> str:
    """
    Interpreta las métricas extendidas del campo informacional para dar un estado cualitativo.
    """
    # Usar .get() con un valor por defecto para evitar KeyErrors y manejar None
    e = float(m.get("entropía", 0.0)) # Asegurarse de que sea float
    v = float(m.get("varianza", 0.0)) # Asegurarse de que sea float
    mx = float(m.get("máximo", 0.0)) # Asegurarse de que sea float
    
    # Manejar los tipos numpy.float64 y numpy.int64
    entropy_change_pct = float(m.get("entropy_change_pct", 0.0))
    symmetry = float(m.get("symmetry", 0.0))
    active_cells = int(m.get("active_cells", 0))
    inhibited_cells = int(m.get("inhibited_cells", 0))
    neutral_cells = int(m.get("neutral_cells", 0))
    
    total_cells = active_cells + inhibited_cells + neutral_cells 
    if total_cells == 0: total_cells = 1 # Evitar división por cero

    decision_context = m.get("decision", "Desconocida")
    applied_attractors_context = ", ".join(m.get("applied_attractors", ["Ninguno"]))

    # Asegurarse de que las métricas son numéricas (excepto los contextos que son strings)
    if not all(isinstance(x, (int, float)) for x in [e, v, mx, entropy_change_pct, symmetry, active_cells, inhibited_cells, neutral_cells]):
        return "IA: Datos de métricas incompletos o no numéricos para una interpretación profunda."

    interpretacion = []

    # 1. Interpretación de la Entropía y su Cambio
    interpretacion.append(f"- Entropía: {e:.4f}")
    if entropy_change_pct < -5.0: # Umbral más realista para "disminución significativa"
        interpretacion.append(f"- Disminución significativa de entropía ({entropy_change_pct:.1f}%)")
        interpretacion.append("- El sistema se está ordenando o equilibrando.")
    elif entropy_change_pct > 5.0: # Umbral más realista para "aumento significativo"
        interpretacion.append(f"- Aumento significativo de entropía (+{entropy_change_pct:.1f}%)")
        interpretacion.append("- El sistema está ganando complejidad, explorando o desordenándose.")
    elif e < 0.1 and abs(entropy_change_pct) < 1.0: # Si la entropía es baja y estable
        interpretacion.append(f"- Entropía muy baja ({e:.4f}), con poco cambio.")
        interpretacion.append("- El sistema está en un estado muy ordenado o estancado (equilibrio pasivo).")
    else:
        interpretacion.append("- Entropía moderada, cambio estable.")
        interpretacion.append("- El sistema busca equilibrio o mantiene patrones existentes (equilibrio dinámico).")

    # 2. Interpretación de la Varianza y Máximo
    interpretacion.append(f"- Varianza del campo: {v:.4f}")
    if v < 0.001 and mx < 0.1: # Para campos casi uniformes
        interpretacion.append("- El campo es muy uniforme o 'plano', sin mucha diversidad.")
    elif v > 0.5: # Alta diversidad
        interpretacion.append("- El campo tiene alta diversidad de valores y actividad.")
    else:
        interpretacion.append("- Diversidad de campo moderada con patrones emergentes.")
    
    # 3. Interpretación de la Simetría y Composición Celular
    interpretacion.append(f"- Simetría del campo: {symmetry:.2f}")
    if symmetry > 0.9:
        interpretacion.append("- El campo exhibe una alta simetría (estructurado).")
    elif symmetry < 0.4:
        interpretacion.append("- El campo es asimétrico o caótico (poca estructura).")
    else:
        interpretacion.append("- Simetría moderada, con algunos patrones visibles.")


    interpretacion.append(f"- Composición: {active_cells} Activas / {inhibited_cells} Inhibidas / {neutral_cells} Neutras.")
    active_ratio = active_cells / total_cells
    inhibited_ratio = inhibited_cells / total_cells
    neutral_ratio = neutral_cells / total_cells

    if active_ratio > 0.5:
        interpretacion.append("- Predominio de celdas activas (energía, exploración).")
    elif inhibited_ratio > 0.5:
        interpretacion.append("- Predominio de celdas inhibidas (supresión, quietud).")
    elif neutral_ratio > 0.5:
        interpretacion.append("- Predominio de celdas neutras (potencial, inactividad pasiva).")
    else:
        interpretacion.append("- Balance de celdas activas, inhibidas y neutras (dinamismo equilibrado).")


    # 4. Contexto de la Decisión del Metamódulo
    interpretacion.append(f"- Última Decisión del Metamódulo: {decision_context.upper()}")
    interpretacion.append(f"- Atractores aplicados: {applied_attractors_context}.")

    return "Análisis del sistema:\n" + "\n".join(interpretacion)






