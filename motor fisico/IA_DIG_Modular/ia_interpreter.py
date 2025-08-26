# ia_interpreter.py

from typing import Dict, Any
import numpy as np 

def interpretar_metrica(m: Dict[str, Any]) -> str:
    """
    Interpreta las métricas extendidas del campo informacional para dar un estado cualitativo.
    Distingue entre diferentes estados del campo: campo muerto, sobrecarga y equilibrio dinámico.
    """
    # Obtener métricas básicas
    e = float(m.get("entropía", 0.0)) 
    v = float(m.get("varianza", 0.0)) 
    mx = float(m.get("máximo", 0.0)) 
    
    # Métricas adicionales
    entropy_change_pct = float(m.get("entropy_change_pct", 0.0))
    symmetry = float(m.get("symmetry", 0.0))
    
    # Conteo de celdas
    active_cells = int(m.get("active_cells", 0))
    inhibited_cells = int(m.get("inhibited_cells", 0))
    neutral_cells = int(m.get("neutral_cells", 0))
    
    total_cells = active_cells + inhibited_cells + neutral_cells 
    if total_cells == 0: total_cells = 1 

    decision_context = m.get("decision", "Desconocida")
    applied_attractors_context = ", ".join(m.get("applied_attractors", ["Ninguno"]))

    # Validación de datos
    if not all(isinstance(x, (int, float)) for x in [e, v, mx, entropy_change_pct, symmetry, active_cells, inhibited_cells, neutral_cells]):
        return "IA: Datos de métricas incompletos o no numéricos para una interpretación profunda."

    # Cálculo de ratios
    active_ratio = active_cells / total_cells
    inhibited_ratio = inhibited_cells / total_cells
    neutral_ratio = neutral_cells / total_cells

    interpretacion = []
    estado_sistema = ""
    recomendacion = ""

    # 1. Determinar el estado principal del sistema
    if e < 0.05 and symmetry > 0.9:
        estado_sistema = "CAMPO MUERTO"
        interpretacion.append(f"🚨 {estado_sistema}: El sistema está en un estado de equilibrio estático con exceso de inhibición.")
        interpretacion.append("- La entropía extremadamente baja y la simetría perfecta indican un campo rígido y sin actividad.")
        recomendacion = "Se recomienda INYECCIÓN DE RUIDO para reactivar el campo."
        
    elif e > 1.0 and neutral_ratio < 0.05:
        estado_sistema = "SOBRECARGA INFORMACIONAL"
        interpretacion.append(f"⚠️ {estado_sistema}: El sistema muestra signos de hiperactividad y desorden.")
        interpretacion.append("- La alta entropía combinada con pocas celdas neutras sugiere un estado caótico.")
        recomendacion = "Se recomienda SUAVIDO GLOBAL para reducir la entropía."
        
    else:
        estado_sistema = "EQUILIBRIO DINÁMICO"
        interpretacion.append(f"✅ {estado_sistema}: El sistema mantiene un balance saludable entre orden y actividad.")
        recomendacion = "Se recomienda continuar con REORGANIZACIÓN LOCAL CON MEMORIA."

    # 2. Detalles de las métricas
    interpretacion.append("\n📊 MÉTRICAS DETALLADAS:")
    interpretacion.append(f"- Entropía: {e:.4f}")
    interpretacion.append(f"- Varianza: {v:.4f}")
    interpretacion.append(f"- Simetría: {symmetry:.2f}")
    interpretacion.append(f"- Composición: {active_cells}A / {inhibited_cells}I / {neutral_cells}N")
    interpretacion.append(f"- Ratios: A:{active_ratio:.1%} I:{inhibited_ratio:.1%} N:{neutral_ratio:.1%}")

    # 3. Análisis de composición
    interpretacion.append("\n🔍 ANÁLISIS DE COMPOSICIÓN:")
    if neutral_ratio > 0.7:
        interpretacion.append("- Predominio de celdas neutras: el campo carece de actividad significativa.")
    elif neutral_ratio < 0.05:
        interpretacion.append("- Muy pocas celdas neutras: el campo puede estar sobrecargado de información.")
    
    if active_ratio > 0.5:
        interpretacion.append("- Mayoría de celdas activas: el sistema está en un estado de alta energía.")
    elif inhibited_ratio > 0.5:
        interpretacion.append("- Mayoría de celdas inhibidas: el sistema está en un estado de baja energía.")

    # 4. Contexto de la decisión
    interpretacion.append("\n🎯 DECISIÓN DEL SISTEMA:")
    interpretacion.append(f"- Decisión actual: {decision_context.upper()}")
    interpretacion.append(f"- Atractores aplicados: {applied_attractors_context}")
    interpretacion.append(f"\n💡 RECOMENDACIÓN: {recomendacion}")

    # 5. Resumen ejecutivo
    resumen = [
        "\n📌 RESUMEN EJECUTIVO:",
        f"Estado: {estado_sistema}",
        f"Entropía: {'BAJA' if e < 0.5 else 'MODERADA' if e < 1.0 else 'ALTA'}",
        f"Simetría: {'ALTA' if symmetry > 0.7 else 'MEDIA' if symmetry > 0.4 else 'BAJA'}",
        f"Balance: {'NEUTRO' if 0.3 < neutral_ratio < 0.7 else 'ACTIVO' if active_ratio > inhibited_ratio else 'INHIBIDO'}"
    ]

    return "\n".join(interpretacion + resumen)



























