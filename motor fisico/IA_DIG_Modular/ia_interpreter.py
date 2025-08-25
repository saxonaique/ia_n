"""
Módulo de interpretación de métricas para el sistema DIG.

Este módulo analiza las métricas del campo informacional y proporciona
interpretaciones cualitativas del estado del sistema.
"""

from typing import Dict, Any, Tuple
import numpy as np

def _evaluate_entropy_pattern(entropy: float, entropy_change: float) -> Tuple[str, str]:
    """Evalúa el patrón de entropía y su tendencia."""
    # Evaluar nivel de entropía
    if entropy > 0.1:
        level = "muy alta"
        implication = "El sistema muestra alta complejidad y desorden"
    elif entropy > 0.05:
        level = "alta"
        implication = "El sistema está en un estado de transición o cambio"
    elif entropy > 0.02:
        level = "moderada"
        implication = "El sistema muestra actividad equilibrada"
    elif entropy > 0.005:
        level = "baja"
        implication = "El sistema está en un estado ordenado"
    else:
        level = "muy baja"
        implication = "El sistema está en un estado muy ordenado o estancado"
    
    # Evaluar tendencia
    if abs(entropy_change) < 1.0:
        trend = "estable"
    elif entropy_change > 0:
        trend = f"aumentando (+{entropy_change:.1f}%)"
    else:
        trend = f"disminuyendo ({entropy_change:+.1f}%)"
    
    return level, trend, implication

def _evaluate_pattern_distribution(metrics: Dict[str, Any]) -> str:
    """Evalúa la distribución de patrones en el campo."""
    total = metrics.get('active_cells', 0) + metrics.get('inhibited_cells', 0) + metrics.get('neutral_cells', 0)
    if total == 0:
        return "Sin actividad detectable"
    
    active_ratio = metrics.get('active_ratio', 0)
    inhibited_ratio = metrics.get('inhibited_ratio', 0)
    neutral_ratio = metrics.get('neutral_ratio', 0)
    
    if neutral_ratio > 0.8:
        return "Mayoría de celdas inactivas"
    
    patterns = []
    if active_ratio > 0.3:
        patterns.append(f"{active_ratio*100:.0f}% activas")
    if inhibited_ratio > 0.3:
        patterns.append(f"{inhibited_ratio*100:.0f}% inhibidas")
    
    if not patterns:
        return "Patrón mixto equilibrado"
    
    return " y ".join(patterns)

def _evaluate_symmetry(symmetry_score: float) -> str:
    """Evalúa el nivel de simetría del patrón."""
    if symmetry_score > 0.9:
        return "Simetría casi perfecta"
    elif symmetry_score > 0.7:
        return "Alta simetría"
    elif symmetry_score > 0.5:
        return "Simetría moderada"
    elif symmetry_score > 0.3:
        return "Baja simetría"
    return "Patrón asimétrico"

def interpretar_metrica(metrics: Dict[str, Any]) -> str:
    """
    Interpreta las métricas del campo informacional para proporcionar un análisis cualitativo.
    
    Args:
        metrics: Diccionario con las métricas del campo informacional
        
    Returns:
        str: Análisis cualitativo del estado del sistema
    """
    # Obtener métricas con valores por defecto seguros
    entropy = metrics.get("entropía", 0.0)
    entropy_change = metrics.get("entropy_change_pct", 0.0)
    variance = metrics.get("varianza", 0.0)
    symmetry = metrics.get("symmetry", 0.0)
    
    # Validar métricas
    if not all(isinstance(x, (int, float)) for x in [entropy, entropy_change, variance, symmetry]):
        return "Error: Datos de métricas incompletos o no numéricos."
    
    # Evaluar componentes individuales
    entropy_level, entropy_trend, entropy_implication = _evaluate_entropy_pattern(entropy, entropy_change)
    pattern_dist = _evaluate_pattern_distribution(metrics)
    symmetry_desc = _evaluate_symmetry(symmetry)
    
    # Construir análisis
    analysis = [
        f"Análisis del sistema:",
        f"- Entropía {entropy_level} ({entropy:.4f}), {entropy_trend}",
        f"- {entropy_implication}",
        f"- {pattern_dist}",
        f"- {symmetry_desc}"
    ]
    
    # Añadir interpretación de decisiones si está disponible
    decisions = metrics.get('applied_attractors', [])
    if decisions:
        last_decision = decisions[-1] if isinstance(decisions, list) else decisions
        analysis.append(f"- Última acción: {last_decision}")
    
    # Añadir recomendación basada en el estado
    if entropy < 0.01 and abs(entropy_change) < 1.0:
        analysis.append("Recomendación: El sistema está muy estable. Considere introducir estímulos.")
    elif entropy > 0.1 and entropy_change > 5.0:
        analysis.append("Recomendación: El sistema está en alto desorden. Considere aplicar atractores conocidos.")
    
    return "\n".join(analysis)




