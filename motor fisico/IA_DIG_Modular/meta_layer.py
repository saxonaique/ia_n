import time
from typing import Dict, List, Optional, Any, Deque
from collections import deque
import numpy as np
import json
import os

class Metamodulo:
    """
    Metamódulo: Capa de auto-evaluación y control de la IA DIG.
    
    Responsable de monitorear el estado interno del sistema, tomar decisiones estratégicas
    y coordinar la interacción entre los diferentes módulos.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el Metamódulo.
        
        Args:
            config: Configuración para el comportamiento del metamódulo
        """
        self.config = config or {
            'entropy_window': 10,
            'decision_thresholds': {
                'high_entropy': 0.8,
                'low_entropy': 0.2,
                'stability': 0.1,
                'confidence': 0.7
            },
            'monitoring': {
                'log_dir': 'logs',
                'save_frequency': 100,
                'verbose': True
            }
        }
        
        # Estado interno
        self.entropy_history: Deque[float] = deque(maxlen=self.config['entropy_window'])
        self.decision_history: List[Dict[str, Any]] = []
        self.state_history: List[Dict[str, Any]] = []
        self.last_decision: Optional[Dict[str, Any]] = None
        self.last_decision_time: float = time.time()
        
        # Crear directorio de logs si no existe
        os.makedirs(self.config['monitoring']['log_dir'], exist_ok=True)
    
    def monitor(self, system_state: Dict[str, Any]) -> None:
        """
        Monitorea el estado actual del sistema.
        
        Args:
            system_state: Estado actual del sistema que incluye métricas clave
        """
        entropy = system_state.get('entropy', 0.5)
        self.entropy_history.append(entropy)
        
        # Guardar estado para análisis
        state_snapshot = {
            'timestamp': time.time(),
            'entropy': entropy,
            'system_state': system_state
        }
        self.state_history.append(state_snapshot)
        
        # Guardar logs periódicamente
        if len(self.state_history) % self.config['monitoring']['save_frequency'] == 0:
            self._save_state_logs()
    
    def decide(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma una decisión basada en el estado actual del sistema.
        
        Returns:
            Dict: Decisión con tipo y metadatos
        """
        self.monitor(system_state)
        
        current_entropy = system_state.get('entropy', 0.5)
        
        # Calcular estadísticas de entropía
        if self.entropy_history:
            # Convertir a array de numpy asegurando que todos los elementos sean escalares
            entropy_array = np.array([e if np.isscalar(e) else 0 for e in self.entropy_history])
            entropy_std = float(np.std(entropy_array))
            entropy_mean = float(np.mean(entropy_array))
        else:
            entropy_std = 0.0
            entropy_mean = 0.0
        
        # Evaluar el estado
        state_eval = self._evaluate_state(current_entropy, entropy_std)
        
        # Tomar decisión
        decision = self._make_decision(state_eval, system_state)
        
        # Registrar decisión
        decision_record = {
            'timestamp': time.time(),
            'decision': decision,
            'state_evaluation': state_eval,
            'system_state': system_state
        }
        self.decision_history.append(decision_record)
        self.last_decision = decision
        self.last_decision_time = time.time()
        
        return decision
    
    def _evaluate_state(self, entropy, entropy_std: float) -> Dict[str, float]:
        """Evalúa el estado actual del sistema.
        
        Args:
            entropy: Valor de entropía (puede ser float o tupla)
            entropy_std: Desviación estándar de la entropía
            
        Returns:
            Dict con las puntuaciones de evaluación
        """
        # Si entropy es una tupla, usar el primer valor
        if isinstance(entropy, (list, tuple, np.ndarray)):
            entropy = float(entropy[0]) if len(entropy) > 0 else 0.5
            
        # Asegurar que entropy sea un número
        try:
            entropy = float(entropy)
        except (TypeError, ValueError):
            entropy = 0.5  # Valor por defecto si no se puede convertir
            
        entropy_score = 1.0 - entropy  # Invertir para que mayor sea mejor
        stability_score = 1.0 / (1.0 + entropy_std) if entropy_std > 0 else 1.0
        
        return {
            'entropy': entropy_score,
            'stability': stability_score,
            'overall': (entropy_score * 0.7 + stability_score * 0.3)
        }
    
    def _make_decision(self, state_eval: Dict[str, float], 
                      system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Toma una decisión basada en la evaluación del estado."""
        overall_score = state_eval['overall']
        entropy = 1.0 - state_eval['entropy']  # Volver a la escala original
        
        # Lógica de decisión básica
        if overall_score > 0.7:
            decision_type = 'continue'
        elif overall_score < 0.3:
            decision_type = 'intervene' if np.random.random() < 0.7 else 'reset'
        else:
            decision_type = np.random.choice(
                ['continue', 'explore', 'adjust'],
                p=[0.6, 0.25, 0.15]
            )
        
        # Ajustar basado en entropía
        if entropy > self.config['decision_thresholds']['high_entropy']:
            if decision_type == 'explore':
                decision_type = 'exploit'
        
        return {
            'type': decision_type,
            'timestamp': time.time(),
            'confidence': state_eval['overall'],
            'reasoning': self._generate_reasoning(decision_type, state_eval)
        }
    
    def _generate_reasoning(self, decision_type: str, 
                          state_eval: Dict[str, float]) -> str:
        """Genera una explicación de la decisión."""
        reasoning = {
            'continue': "Sistema estable. Continuando operación normal.",
            'explore': "Explorando nuevos patrones.",
            'exploit': "Aprovechando conocimiento existente.",
            'adjust': "Ajustando parámetros.",
            'intervene': "Intervención necesaria.",
            'reset': "Reiniciando para recuperar estabilidad."
        }
        return reasoning.get(decision_type, "Decisión basada en el estado actual.")
    
    def _save_state_logs(self) -> None:
        """Guarda los registros de estado en disco."""
        if not self.state_history:
            return
            
        timestamp = int(time.time())
        log_file = os.path.join(
            self.config['monitoring']['log_dir'],
            f'metamodule_log_{timestamp}.json'
        )
        
        try:
            with open(log_file, 'w') as f:
                json.dump({
                    'state_history': self.state_history,
                    'decision_history': self.decision_history,
                    'config': self.config
                }, f, indent=2)
        except Exception as e:
            print(f"Error al guardar logs: {e}")

# Alias para compatibilidad
MetaLayer = Metamodulo
