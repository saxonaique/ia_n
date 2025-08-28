"""
Sistema DIG (Dualidad Información-Gravedad)
Sistema de procesamiento de información basado en los principios de la dualidad
información-gravedad y la danza informacional.
"""
import os
import time
import json
from typing import Dict, Any, Optional
import numpy as np

# Importar módulos del sistema
from sensor_module import SensorModule
from core_nucleus import CoreNucleus
from memory_module import MemoryModule
from action_module import ActionModule
from meta_layer import Metamodulo

class DIGSystem:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializa el sistema DIG con todos sus módulos."""
        # Configuración por defecto
        self.config = config or {
            'system': {'field_size': (64, 64), 'max_cycles': 1000},
            'memory': {'max_attractors': 1000, 'similarity_threshold': 0.8},
            'action': {'output_dir': 'outputs'}
        }
        
        # Inicializar módulos
        self.sensor = SensorModule()
        self.core = CoreNucleus()
        self.memory = MemoryModule({
            'max_attractors': self.config['memory']['max_attractors'],
            'similarity_threshold': self.config['memory']['similarity_threshold']
        })
        self.action = ActionModule({
            'output_dir': self.config['action']['output_dir']
        })
        self.meta = Metamodulo()
        
        # Estado del sistema
        self.current_field = np.zeros(self.config['system']['field_size'])
        self.cycle_count = 0
        self.running = False
        
        # Crear directorios necesarios
        os.makedirs(self.config['action']['output_dir'], exist_ok=True)
    
    def load_input(self, source: str, input_type: str = 'auto') -> None:
        """Carga una entrada en el sistema."""
        self.sensor.load_input(source, input_type)
        self.current_field = self.sensor.translate_to_field()
    
    def process_cycle(self) -> Dict[str, Any]:
        """Ejecuta un ciclo completo de procesamiento."""
        if self.current_field is None:
            raise ValueError("No se ha cargado ningún campo inicial")
        
        # 1. Procesamiento en el núcleo entrópico
        self.core.receive_field(self.current_field)
        processed_field = self.core.reorganize_field()
        entropy = self.core.compute_entropy()
        
        # Consultar memoria
        similar_field, similarity_score = self.memory.find_similar(processed_field)
        
        # Actualizar métricas
        metrics = {
            'cycle': self.cycle_count,
            'entropy': entropy,
            'memory': {
                'similarity': similarity_score,
                'has_similar': similar_field is not None
            }
        }
        system_state = {
            'entropy': entropy,
            'cycle': self.cycle_count,
            'memory': {'similarity': similarity_score if similar_field is not None else 0.0}
        }
        decision = self.meta.decide(system_state)
        
        # 4. Aplicar transformaciones según la decisión
        if decision['type'] == 'continue':
            self.current_field = processed_field
        elif decision['type'] == 'explore':
            noise = np.random.normal(0, 0.1, self.current_field.shape)
            self.current_field = np.clip(self.current_field + noise, -1, 1)
        
        # 5. Actualizar memoria
        self.memory.store_attractor(self.current_field, {
            'cycle': self.cycle_count,
            'entropy': entropy,
            'decision': decision['type']
        })
        
        # 6. Generar salidas
        output = {}
        if self.cycle_count % 10 == 0:
            # Generar timestamp para nombres de archivo
            timestamp = int(time.time())
            
            # Generar imagen
            try:
                image_path = os.path.join(
                    self.config['action']['output_dir'],
                    f'output_{self.cycle_count:04d}_{timestamp}.png'
                )
                self.action.generate_output(
                    self.current_field,
                    output_type='image',
                    output_path=image_path,
                    format='png'
                )
                output['image'] = image_path
            except Exception as e:
                print(f"Error al generar imagen: {str(e)}")
            
            # Generar audio
            try:
                audio_path = os.path.join(
                    self.config['action']['output_dir'],
                    f'audio_{self.cycle_count:04d}_{timestamp}.wav'
                )
                audio_data = self.action.generate_output(
                    self.current_field,
                    output_type='audio',
                    sample_rate=44100,
                    duration=1.0,
                    save=True
                )
                output['audio'] = audio_path
            except Exception as e:
                print(f"Error al generar audio: {str(e)}")
            
            # Generar texto
            try:
                text = self.action.generate_output(
                    self.current_field,
                    output_type='text',
                    max_length=500,
                    save=True
                )
                output['text'] = text
            except Exception as e:
                print(f"Error al generar texto: {str(e)}")
            
            # Ejecutar acción
            try:
                action_result = self.action._execute_action(self.current_field)
                output['action'] = action_result
            except Exception as e:
                print(f"Error al ejecutar acción: {str(e)}")
        
        # Actualizar contador
        self.cycle_count += 1
        
        # Convertir entropy a float si es una tupla
        entropy_value = float(entropy[0]) if isinstance(entropy, (list, tuple)) else float(entropy)
        
        return {
            'cycle': self.cycle_count,
            'entropy': entropy_value,
            'decision': decision['type'],
            'output': output
        }
    
    def run(self, max_cycles: Optional[int] = None) -> None:
        """Ejecuta el sistema de forma continua."""
        max_cycles = max_cycles or self.config['system']['max_cycles']
        self.running = True
        
        try:
            while self.running and self.cycle_count < max_cycles:
                metrics = self.process_cycle()
                
                if self.cycle_count % 10 == 0:
                    print(f"Ciclo {self.cycle_count}: "
                          f"Entropía={metrics['entropy']:.3f}, "
                          f"Decisión={metrics['decision']}")
                
                time.sleep(0.1)  # Pequeña pausa
                
        except KeyboardInterrupt:
            print("\nDeteniendo ejecución...")
        finally:
            self.running = False
    
    @staticmethod
    def convert_numpy(obj):
        """
        Convierte objetos NumPy a tipos nativos de Python para serialización JSON.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: DIGSystem.convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DIGSystem.convert_numpy(item) for item in obj]
        return obj

    def save_state(self, filepath: str) -> None:
        """
        Guarda el estado actual del sistema en un archivo JSON.
        
        Args:
            filepath: Ruta del archivo donde se guardará el estado
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Preparar el estado para serialización
        state = {
            'config': self.config,
            'cycle_count': self.cycle_count,
            'current_field': self.current_field.tolist() if self.current_field is not None else None,
            'running': self.running
        }
        
        # Convertir cualquier objeto NumPy a tipos nativos de Python
        state = self.convert_numpy(state)
        
        # Guardar el estado en el archivo
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, path: str) -> 'DIGSystem':
        """
        Carga un estado guardado del sistema.
        
        Args:
            path: Ruta al archivo de estado guardado
            
        Returns:
            DIGSystem: Instancia del sistema con el estado cargado
        """
        with open(path, 'r') as f:
            state = json.load(f)
        
        # Crear una nueva instancia del sistema con la configuración guardada
        system = cls(state['config'])
        system.cycle_count = state['cycle_count']
        system.running = state.get('running', False)
        
        # Cargar el campo actual si existe
        if 'current_field' in state and state['current_field'] is not None:
            system.current_field = np.array(state['current_field'])
        elif 'field' in state and state['field'] is not None:  # Para compatibilidad con versiones anteriores
            system.current_field = np.array(state['field'])
        else:
            system.current_field = np.zeros(system.config['system']['field_size'])
            
        return system
        
        return system
