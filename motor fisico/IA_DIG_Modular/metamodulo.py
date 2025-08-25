import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple

# Importar todos los módulos de tu IA DIG
from sensor_module import SensorModule # Dependencia: sensor_module.py requiere 'soundfile'. Si obtienes un ModuleNotFoundError, instala con: pip install soundfile
from core_nucleus import CoreNucleus
from memory_module import MemoriaAtractores # Asumiendo que MemoryModule = MemoriaAtractores
from evolution_processor import EvolutionProcessor
from action_module import ModuloAccion # Asumiendo que ActionModule = ModuloAccion
from ia_interpreter import interpretar_metrica # Una función auxiliar para MetaLayer


class Metamodulo:
    """
    Metamódulo: Orquestador principal de la IA DIG.
    Coordina la interacción entre todos los módulos, monitorea el estado
    y toma decisiones estratégicas.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el Metamódulo y todos los submódulos de la IA DIG.

        Args:
            config: Configuración general para el Metamodulo y los submódulos.
        """
        self.config = config or self._default_config()

        # Instanciar submódulos
        # Las configuraciones específicas de cada módulo se pasan desde la config general
        self.sensorium = SensorModule(self.config.get('sensor_module'))
        self.core_nucleus = CoreNucleus(self.config.get('core_nucleus'))
        self.memoria = MemoriaAtractores(self.config.get('memory_module'))
        self.evolution_processor = EvolutionProcessor(self.memoria) # EvolutionProcessor necesita la memoria
        self.action_module = ModuloAccion(self.config.get('action_module'))

        # Estado interno del Metamodulo
        self.current_cycle = 0
        self.last_global_decision = None
        self.log_history = [] # Para registrar eventos clave del Metamodulo

        print("Metamodulo: Todos los módulos DIG inicializados.")

    def _default_config(self) -> Dict[str, Any]:
        """Define una configuración por defecto para todos los módulos."""
        return {
            'metamodule': {
                'entropy_threshold_act': 0.001, # Umbral muy bajo para forzar la acción
                'max_cycles': 100,
                'log_level': 'INFO'
            },
            'sensor_module': {
                'image': {'resize': (64, 64), 'thresholds': (0.3, 0.7)},
                'audio': {'sample_rate': 22050, 'n_fft': 512},
                'text': {
                    'max_length': 256,
                    'embedding_dim': 128,
                    'target_field_size': (64, 64) # Aseguramos un tamaño 2D para texto
                }
            },
            'core_nucleus': {
                'entropy_window': 3,
                'equilibrium_threshold': 0.05,
                'max_history': 10,
                'reorganization_iterations': 15, # <-- NUEVO
                'reorganization_alpha': 0.15, # <-- NUEVO
                'reorganization_noise_factor': 0.03, # <-- NUEVO
                'reorganization_stability_threshold': 0.1 # <-- NUEVO
            },
            'memory_module': {
                'similarity_threshold': 0.85,
                'max_attractors': 500,
                'storage_path': 'data/memory_data.json' # Ruta de almacenamiento para la memoria
            },
            'action_module': {
                'output_dir': 'outputs',
                'image': {'default_size': (256, 256)}, # Ajustado para la visualización inicial
                'feedback': {
                    'enabled': True, 
                    'log_file': 'feedback.log', 
                    'file_path': 'data/feedback.json',
                    'max_entries': 1000 
                },
                'text': {
                    'max_length': 1000,
                    'language': 'es'
                }
            },
            'evolution_processor': {} # Puede tener su propia configuración si es necesario
        }

    def _log(self, message: str, level: str = 'INFO') -> None:
        """Registra un mensaje en el historial del Metamodulo."""
        log_entry = {'timestamp': time.time(), 'cycle': self.current_cycle, 'level': level, 'message': message}
        self.log_history.append(log_entry)
        if self.config['metamodule'].get('log_level') == 'INFO' or level == 'ERROR':
            print(f"[Metamodulo - Ciclo {self.current_cycle}] {level}: {message}")

    def process_cycle(self, raw_input: Any, input_type: str = 'auto') -> Dict[str, Any]:
        """
        Ejecuta un ciclo completo de procesamiento de la IA DIG.

        Args:
            raw_input: Datos brutos de entrada (ej. ruta de archivo, texto).
            input_type: Tipo de los datos de entrada ('image', 'audio', 'text', 'auto').

        Returns:
            Dict: Un resumen del estado y las decisiones del ciclo.
        """
        self.current_cycle += 1
        self._log(f"Iniciando ciclo con entrada tipo: {input_type}")

        cycle_summary = {
            'cycle': self.current_cycle,
            'input_type': input_type,
            'initial_entropy': None,
            'metamodule_decision': 'unknown',
            'applied_attractors': [],
            'reorganized_field_metrics': None,
            'action_output': None,
            'action_feedback_stats': None
        }

        try:
            # 1. Sensores: Capturar y transducir la entrada
            self.sensorium.load_input(raw_input, input_type)
            current_field = self.sensorium.get_ternary_field()
            self._log(f"Sensorium: Campo cargado con forma {current_field.shape}")

            self.core_nucleus.receive_field(current_field)
            global_entropy, entropy_map = self.core_nucleus.compute_entropy()
            cycle_summary['initial_entropy'] = global_entropy
            self._log(f"Núcleo: Entropía inicial calculada: {global_entropy:.3f}")

            # 2. Metamódulo: Decisión global (Actuar/Esperar/Ignorar)
            if global_entropy < self.config['metamodule']['entropy_threshold_act']:
                self.last_global_decision = 'ignore'
                cycle_summary['metamodule_decision'] = 'ignore'
                self._log("Metamodulo: Campo en equilibrio suficiente. No se requiere intervención activa.")
                final_field = current_field
            else:
                self.last_global_decision = 'act'
                cycle_summary['metamodule_decision'] = 'act'
                self._log("Metamodulo: Entropía alta. Preparando intervención.")

                # 3. Memoria & Procesador Evolutivo: Encontrar y aplicar atractores
                similar_attractor, similarity_score = self.memoria.find_similar(current_field)
                
                if similar_attractor is not None and similar_attractor.shape == current_field.shape and similarity_score >= self.config['memory_module']['similarity_threshold']:
                    self._log(f"Memoria: Atractor similar encontrado (score: {similarity_score:.2f}). Guiando reorganización.")
                    cycle_summary['applied_attractors'].append('Atractor Similar')
                    reorganized_field = self.core_nucleus.reorganize_field(similar_attractor)
                else:
                    self._log("Memoria: Sin atractores útiles. Iniciando proceso evolutivo...")
                    self.evolution_processor.receive_field(current_field)
                    evolved_field = self.evolution_processor.evolve() 
                    
                    self.core_nucleus.receive_field(evolved_field)
                    evolved_entropy, _ = self.core_nucleus.compute_entropy()
                    
                    if evolved_entropy < global_entropy:
                        attractor_id = self.memoria.store_attractor(evolved_field, metadata={'source': 'evolution', 'initial_entropy': global_entropy, 'final_entropy': evolved_entropy})
                        self._log(f"Procesador Evolutivo: Nuevo atractor almacenado (ID: {attractor_id}).")
                        cycle_summary['applied_attractors'].append('Evolucionado y Almacenado')
                        reorganized_field = evolved_field
                    else:
                        self._log("Procesador Evolutivo: Evolución no efectiva. Usando reorganización local con influencia de memoria.")
                        reorganized_field = self.core_nucleus.reorganize_field()
                        cycle_summary['applied_attractors'].append('Reorganización Local con Memoria') # Etiquetar la acción

                final_field = reorganized_field

            # --- Recopilar métricas e interpretación después de la decisión ---
            self.core_nucleus.receive_field(final_field)
            final_entropy, _ = self.core_nucleus.compute_entropy()
            final_variance = getattr(self.core_nucleus, 'get_variance', lambda: 0.0)() 
            final_maximo = getattr(self.core_nucleus, 'get_max_value', lambda: 0.0)() 
            entropy_change_pct = (final_entropy - global_entropy) / global_entropy * 100 if global_entropy != 0 else 0.0
            
            # Contar celdas activas, inhibidas y neutras
            active_cells = np.sum(final_field == 1)
            inhibited_cells = np.sum(final_field == -1)
            neutral_cells = np.sum(final_field == 0)
            total_cells = final_field.size

            active_ratio = active_cells / total_cells
            inhibited_ratio = inhibited_cells / total_cells
            neutral_ratio = neutral_cells / total_cells

            symmetry = np.mean(final_field == np.fliplr(final_field)) 


            reorganized_metrics = { 
                'entropía': final_entropy,
                'varianza': final_variance,
                'máximo': final_maximo,
                'entropy_change_pct': entropy_change_pct, 
                'entropy_gradient': np.mean(self.core_nucleus.get_entropy_gradient()), 
                'symmetry': symmetry,
                'active_cells': active_cells,
                'inhibited_cells': inhibited_cells,
                'neutral_cells': neutral_cells,
                'active_ratio': active_ratio,
                'inhibited_ratio': inhibited_ratio,
                'neutral_ratio': neutral_ratio,
                'decision': cycle_summary['metamodule_decision'], 
                'applied_attractors': cycle_summary['applied_attractors'] 
            }
            cycle_summary['reorganized_field_metrics'] = reorganized_metrics 

            log_message_metrics = (
                f"Entropía final: {reorganized_metrics.get('entropía'):.3f} (Δ{reorganized_metrics.get('entropy_change_pct'):.1f}%), "
                f"Varianza: {reorganized_metrics.get('varianza'):.3f}, "
                f"Patrón: {active_cells}A/{inhibited_cells}I/{neutral_cells}N"
            )
            self._log(f"Núcleo: {log_message_metrics}")

            interpretacion_ia = interpretar_metrica(reorganized_metrics)
            cycle_summary['ia_interpretation'] = interpretacion_ia 
            self._log(f"Interpretación IA: {interpretacion_ia}")
            # --- FIN ACTUALIZACIÓN ---


            # 5. Módulo de Acción: Generar salida
            action_output = self.action_module.generate_output(
                final_field,
                output_type='text',
                message=f"Campo procesado en ciclo {self.current_cycle}"
            )
            cycle_summary['action_output'] = action_output
            self._log(f"Acción: Salida generada (tipo texto).")

            # 6. Módulo de Acción (Feedback): Obtener estadísticas
            feedback_stats = self.action_module.get_feedback_stats()
            cycle_summary['action_feedback_stats'] = feedback_stats
            self._log(f"Acción: Estadísticas de feedback: {feedback_stats['success_rate']:.2f} éxito.")
            

        except Exception as e:
            self._log(f"Error durante el ciclo: {e}", level='ERROR')
            cycle_summary['error'] = str(e)
            self.last_global_decision = 'error'

        return cycle_summary

    def run_dig_system(self, input_sources: List[Tuple[Any, str]], max_cycles: Optional[int] = None) -> None:
        """
        Ejecuta el sistema DIG a través de múltiples ciclos.

        Args:
            input_sources: Lista de tuplas (raw_input, input_type) para cada ciclo.
            max_cycles: Número máximo de ciclos a ejecutar.
        """
        if not input_sources:
            self._log("No se proporcionaron fuentes de entrada para la ejecución.", level='WARNING')
            return

        self._log("Iniciando ejecución del sistema DIG...")
        num_inputs = len(input_sources)
        
        for i in range(max_cycles if max_cycles else num_inputs):
            raw_input, input_type = input_sources[i % num_inputs] # Cicla a través de las entradas
            self._log(f"\n--- Ejecutando ciclo {self.current_cycle + 1} con entrada '{input_type}' ---")
            cycle_result = self.process_cycle(raw_input, input_type)
            
            if cycle_result.get('error'):
                self._log(f"Ejecución detenida debido a error en ciclo {self.current_cycle}.", level='ERROR')
                break

        self._log("Ejecución del sistema DIG finalizada.")

# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    print("Iniciando la ejecución del Metamodulo principal...")
    
    # Crear una instancia del Metamodulo
    metamodulo_instance = Metamodulo()

    # Preparar algunas entradas de ejemplo
    example_inputs = [
        ("Este es un texto de prueba para el sistema DIG. Con aaarmonía y eeeequilibrio.", "text"),
        ("Otro texto con datos repetidos, bbb y ccc.", "text"),
        ("Un campo mas equilibrado.", "text"),
    ]

    # Ejecutar el sistema DIG por algunos ciclos
    metamodulo_instance.run_dig_system(input_sources=example_inputs, max_cycles=5) 
    
    print("\nEjecución del Metamodulo principal finalizada.")
