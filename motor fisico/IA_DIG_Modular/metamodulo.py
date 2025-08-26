import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple

# Importar todos los módulos de tu IA DIG
from sensor_module import SensorModule 
from core_nucleus import CoreNucleus
from memory_module import MemoriaAtractores 
from evolution_processor import EvolutionProcessor
from action_module import ModuloAccion 
from ia_interpreter import interpretar_metrica 


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
        self.sensorium = SensorModule(self.config.get('sensor_module'))
        # CoreNucleus ahora se inicializa con su field_shape, ya no recibe el config dict completo
        self.core_nucleus = CoreNucleus(field_shape=self.config['sensor_module']['text']['target_field_size'])
        self.memoria = MemoriaAtractores(self.config.get('memory_module'))
        self.evolution_processor = EvolutionProcessor(self.memoria) 
        self.action_module = ModuloAccion(self.config.get('action_module'))

        # Estado interno del Metamodulo
        self.current_cycle = 0
        self.last_global_decision = None
        self.log_history = [] 

        print("Metamodulo: Todos los módulos DIG inicializados.")

    def _default_config(self) -> Dict[str, Any]:
        """Define una configuración por defecto para todos los módulos."""
        return {
            'metamodule': {
                'entropy_threshold_act': 0.001, 
                'max_cycles': 100,
                'log_level': 'INFO'
            },
            'sensor_module': {
                'image': {'resize': (64, 64)},
                'audio': {
                    'sample_rate': 22050, 'n_fft': 512, 'hop_length': 256,
                    'n_mels': 64
                },
                'text': {
                    'max_length': 256,
                    'embedding_dim': 128,
                    'target_field_size': (64, 64), # Mantener este tamaño para consistencia
                }
            },
            'core_nucleus': { # Configuración para la transformación ternaria en Metamodulo
                'ternary_low_threshold': 0.3, # <-- CAMBIO: de 0.45 a 0.3
                'ternary_high_threshold': 0.7, # <-- CAMBIO: de 0.55 a 0.7
            },
            'memory_module': {
                'similarity_threshold': 0.85,
                'max_attractors': 500,
                'storage_path': 'data/memory_data.json' 
            },
            'action_module': {
                'output_dir': 'outputs',
                'image': {'default_size': (256, 256)}, 
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
            'evolution_processor': {} 
        }

    def _log(self, message: str, level: str = 'INFO') -> None:
        """Registra un mensaje en el historial del Metamodulo."""
        log_entry = {'timestamp': time.time(), 'cycle': self.current_cycle, 'level': level, 'message': message}
        self.log_history.append(log_entry)
        if self.config['metamodule'].get('log_level') == 'INFO' or level == 'ERROR':
            print(f"[Metamodulo - Ciclo {self.current_cycle}] {level}: {message}")

    def _map_01_to_ternary_for_metrics(self, field_01: np.ndarray) -> np.ndarray:
        """
        Mapea un campo de valores en [0, 1] a un campo ternario [-1, 0, 1]
        para el cálculo de métricas específicas del IA_Interpreter.
        """
        ternary_field = np.zeros_like(field_01, dtype=np.int8)
        low_thresh = self.config['core_nucleus']['ternary_low_threshold']
        high_thresh = self.config['core_nucleus']['ternary_high_threshold']

        ternary_field[field_01 < low_thresh] = -1 # Considerar 'caos' o inhibido
        ternary_field[field_01 > high_thresh] = 1 # Considerar 'información' o activo
        # Valores entre low_thresh y high_thresh se quedan como 0 (neutro/equilibrio)
        return ternary_field


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
            # 1. Sensores: Capturar y transducir la entrada a campo [0,1]
            self.sensorium.load_input(raw_input, input_type)
            current_field_01 = self.sensorium.get_ternary_field() # Ahora devuelve [0,1]
            self._log(f"Sensorium: Campo cargado con forma {current_field_01.shape} en rango [0,1]")

            self.core_nucleus.receive_field(current_field_01)
            initial_metrics = self.core_nucleus.get_metrics()
            global_entropy = initial_metrics['entropía']
            cycle_summary['initial_entropy'] = global_entropy
            self._log(f"Núcleo: Entropía inicial calculada: {global_entropy:.3f}")

            # 2. Analizar el estado actual del campo
            ternary_field = self._map_01_to_ternary_for_metrics(current_field_01)
            total_cells = ternary_field.size
            
            # Contar células en cada estado
            active_cells = np.sum(ternary_field == 1)
            inhibited_cells = np.sum(ternary_field == -1)
            neutral_cells = np.sum(ternary_field == 0)
            
            # Calcular proporciones
            active_ratio = active_cells / total_cells
            inhibited_ratio = inhibited_cells / total_cells
            neutral_ratio = neutral_cells / total_cells
            
            # Calcular simetría (comparando el campo con su reflejo vertical)
            symmetry = np.mean(current_field_01 == np.flipud(current_field_01))
            
            # 3. Aplicar reglas de decisión jerárquicas
            if global_entropy > 1.0:  # Entropía muy alta (caos)
                self.last_global_decision = 'inhibir'
                cycle_summary['metamodule_decision'] = 'inhibir'
                self._log(f"Metamodulo: Sobrecarga detectada (E={global_entropy:.3f}, N={neutral_ratio:.1%}). Aplicando suavizado global.")
                
                # Aplicar suavizado gaussiano fuerte para reducir la entropía
                final_field_01 = ndi.gaussian_filter(current_field_01, sigma=2.0)
                applied_attractors_names = ['Suavizado Global']
                
            elif global_entropy < 0.05 and symmetry > 0.9:  # Campo muerto
                self.last_global_decision = 'activar'
                cycle_summary['metamodule_decision'] = 'activar'
                self._log(f"Metamodulo: Campo muerto detectado (E={global_entropy:.3f}, S={symmetry:.2f}). Inyectando ruido.")
                
                # Inyectar ruido estructurado para reactivar el campo
                noise = np.random.normal(loc=0.5, scale=0.3, size=current_field_01.shape)
                final_field_01 = np.clip(current_field_01 + noise * 0.5, 0, 1)
                applied_attractors_names = ['Inyección de Ruido']
                
            elif symmetry < 0.5:  # Baja simetría (desorganización)
                self.last_global_decision = 'reorganizar'
                cycle_summary['metamodule_decision'] = 'reorganizar'
                self._log(f"Metamodulo: Baja simetría detectada (S={symmetry:.2f}). Aplicando reorganización global.")
                
                # Aplicar reorganización global para mejorar la estructura
                temp_field = ndi.median_filter(current_field_01, size=3)
                final_field_01 = ndi.gaussian_filter(temp_field, sigma=0.5)
                applied_attractors_names = ['Reorganización Global']
                
            else:  # Estado normal
                self.last_global_decision = 'act'
                cycle_summary['metamodule_decision'] = 'act'
                self._log(f"Metamodulo: Estado normal (E={global_entropy:.3f}, S={symmetry:.2f}, A/I/N={active_ratio:.1%}/{inhibited_ratio:.1%}/{neutral_ratio:.1%}). Aplicando reorganización local con memoria.")
                
                # Aplicar reorganización local con memoria
                final_field_01 = self.core_nucleus.reorganize_field()
                applied_attractors_names = ['Reorganización Local con Memoria']
            
            # Registrar atractores aplicados
            cycle_summary['applied_attractors'].extend(applied_attractors_names)
            

            # --- Recopilar métricas e interpretación después de la decisión ---
            self.core_nucleus.receive_field(final_field_01) # Asegurarse que el CoreNucleus tiene el campo final
            
            final_metrics_01 = self.core_nucleus.get_metrics()
            final_entropy = final_metrics_01['entropía']
            final_variance = final_metrics_01['varianza']
            final_maximo = final_metrics_01['máximo']
            entropy_change_pct = (final_entropy - global_entropy) / global_entropy * 100 if global_entropy != 0 else 0.0
            
            # --- Mapear campo [0,1] a ternario [-1,0,1] para IA_Interpreter ---
            ternary_field_for_interpretation = self._map_01_to_ternary_for_metrics(final_field_01)
            active_cells = np.sum(ternary_field_for_interpretation == 1)
            inhibited_cells = np.sum(ternary_field_for_interpretation == -1)
            neutral_cells = np.sum(ternary_field_for_interpretation == 0)
            total_cells = ternary_field_for_interpretation.size

            active_ratio = active_cells / total_cells
            inhibited_ratio = inhibited_cells / total_cells
            neutral_ratio = neutral_cells / total_cells

            # Asegurar que get_entropy_gradient exista en CoreNucleus
            entropy_gradient_mean = np.mean(self.core_nucleus.get_entropy_gradient()) if hasattr(self.core_nucleus, 'get_entropy_gradient') else 0.0
            symmetry = np.mean(final_field_01 == np.fliplr(final_field_01)) # Simetría sobre el campo [0,1]

            reorganized_metrics = { 
                'entropía': final_entropy,
                'varianza': final_variance,
                'máximo': final_maximo,
                'entropy_change_pct': entropy_change_pct, 
                'entropy_gradient': entropy_gradient_mean, 
                'symmetry': symmetry,
                'active_cells': active_cells,
                'inhibited_cells': inhibited_cells,
                'neutral_cells': neutral_cells,
                'active_ratio': active_ratio,
                'inhibited_ratio': inhibited_ratio,
                'neutral_ratio': neutral_ratio,
                'decision': self.last_global_decision, 
                'applied_attractors': applied_attractors_names 
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


            # 5. Módulo de Acción: Generar salida
            action_output = self.action_module.generate_output(
                final_field_01, # Pasar el campo [0,1]
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








