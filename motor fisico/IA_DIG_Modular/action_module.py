import os
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import cv2
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import os
import json

class ModuloAccion:
    """
    Módulo de Acción: Interfaz entre el núcleo de la IA DIG y el entorno externo.
    
    Responsable de transformar el campo informacional procesado en acciones concretas
    y salidas comprensibles para el entorno externo.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el Módulo de Acción.
        
        Args:
            config: Configuración para la generación de salidas
        """
        self.config = config or {
            'output_dir': 'outputs',
            'image': {
                'default_size': (512, 512),
                'colormap': cv2.COLORMAP_VIRIDIS,
                'save_format': 'png'
            },
            'audio': {
                'sample_rate': 44100,
                'channels': 1,
                'duration': 5.0,  # segundos
                'normalize': True
            },
            'text': {
                'max_length': 1000,
                'language': 'es'
            },
            'feedback': {
                'enabled': True,
                'log_file': 'feedback.log',
                'max_entries': 1000
            }
        }
        
        # Crear directorio de salida si no existe
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Historial de acciones para retroalimentación
        self.action_history: List[Dict[str, Any]] = []
        self._load_feedback()
    
    def generate_output(self, field: np.ndarray, output_type: str = 'image', **kwargs) -> Any:
        """
        Genera una salida a partir del campo informacional.
        
        Args:
            field: Campo informacional procesado
            output_type: Tipo de salida ('image', 'audio', 'text', 'action')
            **kwargs: Argumentos adicionales específicos del tipo de salida
            
        Returns:
            Depende del tipo de salida:
            - Imagen: ruta al archivo de imagen generado
            - Audio: array de audio generado
            - Texto: cadena de texto generada
            - Acción: resultado de la acción ejecutada
        """
        if not isinstance(field, np.ndarray):
            raise ValueError("El campo debe ser un array de NumPy")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = None
        metadata = {
            'timestamp': timestamp,
            'output_type': output_type,
            'field_shape': field.shape,
            'field_stats': {
                'min': float(np.min(field)) if field.size > 0 else 0,
                'max': float(np.max(field)) if field.size > 0 else 0,
                'mean': float(np.mean(field)) if field.size > 0 else 0,
                'std': float(np.std(field)) if field.size > 0 else 0
            },
            'params': kwargs
        }
        
        try:
            # Obtener configuración para el tipo de salida
            output_config = self.config.get(output_type, {})
            
            # Combinar configuración predeterminada con parámetros proporcionados
            output_params = {**output_config, **kwargs}
            
            if output_type == 'image':
                # Asegurarse de que el directorio de salida existe
                output_dir = self.config.get('output_dir', 'outputs')
                os.makedirs(output_dir, exist_ok=True)
                
                # Generar nombre de archivo si no se proporciona
                if 'output_path' not in output_params:
                    output_params['output_path'] = os.path.join(
                        output_dir,
                        f"image_{timestamp}.{output_params.get('format', 'png')}"
                    )
                
                output = self._generate_image(field, **output_params)
                metadata['output_path'] = output_params['output_path']
                
            elif output_type == 'audio':
                output = self._generate_audio(field, **output_params)
                
                # Guardar el audio si se solicita
                if output_params.get('save', True):
                    output_dir = self.config.get('output_dir', 'outputs')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    output_path = os.path.join(
                        output_dir,
                        f"audio_{timestamp}.wav"
                    )
                    self._save_audio(output, output_path, sample_rate=output_params.get('sample_rate'))
                    metadata['output_path'] = output_path
            elif output_type == 'text':
                # Obtener configuración de texto
                text_config = self.config.get('text', {})
                text_params = {**text_config, **kwargs}
                
                # Generar texto
                output = self._generate_text(field, **text_params)
                metadata['output_text'] = output
                
                # Guardar el texto si se solicita
                if text_params.get('save', True):
                    output_dir = self.config.get('output_dir', 'outputs')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    output_path = os.path.join(
                        output_dir,
                        f"text_{timestamp}.txt"
                    )
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(output)
                        metadata['output_path'] = output_path
                    except Exception as e:
                        print(f"Error al guardar el archivo de texto: {str(e)}")
                        
            elif output_type == 'action':
                # Obtener configuración de acción
                action_config = self.config.get('action', {})
                action_params = {**action_config, **kwargs}
                
                # Ejecutar acción
                output = self._execute_action(field, **action_params)
                metadata['action_result'] = str(output)
                
            else:
                raise ValueError(f"Tipo de salida no soportado: {output_type}")
                
            # Registrar la acción en el historial
            self._log_action(metadata)
            
            return output
            
        except Exception as e:
            error_msg = f"Error al generar salida de tipo '{output_type}': {str(e)}"
            print(error_msg)
            metadata.update({
                'error': str(e),
                'error_type': type(e).__name__,
                'success': False
            })
            self._log_action(metadata)
            raise
    
    def _generate_image(self, field: np.ndarray, timestamp: str = None, **kwargs) -> str:
        """
        Genera una imagen a partir del campo informacional.
        
        Args:
            field: Campo informacional 2D
            timestamp: Marca de tiempo para el nombre del archivo (opcional)
            **kwargs: Argumentos adicionales
                - output_path: Ruta completa del archivo de salida
                - size: Tamaño de salida (ancho, alto)
                - colormap: Mapa de colores a aplicar
                - format: Formato de salida (jpg, png, etc.)
                
        Returns:
            str: Ruta al archivo de imagen generado
        """
        try:
            if not isinstance(field, np.ndarray):
                raise ValueError("El campo debe ser un array de NumPy")
                
            # Obtener ruta de salida o generar una
            output_path = kwargs.get('output_path')
            if not output_path:
                if not timestamp:
                    import time
                    timestamp = str(int(time.time()))
                
                # Obtener configuración de imagen con valores por defecto
                image_config = self.config.get('image', {})
                output_format = kwargs.get('format', image_config.get('save_format', 'png'))
                output_dir = self.config.get('output_dir', 'outputs')
                
                # Crear directorio de salida si no existe
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = os.path.join(
                    output_dir,
                    f'image_{timestamp}.{output_format}'
                )
                
            # Asegurarse de que el campo sea 2D
            if len(field.shape) == 1:
                # Si es 1D, crear una imagen cuadrada
                size = int(np.ceil(np.sqrt(len(field))))
                # Rellenar con ceros si es necesario
                padded = np.pad(field, (0, size*size - len(field)), 'constant')
                field = padded.reshape((size, size))
            elif len(field.shape) == 3:
                # Si es 3D, tomar el canal de mayor intensidad
                if field.shape[2] == 1:
                    field = field[:, :, 0]
                else:
                    # Tomar el canal con mayor varianza
                    channel_var = np.var(field, axis=(0, 1))
                    channel = np.argmax(channel_var)
                    field = field[:, :, channel]
            
            # Normalizar el campo a [0, 1]
            min_val, max_val = np.min(field), np.max(field)
            if np.isclose(min_val, max_val):
                normalized = np.zeros_like(field, dtype=np.float32)
            else:
                normalized = (field - min_val) / (max_val - min_val)
            
            # Convertir a 8-bit [0, 255]
            normalized_8bit = (normalized * 255).astype(np.uint8)
            
            # Aplicar mapa de colores si es necesario
            if kwargs.get('apply_colormap', True):
                colormap = kwargs.get('colormap', self.config['image']['colormap'])
                if isinstance(colormap, str):
                    colormap = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_VIRIDIS)
                colored = cv2.applyColorMap(normalized_8bit, colormap)
            else:
                colored = cv2.cvtColor(normalized_8bit, cv2.COLOR_GRAY2BGR)
            
            # Redimensionar si es necesario
            target_size = kwargs.get('size', self.config['image']['default_size'])
            if isinstance(target_size, int):
                target_size = (target_size, target_size)
                
            if target_size != (colored.shape[1], colored.shape[0]):
                resized = cv2.resize(
                    colored, 
                    target_size, 
                    interpolation=cv2.INTER_CUBIC
                )
            else:
                resized = colored
            
            # Crear directorio de salida si no existe
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Guardar imagen
            cv2.imwrite(output_path, resized)
            return output_path
            
        except Exception as e:
            print(f"Error al generar imagen: {str(e)}")
            raise
    
    def _generate_audio(self, field: np.ndarray, **kwargs) -> np.ndarray:
        """
        Genera una señal de audio a partir del campo informacional.
        
        Args:
            field: Campo informacional
            **kwargs: Argumentos adicionales
                - sample_rate: Tasa de muestreo en Hz (predeterminado: 44100)
                - duration: Duración en segundos (predeterminado: 1.0)
                - channels: Número de canales (predeterminado: 1)
                
        Returns:
            np.ndarray: Señal de audio generada con valores normalizados entre -1 y 1
        """
        try:
            # Obtener configuración de audio con valores por defecto
            audio_config = self.config.get('audio', {})
            
            # Obtener parámetros de configuración con valores por defecto
            sample_rate = kwargs.get('sample_rate', audio_config.get('sample_rate', 44100))
            duration = kwargs.get('duration', audio_config.get('duration', 1.0))
            channels = kwargs.get('channels', audio_config.get('channels', 1))
            
            # Validar parámetros
            if sample_rate <= 0:
                raise ValueError("La tasa de muestreo debe ser mayor que 0")
            if duration <= 0:
                raise ValueError("La duración debe ser mayor que 0")
            if channels not in (1, 2):
                raise ValueError("El número de canales debe ser 1 (mono) o 2 (estéreo)")
            
            # Calcular número total de muestras
            num_samples = int(sample_rate * duration)
            
            # Aplanar el campo y normalizar a [-1, 1]
            flat_field = field.flatten()
            if len(flat_field) == 0:
                # Si el campo está vacío, generar ruido blanco
                audio_data = np.random.uniform(-1, 1, num_samples)
            else:
                # Repetir o truncar el campo para que coincida con el número de muestras
                if len(flat_field) < num_samples:
                    # Repetir el campo si es más corto
                    repeats = int(np.ceil(num_samples / len(flat_field)))
                    flat_field = np.tile(flat_field, repeats)
                
                # Tomar solo las primeras num_samples muestras
                audio_data = flat_field[:num_samples]
                
                # Normalizar a [-1, 1] si está habilitado
                if self.config['audio']['normalize']:
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 1e-10:  # Evitar división por cero
                        audio_data = audio_data / max_val
            
            # Cambiar la forma para múltiples canales si es necesario
            if channels > 1:
                audio_data = np.tile(audio_data.reshape(-1, 1), (1, channels))
            
            return audio_data
            
        except Exception as e:
            print(f"Error al generar audio: {str(e)}")
            # Devolver silencio en caso de error
            channels = int(kwargs.get('channels', self.config['audio']['channels']))
            sample_rate = int(kwargs.get('sample_rate', self.config['audio']['sample_rate']))
            duration = float(kwargs.get('duration', self.config['audio']['duration']))
            return np.zeros((int(sample_rate * duration), channels) if channels > 1 else int(sample_rate * duration))
    
    def _save_audio(self, audio_data: np.ndarray, filepath: str, sample_rate: int = None) -> None:
        """
        Guarda los datos de audio en un archivo.
        
        Args:
            audio_data: Datos de audio a guardar (normalizados entre -1 y 1)
            filepath: Ruta del archivo de salida
            sample_rate: Tasa de muestreo en Hz (opcional, se usará la del config por defecto)
        """
        try:
            if not isinstance(audio_data, np.ndarray):
                raise ValueError("Los datos de audio deben ser un array de NumPy")
                
            # Obtener la tasa de muestreo de los argumentos o de la configuración
            if sample_rate is None:
                audio_config = self.config.get('audio', {})
                sample_rate = audio_config.get('sample_rate', 44100)
            
            # Validar la tasa de muestreo
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError(f"La tasa de muestreo debe ser un entero positivo, se obtuvo: {sample_rate}")
            
            # Asegurarse de que el directorio existe
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Asegurarse de que los datos estén en el rango [-1, 1]
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
            
            # Guardar el archivo de audio
            try:
                sf.write(filepath, audio_data, sample_rate)
            except Exception as e:
                raise RuntimeError(f"Error al guardar el archivo de audio: {str(e)}")
        except Exception as e:
            print(f"Error al guardar audio: {str(e)}")
            raise
    
    def _generate_text(self, field: np.ndarray, **kwargs) -> str:
        """
        Genera texto a partir del campo informacional.
        
        Args:
            field: Campo informacional de entrada
            **kwargs: Argumentos adicionales
                - max_length: Longitud máxima del texto generado (predeterminado: 1000)
                - language: Idioma del texto (predeterminado: 'es')
                - seed: Semilla para generación reproducible (opcional)
                
        Returns:
            str: Texto generado o mensaje descriptivo si hay un error
        """
        try:
            # Obtener configuración de texto con valores por defecto
            text_config = self.config.get('text', {})
            
            # Obtener parámetros de configuración con valores por defecto
            max_length = int(kwargs.get('max_length', text_config.get('max_length', 1000)))
            language = kwargs.get('language', text_config.get('language', 'es'))
            
            # Validar parámetros
            if max_length <= 0:
                raise ValueError("La longitud máxima debe ser mayor que 0")
                
            # Establecer semilla para reproducibilidad si se proporciona
            if 'seed' in kwargs:
                np.random.seed(kwargs['seed'])
            
            # Si el campo está vacío, generar texto aleatorio simple
            if field.size == 0:
                words_es = ["hola", "mundo", "inteligencia", "artificial", "sistema", 
                          "DIG", "campo", "información", "procesando", "datos", 
                          "análisis", "complejo", "sistema", "digital", "entropía"]
                num_words = np.random.randint(5, 15)
                return ' '.join(np.random.choice(words_es, num_words)) + '.'
            
            # Aplanar el campo y normalizar
            flat_field = field.flatten()
            if len(flat_field) == 0:
                return "No se pudo generar texto: campo vacío"
                
            # Convertir el campo a una secuencia de tokens
            tokens = self._field_to_tokens(field, **kwargs)
            
            # Si no hay tokens, devolver texto predeterminado
            if not tokens:
                return "No se pudo generar texto: sin tokens generados"
            
            # Convertir tokens a texto
            text = self._tokens_to_text(tokens, **kwargs)
            
            # Limitar la longitud del texto si es necesario
            if len(text) > max_length:
                text = text[:max_length].rsplit(' ', 1)[0] + '...'
            
            return text
            
        except Exception as e:
            error_msg = f"Error al generar texto: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _field_to_tokens(self, field: np.ndarray, **kwargs) -> List[str]:
        """
        Convierte el campo informacional en una secuencia de tokens.
        
        Args:
            field: Campo informacional de entrada
            **kwargs: Argumentos adicionales
                
        Returns:
            List[str]: Lista de tokens
        """
        try:
            # Aplanar el campo
            flat_field = field.flatten()
            
            # Convertir valores a caracteres ASCII imprimibles (32-126)
            tokens = []
            for val in flat_field:
                # Mapear el valor a un rango de caracteres imprimibles
                char_code = 32 + int((val + 1) * 47)  # Mapear [-1,1] a [32,126]
                char_code = max(32, min(126, char_code))  # Asegurar que esté en rango
                tokens.append(chr(char_code))
                
            return tokens
            
        except Exception as e:
            print(f"Error al convertir campo a tokens: {str(e)}")
            return ["error"]
    
    def _tokens_to_text(self, tokens: List[str], **kwargs) -> str:
        """
        Convierte una secuencia de tokens en texto legible.
        
        Args:
            tokens: Lista de tokens
            **kwargs: Argumentos adicionales
                - max_length: Longitud máxima del texto
                - language: Idioma del texto
                
        Returns:
            str: Texto legible
        """
        try:
            max_length = int(kwargs.get('max_length', self.config['text']['max_length']))
            language = kwargs.get('language', self.config['text']['language'])
            
            # Unir los tokens y truncar a la longitud máxima
            text = ''.join(tokens)[:max_length]
            
            # Dividir en oraciones (para mejor legibilidad)
            # Esto es un enfoque simple, se podría mejorar con un modelo de lenguaje
            if len(text) > 100:
                # Insertar espacios después de puntuación para mejorar legibilidad
                for punct in ['.', '!', '?']:
                    text = text.replace(punct, punct + ' ')
                
                # Dividir en oraciones y unir con saltos de línea
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                text = '.\n'.join(sentences)
            
            return text
            
        except Exception as e:
            print(f"Error al convertir tokens a texto: {str(e)}")
            return "Error al generar texto"
    
    def _execute_action(self, field: np.ndarray, **kwargs) -> str:
        """
        Ejecuta una acción basada en el campo informacional.
        
        Args:
            field: Campo informacional de entrada
            **kwargs: Argumentos adicionales
                
        Returns:
            str: Resultado de la acción
        """
        try:
            # Esta es una implementación de ejemplo
            # En una implementación real, aquí se podrían ejecutar acciones específicas
            # basadas en el contenido del campo
            
            # Calcular algunas métricas simples
            metrics = {
                'min': float(np.min(field)),
                'max': float(np.max(field)),
                'mean': float(np.mean(field)),
                'std': float(np.std(field))
            }
            
            return f"Acción ejecutada: {metrics}"
            
        except Exception as e:
            print(f"Error al ejecutar acción: {str(e)}")
            return "Error al ejecutar acción"
    def _execute_action(self, field: np.ndarray) -> Dict[str, Any]:
        """
        Ejecuta una acción basada en el campo informacional.
        
        Args:
            field: Campo informacional procesado
            
        Returns:
            Dict con información sobre la acción ejecutada
        """
        try:
            # Calcular métricas básicas del campo
            metrics = {
                'min': float(np.min(field)) if field.size > 0 else 0,
                'max': float(np.max(field)) if field.size > 0 else 0,
                'mean': float(np.mean(field)) if field.size > 0 else 0,
                'std': float(np.std(field)) if field.size > 0 else 0,
                'shape': field.shape,
                'non_zero': int(np.count_nonzero(field))
            }
            
            # Determinar el tipo de acción basado en las métricas
            action_type = 'none'
            if metrics['std'] > 0.5:
                action_type = 'high_variation'
            elif metrics['mean'] > 0.7:
                action_type = 'high_intensity'
            elif metrics['mean'] < 0.3:
                action_type = 'low_intensity'
                
            # Registrar la acción
            self._log_event(f"Acción ejecutada: {action_type}", metrics)
            
            return {
                'type': action_type,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Error al ejecutar acción: {str(e)}"
            self._log_error(error_msg)
            return {
                'type': 'error',
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
            
    def _save_audio(self, audio_data: np.ndarray, path: str) -> None:
        """Guarda los datos de audio en un archivo."""
        sample_rate = self.config['audio']['sample_rate']
        sf.write(path, audio_data, sample_rate)
    
    def _generate_text(self, field: np.ndarray, **kwargs) -> str:
        """Genera texto a partir del campo informacional."""
        # Convertir el campo a una secuencia de tokens
        tokens = self._field_to_tokens(field, **kwargs)
        
        # Generar texto a partir de los tokens
        text = self._tokens_to_text(tokens, **kwargs)
        
        # Limitar longitud si es necesario
        max_length = kwargs.get('max_length', self.config['text']['max_length'])
        if len(text) > max_length:
            text = text[:max_length] + '...'
            
        return text
    
    def _field_to_tokens(self, field: np.ndarray, **kwargs) -> List[str]:
        """Convierte el campo informacional en una secuencia de tokens."""
        # Estrategia simple: mapear valores a caracteres
        token_map = {
            -1: 'a',  # Caos
            0: ' ',   # Equilibrio
            1: '1'    # Información
        }
        
        # Aplanar el campo y mapear a tokens
        flat_field = field.flatten()
        tokens = []
        
        for val in flat_field:
            # Encontrar el valor más cercano en el mapa de tokens
            closest_val = min(token_map.keys(), key=lambda x: abs(x - val))
            tokens.append(token_map[closest_val])
        
        return tokens
    
    def _tokens_to_text(self, tokens: List[str], **kwargs) -> str:
        """Convierte una secuencia de tokens en texto legible."""
        # Estrategia simple: unir tokens con espacios
        text = ''.join(tokens)
        
        # Aplicar algunas reglas básicas de formato
        text = ' '.join(text.split())  # Eliminar espacios múltiples
        
        # Capitalizar la primera letra
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _execute_action(self, field: np.ndarray, **kwargs) -> Any:
        """Ejecuta una acción basada en el campo informacional."""
        action_type = kwargs.get('action_type', 'print')
        
        if action_type == 'print':
            # Acción por defecto: imprimir estadísticas del campo
            stats = {
                'shape': field.shape,
                'min': float(np.min(field)),
                'max': float(np.max(field)),
                'mean': float(np.mean(field)),
                'std': float(np.std(field))
            }
            return stats
            
        elif action_type == 'save':
            # Guardar el campo en un archivo
            filename = kwargs.get('filename', f"field_{int(time.time())}.npy")
            np.save(os.path.join(self.config['output_dir'], filename), field)
            return f"Campo guardado en {filename}"
            
        else:
            raise ValueError(f"Tipo de acción no soportado: {action_type}")
    
    def _log_action(self, metadata: Dict[str, Any]) -> None:
        """Registra una acción en el historial."""
        if not self.config['feedback']['enabled']:
            return
            
        self.action_history.append(metadata)
        
        # Limitar el tamaño del historial
        max_entries = self.config['feedback']['max_entries']
        if len(self.action_history) > max_entries:
            self.action_history = self.action_history[-max_entries:]
        
        # Guardar en archivo
        self._save_feedback()
    
    def _save_feedback(self) -> None:
        """Guarda el historial de retroalimentación en un archivo."""
        if not self.config['feedback']['enabled']:
            return
            
        log_file = os.path.join(
            self.config['output_dir'], 
            self.config['feedback']['log_file']
        )
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.action_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error al guardar el registro de retroalimentación: {e}")
    
    def _load_feedback(self):
        """Carga el feedback de archivo si está habilitado."""
        # Verificar si la configuración de feedback existe, si no, usar valores por defecto
        if 'feedback' not in self.config:
            self.config['feedback'] = {
                'enabled': False,
                'file_path': 'feedback.json'
            }
            return
            
        if not self.config['feedback'].get('enabled', False):
            return
            
        feedback_file = self.config['feedback'].get('file_path', 'feedback.json')
        try:
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    self.feedback_history = json.load(f)
        except Exception as e:
            print(f"Error al cargar feedback: {e}")
            self.feedback_history = []
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Devuelve estadísticas sobre las acciones realizadas."""
        if not self.action_history:
            return {
                'total_actions': 0,
                'success_rate': 0.0,
                'output_types': {},
                'errors': {}
            }
        
        total = len(self.action_history)
        success = sum(1 for a in self.action_history if a.get('success', False))
        success_rate = success / total if total > 0 else 0.0
        
        # Contar tipos de salida
        output_types = {}
        for action in self.action_history:
            output_type = action.get('output_type', 'unknown')
            output_types[output_type] = output_types.get(output_type, 0) + 1
        
        # Contar errores
        errors = {}
        for action in self.action_history:
            if not action.get('success', False):
                error_type = action.get('error_type', 'unknown')
                errors[error_type] = errors.get(error_type, 0) + 1
        
        return {
            'total_actions': total,
            'success_rate': success_rate,
            'output_types': output_types,
            'errors': errors
        }

# Alias para compatibilidad con código existente
ActionModule = ModuloAccion
