import numpy as np
import time
import os
import soundfile as sf
from typing import Dict, Any, Optional, List, Tuple, Union
import librosa # Asegúrate de que librosa esté instalado: pip install librosa
from PIL import Image # Importa Pillow aquí para evitar dependencia si no se usa imagen

class SensorModule:
    """
    Sensorium Informacional: Módulo de percepción de la IA DIG
    
    Responsable de la captura y transformación de datos brutos en el campo informacional [0, 1].
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el módulo de Sensorium Informacional.
        
        Args:
            config: Configuración para el procesamiento de diferentes tipos de datos
        """
        self.raw_data = None
        self.field_output = None
        self.config = config or {
            'image': {
                'resize': (64, 64),  # Tamaño estándar para imágenes
            },
            'audio': {
                'sample_rate': 22050,
                'n_fft': 512,
                'hop_length': 256,
                'n_mels': 64
            },
            'text': {
                'max_length': 256,
                'embedding_dim': 128,
                'target_field_size': (64, 64), # Tamaño objetivo para el campo final
            }
        }
        
    def load_input(self, source: Union[str, np.ndarray, Dict], input_type: str = 'auto') -> None:
        """
        Carga y preprocesa datos de entrada de diversas fuentes.
        
        Args:
            source: Fuente de datos (ruta de archivo, array de NumPy o diccionario).
            input_type: Tipo de la entrada ('image', 'audio', 'text', 'auto').
        """
        if input_type == 'auto':
            input_type = self._detect_input_type(source)

        self.raw_data = source
        if input_type == 'text':
            self._process_text(source)
        elif input_type == 'image':
            self._process_image(source)
        elif input_type == 'audio':
            self._process_audio(source)
        else:
            raise ValueError(f"Tipo de entrada '{input_type}' no soportado.")

    def _detect_input_type(self, source: Union[str, np.ndarray, Dict]) -> str:
        """Detecta automáticamente el tipo de entrada."""
        if isinstance(source, str):
            if os.path.exists(source):
                if source.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    return 'image'
                elif source.lower().endswith(('.wav', '.mp3', '.flac')):
                    return 'audio'
                elif source.lower().endswith(('.txt', '.csv')):
                    return 'text' # Asume que .txt y .csv son texto
            return 'text' # Si es string pero no archivo, asume texto directo
        elif isinstance(source, np.ndarray):
            # Asumimos que un ndarray podría ser una imagen o un campo ya procesado
            if source.ndim == 2:
                return 'image' 
            # Podrías añadir lógica para detectar si es audio, pero por simplicidad, imagen.
        elif isinstance(source, Dict):
            if 'text' in source: return 'text'
            if 'image' in source: return 'image'
            if 'audio' in source: return 'audio'
        return 'text' # Default

    def _process_text(self, text_input: str) -> None:
        """Procesa texto para generar un campo informacional en [0, 1]."""
        max_length = self.config['text']['max_length']
        embedding_dim = self.config['text']['embedding_dim']
        target_field_size = self.config['text']['target_field_size']
        
        char_values = np.array([ord(c) for c in text_input.ljust(max_length)[:max_length]])
        
        embedding = np.zeros(embedding_dim)
        for i, val in enumerate(char_values):
            embedding[i % embedding_dim] += val
        
        embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-9)
        
        field_flat = np.resize(embedding, target_field_size[0] * target_field_size[1])
        self.field_output = field_flat.reshape(target_field_size)

    def _process_image(self, image_source: Union[str, np.ndarray]) -> None:
        """Procesa una imagen para generar un campo informacional en [0, 1]."""
        
        if isinstance(image_source, str):
            img = Image.open(image_source).convert('L') # Convertir a escala de grises
        elif isinstance(image_source, np.ndarray):
            img = Image.fromarray(image_source).convert('L')
        else:
            raise ValueError("Formato de imagen no soportado.")
            
        resize_dim = self.config['image']['resize']
        img = img.resize(resize_dim)
        
        self.field_output = np.array(img) / 255.0

    def _process_audio(self, audio_source: str) -> None:
        """Procesa audio para generar un campo informacional (Mel-spectrogram) en [0, 1]."""
        try:
            y, sr = librosa.load(audio_source, sr=self.config['audio']['sample_rate'])
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr,
                n_fft=self.config['audio']['n_fft'],
                hop_length=self.config['audio']['hop_length'],
                n_mels=self.config['audio']['n_mels']
            )
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            normalized_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.min()) / \
                                     (log_mel_spectrogram.max() - log_mel_spectrogram.min() + 1e-9)
            
            target_h, target_w = self.config['text']['target_field_size'] # Reutiliza el tamaño del campo de texto
            
            # Ajustar dimensiones del espectrograma a target_field_size
            current_h, current_w = normalized_spectrogram.shape
            
            # Redimensionar (recortar o expandir) la altura
            if current_h > target_h:
                normalized_spectrogram = normalized_spectrogram[:target_h, :]
            elif current_h < target_h:
                pad_h = target_h - current_h
                normalized_spectrogram = np.pad(normalized_spectrogram, ((0, pad_h), (0,0)), mode='constant')

            # Redimensionar (recortar o expandir) el ancho
            if current_w > target_w:
                normalized_spectrogram = normalized_spectrogram[:, :target_w]
            elif current_w < target_w:
                pad_w = target_w - current_w
                normalized_spectrogram = np.pad(normalized_spectrogram, ((0, 0), (0, pad_w)), mode='constant')

            self.field_output = normalized_spectrogram
            
        except Exception as e:
            print(f"Error al procesar el archivo de audio: {e}")
            self.field_output = np.random.rand(*self.config['text']['target_field_size']) # Campo aleatorio como fallback

    def get_ternary_field(self) -> np.ndarray:
        """
        Devuelve el campo informacional en formato [0, 1].
        Asegura que la salida sea 2D y de la forma esperada.
        """
        target_field_size = self.config['text']['target_field_size']

        if self.field_output is None:
            return np.zeros(target_field_size, dtype=np.float32)

        # Asegurarse de que el campo sea 2D
        if len(self.field_output.shape) == 1:
            size = int(np.ceil(np.sqrt(len(self.field_output))))
            padded = np.pad(self.field_output, (0, size*size - len(self.field_output)), 'constant')
            reshaped_field = padded.reshape((size, size))
        elif len(self.field_output.shape) == 2:
            reshaped_field = self.field_output
        else:
            # Si el campo tiene más de 2 dimensiones (ej. imagen RGB), convertir a 2D (escala de grises)
            # Esto es un fallback, la lógica de _process_image ya debería manejarlo.
            reshaped_field = np.mean(self.field_output, axis=-1) if self.field_output.ndim > 2 else self.field_output


        # Asegurarse de que el campo tenga el target_field_size
        if reshaped_field.shape != target_field_size:
            reshaped_field = np.resize(reshaped_field, target_field_size)

        # Asegurarse de que los valores estén en el rango [0, 1] y sea float32
        return np.clip(reshaped_field, 0, 1).astype(np.float32)


