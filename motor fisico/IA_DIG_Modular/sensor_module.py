
 
import numpy as np
from typing import Union, Dict, List, Any
import cv2
import soundfile as sf # Necesita 'pip install soundfile'
import numpy.fft as fft

class SensorModule:
    """
    Sensorium Informacional: Módulo de percepción de la IA DIG
    
    Responsable de la captura y transformación de datos brutos en el campo informacional ternario.
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
                'thresholds': (0.3, 0.7),  # Umbrales para la conversión ternaria
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
                'target_field_size': (64, 64) # <-- NUEVO: Tamaño de campo 2D objetivo para texto y base para audio
            }
        }
        
    def load_input(self, source: Union[str, np.ndarray, Dict], input_type: str = 'auto') -> None:
        """
        Carga y preprocesa datos de entrada de diversas fuentes.
        
        Args:
            source: Fuente de datos (ruta de archivo, array numpy, etc.)
            input_type: Tipo de entrada ('image', 'audio', 'text', 'auto')
        """
        self.raw_data = source
        
        if input_type == 'auto':
            input_type = self._detect_input_type(source)
            
        if input_type == 'image':
            self._process_image(source)
        elif input_type == 'audio':
            self._process_audio(source)
        elif input_type == 'text':
            self._process_text(source)
        else:
            raise ValueError(f"Tipo de entrada no soportado: {input_type}")
    
    def _detect_input_type(self, source: Any) -> str:
        """Detecta automáticamente el tipo de entrada."""
        if isinstance(source, str):
            if source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                return 'image'
            elif source.lower().endswith(('.wav', '.mp3', '.ogg')):
                return 'audio'
            else:
                return 'text'
        elif isinstance(source, np.ndarray):
            if len(source.shape) == 2 or (len(source.shape) == 3 and source.shape[2] in [1, 3, 4]):
                return 'image'
            return 'audio'  # Asumir audio si es un array 1D (podría ser más sofisticado)
        return 'text'  # Por defecto, tratar como texto
    
    def _process_image(self, image_source: Union[str, np.ndarray]) -> None:
        """Procesa una imagen y la convierte al campo ternario."""
        # Cargar la imagen si es una ruta
        if isinstance(image_source, str):
            image = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"No se pudo cargar la imagen de la ruta: {image_source}")
        else:
            image = image_source.copy()
            if len(image.shape) > 2:  # Convertir a escala de grises si es a color
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar a un tamaño estándar
        target_size = self.config['image']['resize']
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalizar a [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convertir a campo ternario
        low, high = self.config['image']['thresholds']
        self.field_output = np.zeros_like(image, dtype=np.int8)
        self.field_output[image < low] = -1
        self.field_output[image > high] = 1
    
    def _process_audio(self, audio_source: Union[str, np.ndarray]) -> None:
        """Procesa una señal de audio y la convierte al campo ternario."""
        # Cargar el audio si es una ruta
        if isinstance(audio_source, str):
            audio, sr = sf.read(audio_source)
            if len(audio.shape) > 1:  # Tomar solo un canal si es estéreo
                audio = audio[:, 0]
        else:
            audio = audio_source.copy()
            # Asumir sample_rate si se pasa un numpy array directamente
            sr = self.config['audio']['sample_rate'] 
        
        # Calcular el espectrograma de mel
        n_fft = self.config['audio']['n_fft']
        hop_length = self.config['audio']['hop_length']
        n_mels = self.config['audio']['n_mels']
        
        # Aplicar STFT
        # Asegurarse de que el audio tenga la longitud adecuada para STFT
        if len(audio) < n_fft: # Rellenar con ceros si es demasiado corto para FFT
            audio = np.pad(audio, (0, n_fft - len(audio)), 'constant')
        D = np.abs(fft.stft(audio, n_fft=n_fft, hop_length=hop_length))
        
        # Convertir a escala de mel
        mel_basis = self._mel_filterbank(
            sr=sr, # Usar sr del archivo o configurado
            n_fft=n_fft,
            n_mels=n_mels
        )
        # Asegurarse de que las dimensiones para np.dot sean compatibles
        # D.shape[0] es n_fft // 2 + 1
        # mel_basis.shape[1] debe ser D.shape[0]
        if mel_basis.shape[1] != D.shape[0]:
            # Ajuste de tamaño si mel_basis no coincide (puede pasar si n_fft es diferente)
            # Esto es una solución temporal; lo ideal es que mel_basis se genere correctamente
            print(f"Advertencia: mel_basis.shape[1] ({mel_basis.shape[1]}) != D.shape[0] ({D.shape[0]}). Ajustando mel_basis.")
            mel_basis = mel_basis[:, :D.shape[0]] # Truncar o pad, según sea necesario
            
        mel_spec = np.dot(mel_basis, D)
        
        # Escala logarítmica
        mel_spec = 20 * np.log10(np.maximum(1e-5, mel_spec))
        
        # Normalizar
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        
        # Redimensionar el espectrograma para que tenga el tamaño objetivo del campo 2D
        # Esto es crucial para la consistencia en el sistema DIG.
        # Usamos el target_field_size de la configuración de texto como tamaño estándar
        target_field_size = self.config['text']['target_field_size'] 
        mel_spec_resized = cv2.resize(mel_spec, target_field_size, interpolation=cv2.INTER_AREA)


        # Convertir a campo ternario
        low, high = 0.3, 0.7  # Umbrales para audio
        self.field_output = np.zeros_like(mel_spec_resized, dtype=np.int8) # Usar el espectrograma redimensionado
        self.field_output[mel_spec_resized < low] = -1
        self.field_output[mel_spec_resized > high] = 1
        
    def _mel_filterbank(self, sr: int, n_fft: int, n_mels: int = 40, fmin: float = 0.0, fmax: float = None):
        """Crea un banco de filtros de mel."""
        if fmax is None:
            fmax = float(sr) / 2
        
        # Puntos de frecuencia en escala mel
        mel_points = np.linspace(self._hz_to_mel(fmin), self._hz_to_mel(fmax), n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        
        # Convertir a bins de FFT
        bin = np.floor((n_fft + 1) * hz_points / sr)
        
        fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
        
        for m in range(1, n_mels + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right
            
            for k in range(f_m_minus, f_m):
                # Evitar división por cero si bin[m] - bin[m - 1] es 0
                if (bin[m] - bin[m - 1]) != 0:
                    fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
                else:
                    fbank[m - 1, k] = 0.0 # O algún otro valor predeterminado

            for k in range(f_m, f_m_plus):
                # Evitar división por cero si bin[m + 1] - bin[m] es 0
                if (bin[m + 1] - bin[m]) != 0:
                    fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
                else:
                    fbank[m - 1, k] = 0.0 # O algún otro valor predeterminado
        
        return fbank
    
    @staticmethod
    def _hz_to_mel(hz):
        """Convertir de Hz a escala mel."""
        return 2595 * np.log10(1 + hz / 700.0)
    
    @staticmethod
    def _mel_to_hz(mel):
        """Convertir de escala mel a Hz."""
        return 700 * (10 ** (mel / 2595.0) - 1)
    
    def _process_text(self, text: str) -> None:
        """
        Procesa texto y lo convierte al campo ternario, ahora asegurando un campo 2D de tamaño fijo.
        """
        # Tokenización simple
        tokens = text.lower().split()
        
        # Crear un "vocabulario" simple basado en frecuencias y mapeo ternario
        # Esto generará una secuencia de -1, 0, 1
        raw_field_1d = []
        vocab = {}
        for token in tokens:
            vocab[token] = vocab.get(token, 0) + 1
        
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        
        n = len(sorted_vocab)
        for i, (token, count) in enumerate(sorted_vocab):
            if i < n // 3:
                raw_field_1d.append(1)  # Términos más frecuentes
            elif i < 2 * n // 3:
                raw_field_1d.append(0)  # Términos de frecuencia media
            else:
                raw_field_1d.append(-1) # Términos menos frecuentes

        # Rellenar o truncar la secuencia 1D para que tenga el tamaño total del campo 2D objetivo
        target_rows, target_cols = self.config['text']['target_field_size']
        target_total_elements = target_rows * target_cols
        
        if len(raw_field_1d) < target_total_elements:
            # Rellenar con ceros (equilibrio) si es demasiado corto
            padded_field = np.pad(raw_field_1d, (0, target_total_elements - len(raw_field_1d)), 'constant', constant_values=0)
        else:
            # Truncar si es demasiado largo
            padded_field = np.array(raw_field_1d[:target_total_elements])

        # Remodelar a un campo 2D de tamaño fijo
        self.field_output = padded_field.reshape(target_rows, target_cols).astype(np.int8)

    def get_ternary_field(self) -> np.ndarray:
        """
        Devuelve el campo informacional ternario.
        
        Returns:
            np.ndarray: Matriz de valores ternarios (-1, 0, 1)
        """
        if self.field_output is None:
            raise ValueError("No se ha cargado ningún dato. Use load_input() primero.")
        return self.field_output
    
    def get_field_dimensions(self) -> tuple:
        """
        Devuelve las dimensiones del campo de salida.
        
        Returns:
            tuple: Dimensiones del campo
        """
        if self.field_output is None:
            raise ValueError("No hay campo de salida disponible.")
        return self.field_output.shape
