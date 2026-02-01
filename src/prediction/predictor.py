"""
LSTM Time Series Predictor
Optimized for Raspberry Pi with TFLite inference
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from loguru import logger

try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    Model = Any  # Type stub when TF not available

from ..core.base_module import BaseModule
from .tflite_converter import TFLiteConverter


class LSTMPredictor(BaseModule):
    """
    LSTM-based time series predictor.
    Supports both full TensorFlow and optimized TFLite inference.
    """
    
    def __init__(self):
        super().__init__('prediction')
        self.model: Optional[Model] = None
        self.tflite_interpreter = None
        self.use_tflite: bool = False
        
        # LSTM configuration
        self.lstm_config = self.config.get('lstm', {})
        self.sequence_length = self.lstm_config.get('sequence_length', 50)
        self.n_features = self.lstm_config.get('features', 8)
        self.hidden_units = self.lstm_config.get('hidden_units', [64, 32])
        self.prediction_horizon = self.lstm_config.get('prediction_horizon', 10)
        
        # TFLite configuration
        self.tflite_config = self.config.get('tflite', {})
        
        # Data buffer for sequence building
        self._data_buffer: List[np.ndarray] = []
        
    def initialize(self) -> bool:
        """Initialize LSTM model or TFLite interpreter."""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return False
            
        try:
            # Check for TFLite model first (preferred for Pi)
            tflite_path = Path("models/prediction/lstm_predictor.tflite")
            if tflite_path.exists():
                self._load_tflite(tflite_path)
                self.use_tflite = True
                logger.info("Loaded TFLite model for inference")
            else:
                # Build or load full model
                self.model = self._build_model()
                
                weights_path = Path("models/prediction/lstm_predictor.weights.h5")
                if weights_path.exists():
                    self.model.load_weights(str(weights_path))
                    logger.info("Loaded pre-trained LSTM weights")
            
            self.is_initialized = True
            logger.info(f"LSTM Predictor initialized (TFLite: {self.use_tflite})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LSTM Predictor: {e}")
            return False
    
    def _build_model(self) -> Model:
        """Build LSTM model architecture."""
        inputs = keras.Input(
            shape=(self.sequence_length, self.n_features),
            name='sequence_input'
        )
        
        x = inputs
        
        # Stacked LSTM layers
        for i, units in enumerate(self.hidden_units[:-1]):
            x = layers.LSTM(
                units,
                return_sequences=True,
                dropout=self.lstm_config.get('dropout', 0.2),
                recurrent_dropout=0.1,
                name=f'lstm_{i}'
            )(x)
        
        # Final LSTM layer
        x = layers.LSTM(
            self.hidden_units[-1],
            return_sequences=False,
            dropout=self.lstm_config.get('dropout', 0.2),
            name=f'lstm_{len(self.hidden_units)-1}'
        )(x)
        
        # Dense layers for prediction
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.1)(x)
        
        # Output: predict next `prediction_horizon` timesteps
        outputs = layers.Dense(
            self.prediction_horizon * self.n_features,
            activation='linear',
            name='prediction_output'
        )(x)
        
        # Reshape to (prediction_horizon, n_features)
        outputs = layers.Reshape(
            (self.prediction_horizon, self.n_features),
            name='reshape_output'
        )(outputs)
        
        model = Model(inputs, outputs, name='lstm_predictor')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built LSTM model: seq_len={self.sequence_length}, "
                   f"features={self.n_features}, horizon={self.prediction_horizon}")
        
        return model
    
    def _load_tflite(self, path: Path):
        """Load TFLite model for inference."""
        try:
            import tflite_runtime.interpreter as tflite
            self.tflite_interpreter = tflite.Interpreter(model_path=str(path))
        except ImportError:
            # Fall back to TensorFlow Lite
            self.tflite_interpreter = tf.lite.Interpreter(model_path=str(path))
        
        self.tflite_interpreter.allocate_tensors()
        
        # Get input/output details
        self._input_details = self.tflite_interpreter.get_input_details()
        self._output_details = self.tflite_interpreter.get_output_details()
        
        logger.info(f"TFLite model loaded: input shape {self._input_details[0]['shape']}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
            y: Target sequences of shape (n_samples, prediction_horizon, n_features)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation
            
        Returns:
            Training history
        """
        if self.use_tflite:
            raise RuntimeError("Cannot train TFLite model. Use full TensorFlow model.")
        
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        logger.info(f"Training LSTM on {X.shape[0]} sequences")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("LSTM training complete")
        return history.history
    
    def process(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Generate predictions from input sequence.
        
        Args:
            data: Input data. Can be:
                  - Single timestep: (n_features,) - will be buffered
                  - Full sequence: (sequence_length, n_features)
                  - Batch: (batch, sequence_length, n_features)
            
        Returns:
            Prediction results
        """
        if not self.is_initialized:
            raise RuntimeError("Predictor not initialized")
        
        # Handle different input shapes
        if data.ndim == 1:
            # Single timestep - add to buffer
            return self._process_streaming(data)
        elif data.ndim == 2:
            # Single sequence
            data = data.reshape(1, *data.shape)
        
        # Run inference
        if self.use_tflite:
            predictions = self._tflite_predict(data)
        else:
            predictions = self.model.predict(data, verbose=0)
        
        # Calculate confidence based on prediction variance
        confidence = 1.0 / (1.0 + np.std(predictions))
        
        return {
            'forecast': predictions,
            'confidence': float(confidence),
            'horizon': self.prediction_horizon,
            'input_shape': data.shape
        }
    
    def _process_streaming(self, timestep: np.ndarray) -> Dict[str, Any]:
        """Process streaming data by buffering timesteps."""
        self._data_buffer.append(timestep)
        
        # Keep only latest sequence_length timesteps
        if len(self._data_buffer) > self.sequence_length:
            self._data_buffer = self._data_buffer[-self.sequence_length:]
        
        # Not enough data yet
        if len(self._data_buffer) < self.sequence_length:
            return {
                'forecast': None,
                'confidence': 0.0,
                'buffer_size': len(self._data_buffer),
                'ready': False
            }
        
        # Build sequence and predict
        sequence = np.array(self._data_buffer)
        return self.process(sequence)
    
    def _tflite_predict(self, data: np.ndarray) -> np.ndarray:
        """Run TFLite inference."""
        # Ensure correct dtype
        input_dtype = self._input_details[0]['dtype']
        data = data.astype(input_dtype)
        
        # Set input tensor
        self.tflite_interpreter.set_tensor(
            self._input_details[0]['index'],
            data
        )
        
        # Run inference
        self.tflite_interpreter.invoke()
        
        # Get output
        output = self.tflite_interpreter.get_tensor(
            self._output_details[0]['index']
        )
        
        return output
    
    def convert_to_tflite(
        self,
        output_path: str = "models/prediction/lstm_predictor.tflite",
        quantization: str = "int8"
    ) -> str:
        """
        Convert model to TFLite format.
        
        Args:
            output_path: Path to save TFLite model
            quantization: Quantization type ('none', 'float16', 'int8')
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise RuntimeError("No model to convert")
        
        converter = TFLiteConverter(self.model)
        return converter.convert(output_path, quantization)
    
    def prepare_sequences(
        self,
        data: np.ndarray,
        target_offset: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training sequences from raw time series data.
        
        Args:
            data: Raw data of shape (n_timesteps, n_features)
            target_offset: How many steps ahead to predict
            
        Returns:
            X, y arrays for training
        """
        X, y = [], []
        
        total_length = self.sequence_length + self.prediction_horizon
        
        for i in range(len(data) - total_length + 1):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length:i + total_length])
        
        return np.array(X), np.array(y)
    
    def save_model(self, path: str = "models/prediction/lstm_predictor"):
        """Save model weights."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(f"{path}.weights.h5")
        logger.info(f"Saved LSTM model to {path}")
    
    def reset_buffer(self):
        """Reset the streaming data buffer."""
        self._data_buffer = []
    
    def shutdown(self):
        """Clean shutdown."""
        self.model = None
        self.tflite_interpreter = None
        self._data_buffer = []
        self.is_initialized = False
        if TF_AVAILABLE:
            keras.backend.clear_session()
        logger.info("LSTM Predictor shutdown")
    
    def get_status(self) -> Dict[str, Any]:
        """Get predictor status."""
        status = super().get_status()
        status.update({
            'use_tflite': self.use_tflite,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'prediction_horizon': self.prediction_horizon,
            'buffer_size': len(self._data_buffer)
        })
        return status
