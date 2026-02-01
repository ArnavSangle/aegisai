"""
Autoencoder Anomaly Detector
Neural network based anomaly detection using reconstruction error
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
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
    logger.warning("TensorFlow not available, Autoencoder will be disabled")

from ..core.base_module import BaseModule


class AutoencoderDetector(BaseModule):
    """
    Autoencoder based anomaly detector.
    Uses reconstruction error to detect anomalies - high error = anomaly.
    """
    
    def __init__(self):
        super().__init__('anomaly')
        self.model: Optional[Model] = None
        self.encoder: Optional[Model] = None
        self.ae_config = self.config.get('autoencoder', {})
        self.threshold: float = 0.0
        
    def initialize(self) -> bool:
        """Initialize Autoencoder model."""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return False
            
        try:
            # Build autoencoder architecture
            self.model, self.encoder = self._build_model()
            
            # Try to load pre-trained weights
            weights_path = Path("models/anomaly/autoencoder.weights.h5")
            if weights_path.exists():
                self.model.load_weights(str(weights_path))
                logger.info("Loaded pre-trained Autoencoder weights")
            
            # Load threshold
            threshold_path = Path("models/anomaly/ae_threshold.npy")
            if threshold_path.exists():
                self.threshold = float(np.load(threshold_path))
                logger.info(f"Loaded anomaly threshold: {self.threshold}")
            
            self.is_initialized = True
            logger.info("Autoencoder detector initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Autoencoder: {e}")
            return False
    
    def _build_model(self) -> Tuple[Model, Model]:
        """
        Build autoencoder architecture.
        
        Returns:
            Tuple of (autoencoder, encoder) models
        """
        input_dim = self.ae_config.get('input_dim', 64)
        encoding_dim = self.ae_config.get('encoding_dim', 16)
        hidden_layers = self.ae_config.get('hidden_layers', [32, 16, 32])
        activation = self.ae_config.get('activation', 'relu')
        
        # Input layer
        inputs = keras.Input(shape=(input_dim,), name='encoder_input')
        x = inputs
        
        # Encoder layers
        for i, units in enumerate(hidden_layers[:len(hidden_layers)//2 + 1]):
            x = layers.Dense(units, activation=activation, name=f'encoder_{i}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
        
        # Bottleneck (encoding)
        encoded = layers.Dense(encoding_dim, activation=activation, name='encoding')(x)
        
        # Decoder layers
        x = encoded
        for i, units in enumerate(reversed(hidden_layers[len(hidden_layers)//2:])):
            x = layers.Dense(units, activation=activation, name=f'decoder_{i}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
        
        # Output layer
        outputs = layers.Dense(input_dim, activation='linear', name='decoder_output')(x)
        
        # Build models
        autoencoder = Model(inputs, outputs, name='autoencoder')
        encoder = Model(inputs, encoded, name='encoder')
        
        # Compile
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        logger.info(f"Built Autoencoder: {input_dim} -> {encoding_dim} -> {input_dim}")
        
        return autoencoder, encoder
    
    def fit(
        self,
        data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train the Autoencoder model.
        
        Args:
            data: Training data of shape (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            
        Returns:
            Training history
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")
        
        logger.info(f"Training Autoencoder on {data.shape[0]} samples")
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train (autoencoder reconstructs its input)
        history = self.model.fit(
            data, data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Calculate threshold from training data reconstruction errors
        self._calculate_threshold(data)
        
        logger.info("Autoencoder training complete")
        return history.history
    
    def _calculate_threshold(self, data: np.ndarray):
        """Calculate anomaly threshold from training data."""
        reconstructions = self.model.predict(data, verbose=0)
        mse = np.mean(np.square(data - reconstructions), axis=1)
        
        percentile = self.ae_config.get('threshold_percentile', 95)
        self.threshold = np.percentile(mse, percentile)
        
        logger.info(f"Calculated threshold at {percentile}th percentile: {self.threshold}")
    
    def process(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies using reconstruction error.
        
        Args:
            data: Input data of shape (n_samples, n_features) or (n_features,)
            
        Returns:
            Dictionary with anomaly scores and predictions
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")
        
        # Handle single sample
        single_sample = data.ndim == 1
        if single_sample:
            data = data.reshape(1, -1)
        
        # Get reconstruction
        reconstructions = self.model.predict(data, verbose=0)
        
        # Calculate reconstruction error (MSE)
        mse = np.mean(np.square(data - reconstructions), axis=1)
        
        # Determine anomalies based on threshold
        is_anomaly = mse > self.threshold
        
        # Normalize scores
        normalized_scores = mse / (self.threshold + 1e-8)
        
        # Get latent representation
        latent = self.encoder.predict(data, verbose=0)
        
        result = {
            'scores': normalized_scores,
            'reconstruction_error': mse,
            'is_anomaly': is_anomaly,
            'threshold': self.threshold,
            'latent_representation': latent,
            'reconstructions': reconstructions
        }
        
        if single_sample:
            result['scores'] = float(result['scores'][0])
            result['reconstruction_error'] = float(result['reconstruction_error'][0])
            result['is_anomaly'] = bool(result['is_anomaly'][0])
        
        return result
    
    def get_latent(self, data: np.ndarray) -> np.ndarray:
        """Get latent representation of data."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return self.encoder.predict(data, verbose=0)
    
    def save_model(self, path: str = "models/anomaly/autoencoder"):
        """Save trained model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(f"{path}.weights.h5")
        np.save(f"{path}_threshold.npy", self.threshold)
        logger.info(f"Saved Autoencoder model to {path}")
    
    def shutdown(self):
        """Clean shutdown."""
        self.model = None
        self.encoder = None
        self.is_initialized = False
        keras.backend.clear_session()
        logger.info("Autoencoder detector shutdown")
