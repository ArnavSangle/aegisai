"""
Combined Anomaly Detector
Ensemble of Isolation Forest and Autoencoder using cascade/parallel methods
"""

import numpy as np
from typing import Dict, Any, Optional, Literal
from loguru import logger

from ..core.base_module import BaseModule
from .isolation_forest import IsolationForestDetector
from .autoencoder import AutoencoderDetector


class AnomalyDetector(BaseModule):
    """
    Combined anomaly detector using Isolation Forest and Autoencoder.
    
    Supports multiple ensemble methods:
    - cascade: IF first, then AE for borderline cases
    - parallel: Both run simultaneously, results combined
    - voting: Majority voting between detectors
    """
    
    def __init__(self):
        super().__init__('anomaly')
        self.isolation_forest: Optional[IsolationForestDetector] = None
        self.autoencoder: Optional[AutoencoderDetector] = None
        
        # Ensemble configuration
        ensemble_config = self.config.get('ensemble', {})
        self.method: str = ensemble_config.get('method', 'cascade')
        self.weights: list = ensemble_config.get('weights', [0.4, 0.6])
        
        # Thresholds for cascade method
        self.if_low_threshold = 0.3  # Definitely normal
        self.if_high_threshold = 0.7  # Definitely anomaly
        
    def initialize(self) -> bool:
        """Initialize both detectors."""
        try:
            # Initialize Isolation Forest
            self.isolation_forest = IsolationForestDetector()
            if_success = self.isolation_forest.initialize()
            
            # Initialize Autoencoder
            self.autoencoder = AutoencoderDetector()
            ae_success = self.autoencoder.initialize()
            
            if not if_success:
                logger.warning("Isolation Forest failed to initialize")
            if not ae_success:
                logger.warning("Autoencoder failed to initialize, will use IF only")
            
            self.is_initialized = if_success or ae_success
            
            if self.is_initialized:
                logger.info(f"Anomaly Detector initialized with method: {self.method}")
            
            return self.is_initialized
            
        except Exception as e:
            logger.error(f"Failed to initialize Anomaly Detector: {e}")
            return False
    
    def fit(self, data: np.ndarray, **kwargs) -> None:
        """
        Train both detectors.
        
        Args:
            data: Training data of shape (n_samples, n_features)
            **kwargs: Additional arguments for training
        """
        if self.isolation_forest and self.isolation_forest.is_initialized:
            self.isolation_forest.fit(data)
        
        if self.autoencoder and self.autoencoder.is_initialized:
            self.autoencoder.fit(
                data,
                epochs=kwargs.get('epochs', 50),
                batch_size=kwargs.get('batch_size', 32)
            )
    
    def process(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies using ensemble method.
        
        Args:
            data: Input data
            
        Returns:
            Combined anomaly detection results
        """
        if not self.is_initialized:
            raise RuntimeError("Detector not initialized")
        
        # Handle single sample
        single_sample = data.ndim == 1
        if single_sample:
            data = data.reshape(1, -1)
        
        if self.method == 'cascade':
            result = self._cascade_detect(data)
        elif self.method == 'parallel':
            result = self._parallel_detect(data)
        elif self.method == 'voting':
            result = self._voting_detect(data)
        else:
            logger.warning(f"Unknown method {self.method}, using parallel")
            result = self._parallel_detect(data)
        
        # Simplify for single sample
        if single_sample:
            result['score'] = float(result['scores'][0])
            result['is_anomaly'] = bool(result['is_anomaly'][0])
            del result['scores']
        
        return result
    
    def _cascade_detect(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Cascade detection: IF first, then AE for uncertain cases.
        More efficient as AE only runs when needed.
        """
        n_samples = data.shape[0]
        final_scores = np.zeros(n_samples)
        final_anomalies = np.zeros(n_samples, dtype=bool)
        ae_used = np.zeros(n_samples, dtype=bool)
        
        # First pass: Isolation Forest
        if_result = self.isolation_forest.process(data)
        if_scores = if_result['scores']
        
        for i in range(n_samples):
            if if_scores[i] < self.if_low_threshold:
                # Definitely normal
                final_scores[i] = if_scores[i]
                final_anomalies[i] = False
            elif if_scores[i] > self.if_high_threshold:
                # Definitely anomaly
                final_scores[i] = if_scores[i]
                final_anomalies[i] = True
            else:
                # Uncertain - use Autoencoder
                if self.autoencoder and self.autoencoder.is_initialized:
                    ae_result = self.autoencoder.process(data[i:i+1])
                    ae_score = ae_result['scores']
                    if isinstance(ae_score, np.ndarray):
                        ae_score = ae_score[0]
                    
                    # Weighted combination
                    final_scores[i] = (
                        self.weights[0] * if_scores[i] + 
                        self.weights[1] * ae_score
                    )
                    final_anomalies[i] = final_scores[i] > 0.5
                    ae_used[i] = True
                else:
                    # No AE available, use IF result
                    final_scores[i] = if_scores[i]
                    final_anomalies[i] = if_scores[i] > 0.5
        
        return {
            'scores': final_scores,
            'is_anomaly': final_anomalies,
            'method': 'cascade',
            'if_scores': if_scores,
            'ae_used': ae_used
        }
    
    def _parallel_detect(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Parallel detection: Run both and combine scores.
        """
        # Isolation Forest
        if_result = self.isolation_forest.process(data)
        if_scores = if_result['scores']
        
        # Autoencoder
        if self.autoencoder and self.autoencoder.is_initialized:
            ae_result = self.autoencoder.process(data)
            ae_scores = ae_result['scores']
            if isinstance(ae_scores, (int, float)):
                ae_scores = np.array([ae_scores])
            
            # Weighted combination
            final_scores = (
                self.weights[0] * if_scores + 
                self.weights[1] * ae_scores
            )
        else:
            final_scores = if_scores
            ae_scores = None
        
        final_anomalies = final_scores > 0.5
        
        result = {
            'scores': final_scores,
            'is_anomaly': final_anomalies,
            'method': 'parallel',
            'if_scores': if_scores
        }
        
        if ae_scores is not None:
            result['ae_scores'] = ae_scores
        
        return result
    
    def _voting_detect(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Voting detection: Majority vote between detectors.
        """
        # Isolation Forest
        if_result = self.isolation_forest.process(data)
        if_anomalies = if_result['is_anomaly']
        
        # Autoencoder
        if self.autoencoder and self.autoencoder.is_initialized:
            ae_result = self.autoencoder.process(data)
            ae_anomalies = ae_result['is_anomaly']
            if isinstance(ae_anomalies, bool):
                ae_anomalies = np.array([ae_anomalies])
            
            # Both must agree for anomaly (conservative)
            final_anomalies = if_anomalies & ae_anomalies
            
            # Average scores
            final_scores = (if_result['scores'] + ae_result['scores']) / 2
        else:
            final_anomalies = if_anomalies
            final_scores = if_result['scores']
        
        return {
            'scores': final_scores,
            'is_anomaly': final_anomalies,
            'method': 'voting',
            'if_anomalies': if_anomalies,
            'ae_anomalies': ae_anomalies if self.autoencoder else None
        }
    
    def save_models(self, base_path: str = "models/anomaly"):
        """Save all models."""
        if self.isolation_forest:
            self.isolation_forest.save_model(f"{base_path}/isolation_forest.joblib")
        if self.autoencoder:
            self.autoencoder.save_model(f"{base_path}/autoencoder")
    
    def shutdown(self):
        """Clean shutdown of both detectors."""
        if self.isolation_forest:
            self.isolation_forest.shutdown()
        if self.autoencoder:
            self.autoencoder.shutdown()
        self.is_initialized = False
        logger.info("Anomaly Detector shutdown complete")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        status = super().get_status()
        status.update({
            'method': self.method,
            'isolation_forest': self.isolation_forest.get_status() if self.isolation_forest else None,
            'autoencoder': self.autoencoder.get_status() if self.autoencoder else None
        })
        return status
