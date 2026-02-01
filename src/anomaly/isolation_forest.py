"""
Isolation Forest Detector
Unsupervised anomaly detection using sklearn's Isolation Forest
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path
from loguru import logger

from ..core.base_module import BaseModule


class IsolationForestDetector(BaseModule):
    """
    Isolation Forest based anomaly detector.
    Good for detecting global anomalies with low computational cost.
    """
    
    def __init__(self):
        super().__init__('anomaly')
        self.model: Optional[IsolationForest] = None
        self.if_config = self.config.get('isolation_forest', {})
        
    def initialize(self) -> bool:
        """Initialize Isolation Forest model."""
        try:
            self.model = IsolationForest(
                n_estimators=self.if_config.get('n_estimators', 100),
                contamination=self.if_config.get('contamination', 0.1),
                max_samples=self.if_config.get('max_samples', 'auto'),
                random_state=self.if_config.get('random_state', 42),
                n_jobs=-1,  # Use all CPU cores
                warm_start=True  # Allow incremental training
            )
            
            # Try to load pre-trained model
            model_path = Path("models/anomaly/isolation_forest.joblib")
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("Loaded pre-trained Isolation Forest model")
            
            self.is_initialized = True
            logger.info("Isolation Forest detector initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Isolation Forest: {e}")
            return False
    
    def fit(self, data: np.ndarray) -> None:
        """
        Train the Isolation Forest model.
        
        Args:
            data: Training data of shape (n_samples, n_features)
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")
            
        logger.info(f"Training Isolation Forest on {data.shape[0]} samples")
        self.model.fit(data)
        logger.info("Isolation Forest training complete")
    
    def process(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies in input data.
        
        Args:
            data: Input data of shape (n_samples, n_features) or (n_features,)
            
        Returns:
            Dictionary with anomaly scores and predictions
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")
        
        # Handle single sample
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Get anomaly scores (negative = more anomalous)
        scores = self.model.score_samples(data)
        
        # Get predictions (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(data)
        
        # Normalize scores to [0, 1] where 1 = most anomalous
        normalized_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return {
            'scores': normalized_scores,
            'predictions': predictions,
            'is_anomaly': predictions == -1,
            'raw_scores': scores
        }
    
    def save_model(self, path: str = "models/anomaly/isolation_forest.joblib"):
        """Save trained model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved Isolation Forest model to {path}")
    
    def shutdown(self):
        """Clean shutdown."""
        self.model = None
        self.is_initialized = False
        logger.info("Isolation Forest detector shutdown")
