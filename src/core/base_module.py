"""
Base Module for AegisAI Components
Abstract base class for all AI modules
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from loguru import logger

from .config_loader import ConfigLoader


class BaseModule(ABC):
    """
    Abstract base class for all AegisAI modules.
    Provides common interface for initialization, inference, and lifecycle management.
    """
    
    def __init__(self, config_section: str):
        """
        Initialize base module.
        
        Args:
            config_section: Name of the configuration section for this module
        """
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_section(config_section)
        self.is_initialized = False
        self.module_name = self.__class__.__name__
        
        logger.info(f"Initializing module: {self.module_name}")
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the module (load models, setup connections, etc.)
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process input data through the module.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed output
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """Clean shutdown of the module."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get module status information.
        
        Returns:
            Status dictionary
        """
        return {
            'module': self.module_name,
            'initialized': self.is_initialized,
            'config_loaded': bool(self.config)
        }
    
    def validate_input(self, data: np.ndarray, expected_shape: tuple) -> bool:
        """
        Validate input data shape.
        
        Args:
            data: Input numpy array
            expected_shape: Expected shape (use -1 for variable dimensions)
            
        Returns:
            True if valid
        """
        if not isinstance(data, np.ndarray):
            logger.error(f"{self.module_name}: Input must be numpy array")
            return False
            
        if len(data.shape) != len(expected_shape):
            logger.error(f"{self.module_name}: Expected {len(expected_shape)}D array, got {len(data.shape)}D")
            return False
            
        for i, (actual, expected) in enumerate(zip(data.shape, expected_shape)):
            if expected != -1 and actual != expected:
                logger.error(f"{self.module_name}: Dimension {i} mismatch: expected {expected}, got {actual}")
                return False
                
        return True
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
