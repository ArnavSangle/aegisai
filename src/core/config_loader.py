"""
Configuration Loader for AegisAI
Handles YAML configuration loading and validation
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from loguru import logger


class ConfigLoader:
    """Centralized configuration management for AegisAI."""
    
    _instance: Optional['ConfigLoader'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls) -> 'ConfigLoader':
        """Singleton pattern for configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Optional path to config file. Defaults to config/config.yaml
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Find project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)
            
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
            
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            
            # Load additional config files
            self._load_hailo_config(config_path.parent)
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
            
        return self._config
    
    def _load_hailo_config(self, config_dir: Path):
        """Load Hailo-specific configuration."""
        hailo_path = config_dir / "hailo_config.yaml"
        if hailo_path.exists():
            with open(hailo_path, 'r') as f:
                hailo_config = yaml.safe_load(f)
                self._config['hailo'] = hailo_config.get('hailo', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., 'anomaly.isolation_forest.n_estimators')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config
    
    def validate(self) -> bool:
        """Validate configuration has required fields."""
        required_sections = ['system', 'hardware', 'anomaly', 'prediction', 
                           'decision', 'vision', 'fleet', 'mcu']
        
        for section in required_sections:
            if section not in self._config:
                logger.warning(f"Missing required config section: {section}")
                return False
                
        return True


# Global config instance
config = ConfigLoader()
