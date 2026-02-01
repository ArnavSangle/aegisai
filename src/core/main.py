"""
AegisAI Main Entry Point
Orchestrates all AI modules for the Raspberry Pi 5 infrastructure
"""

import asyncio
import signal
import sys
from typing import Dict, Any, Optional
from loguru import logger

from .config_loader import ConfigLoader
from .logger import setup_logger
from ..anomaly import AnomalyDetector
from ..prediction import LSTMPredictor
from ..decision import PPOAgent
from ..vision import VisionPipeline
from ..fleet import FleetManager
from ..mcu import MCUCommunicator


class AegisAI:
    """
    Main orchestrator for the AegisAI system.
    Manages all AI modules and coordinates their execution.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AegisAI system.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = ConfigLoader()
        if config_path:
            self.config.load_config(config_path)
        
        # Setup logging
        log_level = self.config.get('system.log_level', 'INFO')
        log_file = self.config.get('paths.logs', 'logs/') + 'aegis.log'
        setup_logger(log_level=log_level, log_file=log_file)
        
        logger.info("=" * 60)
        logger.info("AegisAI - Raspberry Pi 5 AI Infrastructure")
        logger.info(f"Version: {self.config.get('system.version', '1.0.0')}")
        logger.info("=" * 60)
        
        # Initialize modules (lazy loading)
        self.modules: Dict[str, Any] = {}
        self._running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum}, initiating shutdown...")
        self._running = False
    
    def initialize_modules(self):
        """Initialize all AI modules."""
        logger.info("Initializing AI modules...")
        
        try:
            # Anomaly Detection
            logger.info("Loading Anomaly Detection module...")
            self.modules['anomaly'] = AnomalyDetector()
            self.modules['anomaly'].initialize()
            
            # Prediction Engine
            logger.info("Loading Prediction module...")
            self.modules['prediction'] = LSTMPredictor()
            self.modules['prediction'].initialize()
            
            # Decision Making (PPO)
            logger.info("Loading Decision module...")
            self.modules['decision'] = PPOAgent()
            self.modules['decision'].initialize()
            
            # Vision Pipeline
            if self.config.get('hardware.camera.enabled', False):
                logger.info("Loading Vision module...")
                self.modules['vision'] = VisionPipeline()
                self.modules['vision'].initialize()
            
            # Fleet Management
            if self.config.get('fleet.num_agents', 0) > 1:
                logger.info("Loading Fleet Management module...")
                self.modules['fleet'] = FleetManager()
                self.modules['fleet'].initialize()
            
            # MCU Communication
            logger.info("Loading MCU Communication module...")
            self.modules['mcu'] = MCUCommunicator()
            self.modules['mcu'].initialize()
            
            logger.info(f"Successfully initialized {len(self.modules)} modules")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            return False
    
    async def run_async(self):
        """Run the main processing loop asynchronously."""
        self._running = True
        logger.info("Starting AegisAI main loop...")
        
        while self._running:
            try:
                # Get sensor data from MCU
                sensor_data = await self.modules['mcu'].read_sensors_async()
                
                # Run anomaly detection
                anomaly_result = self.modules['anomaly'].process(sensor_data)
                
                # Run prediction
                prediction = self.modules['prediction'].process(sensor_data)
                
                # Get vision data if available
                vision_data = None
                if 'vision' in self.modules:
                    vision_data = await self.modules['vision'].capture_and_process_async()
                
                # Combine observations for decision making
                observation = self._build_observation(
                    sensor_data, anomaly_result, prediction, vision_data
                )
                
                # Get action from PPO agent
                action = self.modules['decision'].process(observation)
                
                # Execute action via MCU
                await self.modules['mcu'].execute_action_async(action)
                
                # Fleet coordination if enabled
                if 'fleet' in self.modules:
                    await self.modules['fleet'].coordinate_async(action)
                
                # Small delay to control loop rate
                await asyncio.sleep(0.01)  # 100Hz max
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(0.1)
    
    def _build_observation(
        self,
        sensor_data: Dict,
        anomaly_result: Dict,
        prediction: Dict,
        vision_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Build unified observation for decision making."""
        observation = {
            'sensors': sensor_data,
            'anomaly_score': anomaly_result.get('score', 0.0),
            'is_anomaly': anomaly_result.get('is_anomaly', False),
            'prediction': prediction.get('forecast', []),
            'prediction_confidence': prediction.get('confidence', 0.0)
        }
        
        if vision_data:
            observation['vision'] = {
                'detections': vision_data.get('detections', []),
                'features': vision_data.get('features', [])
            }
        
        return observation
    
    def run(self):
        """Run the system (blocking)."""
        if not self.initialize_modules():
            logger.error("Failed to initialize modules, exiting")
            sys.exit(1)
        
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all modules."""
        logger.info("Shutting down AegisAI...")
        
        for name, module in self.modules.items():
            try:
                logger.info(f"Shutting down {name}...")
                module.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
        
        logger.info("AegisAI shutdown complete")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        status = {
            'system': self.config.get('system.name'),
            'version': self.config.get('system.version'),
            'running': self._running,
            'modules': {}
        }
        
        for name, module in self.modules.items():
            status['modules'][name] = module.get_status()
        
        return status


def main():
    """Main entry point."""
    system = AegisAI()
    system.run()


if __name__ == "__main__":
    main()
