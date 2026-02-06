"""
AegisAI Main Entry Point
Autonomous River Sampling Buoy with Agentic AI
Orchestrates all AI modules for water quality monitoring
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
    Main orchestrator for the AegisAI water buoy system.
    Manages all AI modules and coordinates their execution.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AegisAI buoy system.
        
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
        logger.info("AegisAI - Autonomous Water Quality Buoy")
        logger.info(f"Version: {self.config.get('system.version', '1.0.0')}")
        logger.info("=" * 60)
        
        # Initialize modules (lazy loading)
        self.modules: Dict[str, Any] = {}
        self._running = False
        
        # Buoy state
        self._power_mode = 'active'  # active, alert, low_power
        self._samples_taken = 0
        self._vial_capacity = self.config.get('sampling.vial_capacity', 12)
        
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
            
            # Vision Pipeline (optional for visual contaminant detection)
            if self.config.get('vision.enabled', False):
                logger.info("Loading Vision module...")
                self.modules['vision'] = VisionPipeline()
                self.modules['vision'].initialize()
            
            # Fleet Management (for multi-buoy deployments)
            if self.config.get('fleet.num_buoys', 1) > 1:
                logger.info("Loading Fleet Management module...")
                self.modules['fleet'] = FleetManager()
                self.modules['fleet'].initialize()
            
            # MCU Communication (sensors and pump control)
            logger.info("Loading MCU Communication module...")
            self.modules['mcu'] = MCUCommunicator()
            self.modules['mcu'].initialize()
            
            logger.info(f"Successfully initialized {len(self.modules)} modules")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            return False
    
    async def run_async(self):
        """Run the main water quality monitoring loop asynchronously."""
        self._running = True
        logger.info("Starting AegisAI buoy monitoring loop...")
        
        # Get inference interval based on power mode
        inference_interval = self._get_inference_interval()
        
        while self._running:
            try:
                # Get water quality sensor data from MCU
                sensor_data = await self.modules['mcu'].read_water_quality_async()
                
                # Update power mode based on battery level
                self._update_power_mode(sensor_data)
                
                # Run anomaly detection on water quality readings
                anomaly_result = self.modules['anomaly'].process(sensor_data)
                
                # Run prediction for future water quality
                prediction = self.modules['prediction'].process(sensor_data)
                
                # Get visual data if camera is enabled
                vision_data = None
                if 'vision' in self.modules:
                    vision_data = await self.modules['vision'].capture_and_process_async()
                
                # Build observation for decision making
                observation = self._build_observation(
                    sensor_data, anomaly_result, prediction, vision_data
                )
                
                # Get action from PPO agent
                action_result = self.modules['decision'].process(observation)
                action = action_result.get('action', 0)
                
                # Execute action
                await self._execute_action(action, anomaly_result)
                
                # Fleet coordination if enabled
                if 'fleet' in self.modules:
                    await self.modules['fleet'].broadcast_status_async(
                        sensor_data, anomaly_result
                    )
                
                # Wait based on power mode
                inference_interval = self._get_inference_interval()
                await asyncio.sleep(inference_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def _execute_action(self, action: int, anomaly_result: Dict):
        """Execute the action decided by PPO agent."""
        action_names = self.config.get('decision.action_space.actions', [])
        
        if action == 0:  # sample_now
            if self._samples_taken < self._vial_capacity:
                logger.info("Taking water sample...")
                await self.modules['mcu'].trigger_sample_async(self._samples_taken)
                self._samples_taken += 1
                logger.info(f"Sample {self._samples_taken}/{self._vial_capacity} collected")
            else:
                logger.warning("Sample vials full - cannot take more samples")
                
        elif action == 1:  # wait_low_power
            self._power_mode = 'low_power'
            
        elif action == 2:  # wait_normal
            self._power_mode = 'active'
            
        elif action == 3:  # increase_sampling_rate
            self._power_mode = 'alert'
            
        elif action == 4:  # decrease_sampling_rate
            self._power_mode = 'active'
            
        elif action == 5:  # broadcast_alert
            if 'fleet' in self.modules:
                await self.modules['fleet'].broadcast_alert_async(anomaly_result)
            logger.warning(f"ANOMALY ALERT: {anomaly_result}")
    
    def _get_inference_interval(self) -> float:
        """Get inference interval based on current power mode."""
        modes = self.config.get('power.modes', {})
        mode_config = modes.get(self._power_mode, {})
        return mode_config.get('inference_interval_s', 600)  # Default 10 min
    
    def _update_power_mode(self, sensor_data: Dict):
        """Update power mode based on battery level."""
        battery = sensor_data.get('battery_voltage', 12.0)
        battery_pct = (battery - 10.5) / (12.6 - 10.5)  # Approximate for 3S LiPo
        
        thresholds = self.config.get('power.thresholds', {})
        
        if battery_pct < thresholds.get('critical_battery', 0.1):
            self._power_mode = 'low_power'
            logger.warning(f"Critical battery: {battery_pct*100:.1f}%")
        elif battery_pct < thresholds.get('low_battery', 0.2):
            if self._power_mode != 'alert':  # Don't override alert mode
                self._power_mode = 'low_power'
    
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
