"""
Hailo Backend for AI HAT+
Hardware-accelerated inference using Hailo-8L NPU
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger

try:
    from hailo_platform import (
        VDevice, HailoStreamInterface, ConfigureParams,
        InputVStreamParams, OutputVStreamParams,
        FormatType, HailoSchedulingAlgorithm
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.info("Hailo SDK not available")


class HailoBackend:
    """
    Hailo-8L backend for AI HAT+ acceleration.
    Provides up to 13 TOPS for neural network inference.
    """
    
    def __init__(self):
        """Initialize Hailo backend."""
        self.device = None
        self.configured_networks: Dict[str, Any] = {}
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize Hailo device."""
        if not HAILO_AVAILABLE:
            logger.warning("Hailo SDK not installed")
            return False
            
        try:
            # Create virtual device (auto-detects Hailo chip)
            self.device = VDevice()
            
            # Get device info
            device_ids = self.device.get_physical_devices_ids()
            logger.info(f"Found Hailo devices: {device_ids}")
            
            self.is_initialized = True
            logger.info("Hailo backend initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            return False
    
    def load_model(self, hef_path: str, model_name: str) -> bool:
        """
        Load a compiled HEF model onto the Hailo device.
        
        Args:
            hef_path: Path to the .hef file
            model_name: Name to reference this model
            
        Returns:
            True if model loaded successfully
        """
        if not self.is_initialized:
            logger.error("Hailo not initialized")
            return False
            
        hef_path = Path(hef_path)
        if not hef_path.exists():
            logger.error(f"HEF file not found: {hef_path}")
            return False
        
        try:
            # Configure network
            configure_params = ConfigureParams.create_from_hef(
                hef_path=str(hef_path),
                interface=HailoStreamInterface.PCIe
            )
            
            # Configure on device
            network_group = self.device.configure(
                configure_params,
                HailoSchedulingAlgorithm.ROUND_ROBIN
            )[0]
            
            # Get input/output stream info
            input_vstreams_params = InputVStreamParams.make_from_network_group(
                network_group, 
                quantized=False,
                format_type=FormatType.FLOAT32
            )
            
            output_vstreams_params = OutputVStreamParams.make_from_network_group(
                network_group,
                quantized=False,
                format_type=FormatType.FLOAT32
            )
            
            # Store network configuration
            self.configured_networks[model_name] = {
                'network_group': network_group,
                'input_params': input_vstreams_params,
                'output_params': output_vstreams_params,
                'hef_path': str(hef_path)
            }
            
            logger.info(f"Loaded model '{model_name}' from {hef_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return False
    
    def infer(self, model_name: str, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on loaded model.
        
        Args:
            model_name: Name of the loaded model
            input_data: Input tensor (batch, height, width, channels)
            
        Returns:
            Dictionary of output tensors
        """
        if model_name not in self.configured_networks:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        network_config = self.configured_networks[model_name]
        network_group = network_config['network_group']
        
        try:
            # Create input/output virtual streams
            with network_group.create_input_vstreams(network_config['input_params']) as input_vstreams, \
                 network_group.create_output_vstreams(network_config['output_params']) as output_vstreams:
                
                # Send input data
                for input_vstream in input_vstreams:
                    input_vstream.send(input_data)
                
                # Receive outputs
                outputs = {}
                for output_vstream in output_vstreams:
                    output_data = output_vstream.recv()
                    outputs[output_vstream.name] = output_data
                
                return outputs
                
        except Exception as e:
            logger.error(f"Inference failed for '{model_name}': {e}")
            raise
    
    def infer_async(self, model_name: str, input_data: np.ndarray, callback):
        """
        Run async inference with callback.
        
        Args:
            model_name: Name of the loaded model
            input_data: Input tensor
            callback: Function called with results
        """
        # For now, run sync and call callback
        # TODO: Implement true async with Hailo async API
        try:
            results = self.infer(model_name, input_data)
            callback(results, None)
        except Exception as e:
            callback(None, e)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        if model_name not in self.configured_networks:
            return {}
        
        config = self.configured_networks[model_name]
        network_group = config['network_group']
        
        # Get input/output info
        input_info = []
        for param in config['input_params']:
            input_info.append({
                'name': param.name,
                'shape': param.shape,
                'format': str(param.format_type)
            })
        
        output_info = []
        for param in config['output_params']:
            output_info.append({
                'name': param.name,
                'shape': param.shape,
                'format': str(param.format_type)
            })
        
        return {
            'model_name': model_name,
            'hef_path': config['hef_path'],
            'inputs': input_info,
            'outputs': output_info
        }
    
    def unload_model(self, model_name: str):
        """Unload a model from the device."""
        if model_name in self.configured_networks:
            del self.configured_networks[model_name]
            logger.info(f"Unloaded model '{model_name}'")
    
    def benchmark(self, model_name: str, input_data: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            model_name: Name of the loaded model
            input_data: Input tensor
            num_runs: Number of inference runs
            
        Returns:
            Benchmark results
        """
        import time
        
        # Warmup
        for _ in range(10):
            self.infer(model_name, input_data)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.infer(model_name, input_data)
            times.append(time.perf_counter() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            'num_runs': num_runs,
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'fps': 1000.0 / np.mean(times)
        }
    
    def shutdown(self):
        """Clean shutdown of Hailo device."""
        self.configured_networks.clear()
        self.device = None
        self.is_initialized = False
        logger.info("Hailo backend shutdown")


# Fallback for development without Hailo
class MockHailoBackend:
    """Mock Hailo backend for development/testing."""
    
    def __init__(self):
        self.is_initialized = False
        self.configured_networks = {}
        
    def initialize(self) -> bool:
        logger.info("Using Mock Hailo backend (no hardware)")
        self.is_initialized = True
        return True
    
    def load_model(self, hef_path: str, model_name: str) -> bool:
        self.configured_networks[model_name] = {'hef_path': hef_path}
        return True
    
    def infer(self, model_name: str, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        # Return dummy outputs
        batch_size = input_data.shape[0] if input_data.ndim > 0 else 1
        return {
            'boxes': np.random.rand(batch_size, 10, 4).astype(np.float32),
            'scores': np.random.rand(batch_size, 10).astype(np.float32),
            'classes': np.random.randint(0, 80, (batch_size, 10)).astype(np.float32)
        }
    
    def shutdown(self):
        self.is_initialized = False
