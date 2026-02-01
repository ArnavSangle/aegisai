"""
TFLite Model Converter
Optimizes models for Raspberry Pi deployment
"""

import numpy as np
from typing import Optional, Callable
from pathlib import Path
from loguru import logger

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TFLiteConverter:
    """
    Converts TensorFlow models to TFLite format with various optimizations.
    Supports quantization for efficient inference on Raspberry Pi.
    """
    
    def __init__(self, model):
        """
        Initialize converter.
        
        Args:
            model: TensorFlow/Keras model to convert
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for conversion")
        
        self.model = model
        
    def convert(
        self,
        output_path: str,
        quantization: str = "none",
        representative_dataset: Optional[Callable] = None
    ) -> str:
        """
        Convert model to TFLite format.
        
        Args:
            output_path: Path to save TFLite model
            quantization: Quantization type:
                - 'none': No quantization (float32)
                - 'float16': Float16 quantization (2x smaller, good accuracy)
                - 'int8': Full integer quantization (4x smaller, requires calibration)
                - 'dynamic': Dynamic range quantization
            representative_dataset: Generator function for int8 calibration
            
        Returns:
            Path to saved model
        """
        logger.info(f"Converting model with {quantization} quantization")
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply quantization
        if quantization == "none":
            pass  # Default float32
            
        elif quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Int8 requires representative dataset for calibration
            if representative_dataset:
                converter.representative_dataset = representative_dataset
            else:
                logger.warning("No representative dataset provided for int8 quantization, "
                             "using random data for calibration")
                converter.representative_dataset = self._default_representative_dataset()
        
        # Enable experimental optimizations
        converter.experimental_new_converter = True
        
        # Convert
        try:
            tflite_model = converter.convert()
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise
        
        # Save model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Log model size
        size_mb = len(tflite_model) / (1024 * 1024)
        logger.info(f"Saved TFLite model: {output_path} ({size_mb:.2f} MB)")
        
        return str(output_path)
    
    def _default_representative_dataset(self):
        """Generate default representative dataset for int8 calibration."""
        def representative_dataset_gen():
            # Get input shape from model
            input_shape = self.model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            
            # Remove batch dimension
            shape = input_shape[1:]
            
            # Generate random samples
            for _ in range(100):
                sample = np.random.randn(1, *shape).astype(np.float32)
                yield [sample]
        
        return representative_dataset_gen
    
    @staticmethod
    def verify_model(tflite_path: str, test_input: np.ndarray) -> dict:
        """
        Verify TFLite model works correctly.
        
        Args:
            tflite_path: Path to TFLite model
            test_input: Test input array
            
        Returns:
            Verification results
        """
        logger.info(f"Verifying TFLite model: {tflite_path}")
        
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Ensure correct shape and dtype
        expected_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        if test_input.shape != tuple(expected_shape):
            test_input = test_input.reshape(expected_shape)
        
        test_input = test_input.astype(input_dtype)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        return {
            'success': True,
            'input_shape': list(expected_shape),
            'input_dtype': str(input_dtype),
            'output_shape': list(output.shape),
            'output_dtype': str(output.dtype),
            'output_sample': output[0].tolist() if output.ndim > 0 else output.tolist()
        }
    
    @staticmethod
    def benchmark_model(tflite_path: str, test_input: np.ndarray, num_runs: int = 100) -> dict:
        """
        Benchmark TFLite model inference time.
        
        Args:
            tflite_path: Path to TFLite model
            test_input: Test input array
            num_runs: Number of inference runs
            
        Returns:
            Benchmark results
        """
        import time
        
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input
        input_dtype = input_details[0]['dtype']
        test_input = test_input.astype(input_dtype)
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
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
