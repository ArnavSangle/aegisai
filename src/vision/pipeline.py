"""
Vision Pipeline for AegisAI
Manages camera capture and inference pipeline
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

from ..core.base_module import BaseModule
from .mobilenet import MobileNetV3Detector
from .hailo_backend import HailoBackend


class VisionPipeline(BaseModule):
    """
    Complete vision pipeline for Raspberry Pi.
    Supports Pi Camera with AI HAT+ acceleration.
    """
    
    def __init__(self):
        super().__init__('vision')
        self.camera = None
        self.detector: Optional[MobileNetV3Detector] = None
        self.hailo: Optional[HailoBackend] = None
        
        # Configuration
        self.model_config = self.config.get('model', {})
        self.inference_config = self.config.get('inference', {})
        self.preprocess_config = self.config.get('preprocessing', {})
        
        # Resolution
        self.input_size = tuple(self.model_config.get('input_size', [224, 224, 3])[:2])
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Frame buffer for async capture
        self._frame_buffer = None
        self._buffer_lock = asyncio.Lock()
        
    def initialize(self) -> bool:
        """Initialize camera and detection model."""
        try:
            # Initialize camera
            if not self._init_camera():
                logger.warning("Camera initialization failed, using test mode")
            
            # Initialize Hailo backend if available
            use_hailo = self.inference_config.get('delegate', 'hailo') == 'hailo'
            if use_hailo:
                self.hailo = HailoBackend()
                if self.hailo.initialize():
                    logger.info("Hailo AI HAT+ backend initialized")
                else:
                    logger.warning("Hailo not available, falling back to CPU")
                    self.hailo = None
            
            # Initialize detector
            self.detector = MobileNetV3Detector(hailo_backend=self.hailo)
            self.detector.initialize()
            
            self.is_initialized = True
            logger.info("Vision Pipeline initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Vision Pipeline: {e}")
            return False
    
    def _init_camera(self) -> bool:
        """Initialize camera (Pi Camera or USB)."""
        if PICAMERA_AVAILABLE:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": self.input_size, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                logger.info(f"Pi Camera initialized at {self.input_size}")
                return True
            except Exception as e:
                logger.warning(f"Pi Camera failed: {e}")
        
        if CV2_AVAILABLE:
            try:
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_size[0])
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_size[1])
                    logger.info("USB Camera initialized")
                    return True
            except Exception as e:
                logger.warning(f"USB Camera failed: {e}")
        
        return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Frame as numpy array (H, W, 3) in RGB format, or None if failed
        """
        if self.camera is None:
            # Return test image
            return self._generate_test_frame()
        
        try:
            if PICAMERA_AVAILABLE and isinstance(self.camera, Picamera2):
                frame = self.camera.capture_array()
            else:
                ret, frame = self.camera.read()
                if not ret:
                    return None
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None
    
    def _generate_test_frame(self) -> np.ndarray:
        """Generate test frame for development."""
        frame = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
        return frame
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference.
        
        Args:
            frame: Input frame (H, W, 3)
            
        Returns:
            Preprocessed frame
        """
        # Resize if needed
        if frame.shape[:2] != self.input_size:
            frame = cv2.resize(frame, self.input_size) if CV2_AVAILABLE else frame
        
        # Convert to float
        processed = frame.astype(np.float32)
        
        # Normalize
        if self.preprocess_config.get('normalize', True):
            mean = np.array(self.preprocess_config.get('mean', [0.485, 0.456, 0.406]))
            std = np.array(self.preprocess_config.get('std', [0.229, 0.224, 0.225]))
            processed = processed / 255.0
            processed = (processed - mean) / std
        
        return processed
    
    def process(self, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process frame through detection pipeline.
        
        Args:
            frame: Input frame, or None to capture from camera
            
        Returns:
            Detection results
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
        
        # Capture if no frame provided
        if frame is None:
            frame = self.capture_frame()
            if frame is None:
                return {'error': 'Frame capture failed', 'detections': []}
        
        # Preprocess
        processed = self.preprocess(frame)
        
        # Run detection
        result = self.detector.process(processed)
        
        # Apply NMS and filtering
        result = self._post_process(result)
        
        # Add raw frame reference
        result['frame_shape'] = frame.shape
        
        return result
    
    def _post_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply post-processing: NMS, confidence filtering."""
        conf_threshold = self.inference_config.get('confidence_threshold', 0.5)
        nms_threshold = self.inference_config.get('nms_threshold', 0.4)
        
        detections = result.get('detections', [])
        
        # Filter by confidence
        detections = [d for d in detections if d.get('confidence', 0) >= conf_threshold]
        
        # Apply NMS
        if len(detections) > 1 and CV2_AVAILABLE:
            boxes = np.array([d['bbox'] for d in detections])
            scores = np.array([d['confidence'] for d in detections])
            
            # Convert to cv2 format (x, y, w, h)
            boxes_xywh = boxes.copy()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
            
            indices = cv2.dnn.NMSBoxes(
                boxes_xywh.tolist(), 
                scores.tolist(),
                conf_threshold,
                nms_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                detections = [detections[i] for i in indices]
        
        result['detections'] = detections
        result['num_detections'] = len(detections)
        
        return result
    
    async def capture_and_process_async(self) -> Dict[str, Any]:
        """Async version of capture and process."""
        loop = asyncio.get_event_loop()
        
        # Capture in thread pool
        frame = await loop.run_in_executor(self._executor, self.capture_frame)
        
        if frame is None:
            return {'error': 'Frame capture failed', 'detections': []}
        
        # Process in thread pool
        result = await loop.run_in_executor(self._executor, self.process, frame)
        
        return result
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from frame (for RL observations).
        
        Args:
            frame: Input frame
            
        Returns:
            Feature vector
        """
        processed = self.preprocess(frame)
        return self.detector.extract_features(processed)
    
    def start_streaming(self, callback):
        """
        Start continuous frame streaming with callback.
        
        Args:
            callback: Function called with each processed frame result
        """
        self._streaming = True
        
        async def stream_loop():
            while self._streaming:
                result = await self.capture_and_process_async()
                callback(result)
                await asyncio.sleep(1/30)  # 30 FPS
        
        asyncio.create_task(stream_loop())
    
    def stop_streaming(self):
        """Stop continuous streaming."""
        self._streaming = False
    
    def shutdown(self):
        """Clean shutdown of vision pipeline."""
        self._streaming = False
        
        if self.camera is not None:
            if PICAMERA_AVAILABLE and isinstance(self.camera, Picamera2):
                self.camera.stop()
            elif CV2_AVAILABLE:
                self.camera.release()
        
        if self.detector:
            self.detector.shutdown()
        
        if self.hailo:
            self.hailo.shutdown()
        
        self._executor.shutdown(wait=False)
        self.is_initialized = False
        logger.info("Vision Pipeline shutdown")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        status = super().get_status()
        status.update({
            'camera_available': self.camera is not None,
            'hailo_available': self.hailo is not None and self.hailo.is_initialized,
            'input_size': self.input_size,
            'detector': self.detector.get_status() if self.detector else None
        })
        return status
