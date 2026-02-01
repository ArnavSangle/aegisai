"""
MobileNetV3 Detector for AegisAI
Optimized for Raspberry Pi with TFLite and Hailo support
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from loguru import logger

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.base_module import BaseModule


class MobileNetV3Detector(BaseModule):
    """
    MobileNetV3-based object detection and classification.
    Supports TFLite and Hailo acceleration.
    """
    
    # COCO class names for detection
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, hailo_backend=None):
        """
        Initialize MobileNetV3 detector.
        
        Args:
            hailo_backend: Optional HailoBackend for AI HAT+ acceleration
        """
        super().__init__('vision')
        self.hailo = hailo_backend
        self.tflite_interpreter = None
        self.model = None
        self.feature_extractor = None
        
        # Model configuration
        model_config = self.config.get('model', {})
        self.input_size = tuple(model_config.get('input_size', [224, 224, 3]))
        self.num_classes = model_config.get('num_classes', 80)
        
    def initialize(self) -> bool:
        """Initialize detection model."""
        try:
            # Priority: Hailo > TFLite > TensorFlow
            if self.hailo and self.hailo.is_initialized:
                if self._init_hailo():
                    self.is_initialized = True
                    return True
            
            # Try TFLite
            tflite_path = Path("models/vision/mobilenet_v3_ssd.tflite")
            if tflite_path.exists():
                self._load_tflite(tflite_path)
                self.is_initialized = True
                logger.info("Loaded TFLite MobileNetV3 model")
                return True
            
            # Fall back to full TensorFlow
            if TF_AVAILABLE:
                self._build_model()
                self.is_initialized = True
                logger.info("Built TensorFlow MobileNetV3 model")
                return True
            
            logger.error("No backend available for MobileNetV3")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize MobileNetV3: {e}")
            return False
    
    def _init_hailo(self) -> bool:
        """Initialize Hailo backend for detection."""
        hef_path = "models/hailo/mobilenet_v3_ssd.hef"
        if Path(hef_path).exists():
            return self.hailo.load_model(hef_path, "mobilenet_v3")
        return False
    
    def _load_tflite(self, path: Path):
        """Load TFLite model."""
        try:
            import tflite_runtime.interpreter as tflite
            self.tflite_interpreter = tflite.Interpreter(model_path=str(path))
        except ImportError:
            self.tflite_interpreter = tf.lite.Interpreter(model_path=str(path))
        
        self.tflite_interpreter.allocate_tensors()
        self._input_details = self.tflite_interpreter.get_input_details()
        self._output_details = self.tflite_interpreter.get_output_details()
    
    def _build_model(self):
        """Build MobileNetV3 model with TensorFlow."""
        from tensorflow.keras.applications import MobileNetV3Large
        from tensorflow.keras import layers, Model
        
        # Load pretrained MobileNetV3
        base_model = MobileNetV3Large(
            input_shape=self.input_size,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Feature extractor
        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=base_model.output,
            name='feature_extractor'
        )
        
        # Classification head
        inputs = layers.Input(shape=self.input_size)
        features = base_model(inputs)
        outputs = layers.Dense(self.num_classes, activation='softmax')(features)
        
        self.model = Model(inputs, outputs, name='mobilenet_v3_classifier')
        
        # Load custom weights if available
        weights_path = Path("models/vision/mobilenet_v3.weights.h5")
        if weights_path.exists():
            self.model.load_weights(str(weights_path))
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run detection on preprocessed image.
        
        Args:
            image: Preprocessed image (H, W, 3) normalized
            
        Returns:
            Detection results with bounding boxes and classes
        """
        if not self.is_initialized:
            raise RuntimeError("Detector not initialized")
        
        # Add batch dimension
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        # Run inference
        if self.hailo and self.hailo.is_initialized:
            outputs = self.hailo.infer("mobilenet_v3", image)
            return self._parse_hailo_outputs(outputs)
        elif self.tflite_interpreter:
            return self._tflite_detect(image)
        else:
            return self._tf_detect(image)
    
    def _tflite_detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Run TFLite detection."""
        # Ensure correct dtype
        input_dtype = self._input_details[0]['dtype']
        image = image.astype(input_dtype)
        
        # Set input
        self.tflite_interpreter.set_tensor(
            self._input_details[0]['index'], image
        )
        
        # Run inference
        self.tflite_interpreter.invoke()
        
        # Get outputs (SSD format: boxes, classes, scores, num_detections)
        boxes = self.tflite_interpreter.get_tensor(self._output_details[0]['index'])
        classes = self.tflite_interpreter.get_tensor(self._output_details[1]['index'])
        scores = self.tflite_interpreter.get_tensor(self._output_details[2]['index'])
        num_det = int(self.tflite_interpreter.get_tensor(self._output_details[3]['index'])[0])
        
        return self._format_detections(boxes[0], classes[0], scores[0], num_det)
    
    def _tf_detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Run TensorFlow detection (classification mode)."""
        predictions = self.model.predict(image, verbose=0)
        
        # Get top predictions
        top_k = 5
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        detections = []
        for idx in top_indices:
            if predictions[0][idx] > 0.1:  # Confidence threshold
                detections.append({
                    'class_id': int(idx),
                    'class_name': self._get_class_name(idx),
                    'confidence': float(predictions[0][idx]),
                    'bbox': [0, 0, 1, 1]  # Full image for classification
                })
        
        return {
            'detections': detections,
            'raw_predictions': predictions[0].tolist()
        }
    
    def _parse_hailo_outputs(self, outputs: Dict) -> Dict[str, Any]:
        """Parse Hailo inference outputs."""
        # Hailo SSD output format
        boxes = outputs.get('boxes', np.array([]))
        scores = outputs.get('scores', np.array([]))
        classes = outputs.get('classes', np.array([]))
        
        return self._format_detections(
            boxes, classes, scores, len(boxes)
        )
    
    def _format_detections(
        self,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
        num_detections: int
    ) -> Dict[str, Any]:
        """Format raw detections into structured output."""
        detections = []
        
        for i in range(min(num_detections, len(boxes))):
            if scores[i] < 0.1:
                continue
                
            class_id = int(classes[i])
            
            detections.append({
                'class_id': class_id,
                'class_name': self._get_class_name(class_id),
                'confidence': float(scores[i]),
                'bbox': boxes[i].tolist()  # [y1, x1, y2, x2] or [x1, y1, x2, y2]
            })
        
        return {
            'detections': detections,
            'num_detections': len(detections)
        }
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        if 0 <= class_id < len(self.COCO_CLASSES):
            return self.COCO_CLASSES[class_id]
        return f"class_{class_id}"
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Feature vector (1280-dim for MobileNetV3-Large)
        """
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        if self.feature_extractor:
            features = self.feature_extractor.predict(image, verbose=0)
            return features.flatten()
        elif self.tflite_interpreter:
            # Use intermediate layer output if available
            # Fall back to flattened final output
            return self._tflite_detect(image).get('raw_predictions', np.zeros(1280))
        else:
            return np.zeros(1280)
    
    def shutdown(self):
        """Clean shutdown."""
        self.model = None
        self.feature_extractor = None
        self.tflite_interpreter = None
        self.is_initialized = False
        if TF_AVAILABLE:
            tf.keras.backend.clear_session()
        logger.info("MobileNetV3 Detector shutdown")
