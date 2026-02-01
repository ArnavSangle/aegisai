# AegisAI API Reference

## Core Classes

### `AegisAI` (Main Orchestrator)

```python
from src.core.main import AegisAI

# Initialize
aegis = AegisAI()
aegis.initialize()

# Run main loop
await aegis.run()

# Shutdown
aegis.shutdown()
```

#### Methods

| Method | Description |
|--------|-------------|
| `initialize()` | Initialize all modules |
| `run()` | Start async processing loop |
| `shutdown()` | Gracefully stop all modules |
| `get_status()` | Get current system status |

---

## Anomaly Detection

### `AnomalyDetector`

Combined detector using Isolation Forest and Autoencoder ensemble.

```python
from src.anomaly import AnomalyDetector

detector = AnomalyDetector()
detector.initialize()

# Process data
result = detector.process(sensor_data)  # np.ndarray (64,)

# Result format:
# {
#     'score': 0.85,
#     'is_anomaly': False,
#     'isolation_forest_score': 0.12,
#     'autoencoder_score': 0.08,
#     'method': 'cascade'
# }
```

### `IsolationForestDetector`

```python
from src.anomaly import IsolationForestDetector

detector = IsolationForestDetector()
detector.initialize()

# Train on normal data
detector.fit(training_data)  # np.ndarray (n_samples, 64)

# Detect anomalies
result = detector.process(data)
```

### `AutoencoderDetector`

```python
from src.anomaly import AutoencoderDetector

detector = AutoencoderDetector()
detector.initialize()

# Train
detector.train(training_data, epochs=100)

# Detect
result = detector.process(data)
```

---

## Prediction

### `LSTMPredictor`

```python
from src.prediction import LSTMPredictor

predictor = LSTMPredictor()
predictor.initialize()

# Single point (streaming mode)
result = predictor.process(data_point)  # np.ndarray (8,)

# Full sequence
result = predictor.process(sequence)  # np.ndarray (50, 8)

# Result format:
# {
#     'forecast': np.ndarray (10, 8),  # Next 10 timesteps
#     'confidence': 0.92
# }
```

#### Training

```python
predictor.train(
    X_train,  # (n_samples, seq_length, features)
    y_train,  # (n_samples, horizon, features)
    epochs=50,
    batch_size=32
)
```

### `TFLiteConverter`

```python
from src.prediction import TFLiteConverter

converter = TFLiteConverter()

# Convert Keras model to TFLite
converter.convert(
    model,  # Keras model
    output_path='models/prediction/lstm.tflite',
    quantization='int8',  # 'none', 'dynamic', 'int8'
    representative_data=calibration_data  # Required for int8
)
```

---

## Decision Making

### `PPOAgent`

```python
from src.decision import PPOAgent

agent = PPOAgent()
agent.initialize()

# Get action
result = agent.process(observation)  # np.ndarray (64,)

# Result format:
# {
#     'action': np.ndarray (4,),  # Continuous action
#     'action_type': 'forward',   # Mapped action name
#     'confidence': 0.87
# }
```

#### Training

```python
from src.decision import PPOAgent, NavigationEnv

# Create environment
env = NavigationEnv(config)

# Train agent
agent.train(env, total_timesteps=100000)

# Save model
agent.save('models/decision/ppo_navigation')

# Load model
agent.load('models/decision/ppo_navigation')
```

### `NavigationEnv`

Custom Gymnasium environment for robot navigation.

```python
from src.decision import NavigationEnv
import gymnasium as gym

env = NavigationEnv(config)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

#### Observation Space
- Shape: `(64,)`
- Contains: sensor data, anomaly scores, predictions, vision features

#### Action Space
- Shape: `(4,)` continuous
- `[0]`: Linear velocity (-1 to 1)
- `[1]`: Angular velocity (-1 to 1)
- `[2]`: Gripper (0 to 1)
- `[3]`: Special action (0 to 1)

---

## Vision

### `VisionPipeline`

```python
from src.vision import VisionPipeline

pipeline = VisionPipeline()
pipeline.initialize()

# Process frame
result = pipeline.process()  # Captures and processes

# Result format:
# {
#     'detections': [
#         {'class': 'ball', 'bbox': [x1, y1, x2, y2], 'score': 0.95},
#         ...
#     ],
#     'features': np.ndarray (1280,),  # Feature vector
#     'frame_shape': (480, 640, 3),
#     'inference_time_ms': 10.5
# }
```

### `MobileNetV3Detector`

```python
from src.vision import MobileNetV3Detector

detector = MobileNetV3Detector()
detector.initialize()

# Detect objects
result = detector.process(image)  # np.ndarray (H, W, 3)
```

### `HailoBackend`

```python
from src.vision import HailoBackend

backend = HailoBackend()
backend.initialize()

# Load HEF model
backend.load_model('models/vision/mobilenet.hef')

# Run inference
output = backend.infer(input_data)
```

---

## Fleet Management

### `FleetManager`

```python
from src.fleet import FleetManager

manager = FleetManager()
manager.initialize()

# Get coordinated action
result = manager.process(local_state)

# Allocate tasks
manager.allocate_task(task)

# Set formation
manager.set_formation('line', spacing=1.0)
```

### `MARLCoordinator`

```python
from src.fleet import MARLCoordinator

coordinator = MARLCoordinator(num_agents=4)
coordinator.initialize()

# Get joint actions
actions = coordinator.process(observations)  # dict of observations

# Result format:
# {
#     0: {'action': np.ndarray, 'value': 0.5},
#     1: {'action': np.ndarray, 'value': 0.6},
#     ...
# }
```

### `MeshCommunication`

```python
from src.fleet import MeshCommunication

comm = MeshCommunication(agent_id=0)
comm.start()

# Broadcast state
comm.broadcast({'position': [1.0, 2.0], 'status': 'active'})

# Get peer states
peer_states = comm.get_peer_states()

# Stop communication
comm.stop()
```

---

## MCU Communication

### `MCUCommunicator`

```python
from src.mcu import MCUCommunicator

mcu = MCUCommunicator()
mcu.initialize()

# Read sensors
result = mcu.process()

# Result format:
# {
#     'imu': {'accel': [...], 'gyro': [...]},
#     'distance': {'value': 0.5},
#     'battery': {'voltage': 11.2, 'current': 1.5},
#     'timestamp': 1234567890.123
# }

# Execute action
mcu.execute_action(action_dict)
```

### `MCUProtocol`

```python
from src.mcu.protocol import MCUProtocol

protocol = MCUProtocol(config)

# Create command packet
packet = protocol.create_command('motor', {
    'left': 0.5,
    'right': 0.5
})

# Parse response
parsed = protocol.parse_packets(response_bytes)
```

---

## Configuration

### `ConfigLoader`

```python
from src.core.config_loader import ConfigLoader

config = ConfigLoader('config/config.yaml')

# Get value with default
value = config.get('system.name', 'default')

# Get section
hardware = config.get_section('hardware')

# Check if key exists
exists = config.has('vision.enabled')
```

---

## Base Classes

### `BaseModule`

All modules inherit from this:

```python
from src.core.base_module import BaseModule

class MyModule(BaseModule):
    def __init__(self):
        super().__init__("MyModule")
    
    def initialize(self) -> bool:
        # Setup logic
        return True
    
    def process(self, input_data: Any) -> Any:
        # Processing logic
        return output
    
    def shutdown(self) -> None:
        # Cleanup
        pass
    
    def get_status(self) -> Dict[str, Any]:
        return {'healthy': True}
```

---

## Utilities

### Logging

```python
from src.core.logger import get_logger

logger = get_logger('MyModule')
logger.info("Processing started")
logger.warning("Low battery")
logger.error("Connection failed", exc_info=True)
```

### Performance Timing

```python
import time

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self.start) * 1000

with Timer() as t:
    result = heavy_computation()
print(f"Took {t.elapsed:.2f}ms")
```

---

## Type Hints

Common types used across the API:

```python
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

SensorData = Dict[str, Any]
Observation = np.ndarray  # Shape (64,)
Action = np.ndarray       # Shape (4,)
Detection = Dict[str, Any]  # {'class': str, 'bbox': List[float], 'score': float}
```
