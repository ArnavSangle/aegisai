# AegisAI Architecture Documentation

## System Overview

AegisAI is a modular AI infrastructure designed for robotics applications on Raspberry Pi 5 with AI HAT+ acceleration.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              AegisAI System                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Anomaly    │  │  Prediction  │  │   Decision   │  │    Vision    │   │
│  │  Detection   │  │   (LSTM)     │  │    (PPO)     │  │ (MobileNet)  │   │
│  │              │  │              │  │              │  │              │   │
│  │ IF + AE      │  │  TFLite      │  │   SB3        │  │  Hailo/      │   │
│  │ Cascade      │  │  int8        │  │   CPU        │  │  TFLite      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │                 │           │
│         └────────────┬────┴────────────┬────┘                 │           │
│                      │                 │                      │           │
│                      ▼                 ▼                      │           │
│              ┌───────────────────────────────┐                │           │
│              │       Main Orchestrator       │◄───────────────┘           │
│              │        (src/core/main.py)     │                            │
│              └───────────────┬───────────────┘                            │
│                              │                                            │
│         ┌────────────────────┼────────────────────┐                       │
│         │                    │                    │                       │
│         ▼                    ▼                    ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│  │    Fleet     │    │     MCU      │    │   Config     │                │
│  │  Management  │    │ Communicator │    │   Loader     │                │
│  │   (MARL)     │    │  (ESP32-S3)  │    │              │                │
│  └──────────────┘    └──────────────┘    └──────────────┘                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         ESP32-S3 MCU          │
                    │  ┌─────┐ ┌─────┐ ┌─────────┐  │
                    │  │ IMU │ │ TOF │ │ Motors  │  │
                    │  └─────┘ └─────┘ └─────────┘  │
                    └───────────────────────────────┘
```

## Module Details

### 1. Anomaly Detection (`src/anomaly/`)

**Purpose**: Detect abnormal sensor readings or system behavior.

**Architecture**:
```
Input Data (64-dim) ──┬──► Isolation Forest ──► Score
                      │                           │
                      │                           ▼
                      │                      Low/High?
                      │                       │    │
                      │         Low───────────┘    │
                      │         (Normal)           │ Uncertain
                      │                            ▼
                      └──► Autoencoder ───► Reconstruction Error
                                                   │
                                                   ▼
                                            Combined Score
```

**Key Classes**:
- `AnomalyDetector`: Main ensemble detector
- `IsolationForestDetector`: Fast outlier detection
- `AutoencoderDetector`: Deep learning anomaly detection

### 2. Prediction Engine (`src/prediction/`)

**Purpose**: Forecast future sensor values for proactive control.

**Architecture**:
```
Time Series ──► Sliding Window ──► LSTM ──► Future Values
   (t-50)                          │
     to                            │
    (t)              ┌─────────────┘
                     ▼
              TFLite Runtime
              (int8 quantized)
```

**Key Features**:
- Sequence length: 50 timesteps
- Prediction horizon: 10 timesteps
- TFLite optimization for Pi

### 3. Decision Making (`src/decision/`)

**Purpose**: Reinforcement learning for autonomous behavior.

**Architecture**:
```
Observation ──► Policy Network ──► Action Distribution
     │              (MLP)               │
     │                                  ▼
     │                           Sample Action
     │                                  │
     └────────────────┬─────────────────┘
                      ▼
              Execute on Robot
                      │
                      ▼
              Get Reward & Next State
                      │
                      ▼
              Update Policy (Training)
```

**Algorithm**: PPO (Proximal Policy Optimization)
- Stable learning
- Good sample efficiency
- Works well on CPU

### 4. Vision Pipeline (`src/vision/`)

**Purpose**: Real-time object detection and scene understanding.

**Architecture**:
```
Camera ──► Capture ──► Preprocess ──► MobileNetV3 ──► Detections
  │                                       │
  │                       ┌───────────────┘
  │                       ▼
  │               Hailo AI HAT+ (13 TOPS)
  │                    or
  │               TFLite (CPU fallback)
  │
  └──► Feature Extraction ──► RL Observations
```

**Supported Backends**:
1. Hailo-8L NPU (preferred)
2. TFLite with XNNPACK
3. Full TensorFlow (slow)

### 5. Fleet Management (`src/fleet/`)

**Purpose**: Coordinate multiple robots using MARL.

**Architecture**:
```
Robot 1 ──┐                    ┌──► Robot 1 Action
          │    ┌──────────┐    │
Robot 2 ──┼───►│  MAPPO   │────┼──► Robot 2 Action
          │    │ Coord.   │    │
Robot 3 ──┼───►│          │────┼──► Robot 3 Action
          │    └──────────┘    │
Robot 4 ──┘                    └──► Robot 4 Action
          │
          ▼
    Mesh Communication (UDP Broadcast)
```

**Features**:
- Centralized training, decentralized execution
- Formation control
- Collision avoidance
- Task allocation (auction-based)

### 6. MCU Communication (`src/mcu/`)

**Purpose**: Interface with ESP32-S3 for sensor/actuator control.

**Protocol**:
```
┌────────┬────────┬──────┬──────┬─────────────┬───────────┐
│ SYNC1  │ SYNC2  │ TYPE │ LEN  │   PAYLOAD   │   CRC16   │
│ 0xAE   │ 0x5A   │ 1B   │ 1B   │   0-250B    │    2B     │
└────────┴────────┴──────┴──────┴─────────────┴───────────┘
```

**Supported Sensors**:
- IMU (MPU6050/ICM20948)
- Distance (VL53L0X)
- Current (INA219)
- Battery voltage

---

## Data Flow

### Inference Loop (100Hz target)

```python
while running:
    # 1. Get sensor data from MCU
    sensor_data = mcu.read_sensors()           # ~2ms
    
    # 2. Check for anomalies
    anomaly = anomaly_detector.process(sensor_data)  # ~1ms
    
    # 3. Predict future states
    prediction = predictor.process(sensor_data)      # ~5ms
    
    # 4. Get camera frame & detect objects
    detections = vision.process()                    # ~10ms (Hailo)
    
    # 5. Build observation for RL
    observation = build_observation(
        sensor_data, anomaly, prediction, detections
    )
    
    # 6. Get action from policy
    action = decision.process(observation)           # ~2ms
    
    # 7. Coordinate with fleet
    action = fleet.coordinate(action)                # ~5ms
    
    # 8. Execute action
    mcu.execute(action)                              # ~1ms
    
    # Total: ~26ms = ~38Hz (with margin)
```

---

## Configuration System

All modules load configuration from `config/config.yaml`:

```yaml
# Example configuration flow
system:
  name: "AegisAI"

anomaly:
  isolation_forest:
    n_estimators: 100
  autoencoder:
    encoding_dim: 16
  ensemble:
    method: "cascade"
```

Configuration is accessed via:
```python
config = ConfigLoader()
n_estimators = config.get('anomaly.isolation_forest.n_estimators', 100)
```

---

## Model Formats

| Module | Training Format | Inference Format | Size |
|--------|-----------------|------------------|------|
| Anomaly (IF) | sklearn | joblib | ~1 MB |
| Anomaly (AE) | Keras | weights.h5 | ~500 KB |
| Prediction | Keras | TFLite int8 | ~200 KB |
| Decision | PyTorch | SB3 .zip | ~2 MB |
| Vision | Keras | HEF (Hailo) | ~5 MB |
| Fleet | PyTorch | .pt | ~1 MB |

---

## Thread Model

```
Main Thread
    │
    ├──► Async Event Loop
    │       ├── MCU read/write
    │       ├── Fleet communication
    │       └── Vision capture
    │
    ├──► Thread Pool (2 workers)
    │       ├── Heavy inference
    │       └── Model loading
    │
    └──► Background Threads
            ├── Heartbeat (1 Hz)
            └── Sensor polling (100 Hz)
```

---

## Extension Points

### Adding New Sensors

1. Define sensor type in `protocol.py`:
   ```python
   class SensorType(Enum):
       MY_SENSOR = 0x18
   ```

2. Add parsing in `_parse_sensor_data()`

3. Update ESP32 firmware to send data

### Adding New Actions

1. Update action space in config
2. Modify action mapping in `MCUCommunicator.execute_action()`
3. Retrain PPO with new action space

### Adding New Detection Models

1. Create detector class inheriting `BaseModule`
2. Implement `initialize()`, `process()`, `shutdown()`
3. Add to vision pipeline
