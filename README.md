# AegisAI - Autonomous River Sampling Buoy

## Agentic AI for Water Quality Monitoring

AegisAI transforms a simple floating buoy into an intelligent, self-organizing research assistant. Instead of passively recording data on fixed timers, the buoy detects anomalies, predicts water quality changes, decides when to sample, and coordinates with other buoys—all running on solar power with no internet connection.

### Hardware

| Component | Specification |
|-----------|---------------|
| Compute | Raspberry Pi 5 (8GB) |
| Storage | 64GB MicroSD |
| Power | 5W Solar Panel + 20Wh Battery |
| MCU | ESP32-S3 |
| Sensors | pH, Temperature, Turbidity, Conductivity |
| Sampling | Peristaltic Pump, 12-vial Carousel |
| Communication | LoRa Radio (915MHz) |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AegisAI Buoy System                      │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│  Anomaly    │  Prediction │  Decision   │     Vision       │
│  Detection  │    Engine   │   Making    │   (Optional)     │
│             │             │             │                  │
│ Isolation   │   LSTM      │    PPO      │  MobileNetV3     │
│ Forest +    │  (TFLite)   │   (RL)      │  Contaminant     │
│ Autoencoder │             │             │  Detection       │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│                Fleet Coordination (MAPPO via LoRa)          │
├─────────────────────────────────────────────────────────────┤
│                   MCU + Water Quality Sensors               │
│              pH | Temperature | Turbidity | Conductivity    │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
AegisAI/
├── config/                 # Configuration files
├── src/
│   ├── anomaly/           # Cascade anomaly detection (IF + Autoencoder)
│   ├── prediction/        # LSTM water quality prediction
│   ├── decision/          # PPO sampling decisions
│   ├── vision/            # Optional visual contaminant detection
│   ├── fleet/             # Multi-buoy MAPPO coordination
│   ├── mcu/               # ESP32-S3 sensor/pump control
│   └── core/              # Main orchestrator and utilities
├── models/                # Trained models and TFLite exports
├── scripts/               # Setup scripts and diagram generation
├── tests/                 # Unit tests
└── docs/                  # Documentation and presentations
```

### Quick Start

1. **Setup Raspberry Pi:**
   ```bash
   chmod +x scripts/setup_pi5.sh
   ./scripts/setup_pi5.sh
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate presentation diagrams:**
   ```bash
   python scripts/generate_diagrams.py
   ```

4. **Start the buoy system:**
   ```bash
   python -m src.core.main
   ```

### AI Components

| Component | Model | Purpose |
|-----------|-------|---------|
| Anomaly Detection | Isolation Forest + Autoencoder | Detect unusual water quality readings |
| Prediction | Stacked LSTM (TFLite int8) | Forecast water quality changes |
| Decision | PPO (Stable Baselines3) | Decide when to sample, conserve power |
| Vision | MobileNetV3 (optional) | Detect visual contaminants (algae, oil) |
| Fleet | MAPPO | Coordinate multi-buoy deployments |

### Key Features

- **Cascade Anomaly Detection** — Fast IF screening, deep AE analysis only when needed (60% compute savings)
- **Predictive Sampling** — Increases sampling before predicted contamination events
- **Adaptive Power Management** — Learns to schedule inference during peak solar
- **Fleet Coordination** — Buoys converge on anomaly locations over LoRa mesh

### License

MIT License
