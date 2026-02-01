# AegisAI - Raspberry Pi 5 AI Infrastructure

## 🎯 Competition-Ready Full Stack AI System

This project implements an industry-grade AI infrastructure designed for Raspberry Pi 5 with AI HAT+.

### 📋 Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **Board** | Raspberry Pi 5 |
| **RAM** | 16GB |
| **Storage** | 128GB MicroSD (A2 rated recommended) |
| **Cooling** | Active cooling (fan + heatsink) |
| **AI Accelerator** | Raspberry Pi AI HAT+ (Hailo-8L) |
| **MCU** | ESP32-S3 (for peripheral control) |

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AegisAI Core                           │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│  Anomaly    │  Prediction │  Decision   │     Vision       │
│  Detection  │    Engine   │   Making    │    Pipeline      │
│             │             │             │                  │
│ Isolation   │   LSTM      │    PPO      │  MobileNetV3     │
│ Forest +    │  (TFLite)   │   (RL)      │  (AI HAT+)       │
│ Autoencoder │             │             │                  │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│                    Fleet Management (MARL)                  │
├─────────────────────────────────────────────────────────────┤
│                    MCU Communication Layer                  │
│                  (ESP32-S3 Serial/BLE/WiFi)                │
└─────────────────────────────────────────────────────────────┘
```

### 📁 Project Structure

```
AegisAI/
├── config/                 # Configuration files
├── src/
│   ├── anomaly/           # Anomaly detection (Isolation Forest + Autoencoder)
│   ├── prediction/        # LSTM prediction engine
│   ├── decision/          # PPO reinforcement learning
│   ├── vision/            # MobileNetV3 computer vision
│   ├── fleet/             # Multi-agent fleet management
│   ├── mcu/               # ESP32-S3 communication
│   └── core/              # Core utilities and base classes
├── models/                # Trained models and TFLite exports
├── data/                  # Training data and datasets
├── scripts/               # Setup and deployment scripts
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

### 🚀 Quick Start

1. **Flash Raspberry Pi OS (64-bit)**
2. **Run setup script:**
   ```bash
   chmod +x scripts/setup_pi5.sh
   ./scripts/setup_pi5.sh
   ```
3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Start the system:**
   ```bash
   python -m src.core.main
   ```

### 📦 Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Anomaly Detection | Isolation Forest → Autoencoder | Detect system anomalies |
| Prediction | LSTM (TFLite) | Time-series forecasting |
| Decision | PPO | Reinforcement learning decisions |
| Vision | MobileNetV3 | Real-time object detection |
| Fleet | MARL | Multi-agent coordination |
| MCU | ESP32-S3 | Sensor/actuator control |

### 📄 License

MIT License - Built for AI/Robotics competitions
