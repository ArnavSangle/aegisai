# AegisAI Deployment Guide

## Hardware Setup

### Raspberry Pi 5 Configuration

#### Initial Setup

1. **Flash Raspberry Pi OS (64-bit)**
   - Download Raspberry Pi Imager
   - Select Raspberry Pi OS (64-bit) Bookworm
   - Configure hostname, SSH, WiFi in imager settings
   - Flash to 128GB MicroSD (A2 rated recommended)

2. **First Boot**
   ```bash
   # Update system
   sudo apt update && sudo apt full-upgrade -y
   
   # Set hostname
   sudo hostnamectl set-hostname aegis-pi
   
   # Expand filesystem
   sudo raspi-config --expand-rootfs
   ```

### AI HAT+ Installation

1. **Physical Installation**
   - Power off the Pi
   - Align AI HAT+ with GPIO header
   - Press firmly to seat connector
   - Secure with standoffs

2. **Verify Detection**
   ```bash
   lspci | grep -i hailo
   # Should show: Hailo Technologies Ltd. Hailo-8
   ```

3. **Install Hailo SDK**
   ```bash
   # Download from Hailo website
   # https://hailo.ai/developer-zone/
   
   # Install runtime
   sudo dpkg -i hailo-rt_X.X.X_arm64.deb
   
   # Verify installation
   hailortcli fw-control identify
   ```

### Active Cooling Setup

1. **Install Cooler**
   - Attach heatsink with thermal paste
   - Connect fan to GPIO or 5V header
   - Fan pins: 5V (red), GND (black), PWM (optional)

2. **Configure Fan Control**
   ```bash
   # Edit config.txt
   sudo nano /boot/firmware/config.txt
   
   # Add fan control
   dtoverlay=gpio-fan,gpiopin=14,temp=55000
   ```

### ESP32-S3 Connection

1. **Wiring**
   | ESP32-S3 | Raspberry Pi |
   |----------|--------------|
   | TX       | GPIO15 (RX)  |
   | RX       | GPIO14 (TX)  |
   | GND      | GND          |
   | 5V       | 5V (optional)|

2. **USB Connection (Alternative)**
   - Connect ESP32-S3 via USB-C
   - Device appears as `/dev/ttyUSB0` or `/dev/ttyACM0`

---

## Software Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-repo/aegisai.git
cd aegisai
```

### 2. Run Setup Script
```bash
chmod +x scripts/setup_pi5.sh
./scripts/setup_pi5.sh
```

### 3. Activate Environment
```bash
source .venv/bin/activate
```

### 4. Verify Installation
```bash
python -c "import src; print('AegisAI ready!')"
```

---

## Model Training

### On Development Machine (Recommended)

Training is faster on a machine with GPU:

```bash
# Train all models
python scripts/train_models.py --model all

# Train specific model
python scripts/train_models.py --model prediction
```

### On Raspberry Pi (Inference Only)

The Pi is optimized for inference, not training. Use pre-trained models:

```bash
# Copy models from development machine
scp -r models/* pi@aegis-pi:~/aegisai/models/
```

---

## Running AegisAI

### Manual Start
```bash
source .venv/bin/activate
python -m src.core.main
```

### As System Service
```bash
# Enable service
sudo systemctl enable aegisai

# Start service
sudo systemctl start aegisai

# Check status
sudo systemctl status aegisai

# View logs
journalctl -u aegisai -f
```

---

## Configuration

### Main Config: `config/config.yaml`

Key settings to adjust:

```yaml
hardware:
  camera:
    enabled: true       # Enable/disable camera
    resolution: [640, 480]  # Lower for better FPS
    
  mcu:
    port: "/dev/ttyUSB0"  # Adjust for your setup
    
vision:
  inference:
    delegate: "hailo"   # Use "none" if no AI HAT+
    
fleet:
  num_agents: 1         # Set to 1 for single robot
```

---

## Troubleshooting

### Camera Not Working
```bash
# Test camera
libcamera-hello

# Check permissions
sudo usermod -aG video $USER
```

### ESP32 Not Detected
```bash
# List ports
ls /dev/tty*

# Check permissions
sudo usermod -aG dialout $USER

# Test connection
screen /dev/ttyUSB0 115200
```

### Hailo Not Working
```bash
# Check device
lspci | grep -i hailo

# Check driver
hailortcli fw-control identify

# Reset device
sudo modprobe -r hailo_pci && sudo modprobe hailo_pci
```

### Memory Issues
```bash
# Check memory
free -h

# Increase swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=4096
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## Performance Tuning

### For Best Inference Speed

1. **Use TFLite quantized models**
   ```yaml
   # config/config.yaml
   prediction:
     tflite:
       quantization: "int8"
   ```

2. **Enable Hailo acceleration**
   ```yaml
   vision:
     inference:
       delegate: "hailo"
   ```

3. **Reduce camera resolution**
   ```yaml
   hardware:
     camera:
       resolution: [320, 240]
       fps: 15
   ```

4. **Optimize CPU governor**
   ```bash
   echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

---

## Network Setup for Fleet

### Configure Static IP
```bash
# Edit dhcpcd.conf
sudo nano /etc/dhcpcd.conf

# Add:
interface wlan0
static ip_address=192.168.1.10/24
static routers=192.168.1.1
static domain_name_servers=8.8.8.8
```

### Fleet Discovery
Each robot broadcasts on UDP port 5555. Ensure firewall allows:
```bash
sudo ufw allow 5555/udp
```
