#!/bin/bash
# AegisAI Setup Script for Raspberry Pi 5
# Run with: chmod +x setup_pi5.sh && ./setup_pi5.sh

set -e

echo "========================================"
echo "  AegisAI - Raspberry Pi 5 Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please don't run as root. Use a regular user with sudo privileges.${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: System Update${NC}"
sudo apt update && sudo apt upgrade -y

echo -e "${GREEN}Step 2: Install System Dependencies${NC}"
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    cmake \
    build-essential \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk* \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    i2c-tools \
    screen \
    htop

echo -e "${GREEN}Step 3: Configure Hardware Interfaces${NC}"

# Enable I2C
if ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt; then
    echo "dtparam=i2c_arm=on" | sudo tee -a /boot/firmware/config.txt
    echo "I2C enabled"
fi

# Enable SPI
if ! grep -q "^dtparam=spi=on" /boot/firmware/config.txt; then
    echo "dtparam=spi=on" | sudo tee -a /boot/firmware/config.txt
    echo "SPI enabled"
fi

# Enable serial port
if ! grep -q "^enable_uart=1" /boot/firmware/config.txt; then
    echo "enable_uart=1" | sudo tee -a /boot/firmware/config.txt
    echo "UART enabled"
fi

# Add user to required groups
sudo usermod -aG gpio,i2c,spi,video,dialout $USER

echo -e "${GREEN}Step 4: Set Up Python Virtual Environment${NC}"

# Create virtual environment
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

echo -e "${GREEN}Step 5: Install Python Dependencies${NC}"

# Install PyTorch for ARM (Pi 5 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies (tensorflow includes tflite interpreter)
pip install -r requirements.txt

echo -e "${GREEN}Step 6: Configure Raspberry Pi Camera${NC}"

# Check if libcamera is available
if command -v libcamera-hello &> /dev/null; then
    echo "libcamera available"
    pip install picamera2
else
    echo -e "${YELLOW}libcamera not found. Camera support may be limited.${NC}"
fi

echo -e "${GREEN}Step 7: Set Up AI HAT+ (Hailo)${NC}"

# Check for Hailo device
if lspci | grep -i hailo &> /dev/null; then
    echo "Hailo device detected"
    
    # Install Hailo runtime (if available)
    if [ -f /opt/hailo/setup.sh ]; then
        source /opt/hailo/setup.sh
    else
        echo -e "${YELLOW}Hailo SDK not found. Please install manually from:${NC}"
        echo "https://hailo.ai/developer-zone/"
    fi
else
    echo -e "${YELLOW}No Hailo device detected. AI HAT+ features will be disabled.${NC}"
fi

echo -e "${GREEN}Step 8: Configure Active Cooling${NC}"

# Set up fan control for active cooling
cat << 'EOF' | sudo tee /etc/systemd/system/fan-control.service
[Unit]
Description=Raspberry Pi Fan Control
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 -c "
import time
import os

def get_temp():
    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
        return int(f.read()) / 1000

def set_fan(speed):
    # GPIO fan control (adjust pin as needed)
    pass  # Implement based on your fan setup

while True:
    temp = get_temp()
    if temp > 70:
        set_fan(100)
    elif temp > 60:
        set_fan(75)
    elif temp > 50:
        set_fan(50)
    else:
        set_fan(0)
    time.sleep(5)
"
Restart=always

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}Step 9: Create Model Directories${NC}"

mkdir -p models/anomaly
mkdir -p models/prediction
mkdir -p models/decision
mkdir -p models/vision
mkdir -p models/fleet
mkdir -p models/hailo
mkdir -p logs
mkdir -p data
mkdir -p checkpoints

echo -e "${GREEN}Step 10: Set Up Systemd Service${NC}"

# Create service file for AegisAI
cat << EOF | sudo tee /etc/systemd/system/aegisai.service
[Unit]
Description=AegisAI Robot Control System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/.venv/bin:/usr/local/bin:/usr/bin"
ExecStart=$(pwd)/.venv/bin/python -m src.core.main
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

echo ""
echo -e "${GREEN}========================================"
echo "  Setup Complete!"
echo "========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Reboot the Raspberry Pi: sudo reboot"
echo "2. Activate the virtual environment: source .venv/bin/activate"
echo "3. Run AegisAI: python -m src.core.main"
echo ""
echo "To enable auto-start on boot:"
echo "  sudo systemctl enable aegisai"
echo ""
echo "To check system status:"
echo "  sudo systemctl status aegisai"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/aegis.log"
echo ""
