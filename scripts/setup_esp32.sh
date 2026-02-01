#!/bin/bash
# ESP32-S3 Firmware Build and Upload Script
# Requires PlatformIO

set -e

echo "========================================"
echo "  ESP32-S3 Firmware Setup"
echo "========================================"

# Check for PlatformIO
if ! command -v pio &> /dev/null; then
    echo "PlatformIO not found. Installing..."
    pip install platformio
fi

# Generate firmware project
FIRMWARE_DIR="firmware/esp32_aegis"
mkdir -p "$FIRMWARE_DIR"

# Generate firmware using Python
python3 << 'EOF'
from src.mcu.esp32_firmware import ESP32FirmwareManager

manager = ESP32FirmwareManager()
manager.generate_firmware_files("firmware/esp32_aegis")
EOF

echo "Firmware project generated at: $FIRMWARE_DIR"

# Build firmware
echo ""
echo "Building firmware..."
cd "$FIRMWARE_DIR"
pio run

# List available upload ports
echo ""
echo "Available serial ports:"
pio device list

# Prompt for upload
read -p "Upload firmware now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Uploading firmware..."
    pio run -t upload
    
    echo ""
    echo "Firmware uploaded successfully!"
    echo "Monitor serial output with: pio device monitor"
fi

cd ../..
echo ""
echo "ESP32-S3 setup complete!"
