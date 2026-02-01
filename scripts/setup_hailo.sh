#!/bin/bash
# Hailo AI HAT+ Setup Script
# For Raspberry Pi 5 with AI HAT+

set -e

echo "========================================"
echo "  Hailo AI HAT+ Setup"
echo "========================================"

# Check for root
if [ "$EUID" -eq 0 ]; then
    echo "Don't run as root"
    exit 1
fi

# Check for Hailo device
if ! lspci | grep -i hailo &> /dev/null; then
    echo "Error: No Hailo device detected!"
    echo "Make sure the AI HAT+ is properly connected."
    exit 1
fi

echo "Hailo device found!"
lspci | grep -i hailo

# Download and install Hailo SDK
HAILO_VERSION="4.16.0"
HAILO_SDK_URL="https://hailo.ai/developer-zone/software-downloads/"

echo ""
echo "To install the Hailo SDK:"
echo "1. Visit: $HAILO_SDK_URL"
echo "2. Download the HailoRT for Raspberry Pi"
echo "3. Run the installer"
echo ""

# Check if HailoRT is installed
if command -v hailortcli &> /dev/null; then
    echo "HailoRT is installed:"
    hailortcli fw-control identify
else
    echo "HailoRT not found. Please install from the Hailo website."
fi

# Download pre-compiled models (if available)
MODELS_DIR="models/hailo"
mkdir -p "$MODELS_DIR"

echo ""
echo "Downloading pre-compiled HEF models..."

# MobileNetV3 for object detection
if [ ! -f "$MODELS_DIR/mobilenet_v3_ssd.hef" ]; then
    echo "Note: Pre-compiled models need to be downloaded from Hailo Model Zoo"
    echo "Visit: https://github.com/hailo-ai/hailo_model_zoo"
fi

# Create Hailo test script
cat << 'EOF' > scripts/test_hailo.py
#!/usr/bin/env python3
"""Test Hailo AI HAT+ functionality."""

import sys

def test_hailo():
    try:
        from hailo_platform import VDevice
        
        # Create device
        device = VDevice()
        
        # Get device info
        device_ids = device.get_physical_devices_ids()
        print(f"Hailo devices found: {device_ids}")
        
        # Print device info
        for dev_id in device_ids:
            print(f"  Device {dev_id}: Hailo-8L")
        
        print("\n✓ Hailo AI HAT+ is working correctly!")
        return True
        
    except ImportError:
        print("✗ Hailo SDK not installed")
        print("  Install from: https://hailo.ai/developer-zone/")
        return False
        
    except Exception as e:
        print(f"✗ Hailo error: {e}")
        return False

if __name__ == "__main__":
    success = test_hailo()
    sys.exit(0 if success else 1)
EOF

chmod +x scripts/test_hailo.py

echo ""
echo "Setup complete!"
echo "Run 'python scripts/test_hailo.py' to test the Hailo AI HAT+"
