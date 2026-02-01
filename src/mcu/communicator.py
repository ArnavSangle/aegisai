"""
MCU Communicator for AegisAI
Handles communication with ESP32-S3 microcontroller
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from loguru import logger
import struct
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    from bleak import BleakClient, BleakScanner
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False

from ..core.base_module import BaseModule
from .protocol import MCUProtocol, SensorData, ActuatorCommand


class MCUCommunicator(BaseModule):
    """
    Communication interface for ESP32-S3 microcontroller.
    Supports Serial, BLE, and WiFi connections.
    """
    
    def __init__(self):
        super().__init__('mcu')
        
        # Connection settings
        self.connection_type = self.config.get('protocol', {}).get('type', 'serial')
        self.serial_port: Optional[serial.Serial] = None
        self.ble_client: Optional[BleakClient] = None
        
        # Protocol handler
        self.protocol = MCUProtocol(self.config.get('protocol', {}))
        
        # Sensor data cache
        self._sensor_cache: Dict[str, SensorData] = {}
        self._last_update: float = 0.0
        
        # Async communication
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._read_buffer = bytearray()
        
        # Connection state
        self._connected = False
        
    def initialize(self) -> bool:
        """Initialize MCU communication."""
        try:
            if self.connection_type == 'serial':
                success = self._init_serial()
            elif self.connection_type == 'ble':
                success = self._init_ble()
            elif self.connection_type == 'wifi':
                success = self._init_wifi()
            else:
                logger.warning(f"Unknown connection type: {self.connection_type}")
                success = self._init_serial()  # Default to serial
            
            if success:
                # Send initialization command to MCU
                self._send_init_command()
                self._connected = True
            
            self.is_initialized = success
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize MCU Communicator: {e}")
            return False
    
    def _init_serial(self) -> bool:
        """Initialize serial connection."""
        if not SERIAL_AVAILABLE:
            logger.warning("pyserial not available")
            return False
        
        port = self.config.get('protocol', {}).get('port', '/dev/ttyUSB0')
        baudrate = self.config.get('protocol', {}).get('baudrate', 115200)
        
        # Auto-detect port if not specified
        if port == 'auto':
            port = self._find_esp32_port()
            if not port:
                logger.error("Could not auto-detect ESP32 port")
                return False
        
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=0.1,
                write_timeout=0.1
            )
            
            # Clear buffers
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            logger.info(f"Serial connected: {port} @ {baudrate} baud")
            return True
            
        except serial.SerialException as e:
            logger.error(f"Serial connection failed: {e}")
            return False
    
    def _find_esp32_port(self) -> Optional[str]:
        """Auto-detect ESP32 serial port."""
        ports = serial.tools.list_ports.comports()
        
        esp32_identifiers = [
            'CP210', 'CH340', 'FTDI', 'USB Serial',
            'Silicon Labs', 'ESP32', 'USB-SERIAL'
        ]
        
        for port in ports:
            for identifier in esp32_identifiers:
                if identifier.lower() in port.description.lower():
                    logger.info(f"Found ESP32 at {port.device}: {port.description}")
                    return port.device
        
        # Return first available port as fallback
        if ports:
            return ports[0].device
        
        return None
    
    def _init_ble(self) -> bool:
        """Initialize BLE connection."""
        if not BLE_AVAILABLE:
            logger.warning("bleak not available for BLE")
            return False
        
        # BLE initialization is async, handled separately
        logger.info("BLE mode enabled - will connect on first use")
        return True
    
    def _init_wifi(self) -> bool:
        """Initialize WiFi/TCP connection."""
        import socket
        
        host = self.config.get('wifi', {}).get('host', '192.168.4.1')
        port = self.config.get('wifi', {}).get('port', 8080)
        
        try:
            self._tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._tcp_socket.settimeout(5.0)
            self._tcp_socket.connect((host, port))
            self._tcp_socket.setblocking(False)
            
            logger.info(f"WiFi connected: {host}:{port}")
            return True
            
        except socket.error as e:
            logger.error(f"WiFi connection failed: {e}")
            return False
    
    def _send_init_command(self):
        """Send initialization command to MCU."""
        init_cmd = self.protocol.create_command('init', {
            'version': '1.0.0',
            'sample_rate': 100  # Hz
        })
        self._write(init_cmd)
    
    def process(self, data: Any = None) -> Dict[str, Any]:
        """
        Process communication cycle - read sensors and return data.
        
        Args:
            data: Optional command to send
            
        Returns:
            Sensor data dictionary
        """
        if not self.is_initialized:
            raise RuntimeError("MCU Communicator not initialized")
        
        # Send command if provided
        if data is not None:
            self.send_command(data)
        
        # Read sensor data
        sensor_data = self.read_sensors()
        
        return {
            'sensors': sensor_data,
            'timestamp': time.time(),
            'connected': self._connected
        }
    
    def read_sensors(self) -> Dict[str, Any]:
        """
        Read all sensor data from MCU.
        
        Returns:
            Dictionary of sensor readings
        """
        # Read raw data
        raw_data = self._read()
        
        if raw_data:
            # Parse packets
            packets = self.protocol.parse_packets(raw_data)
            
            for packet in packets:
                if packet['type'] == 'sensor':
                    sensor = SensorData.from_packet(packet)
                    self._sensor_cache[sensor.name] = sensor
                    self._last_update = time.time()
        
        # Build sensor dictionary
        result = {}
        for name, sensor in self._sensor_cache.items():
            result[name] = {
                'value': sensor.value,
                'timestamp': sensor.timestamp,
                'unit': sensor.unit
            }
        
        return result
    
    async def read_sensors_async(self) -> Dict[str, Any]:
        """Async version of read_sensors."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.read_sensors)
    
    def send_command(self, command: Union[Dict, ActuatorCommand]):
        """
        Send command to MCU.
        
        Args:
            command: Command dictionary or ActuatorCommand object
        """
        if isinstance(command, ActuatorCommand):
            packet = command.to_packet()
        else:
            packet = self.protocol.create_command(
                command.get('type', 'generic'),
                command.get('data', {})
            )
        
        self._write(packet)
    
    def execute_action(self, action: Union[int, Dict]) -> bool:
        """
        Execute action on MCU (convert RL action to motor commands).
        
        Args:
            action: Action from decision module
            
        Returns:
            True if command sent successfully
        """
        # Map discrete actions to motor commands
        action_map = {
            0: {'type': 'stop', 'motors': [0, 0, 0, 0]},
            1: {'type': 'forward', 'motors': [100, 100, 100, 100]},
            2: {'type': 'backward', 'motors': [-100, -100, -100, -100]},
            3: {'type': 'left', 'motors': [-50, 50, -50, 50]},
            4: {'type': 'right', 'motors': [50, -50, 50, -50]},
            5: {'type': 'up', 'motors': [120, 120, 120, 120]},
            6: {'type': 'down', 'motors': [80, 80, 80, 80]},
            7: {'type': 'rotate_cw', 'motors': [50, -50, -50, 50]},
        }
        
        if isinstance(action, int):
            command = action_map.get(action, action_map[0])
        else:
            command = action
        
        # Create actuator command
        cmd = ActuatorCommand(
            actuator_type='motors',
            values=command.get('motors', [0, 0, 0, 0]),
            duration_ms=100
        )
        
        self.send_command(cmd)
        return True
    
    async def execute_action_async(self, action: Union[int, Dict]) -> bool:
        """Async version of execute_action."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.execute_action, action)
    
    def _read(self) -> bytes:
        """Read data from MCU."""
        data = b''
        
        try:
            if self.serial_port and self.serial_port.in_waiting:
                data = self.serial_port.read(self.serial_port.in_waiting)
            elif hasattr(self, '_tcp_socket'):
                try:
                    data = self._tcp_socket.recv(1024)
                except BlockingIOError:
                    pass
        except Exception as e:
            logger.error(f"Read error: {e}")
            self._connected = False
        
        return data
    
    def _write(self, data: bytes) -> bool:
        """Write data to MCU."""
        try:
            if self.serial_port:
                self.serial_port.write(data)
                return True
            elif hasattr(self, '_tcp_socket'):
                self._tcp_socket.send(data)
                return True
        except Exception as e:
            logger.error(f"Write error: {e}")
            self._connected = False
        
        return False
    
    def get_imu_data(self) -> Optional[Dict]:
        """Get IMU sensor data."""
        imu = self._sensor_cache.get('imu')
        if imu:
            return {
                'accel': imu.value[:3],
                'gyro': imu.value[3:6],
                'timestamp': imu.timestamp
            }
        return None
    
    def get_distance_data(self) -> Optional[float]:
        """Get distance sensor data."""
        dist = self._sensor_cache.get('distance')
        return dist.value if dist else None
    
    def calibrate_sensors(self) -> bool:
        """Send calibration command to MCU."""
        cmd = self.protocol.create_command('calibrate', {})
        self._write(cmd)
        
        # Wait for acknowledgment
        time.sleep(0.5)
        response = self._read()
        
        return b'CAL_OK' in response
    
    def shutdown(self):
        """Clean shutdown."""
        # Send stop command
        self.execute_action(0)  # Stop
        
        if self.serial_port:
            self.serial_port.close()
        
        if hasattr(self, '_tcp_socket'):
            self._tcp_socket.close()
        
        self._executor.shutdown(wait=False)
        self._connected = False
        self.is_initialized = False
        
        logger.info("MCU Communicator shutdown")
    
    def get_status(self) -> Dict[str, Any]:
        """Get communicator status."""
        status = super().get_status()
        status.update({
            'connection_type': self.connection_type,
            'connected': self._connected,
            'last_update': self._last_update,
            'sensors_active': list(self._sensor_cache.keys())
        })
        return status
