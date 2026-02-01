"""
MCU Protocol Definition
Binary protocol for ESP32-S3 communication
"""

import struct
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import time


class PacketType(Enum):
    """Packet types for MCU communication."""
    COMMAND = 0x01
    SENSOR_DATA = 0x02
    ACK = 0x03
    ERROR = 0x04
    CONFIG = 0x05
    CALIBRATION = 0x06


class SensorType(Enum):
    """Sensor types supported by MCU."""
    IMU = 0x10          # MPU6050/ICM20948
    DISTANCE = 0x11     # VL53L0X
    CURRENT = 0x12      # INA219
    TEMPERATURE = 0x13
    PRESSURE = 0x14
    GPS = 0x15
    ENCODER = 0x16
    BATTERY = 0x17


class ActuatorType(Enum):
    """Actuator types supported by MCU."""
    MOTOR_PWM = 0x20
    SERVO = 0x21
    LED = 0x22
    BUZZER = 0x23


@dataclass
class SensorData:
    """Sensor data structure."""
    name: str
    sensor_type: SensorType
    value: Union[float, List[float]]
    timestamp: float
    unit: str = ""
    
    @classmethod
    def from_packet(cls, packet: Dict) -> 'SensorData':
        """Create SensorData from parsed packet."""
        return cls(
            name=packet.get('name', 'unknown'),
            sensor_type=SensorType(packet.get('sensor_type', 0x10)),
            value=packet.get('value', 0.0),
            timestamp=packet.get('timestamp', time.time()),
            unit=packet.get('unit', '')
        )
    
    def to_numpy(self):
        """Convert value to numpy array."""
        import numpy as np
        if isinstance(self.value, list):
            return np.array(self.value, dtype=np.float32)
        return np.array([self.value], dtype=np.float32)


@dataclass
class ActuatorCommand:
    """Actuator command structure."""
    actuator_type: str
    values: List[float]
    duration_ms: int = 0
    
    def to_packet(self) -> bytes:
        """Convert command to binary packet."""
        protocol = MCUProtocol({})
        return protocol.create_command('actuator', {
            'type': self.actuator_type,
            'values': self.values,
            'duration': self.duration_ms
        })


class MCUProtocol:
    """
    Binary protocol for MCU communication.
    
    Packet Format:
    +--------+--------+--------+--------+--------+--------+...+--------+--------+
    | SYNC1  | SYNC2  |  TYPE  |  LEN   |     PAYLOAD     |  CRC16-L | CRC16-H |
    +--------+--------+--------+--------+--------+--------+...+--------+--------+
    | 0xAE   | 0x5A   | 1 byte | 1 byte |  0-250 bytes    |  2 bytes          |
    +--------+--------+--------+--------+--------+--------+...+--------+--------+
    """
    
    SYNC1 = 0xAE
    SYNC2 = 0x5A
    MAX_PAYLOAD = 250
    
    def __init__(self, config: Dict):
        """
        Initialize protocol handler.
        
        Args:
            config: Protocol configuration
        """
        self.config = config
        self.checksum_type = config.get('checksum', 'crc16')
        self._packet_buffer = bytearray()
        self._sequence = 0
    
    def create_command(self, cmd_type: str, data: Dict) -> bytes:
        """
        Create command packet.
        
        Args:
            cmd_type: Command type string
            data: Command data
            
        Returns:
            Binary packet
        """
        # Map command types
        cmd_map = {
            'init': 0x01,
            'stop': 0x02,
            'start': 0x03,
            'calibrate': 0x04,
            'config': 0x05,
            'actuator': 0x10,
            'motor': 0x11,
            'servo': 0x12,
            'generic': 0xFF
        }
        
        cmd_byte = cmd_map.get(cmd_type, 0xFF)
        
        # Build payload based on command type
        if cmd_type == 'actuator':
            payload = self._encode_actuator_command(data)
        elif cmd_type == 'motor':
            payload = self._encode_motor_command(data)
        elif cmd_type == 'servo':
            payload = self._encode_servo_command(data)
        else:
            payload = self._encode_generic(data)
        
        return self._build_packet(PacketType.COMMAND.value, cmd_byte, payload)
    
    def _encode_actuator_command(self, data: Dict) -> bytes:
        """Encode actuator command payload."""
        actuator_type = data.get('type', 'motors')
        values = data.get('values', [0, 0, 0, 0])
        duration = data.get('duration', 0)
        
        # Pack: type (1), num_values (1), values (4 * num), duration (2)
        num_values = len(values)
        
        # Clamp values to int16 range
        clamped_values = [max(-32768, min(32767, int(v))) for v in values]
        
        payload = struct.pack(
            f'!BB{num_values}hH',
            ActuatorType.MOTOR_PWM.value if actuator_type == 'motors' else ActuatorType.SERVO.value,
            num_values,
            *clamped_values,
            duration
        )
        
        return payload
    
    def _encode_motor_command(self, data: Dict) -> bytes:
        """Encode motor-specific command."""
        speeds = data.get('speeds', [0, 0, 0, 0])
        return struct.pack('!4h', *[int(s) for s in speeds])
    
    def _encode_servo_command(self, data: Dict) -> bytes:
        """Encode servo command."""
        angles = data.get('angles', [90, 90])
        return struct.pack('!BB', *[int(a) for a in angles[:2]])
    
    def _encode_generic(self, data: Dict) -> bytes:
        """Encode generic command as JSON."""
        import json
        return json.dumps(data).encode()[:self.MAX_PAYLOAD]
    
    def _build_packet(self, pkt_type: int, cmd: int, payload: bytes) -> bytes:
        """Build complete packet with header and CRC."""
        # Header
        header = bytes([
            self.SYNC1,
            self.SYNC2,
            pkt_type,
            len(payload) + 1  # +1 for command byte
        ])
        
        # Payload with command byte
        full_payload = bytes([cmd]) + payload
        
        # Calculate CRC
        crc = self._calculate_crc16(full_payload)
        
        return header + full_payload + struct.pack('!H', crc)
    
    def parse_packets(self, data: bytes) -> List[Dict]:
        """
        Parse received data into packets.
        
        Args:
            data: Raw received bytes
            
        Returns:
            List of parsed packets
        """
        self._packet_buffer.extend(data)
        packets = []
        
        while len(self._packet_buffer) >= 6:  # Minimum packet size
            # Find sync bytes
            sync_idx = self._find_sync()
            if sync_idx < 0:
                self._packet_buffer.clear()
                break
            
            # Remove bytes before sync
            if sync_idx > 0:
                del self._packet_buffer[:sync_idx]
            
            # Check if we have enough data for header
            if len(self._packet_buffer) < 4:
                break
            
            pkt_type = self._packet_buffer[2]
            payload_len = self._packet_buffer[3]
            
            # Check if we have complete packet (header + payload + crc)
            total_len = 4 + payload_len + 2
            if len(self._packet_buffer) < total_len:
                break
            
            # Extract packet
            packet_data = bytes(self._packet_buffer[:total_len])
            del self._packet_buffer[:total_len]
            
            # Verify CRC
            payload = packet_data[4:4+payload_len]
            received_crc = struct.unpack('!H', packet_data[-2:])[0]
            calculated_crc = self._calculate_crc16(payload)
            
            if received_crc != calculated_crc:
                continue  # Skip invalid packet
            
            # Parse payload
            parsed = self._parse_payload(pkt_type, payload)
            if parsed:
                packets.append(parsed)
        
        return packets
    
    def _find_sync(self) -> int:
        """Find sync bytes in buffer."""
        for i in range(len(self._packet_buffer) - 1):
            if self._packet_buffer[i] == self.SYNC1 and self._packet_buffer[i+1] == self.SYNC2:
                return i
        return -1
    
    def _parse_payload(self, pkt_type: int, payload: bytes) -> Optional[Dict]:
        """Parse packet payload based on type."""
        if pkt_type == PacketType.SENSOR_DATA.value:
            return self._parse_sensor_data(payload)
        elif pkt_type == PacketType.ACK.value:
            return {'type': 'ack', 'status': payload[0] if payload else 0}
        elif pkt_type == PacketType.ERROR.value:
            return {'type': 'error', 'code': payload[0] if payload else 0}
        else:
            return {'type': 'unknown', 'raw': payload.hex()}
    
    def _parse_sensor_data(self, payload: bytes) -> Dict:
        """Parse sensor data payload."""
        if len(payload) < 2:
            return {}
        
        sensor_type = payload[0]
        num_values = payload[1]
        
        # Parse values based on sensor type
        if sensor_type == SensorType.IMU.value:
            # IMU: 6 floats (ax, ay, az, gx, gy, gz)
            if len(payload) >= 2 + 24:
                values = struct.unpack('!6f', payload[2:26])
                return {
                    'type': 'sensor',
                    'name': 'imu',
                    'sensor_type': sensor_type,
                    'value': list(values),
                    'timestamp': time.time(),
                    'unit': 'g,deg/s'
                }
        
        elif sensor_type == SensorType.DISTANCE.value:
            # Distance: 1 uint16 (mm)
            if len(payload) >= 4:
                distance_mm = struct.unpack('!H', payload[2:4])[0]
                return {
                    'type': 'sensor',
                    'name': 'distance',
                    'sensor_type': sensor_type,
                    'value': distance_mm / 1000.0,  # Convert to meters
                    'timestamp': time.time(),
                    'unit': 'm'
                }
        
        elif sensor_type == SensorType.CURRENT.value:
            # Current: 2 int16 (mA, mV)
            if len(payload) >= 6:
                current_ma, voltage_mv = struct.unpack('!2h', payload[2:6])
                return {
                    'type': 'sensor',
                    'name': 'current',
                    'sensor_type': sensor_type,
                    'value': [current_ma / 1000.0, voltage_mv / 1000.0],
                    'timestamp': time.time(),
                    'unit': 'A,V'
                }
        
        elif sensor_type == SensorType.BATTERY.value:
            # Battery: voltage (mV) and percentage
            if len(payload) >= 5:
                voltage_mv = struct.unpack('!H', payload[2:4])[0]
                percentage = payload[4]
                return {
                    'type': 'sensor',
                    'name': 'battery',
                    'sensor_type': sensor_type,
                    'value': [voltage_mv / 1000.0, percentage],
                    'timestamp': time.time(),
                    'unit': 'V,%'
                }
        
        # Generic parsing for unknown sensors
        return {
            'type': 'sensor',
            'name': f'sensor_{sensor_type:02x}',
            'sensor_type': sensor_type,
            'value': list(payload[2:]),
            'timestamp': time.time()
        }
    
    def _calculate_crc16(self, data: bytes) -> int:
        """Calculate CRC-16/CCITT checksum."""
        crc = 0xFFFF
        
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        
        return crc
