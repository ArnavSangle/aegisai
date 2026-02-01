"""
Fleet Communication Module
Mesh networking for multi-agent coordination
"""

import asyncio
import json
import struct
from typing import Dict, Any, Optional, List, Callable
from loguru import logger
import socket
import threading
from queue import Queue
from dataclasses import dataclass
from enum import Enum

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class MessageType(Enum):
    """Fleet communication message types."""
    HEARTBEAT = 0x01
    STATE_UPDATE = 0x02
    ACTION_BROADCAST = 0x03
    TASK_ALLOCATION = 0x04
    EMERGENCY = 0x05
    SYNC_REQUEST = 0x06
    SYNC_RESPONSE = 0x07


@dataclass
class FleetMessage:
    """Fleet communication message."""
    msg_type: MessageType
    sender_id: int
    payload: Dict[str, Any]
    timestamp: float = 0.0
    sequence: int = 0


class FleetCommunication:
    """
    Communication layer for fleet coordination.
    Supports mesh networking over WiFi, serial, or BLE.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize fleet communication.
        
        Args:
            config: Communication configuration
        """
        self.config = config
        self.protocol = config.get('protocol', 'mesh')
        self.range_m = config.get('range_m', 50)
        
        # Agent identification
        self._agent_id: int = 0
        self._sequence: int = 0
        
        # Message queues
        self._inbox: Queue = Queue()
        self._outbox: Queue = Queue()
        
        # Peer tracking
        self._peers: Dict[int, Dict] = {}
        
        # Communication backends
        self._udp_socket: Optional[socket.socket] = None
        self._serial_port = None
        self._running = False
        
        # Callbacks
        self._message_handlers: Dict[MessageType, List[Callable]] = {}
        
    def initialize(self) -> bool:
        """Initialize communication backends."""
        try:
            # Generate agent ID (could be from config, MAC address, etc.)
            self._agent_id = self._generate_agent_id()
            
            # Initialize UDP for WiFi mesh
            if self.protocol in ['mesh', 'wifi']:
                self._init_udp()
            
            # Initialize serial for direct connection
            if self.protocol in ['serial', 'mesh']:
                self._init_serial()
            
            # Start communication threads
            self._running = True
            self._start_comm_threads()
            
            logger.info(f"Fleet Communication initialized (Agent ID: {self._agent_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Fleet Communication: {e}")
            return False
    
    def _generate_agent_id(self) -> int:
        """Generate unique agent ID."""
        # Try to get from config first
        if 'agent_id' in self.config:
            return self.config['agent_id']
        
        # Generate from hostname hash
        import hashlib
        hostname = socket.gethostname()
        hash_val = hashlib.md5(hostname.encode()).hexdigest()
        return int(hash_val[:4], 16) % 256
    
    def _init_udp(self):
        """Initialize UDP socket for WiFi communication."""
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        port = self.config.get('port', 5555)
        self._udp_socket.bind(('', port))
        self._udp_socket.setblocking(False)
        
        self._broadcast_addr = ('<broadcast>', port)
        logger.info(f"UDP socket initialized on port {port}")
    
    def _init_serial(self):
        """Initialize serial port for direct communication."""
        if not SERIAL_AVAILABLE:
            return
        
        port = self.config.get('serial_port', '/dev/ttyUSB0')
        baudrate = self.config.get('baudrate', 115200)
        
        try:
            self._serial_port = serial.Serial(port, baudrate, timeout=0.1)
            logger.info(f"Serial port initialized: {port}")
        except Exception as e:
            logger.warning(f"Serial port not available: {e}")
    
    def _start_comm_threads(self):
        """Start communication threads."""
        # Receiver thread
        self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._recv_thread.start()
        
        # Sender thread
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()
        
        # Heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
    
    def _receive_loop(self):
        """Background thread for receiving messages."""
        while self._running:
            try:
                # Check UDP
                if self._udp_socket:
                    try:
                        data, addr = self._udp_socket.recvfrom(4096)
                        message = self._decode_message(data)
                        if message and message.sender_id != self._agent_id:
                            self._process_received(message, addr)
                    except BlockingIOError:
                        pass
                
                # Check serial
                if self._serial_port and self._serial_port.in_waiting:
                    data = self._serial_port.read(self._serial_port.in_waiting)
                    message = self._decode_message(data)
                    if message:
                        self._process_received(message, None)
                
                asyncio.sleep(0.001)  # Small delay
                
            except Exception as e:
                logger.error(f"Receive error: {e}")
    
    def _send_loop(self):
        """Background thread for sending messages."""
        while self._running:
            try:
                if not self._outbox.empty():
                    message = self._outbox.get()
                    self._transmit(message)
                else:
                    asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Send error: {e}")
    
    def _heartbeat_loop(self):
        """Background thread for heartbeat messages."""
        import time
        while self._running:
            self.broadcast({
                'type': 'heartbeat',
                'agent_id': self._agent_id,
                'timestamp': time.time()
            }, MessageType.HEARTBEAT)
            time.sleep(1.0)  # 1 Hz heartbeat
    
    def _process_received(self, message: FleetMessage, addr):
        """Process received message."""
        # Update peer tracking
        self._peers[message.sender_id] = {
            'last_seen': message.timestamp,
            'address': addr
        }
        
        # Add to inbox
        self._inbox.put(message)
        
        # Call registered handlers
        if message.msg_type in self._message_handlers:
            for handler in self._message_handlers[message.msg_type]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Handler error: {e}")
    
    def _encode_message(self, message: FleetMessage) -> bytes:
        """Encode message for transmission."""
        # Header: type (1), sender (1), sequence (2), payload_len (2)
        payload_json = json.dumps(message.payload).encode()
        
        header = struct.pack(
            '!BBHH',
            message.msg_type.value,
            message.sender_id,
            message.sequence,
            len(payload_json)
        )
        
        return header + payload_json
    
    def _decode_message(self, data: bytes) -> Optional[FleetMessage]:
        """Decode received message."""
        if len(data) < 6:  # Minimum header size
            return None
        
        try:
            msg_type, sender_id, sequence, payload_len = struct.unpack('!BBHH', data[:6])
            payload_json = data[6:6+payload_len]
            payload = json.loads(payload_json.decode())
            
            return FleetMessage(
                msg_type=MessageType(msg_type),
                sender_id=sender_id,
                payload=payload,
                sequence=sequence
            )
        except Exception as e:
            logger.warning(f"Message decode error: {e}")
            return None
    
    def _transmit(self, message: FleetMessage):
        """Transmit message over all available channels."""
        data = self._encode_message(message)
        
        # Send via UDP broadcast
        if self._udp_socket:
            try:
                self._udp_socket.sendto(data, self._broadcast_addr)
            except Exception as e:
                logger.warning(f"UDP send failed: {e}")
        
        # Send via serial
        if self._serial_port:
            try:
                self._serial_port.write(data)
            except Exception as e:
                logger.warning(f"Serial send failed: {e}")
    
    def broadcast(self, payload: Dict, msg_type: MessageType = MessageType.STATE_UPDATE):
        """
        Broadcast message to all peers.
        
        Args:
            payload: Message payload
            msg_type: Message type
        """
        import time
        self._sequence += 1
        
        message = FleetMessage(
            msg_type=msg_type,
            sender_id=self._agent_id,
            payload=payload,
            timestamp=time.time(),
            sequence=self._sequence
        )
        
        self._outbox.put(message)
    
    async def broadcast_async(self, payload: Dict, msg_type: MessageType = MessageType.STATE_UPDATE):
        """Async version of broadcast."""
        self.broadcast(payload, msg_type)
    
    def receive(self, timeout: float = 0.1) -> Optional[FleetMessage]:
        """
        Receive a message.
        
        Args:
            timeout: Receive timeout in seconds
            
        Returns:
            Received message or None
        """
        try:
            return self._inbox.get(timeout=timeout)
        except:
            return None
    
    async def receive_all_async(self, timeout: float = 0.1) -> List[Dict]:
        """
        Receive all pending messages asynchronously.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            List of message payloads
        """
        messages = []
        
        while not self._inbox.empty():
            try:
                msg = self._inbox.get_nowait()
                messages.append(msg.payload)
            except:
                break
        
        return messages
    
    def register_handler(self, msg_type: MessageType, handler: Callable):
        """Register a message handler."""
        if msg_type not in self._message_handlers:
            self._message_handlers[msg_type] = []
        self._message_handlers[msg_type].append(handler)
    
    def get_agent_id(self) -> int:
        """Get this agent's ID."""
        return self._agent_id
    
    def get_peers(self) -> Dict[int, Dict]:
        """Get known peers."""
        return self._peers.copy()
    
    def get_active_peers(self, max_age: float = 5.0) -> List[int]:
        """Get list of recently active peer IDs."""
        import time
        now = time.time()
        return [
            peer_id for peer_id, info in self._peers.items()
            if now - info.get('last_seen', 0) < max_age
        ]
    
    def shutdown(self):
        """Clean shutdown."""
        self._running = False
        
        if self._udp_socket:
            self._udp_socket.close()
        
        if self._serial_port:
            self._serial_port.close()
        
        logger.info("Fleet Communication shutdown")
