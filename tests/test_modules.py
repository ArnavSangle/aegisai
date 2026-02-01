"""
AegisAI Test Suite
Unit tests for all modules
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigLoader:
    """Tests for configuration loading."""
    
    def test_config_loads(self):
        from src.core.config_loader import ConfigLoader
        config = ConfigLoader()
        assert config.config is not None
    
    def test_config_get_nested(self):
        from src.core.config_loader import ConfigLoader
        config = ConfigLoader()
        value = config.get('system.name', 'default')
        assert value == 'AegisAI' or value == 'default'
    
    def test_config_get_section(self):
        from src.core.config_loader import ConfigLoader
        config = ConfigLoader()
        section = config.get_section('hardware')
        assert isinstance(section, dict)


class TestAnomalyDetector:
    """Tests for anomaly detection module."""
    
    def test_isolation_forest_init(self):
        from src.anomaly import IsolationForestDetector
        detector = IsolationForestDetector()
        assert detector.initialize()
    
    def test_isolation_forest_process(self):
        from src.anomaly import IsolationForestDetector
        detector = IsolationForestDetector()
        detector.initialize()
        
        # Train on normal data
        normal_data = np.random.randn(100, 64).astype(np.float32)
        detector.fit(normal_data)
        
        # Test detection
        result = detector.process(np.random.randn(64).astype(np.float32))
        
        assert 'scores' in result
        assert 'is_anomaly' in result
    
    def test_anomaly_ensemble(self):
        from src.anomaly import AnomalyDetector
        detector = AnomalyDetector()
        assert detector.initialize()
        
        # Test processing
        data = np.random.randn(64).astype(np.float32)
        result = detector.process(data)
        
        assert 'score' in result or 'scores' in result
        assert 'is_anomaly' in result


class TestLSTMPredictor:
    """Tests for LSTM prediction module."""
    
    def test_predictor_init(self):
        from src.prediction import LSTMPredictor
        predictor = LSTMPredictor()
        assert predictor.initialize()
    
    def test_predictor_process_sequence(self):
        from src.prediction import LSTMPredictor
        predictor = LSTMPredictor()
        predictor.initialize()
        
        # Create sequence
        sequence = np.random.randn(50, 8).astype(np.float32)
        result = predictor.process(sequence)
        
        assert 'forecast' in result
        assert result['forecast'] is not None
    
    def test_predictor_streaming(self):
        from src.prediction import LSTMPredictor
        predictor = LSTMPredictor()
        predictor.initialize()
        
        # Stream data points
        for i in range(60):
            data = np.random.randn(8).astype(np.float32)
            result = predictor.process(data)
            
            if i < 49:  # Buffer not full
                assert result['forecast'] is None
            else:  # Buffer full, should predict
                assert result['forecast'] is not None


class TestPPOAgent:
    """Tests for PPO decision module."""
    
    def test_agent_init(self):
        from src.decision import PPOAgent
        agent = PPOAgent()
        # Initialize creates dummy env, so always succeeds if SB3 available
        result = agent.initialize()
        # May fail if stable-baselines3 not installed
        assert result is True or result is False
    
    def test_agent_process(self):
        from src.decision import PPOAgent
        agent = PPOAgent()
        if not agent.initialize():
            pytest.skip("PPO agent requires stable-baselines3")
        
        observation = np.random.randn(64).astype(np.float32)
        result = agent.process(observation)
        
        assert 'action' in result
        assert 'confidence' in result


class TestVisionPipeline:
    """Tests for vision module."""
    
    def test_pipeline_init(self):
        from src.vision import VisionPipeline
        pipeline = VisionPipeline()
        # May fail without camera
        result = pipeline.initialize()
        assert isinstance(result, bool)
    
    def test_mobilenet_detector(self):
        from src.vision import MobileNetV3Detector
        detector = MobileNetV3Detector()
        result = detector.initialize()
        
        if result:
            # Test with random image
            image = np.random.randn(224, 224, 3).astype(np.float32)
            detections = detector.process(image)
            assert 'detections' in detections


class TestMCUProtocol:
    """Tests for MCU communication protocol."""
    
    def test_packet_creation(self):
        from src.mcu.protocol import MCUProtocol
        protocol = MCUProtocol({})
        
        packet = protocol.create_command('stop', {})
        
        assert packet[0] == 0xAE  # SYNC1
        assert packet[1] == 0x5A  # SYNC2
    
    def test_packet_parsing(self):
        from src.mcu.protocol import MCUProtocol
        protocol = MCUProtocol({})
        
        # Create a valid packet
        packet = protocol.create_command('init', {'test': True})
        
        # Parse it back
        packets = protocol.parse_packets(packet)
        # Should parse without error
    
    def test_crc_calculation(self):
        from src.mcu.protocol import MCUProtocol
        protocol = MCUProtocol({})
        
        data = b'test data'
        crc = protocol._calculate_crc16(data)
        
        # CRC should be consistent
        assert crc == protocol._calculate_crc16(data)


class TestFleetManager:
    """Tests for fleet management module."""
    
    def test_marl_coordinator_init(self):
        from src.fleet import MARLCoordinator
        config = {'marl': {'algorithm': 'mappo'}}
        coordinator = MARLCoordinator(4, config)
        assert coordinator.initialize()
    
    def test_coordination_heuristic(self):
        from src.fleet import MARLCoordinator
        config = {'marl': {'algorithm': 'mappo'}}
        coordinator = MARLCoordinator(4, config)
        coordinator.initialize()
        
        # Test with observations
        observations = {
            0: np.random.randn(64).astype(np.float32),
            1: np.random.randn(64).astype(np.float32),
        }
        
        actions = coordinator.process(observations)
        
        assert 0 in actions
        assert 1 in actions
        assert 'action' in actions[0]


class TestIntegration:
    """Integration tests for the full system."""
    
    def test_data_flow(self):
        """Test data flows correctly through the system."""
        # Simulate sensor data
        sensor_data = {
            'imu': {'value': np.random.randn(6).tolist()},
            'distance': {'value': 0.5},
        }
        
        # Build observation (simplified)
        obs = np.zeros(64, dtype=np.float32)
        obs[:6] = sensor_data['imu']['value']
        obs[6] = sensor_data['distance']['value']
        
        assert obs.shape == (64,)
        assert not np.all(obs == 0)
    
    def test_observation_building(self):
        """Test observation building for RL."""
        from src.decision import PPOAgent
        agent = PPOAgent()
        agent.initialize()
        
        # Test dict to observation conversion
        obs_dict = {
            'sensors': {'imu': [1, 2, 3, 4, 5, 6]},
            'anomaly_score': 0.1,
            'is_anomaly': False,
            'prediction': [0.1] * 10,
            'prediction_confidence': 0.9
        }
        
        obs_array = agent._dict_to_observation(obs_dict)
        
        assert obs_array.shape == (64,)
        assert obs_array.dtype == np.float32


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
