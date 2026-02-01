#!/usr/bin/env python3
"""
AegisAI Quick Start Example
Demonstrates basic usage of the system
"""

import asyncio
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def example_anomaly_detection():
    """Example: Using anomaly detection module."""
    print("\n=== Anomaly Detection Example ===\n")
    
    from src.anomaly import AnomalyDetector, IsolationForestDetector
    
    # Option 1: Use combined detector
    detector = AnomalyDetector()
    detector.initialize()
    
    # Option 2: Use standalone Isolation Forest
    if_detector = IsolationForestDetector()
    if_detector.initialize()
    
    # Simulate normal sensor data
    normal_data = np.random.randn(100, 64).astype(np.float32)
    if_detector.fit(normal_data)
    
    # Test with normal sample
    normal_sample = np.random.randn(64).astype(np.float32)
    result = if_detector.process(normal_sample)
    print(f"Normal sample - Score: {result['scores'][0]:.3f}, Anomaly: {result['is_anomaly']}")
    
    # Test with anomalous sample (outlier)
    anomaly_sample = np.random.randn(64).astype(np.float32) * 10
    result = if_detector.process(anomaly_sample)
    print(f"Anomaly sample - Score: {result['scores'][0]:.3f}, Anomaly: {result['is_anomaly']}")


async def example_prediction():
    """Example: Using LSTM predictor."""
    print("\n=== Prediction Example ===\n")
    
    from src.prediction import LSTMPredictor
    
    predictor = LSTMPredictor()
    predictor.initialize()
    
    # Simulate streaming sensor data
    print("Streaming data points...")
    for i in range(55):
        data_point = np.sin(i * 0.1) + np.random.randn(8).astype(np.float32) * 0.1
        result = predictor.process(data_point)
        
        if i < 49:
            status = "Buffering..."
        elif result['forecast'] is not None:
            status = f"Prediction available! Horizon: {result['forecast'].shape[0]} steps"
        else:
            status = "Processing..."
        
        if i % 10 == 0 or i >= 49:
            print(f"  Point {i}: {status}")


async def example_decision_making():
    """Example: Using PPO agent."""
    print("\n=== Decision Making Example ===\n")
    
    from src.decision import PPOAgent
    
    agent = PPOAgent()
    if not agent.initialize():
        print("PPO agent initialization failed (stable-baselines3 required)")
        return
    
    # Get action from random observation
    observation = np.random.randn(64).astype(np.float32)
    result = agent.process(observation)
    
    print(f"Observation shape: {observation.shape}")
    print(f"Action: {result['action']}")
    print(f"Action type: {result['action_type']}")
    print(f"Confidence: {result['confidence']:.3f}")


async def example_mcu_protocol():
    """Example: MCU communication protocol."""
    print("\n=== MCU Protocol Example ===\n")
    
    from src.mcu.protocol import MCUProtocol, PacketType
    
    protocol = MCUProtocol({})
    
    # Create motor command
    motor_cmd = protocol.create_command('motor', {
        'left': 0.5,
        'right': 0.5
    })
    print(f"Motor command packet: {motor_cmd.hex()}")
    print(f"  Length: {len(motor_cmd)} bytes")
    
    # Create stop command
    stop_cmd = protocol.create_command('stop', {})
    print(f"Stop command packet: {stop_cmd.hex()}")


async def example_fleet_coordination():
    """Example: Fleet coordination."""
    print("\n=== Fleet Coordination Example ===\n")
    
    from src.fleet import MARLCoordinator
    
    num_agents = 4
    config = {'marl': {'algorithm': 'mappo'}}
    
    coordinator = MARLCoordinator(num_agents, config)
    coordinator.initialize()
    
    # Create observations for each agent
    observations = {
        i: np.random.randn(64).astype(np.float32)
        for i in range(num_agents)
    }
    
    # Get coordinated actions
    actions = coordinator.process(observations)
    
    for agent_id, action_info in actions.items():
        print(f"Agent {agent_id}: action shape {action_info['action'].shape}")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("AegisAI Quick Start Examples")
    print("=" * 60)
    
    await example_anomaly_detection()
    await example_prediction()
    await example_decision_making()
    await example_mcu_protocol()
    await example_fleet_coordination()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
