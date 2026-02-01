#!/usr/bin/env python3
"""
AegisAI Full System Demo
Demonstrates running the complete AI pipeline
"""

import asyncio
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockMCU:
    """Mock MCU for demo purposes."""
    
    def __init__(self):
        self.step = 0
    
    def read_sensors(self):
        """Generate mock sensor data."""
        self.step += 1
        
        # Simulate IMU
        imu = {
            'accel': [
                np.sin(self.step * 0.1) + np.random.randn() * 0.1,
                np.cos(self.step * 0.1) + np.random.randn() * 0.1,
                9.81 + np.random.randn() * 0.1
            ],
            'gyro': [
                np.random.randn() * 0.01,
                np.random.randn() * 0.01,
                np.random.randn() * 0.01
            ]
        }
        
        # Simulate distance sensor
        distance = {
            'value': 0.5 + np.sin(self.step * 0.05) * 0.3
        }
        
        # Simulate battery
        battery = {
            'voltage': 11.2 - self.step * 0.001,
            'current': 1.5 + np.random.randn() * 0.1
        }
        
        return {
            'imu': imu,
            'distance': distance,
            'battery': battery,
            'timestamp': time.time()
        }
    
    def send_command(self, command):
        """Mock command execution."""
        pass


class AegisDemo:
    """Demonstration of complete AegisAI pipeline."""
    
    def __init__(self):
        self.mcu = MockMCU()
        self.anomaly_detector = None
        self.predictor = None
        self.decision_agent = None
        self.running = False
    
    async def initialize(self):
        """Initialize all modules."""
        print("Initializing AegisAI Demo...")
        
        # Import modules
        from src.anomaly import AnomalyDetector, IsolationForestDetector
        from src.prediction import LSTMPredictor
        from src.decision import PPOAgent
        
        # Initialize anomaly detection
        print("  [1/3] Anomaly Detection...", end=" ")
        self.anomaly_detector = AnomalyDetector()
        if self.anomaly_detector.initialize():
            # Train with random "normal" data for demo
            normal_data = np.random.randn(200, 64).astype(np.float32)
            if hasattr(self.anomaly_detector, 'isolation_forest') and self.anomaly_detector.isolation_forest:
                self.anomaly_detector.isolation_forest.fit(normal_data)
            print("✓")
        else:
            print("✗")
        
        # Initialize prediction
        print("  [2/3] Prediction Engine...", end=" ")
        self.predictor = LSTMPredictor()
        if self.predictor.initialize():
            print("✓")
        else:
            print("✗")
        
        # Initialize decision making
        print("  [3/3] Decision Agent...", end=" ")
        self.decision_agent = PPOAgent()
        if self.decision_agent.initialize():
            print("✓")
        else:
            print("✗ (requires stable-baselines3)")
        
        print("Initialization complete!\n")
    
    def build_observation(self, sensor_data, anomaly_result, prediction_result):
        """Build observation vector for RL agent."""
        obs = np.zeros(64, dtype=np.float32)
        
        # Sensor data (0-15)
        obs[0:3] = sensor_data['imu']['accel']
        obs[3:6] = sensor_data['imu']['gyro']
        obs[6] = sensor_data['distance']['value']
        obs[7] = sensor_data['battery']['voltage']
        
        # Anomaly features (16-23)
        obs[16] = anomaly_result.get('score', 0)
        obs[17] = 1.0 if anomaly_result.get('is_anomaly', False) else 0.0
        
        # Prediction features (24-39)
        if prediction_result.get('forecast') is not None:
            forecast = prediction_result['forecast'].flatten()[:16]
            obs[24:24+len(forecast)] = forecast
        
        return obs
    
    async def run_loop(self, num_steps=100):
        """Run main processing loop."""
        print("Starting main loop...")
        print("-" * 60)
        
        self.running = True
        loop_times = []
        
        for step in range(num_steps):
            if not self.running:
                break
            
            start_time = time.perf_counter()
            
            # 1. Read sensors
            sensor_data = self.mcu.read_sensors()
            
            # 2. Build sensor array for anomaly/prediction
            sensor_array = np.array(
                sensor_data['imu']['accel'] + 
                sensor_data['imu']['gyro'] + 
                [sensor_data['distance']['value'], sensor_data['battery']['voltage']],
                dtype=np.float32
            )
            
            # Pad to expected size
            if len(sensor_array) < 64:
                sensor_array = np.pad(sensor_array, (0, 64 - len(sensor_array)))
            
            # 3. Check for anomalies
            anomaly_result = self.anomaly_detector.process(sensor_array)
            
            # 4. Predict future states
            prediction_input = sensor_array[:8]  # Use first 8 features
            prediction_result = self.predictor.process(prediction_input)
            
            # 5. Build observation
            observation = self.build_observation(
                sensor_data, anomaly_result, prediction_result
            )
            
            # 6. Get action from agent
            if self.decision_agent and self.decision_agent.model:
                action_result = self.decision_agent.process(observation)
                action = action_result['action']
                action_type = action_result['action_type']
            else:
                # Fallback: simple reactive control
                action = np.zeros(4, dtype=np.float32)
                action_type = 'stop'
                if sensor_data['distance']['value'] > 0.3:
                    action[0] = 0.5  # Forward
                    action_type = 'forward'
            
            # 7. Execute action
            self.mcu.send_command({
                'action': action.tolist(),
                'type': action_type
            })
            
            # Track timing
            loop_time = (time.perf_counter() - start_time) * 1000
            loop_times.append(loop_time)
            
            # Print status every 10 steps
            if step % 10 == 0:
                is_anomaly = anomaly_result.get('is_anomaly', False)
                anomaly_str = "⚠ ANOMALY" if is_anomaly else "Normal"
                
                has_prediction = prediction_result.get('forecast') is not None
                pred_str = "Yes" if has_prediction else "Buffering"
                
                print(f"Step {step:4d} | "
                      f"Action: {action_type:8s} | "
                      f"Status: {anomaly_str:9s} | "
                      f"Prediction: {pred_str:9s} | "
                      f"Time: {loop_time:.1f}ms")
            
            # Small delay to simulate real-time
            await asyncio.sleep(0.01)
        
        print("-" * 60)
        print(f"\nLoop Statistics:")
        print(f"  Total steps: {len(loop_times)}")
        print(f"  Avg loop time: {np.mean(loop_times):.2f}ms")
        print(f"  Max loop time: {np.max(loop_times):.2f}ms")
        print(f"  Min loop time: {np.min(loop_times):.2f}ms")
        print(f"  Theoretical max Hz: {1000/np.mean(loop_times):.1f}")
    
    def stop(self):
        """Stop the loop."""
        self.running = False


async def main():
    """Run the demo."""
    print("=" * 60)
    print("AegisAI Full System Demo")
    print("=" * 60)
    print()
    
    demo = AegisDemo()
    await demo.initialize()
    await demo.run_loop(num_steps=100)
    
    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
