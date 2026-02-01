#!/usr/bin/env python3
"""
AegisAI Training Script
Train all AI models for the robot
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config_loader import ConfigLoader
from src.core.logger import setup_logger
from loguru import logger


def train_anomaly_detector(config, data_path: str):
    """Train anomaly detection models."""
    from src.anomaly import AnomalyDetector
    import numpy as np
    
    logger.info("Training Anomaly Detector...")
    
    # Load or generate training data
    if Path(data_path).exists():
        data = np.load(data_path)
    else:
        logger.info("Generating synthetic training data...")
        # Generate normal operation data
        data = np.random.randn(10000, 64).astype(np.float32)
    
    detector = AnomalyDetector()
    detector.initialize()
    detector.fit(data, epochs=50)
    detector.save_models()
    
    logger.info("Anomaly Detector training complete!")


def train_lstm_predictor(config, data_path: str):
    """Train LSTM prediction model."""
    from src.prediction import LSTMPredictor
    import numpy as np
    
    logger.info("Training LSTM Predictor...")
    
    predictor = LSTMPredictor()
    predictor.initialize()
    
    # Load or generate time series data
    if Path(data_path).exists():
        raw_data = np.load(data_path)
    else:
        logger.info("Generating synthetic time series data...")
        # Generate synthetic sensor time series
        t = np.linspace(0, 100, 10000)
        raw_data = np.column_stack([
            np.sin(t * 0.1) + np.random.randn(10000) * 0.1,
            np.cos(t * 0.15) + np.random.randn(10000) * 0.1,
            np.sin(t * 0.2 + 1) + np.random.randn(10000) * 0.1,
            np.cos(t * 0.25 + 2) + np.random.randn(10000) * 0.1,
            np.sin(t * 0.3) * 0.5 + np.random.randn(10000) * 0.05,
            np.cos(t * 0.35) * 0.5 + np.random.randn(10000) * 0.05,
            np.sin(t * 0.4 + 3) * 0.3 + np.random.randn(10000) * 0.05,
            np.cos(t * 0.45 + 4) * 0.3 + np.random.randn(10000) * 0.05,
        ]).astype(np.float32)
    
    # Prepare sequences
    X, y = predictor.prepare_sequences(raw_data)
    
    # Train
    predictor.fit(X, y, epochs=100, batch_size=64)
    predictor.save_model()
    
    # Convert to TFLite
    predictor.convert_to_tflite(
        quantization=config.get('prediction.tflite.quantization', 'int8')
    )
    
    logger.info("LSTM Predictor training complete!")


def train_ppo_agent(config, env_name: str = 'navigation'):
    """Train PPO reinforcement learning agent."""
    from src.decision import PPOAgent, AegisEnvironment, AegisNavigationEnv
    
    logger.info("Training PPO Agent...")
    
    # Create environments
    if env_name == 'navigation':
        env = AegisNavigationEnv(config={
            'max_steps': 500,
            'n_obstacles': 5
        })
        eval_env = AegisNavigationEnv(config={
            'max_steps': 500,
            'n_obstacles': 5
        })
    else:
        env = AegisEnvironment()
        eval_env = AegisEnvironment()
    
    agent = PPOAgent()
    agent.train(
        env=env,
        total_timesteps=500000,
        eval_env=eval_env,
        eval_freq=10000
    )
    agent.save_model()
    
    logger.info("PPO Agent training complete!")


def train_fleet_marl(config):
    """Train MARL for fleet coordination."""
    from src.fleet import MARLCoordinator
    
    logger.info("Training Fleet MARL...")
    
    num_agents = config.get('fleet.num_agents', 4)
    coordinator = MARLCoordinator(num_agents, config.config)
    coordinator.initialize()
    
    # MARL training would require a multi-agent environment
    # This is a placeholder for now
    logger.warning("MARL training requires multi-agent simulation environment")
    logger.info("Using pre-trained or heuristic coordination")
    
    coordinator.save_models()
    
    logger.info("Fleet MARL setup complete!")


def main():
    parser = argparse.ArgumentParser(description='AegisAI Model Training')
    parser.add_argument('--model', type=str, default='all',
                       choices=['anomaly', 'prediction', 'decision', 'fleet', 'all'],
                       help='Which model to train')
    parser.add_argument('--data-dir', type=str, default='data/',
                       help='Directory containing training data')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level=args.log_level, log_file='logs/training.log')
    
    # Load configuration
    config = ConfigLoader()
    
    logger.info("="*60)
    logger.info("AegisAI Model Training")
    logger.info("="*60)
    
    # Train selected models
    if args.model in ['anomaly', 'all']:
        train_anomaly_detector(config, f"{args.data_dir}/anomaly_data.npy")
    
    if args.model in ['prediction', 'all']:
        train_lstm_predictor(config, f"{args.data_dir}/timeseries_data.npy")
    
    if args.model in ['decision', 'all']:
        train_ppo_agent(config)
    
    if args.model in ['fleet', 'all']:
        train_fleet_marl(config)
    
    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
