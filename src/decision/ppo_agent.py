"""
PPO Agent for Decision Making
Proximal Policy Optimization using Stable Baselines3
"""

import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
from loguru import logger

try:
    import torch
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("Stable Baselines3 not available, PPO will be disabled")

from ..core.base_module import BaseModule


class PPOAgent(BaseModule):
    """
    PPO-based decision making agent using Stable Baselines3.
    Optimized for real-time inference on Raspberry Pi.
    """
    
    def __init__(self):
        super().__init__('decision')
        self.agent: Optional[PPO] = None
        self.env = None
        
        # PPO configuration
        self.ppo_config = self.config.get('ppo', {})
        
        # Action/observation space config
        self.action_config = self.config.get('action_space', {})
        self.obs_config = self.config.get('observation_space', {})
        
        # Inference mode (no training overhead)
        self.inference_mode = True
        
    def initialize(self) -> bool:
        """Initialize PPO agent."""
        if not SB3_AVAILABLE:
            logger.error("Stable Baselines3 not available")
            return False
            
        try:
            # Check for pre-trained model
            model_path = Path("models/decision/ppo_agent.zip")
            
            if model_path.exists():
                self.agent = PPO.load(str(model_path))
                logger.info("Loaded pre-trained PPO agent")
            else:
                # Create new agent (will need training)
                self._create_agent()
                logger.info("Created new PPO agent (requires training)")
            
            # Set to evaluation mode
            if self.agent is not None:
                self.agent.policy.eval()
            
            self.is_initialized = True
            logger.info("PPO Agent initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PPO Agent: {e}")
            return False
    
    def _create_agent(self, env=None):
        """Create new PPO agent with configured parameters."""
        # Create dummy environment for initialization
        if env is None:
            env = self._create_dummy_env()
        
        # PPO hyperparameters from config
        self.agent = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.ppo_config.get('learning_rate', 3e-4),
            n_steps=self.ppo_config.get('n_steps', 2048),
            batch_size=self.ppo_config.get('batch_size', 64),
            n_epochs=self.ppo_config.get('n_epochs', 10),
            gamma=self.ppo_config.get('gamma', 0.99),
            gae_lambda=self.ppo_config.get('gae_lambda', 0.95),
            clip_range=self.ppo_config.get('clip_range', 0.2),
            ent_coef=self.ppo_config.get('ent_coef', 0.01),
            vf_coef=self.ppo_config.get('vf_coef', 0.5),
            max_grad_norm=self.ppo_config.get('max_grad_norm', 0.5),
            verbose=1,
            device='cpu',  # Use CPU for Pi compatibility
            tensorboard_log="logs/tensorboard/ppo"
        )
    
    def _create_dummy_env(self):
        """Create a dummy environment matching the expected spaces."""
        obs_shape = tuple(self.obs_config.get('shape', [64]))
        n_actions = self.action_config.get('n_actions', 8)
        action_type = self.action_config.get('type', 'discrete')
        
        class DummyEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=obs_shape, dtype=np.float32
                )
                if action_type == 'discrete':
                    self.action_space = gym.spaces.Discrete(n_actions)
                else:
                    self.action_space = gym.spaces.Box(
                        low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32
                    )
            
            def reset(self, seed=None, options=None):
                return self.observation_space.sample(), {}
            
            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}
        
        return DummyEnv()
    
    def process(self, observation: Union[np.ndarray, Dict]) -> Dict[str, Any]:
        """
        Get action from observation.
        
        Args:
            observation: Either numpy array or dict with observation data
            
        Returns:
            Action and metadata
        """
        if not self.is_initialized or self.agent is None:
            raise RuntimeError("Agent not initialized")
        
        # Extract observation array from dict if needed
        if isinstance(observation, dict):
            obs_array = self._dict_to_observation(observation)
        else:
            obs_array = observation
        
        # Ensure correct shape
        if obs_array.ndim == 1:
            obs_array = obs_array.reshape(1, -1)
        
        # Get action (deterministic for inference)
        with torch.no_grad():
            action, _ = self.agent.predict(obs_array, deterministic=self.inference_mode)
        
        # Get action probabilities for confidence
        action_probs = self._get_action_probs(obs_array)
        
        return {
            'action': int(action) if np.isscalar(action) else action.tolist(),
            'confidence': float(np.max(action_probs)) if action_probs is not None else 1.0,
            'action_probs': action_probs.tolist() if action_probs is not None else None
        }
    
    def _dict_to_observation(self, obs_dict: Dict) -> np.ndarray:
        """Convert observation dictionary to flat array."""
        obs_parts = []
        
        # Sensor data
        if 'sensors' in obs_dict:
            sensors = obs_dict['sensors']
            if isinstance(sensors, dict):
                for key in sorted(sensors.keys()):
                    value = sensors[key]
                    if isinstance(value, (list, np.ndarray)):
                        obs_parts.extend(np.asarray(value).flatten())
                    else:
                        obs_parts.append(float(value))
            else:
                obs_parts.extend(np.asarray(sensors).flatten())
        
        # Anomaly info
        if 'anomaly_score' in obs_dict:
            obs_parts.append(float(obs_dict['anomaly_score']))
        if 'is_anomaly' in obs_dict:
            obs_parts.append(1.0 if obs_dict['is_anomaly'] else 0.0)
        
        # Prediction info
        if 'prediction' in obs_dict and obs_dict['prediction'] is not None:
            pred = np.asarray(obs_dict['prediction']).flatten()
            obs_parts.extend(pred[:10])  # Limit prediction features
        
        if 'prediction_confidence' in obs_dict:
            obs_parts.append(float(obs_dict['prediction_confidence']))
        
        # Vision info
        if 'vision' in obs_dict and obs_dict['vision'] is not None:
            vision = obs_dict['vision']
            if 'features' in vision:
                features = np.asarray(vision['features']).flatten()
                obs_parts.extend(features[:16])  # Limit vision features
        
        # Pad or truncate to expected size
        expected_size = self.obs_config.get('shape', [64])[0]
        obs_array = np.array(obs_parts, dtype=np.float32)
        
        if len(obs_array) < expected_size:
            obs_array = np.pad(obs_array, (0, expected_size - len(obs_array)))
        elif len(obs_array) > expected_size:
            obs_array = obs_array[:expected_size]
        
        return obs_array
    
    def _get_action_probs(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """Get action probabilities from policy."""
        try:
            obs_tensor = torch.as_tensor(observation).float()
            with torch.no_grad():
                dist = self.agent.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.numpy()
            return probs.flatten()
        except Exception:
            return None
    
    def train(
        self,
        env,
        total_timesteps: int = 100000,
        eval_env=None,
        eval_freq: int = 10000
    ) -> Dict[str, Any]:
        """
        Train the PPO agent.
        
        Args:
            env: Training environment
            total_timesteps: Total training steps
            eval_env: Optional evaluation environment
            eval_freq: Evaluation frequency
            
        Returns:
            Training info
        """
        if not SB3_AVAILABLE:
            raise RuntimeError("Training requires Stable Baselines3")
        
        logger.info(f"Training PPO agent for {total_timesteps} timesteps")
        
        # Create agent with real environment
        self._create_agent(env)
        
        # Callbacks
        callbacks = []
        
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path="models/decision/",
                log_path="logs/ppo_eval/",
                eval_freq=eval_freq,
                deterministic=True
            )
            callbacks.append(eval_callback)
        
        # Train
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        logger.info("PPO training complete")
        
        return {
            'total_timesteps': total_timesteps,
            'model_path': 'models/decision/ppo_agent.zip'
        }
    
    def save_model(self, path: str = "models/decision/ppo_agent"):
        """Save trained agent."""
        if self.agent is None:
            raise RuntimeError("No agent to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(path)
        logger.info(f"Saved PPO agent to {path}")
    
    def set_inference_mode(self, deterministic: bool = True):
        """Set inference mode (deterministic vs stochastic actions)."""
        self.inference_mode = deterministic
        if self.agent is not None:
            self.agent.policy.eval()
    
    def shutdown(self):
        """Clean shutdown."""
        self.agent = None
        self.env = None
        self.is_initialized = False
        logger.info("PPO Agent shutdown")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        status = super().get_status()
        status.update({
            'inference_mode': self.inference_mode,
            'n_actions': self.action_config.get('n_actions', 8),
            'obs_shape': self.obs_config.get('shape', [64])
        })
        return status
