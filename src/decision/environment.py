"""
Custom Gymnasium Environment for AegisAI
Defines the observation and action spaces for RL training
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from loguru import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


class AegisEnvironment(gym.Env):
    """
    Custom Gymnasium environment for AegisAI robot/drone control.
    Can be connected to simulation or real hardware.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize environment.
        
        Args:
            render_mode: Rendering mode
            config: Environment configuration
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.config = config or {}
        
        # Environment parameters
        self.max_steps = self.config.get('max_steps', 1000)
        self.n_sensors = self.config.get('n_sensors', 8)
        self.n_actions = self.config.get('n_actions', 8)
        
        # Define action space
        action_type = self.config.get('action_type', 'discrete')
        if action_type == 'discrete':
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, 
                shape=(self.n_actions,), 
                dtype=np.float32
            )
        
        # Define observation space
        obs_dim = self.config.get('obs_dim', 64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # State
        self._state = None
        self._step_count = 0
        self._episode_reward = 0.0
        
        # Hardware interface (can be None for simulation)
        self._hardware = None
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        self._step_count = 0
        self._episode_reward = 0.0
        
        # Initialize state
        if self._hardware is not None:
            # Get real sensor data
            self._state = self._hardware.get_sensors()
        else:
            # Simulation: random initial state
            self._state = self.np_random.uniform(
                low=-0.5, high=0.5, 
                size=(self.observation_space.shape[0],)
            ).astype(np.float32)
        
        observation = self._get_observation()
        info = {'step': 0}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return results.
        
        Args:
            action: Action to execute
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self._step_count += 1
        
        # Execute action
        if self._hardware is not None:
            # Send to real hardware
            self._hardware.execute_action(action)
            # Wait for effect and get new state
            self._state = self._hardware.get_sensors()
        else:
            # Simulation: apply action effect
            self._simulate_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self._episode_reward += reward
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self._step_count >= self.max_steps
        
        observation = self._get_observation()
        
        info = {
            'step': self._step_count,
            'episode_reward': self._episode_reward,
            'action': action
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return self._state.copy()
    
    def _simulate_action(self, action: int):
        """Simulate action effect on state."""
        # Simple dynamics for simulation
        if isinstance(action, (int, np.integer)):
            # Discrete action: map to direction
            direction = np.zeros(self._state.shape)
            if action < len(direction):
                direction[action] = 0.1
        else:
            # Continuous action
            direction = action * 0.1
        
        # Update state with noise
        noise = self.np_random.normal(0, 0.01, self._state.shape)
        self._state = np.clip(self._state + direction + noise, -1.0, 1.0).astype(np.float32)
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward based on current state and action.
        Override this for specific tasks.
        """
        # Default: reward for staying near origin
        distance = np.linalg.norm(self._state[:3])  # First 3 dims as position
        reward = -distance
        
        # Penalty for large actions (energy efficiency)
        if isinstance(action, np.ndarray):
            reward -= 0.01 * np.sum(np.abs(action))
        
        return float(reward)
    
    def _check_terminated(self) -> bool:
        """
        Check if episode should terminate.
        Override for specific termination conditions.
        """
        # Terminate if state goes out of bounds
        if np.any(np.abs(self._state) > 10.0):
            return True
        return False
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            # Print state summary
            print(f"Step: {self._step_count}, State norm: {np.linalg.norm(self._state):.3f}")
        elif self.render_mode == 'rgb_array':
            # Return image representation
            return self._render_frame()
        
    def _render_frame(self) -> np.ndarray:
        """Render frame for rgb_array mode."""
        # Simple visualization: state as grayscale image
        size = int(np.sqrt(len(self._state)))
        if size * size < len(self._state):
            size += 1
        
        # Pad state to square
        padded = np.zeros(size * size)
        padded[:len(self._state)] = self._state
        
        # Reshape and normalize
        frame = padded.reshape(size, size)
        frame = ((frame + 1) * 127.5).astype(np.uint8)
        
        # Convert to RGB
        frame = np.stack([frame, frame, frame], axis=-1)
        
        return frame
    
    def close(self):
        """Clean up environment."""
        if self._hardware is not None:
            self._hardware.shutdown()
    
    def set_hardware(self, hardware):
        """Connect to hardware interface."""
        self._hardware = hardware
        logger.info("Connected environment to hardware interface")


class AegisNavigationEnv(AegisEnvironment):
    """
    Navigation-specific environment.
    Robot must navigate to target while avoiding obstacles.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._target = None
        self._obstacles = []
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Set random target
        self._target = self.np_random.uniform(-0.8, 0.8, size=3).astype(np.float32)
        
        # Set random obstacles
        n_obstacles = self.config.get('n_obstacles', 3)
        self._obstacles = [
            self.np_random.uniform(-0.8, 0.8, size=3).astype(np.float32)
            for _ in range(n_obstacles)
        ]
        
        info['target'] = self._target.tolist()
        return obs, info
    
    def _calculate_reward(self, action) -> float:
        """Navigation reward: reach target, avoid obstacles."""
        position = self._state[:3]
        
        # Distance to target
        dist_to_target = np.linalg.norm(position - self._target)
        reward = -dist_to_target
        
        # Bonus for reaching target
        if dist_to_target < 0.1:
            reward += 10.0
        
        # Penalty for obstacle collision
        for obs in self._obstacles:
            dist_to_obs = np.linalg.norm(position - obs)
            if dist_to_obs < 0.15:
                reward -= 5.0
        
        return float(reward)
    
    def _check_terminated(self) -> bool:
        """Terminate on target reached or collision."""
        position = self._state[:3]
        
        # Reached target
        if np.linalg.norm(position - self._target) < 0.1:
            return True
        
        # Collision
        for obs in self._obstacles:
            if np.linalg.norm(position - obs) < 0.1:
                return True
        
        return super()._check_terminated()
