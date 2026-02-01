"""
MARL Coordinator for Fleet Management
Multi-Agent PPO (MAPPO) for cooperative decision making
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class MARLCoordinator:
    """
    Multi-Agent Reinforcement Learning coordinator using MAPPO.
    Supports both centralized training with decentralized execution (CTDE)
    and fully decentralized operation.
    """
    
    def __init__(self, num_agents: int, config: Dict):
        """
        Initialize MARL coordinator.
        
        Args:
            num_agents: Number of agents in the fleet
            config: Fleet configuration
        """
        self.num_agents = num_agents
        self.config = config
        
        # MARL config
        marl_config = config.get('marl', {})
        self.algorithm = marl_config.get('algorithm', 'mappo')
        self.shared_policy = marl_config.get('shared_policy', True)
        self.centralized_critic = marl_config.get('centralized_critic', True)
        
        # Policy networks
        self.policy_net: Optional[nn.Module] = None
        self.critic_net: Optional[nn.Module] = None
        
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize MARL networks."""
        try:
            if TORCH_AVAILABLE:
                self._build_networks()
                
                # Load pre-trained weights if available
                policy_path = Path("models/fleet/mappo_policy.pt")
                if policy_path.exists():
                    self.policy_net.load_state_dict(torch.load(policy_path))
                    logger.info("Loaded pre-trained MAPPO policy")
                
                self.is_initialized = True
                logger.info(f"MARL Coordinator initialized ({self.algorithm})")
                return True
            else:
                logger.warning("PyTorch not available, using heuristic coordination")
                self.is_initialized = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize MARL Coordinator: {e}")
            return False
    
    def _build_networks(self):
        """Build MAPPO policy and critic networks."""
        obs_dim = 64  # Per-agent observation dimension
        action_dim = 8  # Per-agent action dimension
        hidden_dim = 128
        
        # Policy network (actor)
        if self.shared_policy:
            self.policy_net = MAPPOPolicy(
                obs_dim, action_dim, hidden_dim, self.num_agents
            )
        else:
            # Separate policy per agent
            self.policy_net = nn.ModuleList([
                MAPPOPolicy(obs_dim, action_dim, hidden_dim, 1)
                for _ in range(self.num_agents)
            ])
        
        # Critic network (centralized if enabled)
        if self.centralized_critic:
            # Centralized critic sees all observations
            self.critic_net = CentralizedCritic(
                obs_dim * self.num_agents, hidden_dim
            )
        else:
            self.critic_net = CentralizedCritic(obs_dim, hidden_dim)
        
        # Set to eval mode
        self.policy_net.eval()
        self.critic_net.eval()
    
    def process(self, observations: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        """
        Get coordinated actions for all agents.
        
        Args:
            observations: Dictionary mapping agent_id to observation
            
        Returns:
            Dictionary mapping agent_id to action info
        """
        if not self.is_initialized:
            raise RuntimeError("Coordinator not initialized")
        
        if not TORCH_AVAILABLE or self.policy_net is None:
            return self._heuristic_coordination(observations)
        
        actions = {}
        
        with torch.no_grad():
            # Stack observations
            obs_list = []
            agent_ids = sorted(observations.keys())
            
            for agent_id in agent_ids:
                obs = observations.get(agent_id, np.zeros(64))
                obs_list.append(torch.tensor(obs, dtype=torch.float32))
            
            obs_tensor = torch.stack(obs_list)  # (num_agents, obs_dim)
            
            # Get actions from policy
            if self.shared_policy:
                action_logits = self.policy_net(obs_tensor)
            else:
                action_logits = torch.stack([
                    self.policy_net[i](obs_tensor[i:i+1])
                    for i in range(len(agent_ids))
                ])
            
            # Sample actions
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            sampled_actions = action_dist.sample()
            
            # Build action dictionary
            for i, agent_id in enumerate(agent_ids):
                actions[agent_id] = {
                    'action': int(sampled_actions[i].item()),
                    'action_probs': action_probs[i].numpy().tolist(),
                    'confidence': float(action_probs[i].max().item())
                }
        
        return actions
    
    def _heuristic_coordination(self, observations: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        """Fallback heuristic coordination without neural networks."""
        actions = {}
        
        # Extract positions from observations
        positions = {}
        for agent_id, obs in observations.items():
            positions[agent_id] = obs[:3]  # First 3 elements are position
        
        # Calculate centroid
        if positions:
            centroid = np.mean(list(positions.values()), axis=0)
        else:
            centroid = np.zeros(3)
        
        for agent_id, obs in observations.items():
            pos = positions.get(agent_id, np.zeros(3))
            
            # Simple heuristic: move towards formation position
            formation_offset = self._get_formation_offset(agent_id)
            target = centroid + formation_offset
            
            direction = target - pos
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0.1:
                # Map direction to discrete action
                action = self._direction_to_action(direction)
            else:
                action = 0  # Stay
            
            actions[agent_id] = {
                'action': action,
                'confidence': 0.8,
                'heuristic': True
            }
        
        return actions
    
    def _get_formation_offset(self, agent_id: int) -> np.ndarray:
        """Get formation offset for agent."""
        angle = agent_id * 2 * np.pi / self.num_agents
        radius = 1.0
        return np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.0
        ])
    
    def _direction_to_action(self, direction: np.ndarray) -> int:
        """Map direction vector to discrete action."""
        # 8 directions + stay
        angle = np.arctan2(direction[1], direction[0])
        action = int((angle + np.pi) / (2 * np.pi / 8)) % 8
        return action
    
    def allocate_task(self, task_obs: np.ndarray) -> np.ndarray:
        """
        Use policy to allocate task to agents.
        
        Args:
            task_obs: Task observation vector
            
        Returns:
            Allocation probabilities for each agent
        """
        if not TORCH_AVAILABLE or self.policy_net is None:
            # Uniform allocation
            return np.ones(self.num_agents) / self.num_agents
        
        with torch.no_grad():
            task_tensor = torch.tensor(task_obs, dtype=torch.float32)
            
            # Broadcast task to all agents and get preference
            task_expanded = task_tensor.unsqueeze(0).expand(self.num_agents, -1)
            
            allocation_logits = self.policy_net(task_expanded)
            allocation_probs = F.softmax(allocation_logits.mean(dim=-1), dim=0)
            
            return allocation_probs.numpy()
    
    def train_step(
        self,
        observations: Dict[int, np.ndarray],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_observations: Dict[int, np.ndarray],
        dones: Dict[int, bool]
    ) -> Dict[str, float]:
        """
        Perform one training step (for online learning).
        
        Args:
            observations: Current observations
            actions: Taken actions
            rewards: Received rewards
            next_observations: Next observations
            dones: Episode done flags
            
        Returns:
            Training metrics
        """
        if not TORCH_AVAILABLE:
            return {}
        
        # This would implement MAPPO training step
        # For now, return empty (inference-only mode)
        return {}
    
    def save_models(self, path: str = "models/fleet"):
        """Save trained models."""
        if self.policy_net is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(self.policy_net.state_dict(), f"{path}/mappo_policy.pt")
            torch.save(self.critic_net.state_dict(), f"{path}/mappo_critic.pt")
            logger.info(f"Saved MARL models to {path}")
    
    def shutdown(self):
        """Clean shutdown."""
        self.policy_net = None
        self.critic_net = None
        self.is_initialized = False


# Neural Network Definitions
if TORCH_AVAILABLE:
    class MAPPOPolicy(nn.Module):
        """MAPPO Policy Network."""
        
        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_agents: int):
            super().__init__()
            
            self.fc1 = nn.Linear(obs_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
            
            # Layer normalization for stability
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        def forward(self, x):
            x = F.relu(self.ln1(self.fc1(x)))
            x = F.relu(self.ln2(self.fc2(x)))
            return self.fc3(x)
    
    class CentralizedCritic(nn.Module):
        """Centralized Critic for MAPPO."""
        
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
            
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        def forward(self, x):
            x = F.relu(self.ln1(self.fc1(x)))
            x = F.relu(self.ln2(self.fc2(x)))
            return self.fc3(x)
