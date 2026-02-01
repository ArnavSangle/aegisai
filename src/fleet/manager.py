"""
Fleet Manager for AegisAI
Coordinates multiple agents using MARL and distributed communication
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger
import asyncio

from ..core.base_module import BaseModule
from .marl_coordinator import MARLCoordinator
from .communication import FleetCommunication


class FleetManager(BaseModule):
    """
    Fleet management system for coordinating multiple robots/drones.
    Uses MARL for cooperative decision making and mesh communication.
    """
    
    def __init__(self):
        super().__init__('fleet')
        self.num_agents = self.config.get('num_agents', 4)
        self.coordinator: Optional[MARLCoordinator] = None
        self.comm: Optional[FleetCommunication] = None
        
        # Agent states
        self.agent_states: Dict[int, Dict] = {}
        self.agent_positions: Dict[int, np.ndarray] = {}
        
        # My agent ID (set during initialization)
        self.my_agent_id: int = 0
        
        # Coordination config
        coord_config = self.config.get('coordination', {})
        self.task_allocation = coord_config.get('task_allocation', 'auction')
        self.collision_avoidance = coord_config.get('collision_avoidance', True)
        self.formation_control = coord_config.get('formation_control', True)
        
    def initialize(self) -> bool:
        """Initialize fleet management system."""
        try:
            # Initialize MARL coordinator
            self.coordinator = MARLCoordinator(self.num_agents, self.config)
            self.coordinator.initialize()
            
            # Initialize communication
            self.comm = FleetCommunication(self.config.get('communication', {}))
            self.comm.initialize()
            
            # Get my agent ID from communication layer
            self.my_agent_id = self.comm.get_agent_id()
            
            # Initialize agent states
            for i in range(self.num_agents):
                self.agent_states[i] = {
                    'active': i == self.my_agent_id,
                    'position': np.zeros(3),
                    'velocity': np.zeros(3),
                    'task': None,
                    'status': 'idle'
                }
            
            self.is_initialized = True
            logger.info(f"Fleet Manager initialized (Agent {self.my_agent_id}/{self.num_agents})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Fleet Manager: {e}")
            return False
    
    def process(self, observation: Dict) -> Dict[str, Any]:
        """
        Process observation and return coordinated action.
        
        Args:
            observation: Local observation from sensors
            
        Returns:
            Coordinated action and fleet info
        """
        if not self.is_initialized:
            raise RuntimeError("Fleet Manager not initialized")
        
        # Update my state
        self._update_my_state(observation)
        
        # Get observations from other agents
        fleet_obs = self._gather_fleet_observations()
        
        # Run MARL for coordinated decision
        joint_action = self.coordinator.process(fleet_obs)
        
        # Get my action from joint action
        my_action = joint_action.get(self.my_agent_id, {'action': 0})
        
        # Apply collision avoidance if enabled
        if self.collision_avoidance:
            my_action = self._apply_collision_avoidance(my_action)
        
        # Apply formation control if enabled
        if self.formation_control:
            my_action = self._apply_formation_control(my_action)
        
        return {
            'action': my_action['action'],
            'fleet_status': self._get_fleet_status(),
            'coordination_info': {
                'task_allocation': self.task_allocation,
                'my_task': self.agent_states[self.my_agent_id].get('task'),
                'formation_target': my_action.get('formation_target')
            }
        }
    
    async def coordinate_async(self, local_action: Dict) -> Dict[str, Any]:
        """
        Async fleet coordination.
        
        Args:
            local_action: Action from local decision module
            
        Returns:
            Coordinated action
        """
        # Broadcast my intended action
        await self.comm.broadcast_async({
            'agent_id': self.my_agent_id,
            'action': local_action,
            'position': self.agent_positions.get(self.my_agent_id, [0, 0, 0]).tolist(),
            'status': self.agent_states[self.my_agent_id].get('status')
        })
        
        # Receive messages from other agents
        messages = await self.comm.receive_all_async(timeout=0.05)
        
        # Update fleet state from messages
        for msg in messages:
            agent_id = msg.get('agent_id')
            if agent_id is not None and agent_id != self.my_agent_id:
                self._update_agent_state(agent_id, msg)
        
        # Return coordinated action
        return self.process({'local_action': local_action})
    
    def _update_my_state(self, observation: Dict):
        """Update local agent state from observation."""
        if 'position' in observation:
            self.agent_positions[self.my_agent_id] = np.array(observation['position'])
        
        if 'sensors' in observation:
            self.agent_states[self.my_agent_id]['sensors'] = observation['sensors']
        
        self.agent_states[self.my_agent_id]['status'] = 'active'
    
    def _update_agent_state(self, agent_id: int, data: Dict):
        """Update state of remote agent."""
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = {}
        
        self.agent_states[agent_id]['active'] = True
        self.agent_states[agent_id]['last_update'] = asyncio.get_event_loop().time()
        
        if 'position' in data:
            self.agent_positions[agent_id] = np.array(data['position'])
        
        if 'status' in data:
            self.agent_states[agent_id]['status'] = data['status']
        
        if 'action' in data:
            self.agent_states[agent_id]['last_action'] = data['action']
    
    def _gather_fleet_observations(self) -> Dict[int, np.ndarray]:
        """Gather observations from all known agents."""
        fleet_obs = {}
        
        for agent_id, state in self.agent_states.items():
            if state.get('active', False):
                # Build observation vector for this agent
                obs = np.zeros(64)  # Standard observation size
                
                # Position
                if agent_id in self.agent_positions:
                    obs[:3] = self.agent_positions[agent_id]
                
                # Status encoding
                status = state.get('status', 'unknown')
                status_map = {'idle': 0, 'active': 1, 'busy': 2, 'error': 3}
                obs[3] = status_map.get(status, -1)
                
                fleet_obs[agent_id] = obs
        
        return fleet_obs
    
    def _apply_collision_avoidance(self, action: Dict) -> Dict:
        """Apply collision avoidance to action."""
        my_pos = self.agent_positions.get(self.my_agent_id, np.zeros(3))
        
        min_distance = 0.5  # Minimum safe distance
        
        for agent_id, pos in self.agent_positions.items():
            if agent_id == self.my_agent_id:
                continue
            
            distance = np.linalg.norm(my_pos - pos)
            
            if distance < min_distance:
                # Calculate repulsion vector
                repulsion = (my_pos - pos) / (distance + 1e-6)
                
                # Modify action to avoid collision
                if 'velocity_adjustment' not in action:
                    action['velocity_adjustment'] = np.zeros(3)
                
                action['velocity_adjustment'] += repulsion * (min_distance - distance)
                action['collision_warning'] = True
                
                logger.warning(f"Collision warning with agent {agent_id}, distance: {distance:.2f}")
        
        return action
    
    def _apply_formation_control(self, action: Dict) -> Dict:
        """Apply formation control to action."""
        # Define formation positions (relative to centroid)
        formations = {
            'line': [
                np.array([i * 1.0, 0, 0]) 
                for i in range(self.num_agents)
            ],
            'square': [
                np.array([i % 2 * 1.0, i // 2 * 1.0, 0])
                for i in range(self.num_agents)
            ],
            'diamond': [
                np.array([np.cos(i * 2 * np.pi / self.num_agents),
                         np.sin(i * 2 * np.pi / self.num_agents), 0])
                for i in range(self.num_agents)
            ]
        }
        
        current_formation = 'diamond'  # Could be made configurable
        
        # Calculate fleet centroid
        active_positions = [
            self.agent_positions.get(i, np.zeros(3))
            for i in range(self.num_agents)
            if self.agent_states.get(i, {}).get('active', False)
        ]
        
        if active_positions:
            centroid = np.mean(active_positions, axis=0)
            
            # Target position for this agent in formation
            target_offset = formations[current_formation][self.my_agent_id]
            target_pos = centroid + target_offset
            
            action['formation_target'] = target_pos.tolist()
        
        return action
    
    def allocate_task(self, task: Dict) -> int:
        """
        Allocate task to best agent.
        
        Args:
            task: Task description with location, priority, etc.
            
        Returns:
            Agent ID assigned to task
        """
        if self.task_allocation == 'auction':
            return self._auction_allocation(task)
        elif self.task_allocation == 'greedy':
            return self._greedy_allocation(task)
        else:
            return self._learned_allocation(task)
    
    def _auction_allocation(self, task: Dict) -> int:
        """Auction-based task allocation."""
        task_pos = np.array(task.get('position', [0, 0, 0]))
        
        # Each agent bids based on distance and current load
        best_agent = self.my_agent_id
        best_bid = float('inf')
        
        for agent_id, state in self.agent_states.items():
            if not state.get('active', False):
                continue
            
            pos = self.agent_positions.get(agent_id, np.zeros(3))
            distance = np.linalg.norm(pos - task_pos)
            
            # Bid = distance + penalty for existing tasks
            current_tasks = 1 if state.get('task') else 0
            bid = distance + current_tasks * 2.0
            
            if bid < best_bid:
                best_bid = bid
                best_agent = agent_id
        
        # Assign task
        self.agent_states[best_agent]['task'] = task
        
        return best_agent
    
    def _greedy_allocation(self, task: Dict) -> int:
        """Greedy nearest-agent allocation."""
        task_pos = np.array(task.get('position', [0, 0, 0]))
        
        nearest_agent = self.my_agent_id
        min_distance = float('inf')
        
        for agent_id, pos in self.agent_positions.items():
            if not self.agent_states.get(agent_id, {}).get('active', False):
                continue
            
            distance = np.linalg.norm(pos - task_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_agent = agent_id
        
        self.agent_states[nearest_agent]['task'] = task
        return nearest_agent
    
    def _learned_allocation(self, task: Dict) -> int:
        """Use MARL policy for task allocation."""
        # Encode task as observation
        task_obs = np.zeros(32)
        if 'position' in task:
            task_obs[:3] = task['position']
        task_obs[3] = task.get('priority', 1)
        
        # Get allocation from coordinator
        allocation = self.coordinator.allocate_task(task_obs)
        
        assigned_agent = int(np.argmax(allocation))
        self.agent_states[assigned_agent]['task'] = task
        
        return assigned_agent
    
    def _get_fleet_status(self) -> Dict[str, Any]:
        """Get overall fleet status."""
        active_count = sum(
            1 for s in self.agent_states.values() 
            if s.get('active', False)
        )
        
        return {
            'total_agents': self.num_agents,
            'active_agents': active_count,
            'my_agent_id': self.my_agent_id,
            'formation': 'diamond',
            'agents': {
                agent_id: {
                    'active': state.get('active', False),
                    'status': state.get('status', 'unknown'),
                    'task': state.get('task') is not None
                }
                for agent_id, state in self.agent_states.items()
            }
        }
    
    def shutdown(self):
        """Clean shutdown."""
        if self.coordinator:
            self.coordinator.shutdown()
        if self.comm:
            self.comm.shutdown()
        
        self.is_initialized = False
        logger.info("Fleet Manager shutdown")
