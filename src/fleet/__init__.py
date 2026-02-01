"""
AegisAI Fleet Management Module
Multi-Agent Reinforcement Learning (MARL) for fleet coordination
"""

from .manager import FleetManager
from .marl_coordinator import MARLCoordinator
from .communication import FleetCommunication

__all__ = ['FleetManager', 'MARLCoordinator', 'FleetCommunication']
