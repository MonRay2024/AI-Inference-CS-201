import torch
import torch.nn as nn
from collections import namedtuple
import random

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])

class EditOptimizer(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.memory = []
        self.gamma = 0.99
        self.optimizer = torch.optim.Adam(self.parameters())
        
    def forward(self, x):
        return self.network(x)
    
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.network[-1].out_features - 1)
        
        with torch.no_grad():
            q_values = self(state)
            return torch.argmax(q_values).item()
    
    def optimize_edit(self, edit_sequence: List[str]) -> List[str]:
        """Optimize edit sequence using RL"""
        state = self._encode_edit_state(edit_sequence)
        optimized_sequence = []
        
        for edit in edit_sequence:
            action = self.select_action(state)
            optimized_edit = self._apply_optimization(edit, action)
            reward = self._calculate_reward(optimized_edit)
            
            next_state = self._encode_edit_state(optimized_sequence + [optimized_edit])
            self.memory.append(Experience(state, action, reward, next_state))
            
            state = next_state
            optimized_sequence.append(optimized_edit)
            
        self._train()
        return optimized_sequence 