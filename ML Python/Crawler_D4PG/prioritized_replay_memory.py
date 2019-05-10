import numpy as np
from collections import deque
import random

class PrioritizedReplayMemory():
    def __init__(self, keys, buffer_size, batch_size):
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.KEYS = keys

        self.memory = {}
        self.priorities = {}

        for key in self.KEYS:
            self.memory[key] = deque(maxlen=buffer_size)
            self.priorities[key] = deque(maxlen=buffer_size)
        
    def add(self, key, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = {
            "state" : state,
            "action" : action,
            "reward" : reward,
            "next_state" : next_state,
            "done" : done,
        }

        self.memory[key].append(e)
        self.priorities[key].append(max(self.priorities, default=1))
        
    def _get_probabilities(self, key, priority_scale):
        scaled_priorities = np.array(self.priorities[key]) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)

        return sample_probabilities
    
    def _get_importance(self, key, probabilities):
        importance = 1/len(self.memory[key]) * 1/probabilities
        importance_normalized = importance / max(importance)

        return importance_normalized
        
    def enougth_samples(self, key):
        return len( self.memory[key] ) >= self.BATCH_SIZE

    def sample(self, key, priority_scale=1.0):
        # sample_size = min(len(self.memory), self.BATCH_SIZE)
        sample_probs = self._get_probabilities(key , priority_scale)
        sample_indices = random.choices( range( len(self.memory[key]) ), k=self.BATCH_SIZE, weights=sample_probs)
        samples = np.array( self.memory[key] )[sample_indices]
        importance = self._get_importance( key , sample_probs[sample_indices] )

        return samples, importance, sample_indices
    
    def set_priorities(self, key, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[key][i] = abs(e) + offset