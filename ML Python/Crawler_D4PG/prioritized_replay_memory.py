import numpy as np
from collections import namedtuple

class PrioritizedReplayMemory(object):  
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.prob_alpha = alpha
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)

        max_prio = self.priorities.max() if self.buffer else 1.0**self.prob_alpha
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(e)
        else:
            self.buffer[self.pos] = e
        
        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self):
        if len(self.buffer) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        total = len(self.buffer)

        probs = prios / prios.sum()

        indices = np.random.choice(total, self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        states = np.vstack([e.state for e in samples if e is not None])
        actions = np.vstack([e.action for e in samples if e is not None])
        rewards = np.vstack([e.reward for e in samples if e is not None])
        next_states = np.vstack([e.next_state for e in samples if e is not None])
        dones = np.vstack([e.done for e in samples if e is not None])

        
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        #min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min*total)**(-beta)

        weights  = (total * probs[indices]) ** (-beta)
        weights /= max_weight
        # weights  = torch.tensor(weights, device=device, dtype=torch.float)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + 1e-5)**self.prob_alpha

    def enough_experiences(self):
        return len(self) >= self.batch_size

    def __len__(self):
        return len(self.buffer)
