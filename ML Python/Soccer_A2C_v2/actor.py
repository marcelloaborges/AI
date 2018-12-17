import numpy as np
import torch

from memory import Memory

class Actor:

    def __init__(self,
        device,
        key,
        model, 
        shared_memory,
        n_step,
        gamma):

        self.DEVICE = device
        self.KEY = key                

        # Neural model
        self.model = model        

        # Memory
        self.memory = Memory()
        self.shared_memory = shared_memory

        # HYPERPARAMETERS
        self.N_STEP = n_step
        self.GAMMA = gamma

        self.t_step = 0

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)
                
        self.model.eval()
        with torch.no_grad():
            action, log_prob, _ = self.model(state)            
        self.model.train()

        action = action.cpu().detach().numpy().squeeze(1)
        log_prob = log_prob.cpu().detach().numpy().squeeze(1)

        return action, log_prob

    def step(self, state, teammate_state, action, action_prob, reward):        
        # Save experience / reward
        self.memory.add(state, teammate_state, action, action_prob, reward)

        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step == 0:

            states, teammate_states, actions, actions_probs, rewards = self.memory.experiences()

            # Calc the discounted rewards
            discounts = self.GAMMA ** np.arange( len( rewards ) )
            rewards = np.asarray( rewards ) * discounts[:,np.newaxis]
            future_rewards = rewards[::-1].cumsum(axis=0)[::-1]  

            # copy the local experiences to the shared memory with the future reward adjustment
            for i in range( len(states) ):
                self.shared_memory.add(states[i], teammate_states[i], actions[i], actions_probs[i], future_rewards[i])    

