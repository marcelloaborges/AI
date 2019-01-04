import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import ActorModel, CriticModel
from memory import Memory

class Agent:

    def __init__(
        self, 
        device,
        key,
        agent_type,
        state_size,
        action_size,
        teammate_state_size,
        adversary_state_size, 
        adversary_teammate_state_size,
        lr,
        n_step,        
        gamma,
        epsilon,
        entropy_weight,
        gradient_clip,
        checkpoint_folder
        ):

        self.DEVICE = device
        self.KEY = key
        self.TYPE = agent_type

        self.CHECKPOINT_ACTOR_FILE = checkpoint_folder + 'checkpoint' + '_' + self.TYPE + '_' + str(self.KEY) + '.pth'
        self.CHECKPOINT_CRITIC_FILE = checkpoint_folder + 'checkpoint_critic_' + '_' + self.TYPE + '_' + str(self.KEY) + '.pth'

        # NEURAL MODEL
        self.actor_model = ActorModel( state_size, action_size ).to(self.DEVICE)
        self.critic_model = CriticModel( state_size + teammate_state_size + adversary_state_size + adversary_teammate_state_size ).to(self.DEVICE)
        self.optimizer = optim.Adam( list( self.actor_model.parameters() ) + list( self.critic_model.parameters() ), lr=lr )
        # self.optimizer = optim.RMSprop( list( self.actor_model.parameters() ) + list( self.critic_model.parameters() ), lr=lr, alpha=0.99, eps=1e-5 )

        self.actor_model.load(self.CHECKPOINT_ACTOR_FILE)
        self.critic_model.load(self.CHECKPOINT_CRITIC_FILE)

        # N_STEP MEMORY
        self.memory = Memory()        

        # HYPERPARAMETERS
        self.N_STEP = n_step        
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.ENTROPY_WEIGHT = entropy_weight
        self.GRADIENT_CLIP = gradient_clip        

        self.t_step = 0
        self.loss = 0

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)

        self.actor_model.eval()
        with torch.no_grad():                
            action, log_prob, _ = self.actor_model(state)                    
        self.actor_model.train()

        action = action.cpu().detach().numpy()
        log_prob = log_prob.cpu().detach().numpy()

        return action, log_prob

    def step(self, state, teammate_state, adversary_state, adversary_teammate_state, action, log_prob, reward):                
        self.memory.add( state, teammate_state, adversary_state, adversary_teammate_state, action, log_prob, reward )

        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step != 0:
            return self.loss
            
        # LEARN
        states, teammate_states, adversary_states, adversary_teammate_states, actions, log_probs, rewards, n_exp = self.memory.experiences()

        
        discount = self.GAMMA**np.arange(n_exp).reshape(-1, 1)
        rewards = rewards * discount
        rewards_future = rewards[::-1].cumsum(axis=1)[::-1]


        states = torch.from_numpy(states).float().to(self.DEVICE)
        teammate_states = torch.from_numpy(teammate_states).float().to(self.DEVICE)
        adversary_states = torch.from_numpy(adversary_states).float().to(self.DEVICE)
        adversary_teammate_states = torch.from_numpy(adversary_teammate_states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).long().to(self.DEVICE)
        log_probs = torch.from_numpy(log_probs).float().to(self.DEVICE)
        rewards = torch.from_numpy(rewards_future.copy()).float().to(self.DEVICE)


        values = self.critic_model( torch.cat( (states, teammate_states, adversary_states, adversary_teammate_states), dim=1 ) )
                        

        advantages = (rewards - values).detach()
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        advantages_normalized = torch.tensor(advantages_normalized).float().to(self.DEVICE)


        _, new_log_probs, entropies = self.actor_model(states, actions)

        ratio = ( new_log_probs - log_probs ).exp()
        clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )
        
        policy_loss = torch.min( ratio * advantages, clip * advantages )
        policy_loss = - torch.mean( policy_loss )

        entropy = torch.mean(entropies)
        
        values = self.critic_model( torch.cat( (states, teammate_states, adversary_states, adversary_teammate_states), dim=1 ) )
        value_loss = F.mse_loss( rewards, values )


        self.optimizer.zero_grad()

        loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT)        
        loss.backward()
        # nn.utils.clip_grad_norm_( self.model.parameters(), self.GRADIENT_CLIP )

        self.optimizer.step()


        self.loss = loss.cpu().detach().numpy()

        return self.loss

    def checkpoint(self):
        self.actor_model.checkpoint(self.CHECKPOINT_ACTOR_FILE)
        self.critic_model.checkpoint(self.CHECKPOINT_CRITIC_FILE)