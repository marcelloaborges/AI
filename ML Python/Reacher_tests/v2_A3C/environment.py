import numpy as np

class Environment:

    def __init__(self, environment, show_info=True, no_grafics=True):
        # Number of actions: 4
        # States have length: 33
        
        self.env = environment        
       
        self.BRAIN_NAME = self.env.brain_names[0]
        self.action_size = self.env.brains[self.BRAIN_NAME].vector_action_space_size

        if show_info:
            self.info()        

    def info(self):        
        # get the default brain            
        brain = self.env.brains[self.BRAIN_NAME]

        # reset the environment
        env_info = self.env.reset(train_mode=True)[self.BRAIN_NAME]
        # number of agents in the environment
        print('Number of agents:', len(env_info.agents))
        # number of actions
        action_size = brain.vector_action_space_size
        print('Number of actions:', action_size)
        # examine the state space 
        state = env_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)     

        return state_size , action_size

    def reset(self):
        env_info = self.env.reset(train_mode=False)[self.BRAIN_NAME]
        state = env_info.vector_observations
        
        return state

    def step(self, agent_id, action):       
        # FORMAT THE ACTION TO THE ENV's INPUT
        # formated_action = self._format_action(agent_id, action)         

        env_info = self.env.step( action )[self.BRAIN_NAME]     # send the action to the environment        
        next_state = env_info.vector_observations               # get the next state        
        reward = env_info.rewards                               # get the reward        
        done = env_info.local_done                              # see if episode has finished           

        return next_state, reward, done

    def _format_action(self, agent_id, action):
        # FIXED AGENTS NUMBER BECAUSE THIS ENVIRONMENT ALLOWS JUST ONE INSTANCE RUNNING AT TIME
        n_agents = 20
        actions = np.random.randn(n_agents, self.action_size)
        actions = np.clip(actions, -1, 1)

        actions[agent_id] = action

        return actions