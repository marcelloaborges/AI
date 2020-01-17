from monte_carlo_tree_search import MonteCarloTreeSearch
import numpy as np
import copy
import os
import pickle

class Player:

    def __init__(self, mark, action_size = 9, rollout_steps = 50, simulation_steps = 5, checkpoint='checkpoint.mcts'):
        self.mark = mark            

        self.mcts = MonteCarloTreeSearch( action_size, rollout_steps, simulation_steps )

        if os.path.isfile( checkpoint ):
            with open(checkpoint, 'rb') as config:
                mcts = pickle.load(config)                            
                self.mcts = mcts

    def act(self, env):
        action = self.mcts.act(env)
        
        return action

    def step(self, env):
        env_copy = copy.deepcopy(env)        

        action = self.mcts.step(env_copy)
        
        return action

    def checkpoint(self, checkpoint='checkpoint.mcts'):
        with open( checkpoint, 'wb' ) as config:
            pickle.dump( self.mcts, config )    