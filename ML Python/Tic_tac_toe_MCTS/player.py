from monte_carlo_tree_search import MonteCarloTreeSearch
import numpy as np

class Player:

    def __init__(self, mark, action_size = 9, rollout_steps = 50, simulation_steps = 5):

        self.mark = mark
        self.mcts = MonteCarloTreeSearch( mark, action_size, rollout_steps, simulation_steps )

    def act(self, env):
        action = self.mcts.act(env)
        
        return action

    def step(self, env):
        action = self.mcts.step(env)
        
        return action
