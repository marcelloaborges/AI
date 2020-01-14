from monte_carlo_tree_search import MCTSNode
import random

class RandomPlayer:    

    def __init__(self, mark):
        self.mark = mark

    def step(self, env):
        action = random.choice( MCTSNode( None, env, None ).get_available_actions() )
                    
        return action
