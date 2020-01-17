from monte_carlo_tree_search import MCTSNode
import random

class RandomPlayer:

    def __init__(self, mark):
        self.mark = mark

    def act(self, env):
        current_node = MCTSNode( None, env)
        available_actions = [action for action in current_node.children if current_node.children[action] == None]
        
        action = random.choice( available_actions )
        
        return action

    def step(self, env):
        return self.act(env)
        
