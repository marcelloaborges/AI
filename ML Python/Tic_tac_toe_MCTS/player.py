from monte_carlo_tree_search import MonteCarloTreeSearch

class Player:

    def __init__(self, action_size = 9, rollout_size = 5, simulation_size = 5):

        self.mcts = MonteCarloTreeSearch( action_size, rollout_size, simulation_size)        

    def act(self, state):                
        action = self.mcts.choose(state)
        
        return action

    def step(self, env, s):
        
        a = self.mcts.step(env, s)
        
        return a
