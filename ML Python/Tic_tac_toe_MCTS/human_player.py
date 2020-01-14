from monte_carlo_tree_search import MCTSNode

class HumanPlayer:    

    def __init__(self, mark):
        self.mark = mark

    def act(self, env):
        print('What position do you want to play?')
        action = int(input())
        
        return action - 1
