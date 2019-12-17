import math
import numpy as np
import copy
import random

class MonteCarloTreeSearch:
    
    def __init__(self, action_size, rollout_steps = 5, simulation_steps = 5):
        self.action_size = action_size
        self.rollout_steps = rollout_steps
        self.simulation_steps = simulation_steps

        self.root = None        
    
    def _choose(self, mcts_s):
        best_action = None
        score = 0

        for child in mcts_s.children:
            if child.V > score:
                best_action = child.parent_action
                score = child.V

        return best_action
    
    def step(self, env, s):

        mcts_node = MCTSNode( None, s, None )
        
        if not self.root:
            self.root = mcts_node

        current_node = self._find_current_state( self.root, mcts_node )

        self._search_tree(env, current_node)

        return self._choose( s )

    def _search_tree(self, env, current_node):
        if current_node.is_leaf_node():

            value = 0
            if current_node.N == 0:
                value = self._simulate( env, current_node, False )

            else:
                for action in current_node.available_actions:
                    env_copy = copy.copy( env )
                    _, s_, _ = env_copy.step( action )
                    new_node = MCTSNode( current_node, s_, action )
                    current_node.children.append( new_node )
                
                current_node = current_node.children[0]

                value = self._simulate( env, current_node, False )

            self._backpropagate( current_node, value )
        else:
            current_node = self.current_none.UCB1_selection()

            self._search_tree( current_node )
        
    def _simulate(self, env, current_node, terminal):
        while True:
            if terminal:
                return current_node.V

            random_a = random.choice( current_node.available_actions )

            r, s_, done = env.step( random_a )

            temp_current_node = MCTSNode( current_node, s_, random_a )

            temp_current_node.V = r

            return self._simulate( env, temp_current_node, done )

    def _backpropagate(self, current_node, value):
        current_node.N += 1
        current_node.V += value

        if current_node.parent:
            self._backpropagate( current_node.parent, value )

    def _find_current_state(self, parent, current_node):
        if np.array_equal( current_node.state, parent.state ):
            return current_node
        else:            
            for child in current_node.children:
                return self._find_current_state( current_node, child )    

class MCTSNode:

    def __init__(self, parent, state, parent_action):
        self.N = 0
        self.V = 0
                
        self.parent = parent

        self.state = state
        self.parent_action = parent_action        

        self.available_actions = []
        self.children = []

        for i, position in enumerate(self.state):
            if position == 0:
                self.available_actions.append( i )

    def is_leaf_node(self):
        return not self.children

    def UCB1_selection(self):
        selected_node = None
        temp_score = 0
        for child in self.children:
            ucb_score = child.V + 2 * math.sqrt( math.log( self.N ) / child.N  )

            if ucb_score > temp_score:
                temp_score = ucb_score
                selected_node = child

        return selected_node




# def _uct_selection(self):
#     uct = 0
#     selected_action = None

#     for action in self.actions:
#         log_N_vertex = math.log(self.state.N)
        
#         # Upper confidence bound for trees
#         uct_action = action.V / action.N + 0.1 * math.sqrt( log_N_vertex / action.N )

#         if uct_action > uct:
#             uct = uct_action
#             selected_action = action
    
#     return selected_action
