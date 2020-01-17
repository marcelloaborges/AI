import math
import numpy as np
import copy
import random

class MonteCarloTreeSearch:
    
    def __init__(self, action_size, rollout_steps, simulation_steps):
        self.action_size = action_size
        self.rollout_steps = rollout_steps
        self.simulation_steps = simulation_steps
        
        self.states = {} # every env becomes a "root"
    
    def act(self, env):
        key = str(env.board)

        if key not in self.states:   
            current_node = MCTSNode( None, env )

            available_actions = [action for action in current_node.children if current_node.children[action] == None]     

            a = random.choice( available_actions )

            return a


        current_node = self.states[key]

        selected_child = None
        selected_action = 0

        for action, child in current_node.children.items():
            if selected_child == None:
                selected_child = child
                selected_action = action
                continue

            if child.V / child.N > selected_child.V / selected_child.N:
                selected_child = child
                selected_action = action

        return selected_action
    
    def step(self, env):
        key = str(env.board)

        if key not in self.states:
            self.states[key] = MCTSNode( None, env)

        current_node = self.states[key]
        
        # learn
        for _ in range(self.rollout_steps):
            selected_child = self._select( current_node )            
            
            # everytime the winner is different from myself, I've lost, so the reward must be inverted
            winner = self._simulate( copy.deepcopy( selected_child ) )
            v = 1 if winner != selected_child.env.current_player else -1

            self._update( selected_child, v )

        # select
        return self.act(env)        

    def _select(self, current_node):
        if current_node.is_terminal():
            return current_node

        # expansion
        if current_node.is_leaf_node():            
            if current_node.N == 0:
                return current_node

            else:
                for action, child in current_node.children.items():

                    env_copy = copy.deepcopy( current_node.env )
                    env_copy.step( action )

                    child = MCTSNode( current_node, env_copy )

                    current_node.children[action] = child

                for action, child in current_node.children.items():
                    return child                
        
        else:
            for action, child in current_node.children.items():
                if child.N == 0:
                    return child

            # get the children with the biggest score
            selected_child = self._UCB_selection( current_node )
            return self._select( selected_child )

    def _simulate(self, current_node):
        while True:                            
            if current_node.is_terminal():
                return current_node.env.winner

            available_actions = [action for action in current_node.children if current_node.children[action] == None]

            random_a = random.choice( available_actions )

            current_node.env.step( random_a )

            return self._simulate( current_node )
            
    def _update(self, current_node, v):
        current_node.N += 1
        current_node.V += v

        if current_node.parent:
            self._update(current_node.parent, v * -1)

    def _UCB_selection(self, current_node):        
        N = self._get_root(current_node).N

        best_score = float('-inf')        
        selected_child = None
        for action, child in current_node.children.items():
            score = child.V / child.N + 1 * math.sqrt( math.log( N ) / child.N )

            if score > best_score:
                best_score = score
                selected_child = child                
        
        return selected_child

    def _get_root(self, current_node):
        if not current_node.parent:
            return current_node

        return self._get_root(current_node.parent)

class MCTSNode:

    def __init__(self, parent, env):
        self.N = 0
        self.V = 0
        self.terminal = env.done
                
        self.parent = parent
        self.env = env        

        # action => MTCS child (env)
        self.children = {}

        for i, position in enumerate(self.env.board[:-1]):
            if position == 0:
                self.children[i] = None

    def is_leaf_node(self):
        for action, child in self.children.items():
            if child != None:
                return False

        return True

    def is_terminal(self):
        return self.env.done