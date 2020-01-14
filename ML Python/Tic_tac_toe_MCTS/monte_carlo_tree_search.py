import math
import numpy as np
import copy
import random

class MonteCarloTreeSearch:
    
    def __init__(self, mark, action_size, rollout_steps, simulation_steps):
        self.mark = mark

        self.action_size = action_size
        self.rollout_steps = rollout_steps
        self.simulation_steps = simulation_steps
        
        self.envs = {} # every env becomes a "root"
    
    def act(self, env):
        key = hash(str(env))

        if key not in self.envs:        
            a = random.choice( MCTSNode( None, env ) )
            return a

        current_node = self.envs[key]

        selected_child = current_node.children[0]

        for action, child in current_node.children.items()[1:]:
            if child.V / child.N > selected_child.V / selected_child.N:
                selected_child = child

        return selected_child.parent_action
    
    def step(self, env):
        key = hash(str(env))

        if key not in self.envs:
            self.envs[key] = MCTSNode( None, env)

        current_node = self.envs[key]
        
        # learn
        for _ in range(self.rollout_steps):
            action = self._select( current_node )
            selected_child = current_node.children[action]
            
            # everytime the winner is different from myself, I've lost, so the reward must be inverted
            winner = self._simulate( copy.deepcopy( selected_child ), self.mark * -1 )
            v = 1 if winner == self.mark else -1

            self._update( selected_child, v )

        # select
        return self.act(env)        

    def _select(self, current_node):        
        # expansion
        unexpanded_children = [action for action in current_node.children if current_node.children[action] == None]

        # return an unexplored child if exists
        if unexpanded_children:
            action = random.choice( unexpanded_children )

            env_copy = copy.deepcopy( current_node.env )
            env_copy.step( self.mark, action )

            child = MCTSNode( current_node, env_copy )

            current_node.children[action] = child

            return action
                
        # get the children with the biggest score
        _, selected_child = self._UCB_selection( current_node )
        selected_child.N += 1

        if not selected_child.children:
            return selected_child

        return self._select(selected_child)

    def _simulate(self, current_node, mark):
        while True:                            
            if current_node.is_terminal():
                return current_node.env.winner

            available_actions = [action for action in current_node.children if current_node.children[action] == None]

            random_a = random.choice( available_actions )

            current_node.env.step( mark, random_a )

            return self._simulate( current_node, mark * -1 )
            
    def _update(self, current_node, v):
        current_node.V += v        

        if current_node.parent:
            self._update(current_node.parent, v * -1)

    def _UCB_selection(self, current_node):
        N = self._get_root(current_node).N

        best_score = float('-inf')
        selected_action = 0
        selected_child = None
        for action, child in current_node.children.items():
            score = child.V / child.N + 1 * math.sqrt( math.log( N ) / child.N )

            if score > best_score:
                best_score = score
                selected_action = action
                selected_child = child                
        
        return selected_action, selected_child

    def _get_root(self, current_node):
        if not current_node.parent:
            return current_node

        return self._get_root(current_node.parent)

class MCTSNode:

    def __init__(self, parent, env):
        self.N = 1
        self.V = 0
        self.terminal = env.done
                
        self.parent = parent
        self.env = env        

        # action => MTCS child (env)
        self.children = {}

        for i, position in enumerate(self.env.board[:-1]):
            if position == 0:
                self.children[i] = None

    def is_terminal(self):
        return self.env.done