from tic_tac_toe import TicTacToe
from player import Player
from random_player import RandomPlayer
from human_player import HumanPlayer
import random

# HYPERPARAMETERS
action_size = 9
rollout_steps = 100
simulation_steps = 5
checkpoint='checkpoint.mcts'

# PLAYERS
p1 = Player(1, action_size, rollout_steps, simulation_steps, checkpoint)  

c_player = Player(-1, action_size, rollout_steps, simulation_steps, checkpoint)
r_player = RandomPlayer(-1) 

# GAMING
env = TicTacToe()

def train():    
    games = 100

    initial_player = p1
    p2 = c_player
    for i in range(games):
        current_player = initial_player
        env.reset(current_player.mark)
        
        # env.print_board(i)

        while True:                
            # copy env to simulate the mcts        
            a = current_player.step(env)

            env.step( a )

            if current_player.mark == p1.mark:
                current_player = p2
            else:
                current_player = p1
            
            if env.done:
                env.print_board(i)
                break        
        
            # env.print_board(i)

        p1.checkpoint()

        p = random.choice([0,1])
        if p == 0:
            p2 = r_player
        else:
            p2 = c_player

        if initial_player.mark == p1.mark:
            initial_player = p2
        else:
            initial_player = p1        


def play():
    games = 10

    initial_player = p1
    p2 = HumanPlayer(-1)
    for i in range(games):
        current_player = initial_player
        env.reset(current_player.mark)    
        env.print_board(i, True)

        while True:
            a = current_player.act(env)

            env.step( a )

            if current_player.mark == p1.mark:
                current_player = p2
            else:
                current_player = p1
            
            if env.done:
                env.print_board(i, True)
                break
        
            env.print_board(i, True)

        if initial_player.mark == p1.mark:
            initial_player = p2
        else:
            initial_player = p1 

# train()
play()