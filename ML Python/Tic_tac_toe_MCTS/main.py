from tic_tac_toe import TicTacToe
from player import Player
from random_player import RandomPlayer
from human_player import HumanPlayer
import copy

env = TicTacToe()

p1 = Player(1)  #  1 - X
# p2 = Player(-1) # -1 - O
p2 = RandomPlayer(-1) # O

current_player = p1
games = 100
for i in range(games):        
    env.reset(current_player.mark)
    
    # env.print_board(i)

    while True:                
        # copy env to simulate the mcts
        env_copy = copy.deepcopy(env)        
        a = current_player.step(env_copy)

        env.step( current_player.mark, a )

        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
        
        if env.done:
            env.print_board(i)
            break        
    
        # env.print_board(i)

    if current_player == p1:
        current_player = p2
    else:
        current_player = p1

p2 = HumanPlayer(-1)
games = 10
for i in range(games):
    env.reset()
    current_player = p1
    env.print_board(i)

    while True:
        a = current_player.act(env)

        r, done = env.step( current_player.mark, a )

        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
        
        if done:
            env.print_board(i)
            break
    
        env.print_board(i)