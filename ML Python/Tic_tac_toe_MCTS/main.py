from tic_tac_toe import TicTacToe
from player import Player
import copy

env = TicTacToe()
current_player = None
p1 = Player(1)  # X
p2 = Player(-1) # O

games = 1
current_player = p1

for i in range(games):        
    s = env.reset()    
    env.print_board()    

    while True:                
        # copy env to simulate the mcts
        env_copy = copy.copy(env)
        a = current_player.step(env_copy, s)

        r, s_, done = env.step( a )

        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
        
        if done:
            break

        s = s_
    
        env.print_board()


    if current_player == p1:
        current_player = p2
    else:
        current_player = p1