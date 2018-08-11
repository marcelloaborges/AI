from environment import Environment
from player import Player

environment = Environment()

p1 = Player('X')
p2 = Player('O')

games = 1
current_player = p1

environment.print_board()
for i in range(games):        
    while not environment.check_end_game():    
        state = environment.state()
        action = current_player.play(state)                
        reward = environment.apply_action(current_player, action)             
        current_player.add_memory(state, action, reward, environment.state())

        if current_player == p1:
            current_player = p2            
        else:
            current_player = p1            
    
    environment.print_board()
    environment.clear_board()

    p1.learn()
    p2.learn()

# p1.print_moves()
# p2.print_moves()