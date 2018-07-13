from environment import Environment
from player import Player

environment = Environment()

p1 = Player('X')
p2 = Player('O')

current_player = p1

games = 10000
current_player.play(environment)

for i in range(0, games):
    while not environment.check_end_game():        
        current_player.play(environment)

        if current_player == p1:
            current_player = p2            
        else:
            current_player = p1                        
    
    environment.print_board()
    environment.clear_board()
