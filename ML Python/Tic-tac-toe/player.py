import math
from state import State

class Player:

    def __init__(self, action):
        self.action = action    

    #key | (picks, wins)
    states = {}    

    def print_states(self):
        for key in self.states:
            print(self.states[key].picks, self.states[key].wins)

    def play_UCB(self, environment):                
        available_moves = environment.available_moves()

        best_move = None
        best_move_key = None
        best_move_reward = 0

        for move in available_moves:            
            environment.apply_move(move, self.action)
            key = environment.get_board_state_key()            
            environment.reset_move(move)

            if not key in self.states:
                self.states[key] = State(1, 0)
                        
            # WINS / PICKS
            avarage_move_reward = self.states[key].wins / self.states[key].picks                 
            # SQRT( 1.5 * LOG(GAMES) / PICKS)
            deltaI = math.sqrt(1.5 * math.log(environment.games + 1) / self.states[key].picks)                        
            move_reward = avarage_move_reward + deltaI            

            if move_reward > best_move_reward:                                
                best_move = move
                best_move_key = key
                best_move_reward = move_reward
                
        environment.apply_move(best_move, self.action)

        self.states[best_move_key].picks += 1
        if environment.check_win(self.action):
            self.states[best_move_key].wins += 1        

    def play(self, environment):
        self.play_UCB(environment)         
