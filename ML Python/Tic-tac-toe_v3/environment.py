import numpy as np

class Environment:

    X = 'X'
    O = 'O'
    _ = ''

    LENGTH = 3
    games = 1
    winner = _    

    actions_matrix_array = {
        (0,0) : 1,
        (0,1) : 2,
        (0,2) : 3,
        (1,0) : 4,
        (1,1) : 5,
        (1,2) : 6,
        (2,0) : 7,
        (2,1) : 8,
        (2,2) : 9,
        }

    actions_array_matrix = {
        1 : (0,0),
        2 : (0,1),
        3 : (0,2),
        4 : (1,0),
        5 : (1,1),
        6 : (1,2),
        7 : (2,0),
        8 : (2,1),
        9 : (2,2),
        }

    board = np.array([
        [_, _, _],
        [_, _, _],
        [_, _, _]
        ])    

    def __init__(self, length = 3):
        self.LENGTH = length        

    def print_board(self):
        print('-------------------------')           
        print('GAME: ', self.games)                  
        print('-----')
        for i in range(0, self.LENGTH):
            for j in range(0, self.LENGTH):
                print(' ' if self.board[i, j] == self._ else self.board[i, j], end="", flush=True)                
                print(' ', end="", flush=True)
            print()
        print('-----')                 
        print('WINNER:', self.winner if self.winner != self._ else 'NONE')
        print('-------------------------')  
        print()       

    def check_win(self, player):
        win = False

        #LINES                        
        for i in range(0, self.LENGTH):            
            hits = 0
            for j in range(0, self.LENGTH):                      
                if self.board[i, j] == player:                    
                    hits += 1            
            if hits == self.LENGTH:
                win = True

        if win:
            self.winner = player 
            return win
        
        #COLUMNS        
        for i in range(0, self.LENGTH):
            hits = 0
            for j in range(0, self.LENGTH):
                if self.board[j, i] == player:
                    hits += 1
            if hits == self.LENGTH:
                win = True
        
        if win:
            self.winner = player 
            return win

        #MAIN DIAGONAL       
        hits = 0 
        for i in range(0, self.LENGTH):            
            if self.board[i, i] == player:
                hits += 1        
        if hits == self.LENGTH:
            win = True

        if win:
            self.winner = player 
            return win

        #SECOND DIAGONAL        
        hits = 0
        for i, j in zip(range(0, self.LENGTH), range(self.LENGTH - 1, -1, -1)):                   
            if self.board[i, j] == player:
                hits += 1
        if hits == self.LENGTH:            
            win = True
        
        if win:
            self.winner = player 
            return win           

        return win

    def check_draw(self):
        #DRAW        
        for i in range(0, self.LENGTH):
            for j in range(0, self.LENGTH):
                if self.board[i, j] == self._:
                    return False

        return True

    def check_end_game(self):
        if self.check_win(self.X):
            return True
        if self.check_win(self.O):
            return True
        if self.check_draw():
            return True
        return False

    def clear_board(self):
        self.board = np.array([
        [self._, self._, self._],
        [self._, self._, self._],
        [self._, self._, self._]
        ])
        self.games += 1
        self.winner = self._

    def state(self):
        state = []
        for i in range(0, self.LENGTH):
            for j in range(0, self.LENGTH):  
                if self.board[i, j] == self._:
                    state.append(-1)
                elif self.board[i, j] == self.O:
                    state.append(0)
                elif self.board[i, j] == self.X:
                    state.append(1)
        return state

    def apply_action(self, player, action):        
        action_matrix = self.actions_array_matrix[action]        
        self.board[action_matrix[0], action_matrix[1]] = player.action

        if self.check_end_game():
            if self.winner == player:
                return 1
            elif self.check_draw():
                return 0.5
            else:
                return 0
        else:
            return 0.5