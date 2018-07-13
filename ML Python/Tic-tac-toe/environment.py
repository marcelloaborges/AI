import numpy as np

class Environment:

    LENGTH = 3
    games = 1
    winner = ''

    board = np.array([
        ['', '', ''],
        ['', '', ''],
        ['', '', '']
        ])    

    def __init__(self, length = 3):
        self.LENGTH = length        

    def print_board(self):
        print('-------------------------')           
        print('GAME: ', self.games)                  
        print('-----')
        for i in range(0, self.LENGTH):
            for j in range(0, self.LENGTH):
                print(' ' if self.board[i, j] == '' else self.board[i, j], end="", flush=True)                
                print(' ', end="", flush=True)
            print()
        print('-----')                 
        print('WINNER:', self.winner if self.winner != '' else 'NONE')
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
        
        #COLUMNS        
        for i in range(0, self.LENGTH):
            hits = 0
            for j in range(0, self.LENGTH):
                if self.board[j, i] == player:
                    hits += 1
            if hits == self.LENGTH:
                win = True
        
        #MAIN DIAGONAL       
        hits = 0 
        for i in range(0, self.LENGTH):            
            if self.board[i, i] == player:
                hits += 1        
        if hits == self.LENGTH:
            win = True

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

    def check_draw(self):
        #DRAW        
        for i in range(0, self.LENGTH):
            for j in range(0, self.LENGTH):
                if self.board[i, j] == '':
                    return False

        return True

    def check_end_game(self):
        if self.winner != '':
            return True
        if self.check_draw():
            return True
        return False

    def clear_board(self):
        self.board = np.array([
        ['', '', ''],
        ['', '', ''],
        ['', '', '']
        ])
        self.games += 1
        self.winner = ''

    def available_moves(self):
        moves = []
        for i in range(0, self.LENGTH):
            for j in range(0, self.LENGTH):  
                if self.board[i, j] == '':
                    moves.append((i, j))
        return moves

    def apply_move(self, move, player):        
        self.board[move[0], move[1]] = player

    def reset_move(self, move):
        self.board[move[0], move[1]] = ''

    def calculate_board_state_value(self, player):
        if self.check_end_game():
            if self.winner == player:
                return 1
            else:
                return 0
        else:
            return 0.5

    def get_board_state_key(self):
        k = 0
        h = 0
        v = 0
        for i in range(self.LENGTH):
            for j in range(self.LENGTH):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == 'X':
                    v = 1
                elif self.board[i, j] == 'O':
                    v = 2
                h += (3**k) * v
                k += 1
        return h