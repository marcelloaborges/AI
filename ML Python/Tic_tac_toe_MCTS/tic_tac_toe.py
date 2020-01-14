import numpy as np

class TicTacToe:    

    def __init__(self, length = 3):
        self.LENGTH = length

        self.LENGTH = 3

        #  1: X 
        # -1: O
        #  0: None        
        self.winner = 0 
        self.done = False        
        
        self.board = []

    def print_board(self, game=0):
        print('-------------------------')
        print('GAME: {}'.format(game) )
        print('-----')              
        for i in range(0, self.LENGTH * self.LENGTH):           
            # print('X', end='', flush=True)
            print( '{}'.format(' ') if self.board[i] == 0 else 'X' if self.board[i] == 1 else 'O', end='', flush=True)            

            if (i + 1) % self.LENGTH == 0:
                print()
            else:            
                print('|', end='', flush=True)
        print('-----')
        print('WINNER: {}'.format( 'None' if self.winner == 0 else 'X' if self.winner == 1 else 'O' ) )
        print('-------------------------')  
        print()       

    def check_win(self, player):
        win = False

        #LINES                        
        for i in range(0, self.LENGTH):
            hits = 0
            for j in range(0, self.LENGTH):
                if self.board[i * self.LENGTH + j] == player:
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
                if self.board[ (j * self.LENGTH) + i ] == player:
                    hits += 1
            if hits == self.LENGTH:
                win = True
        
        if win:
            self.winner = player 
            return win

        #MAIN DIAGONAL       
        hits = 0 
        for i in range(0, self.LENGTH):            
            if self.board[i * self.LENGTH + i] == player:
                hits += 1        
        if hits == self.LENGTH:
            win = True

        if win:
            self.winner = player 
            return win

        #SECOND DIAGONAL        
        hits = 0        
        for i, j in zip(range(0, self.LENGTH), range(self.LENGTH - 1, -1, -1)):
            if self.board[i * self.LENGTH + j] == player:
                hits += 1
        if hits == self.LENGTH:
            win = True
        
        if win:
            self.winner = player 
            return win           

        return win

    def check_draw(self):
        #DRAW        
        for i in range(0, self.LENGTH * self.LENGTH):
            if self.board[i] == 0:
                return False

        return True

    def _update_done(self):
        if self.check_win(1): # X
            self.winner = 1
            self.done = True
        if self.check_win(-1): # O
            self.winner = -1
            self.done = True 
        if self.check_draw(): # 0
            self.done = True

    def reset(self, current_player_mark):
        self.winner = 0 
        self.done = False

        self.board = np.array([
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        current_player_mark
        ])

    def step(self, mark, position):
        # action
        self.board[position] = mark
        # current_player
        self.board[-1] = mark * -1

        self._update_done()