import numpy as np

class Environment:

    X = 'X'
    O = 'O'
    _ = ''

    LENGTH = 3
    games = 1
    winner = _    

    moves_matrix_array = {
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

    moves_array_matrix = {
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

    def available_moves(self):
        moves = []
        for i in range(0, self.LENGTH):
            for j in range(0, self.LENGTH):  
                if self.board[i, j] == self._:
                    moves.append(self.moves_matrix_array[(i, j)])
        return moves

    def apply_move(self, move, player):
        move_matrix = self.moves_array_matrix[move]        
        self.board[move_matrix[0], move_matrix[1]] = player.action