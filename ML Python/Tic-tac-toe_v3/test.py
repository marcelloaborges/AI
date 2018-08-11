import numpy as np
from environment import Environment

X = 'X'
O = 'O'
_ = ''

#TESTS FOR WIN
print()
print('------')
print('CHECK WIN')
print('------')

### WIN LINE
env_win_x = Environment()
env_win_x.board = np.array([
        [_, _, _],
        [X, X, X],
        [_, _, _]
        ])    

result = env_win_x.check_win(X)

if result and env_win_x.winner == X:
    print('WIN X LINE => OK')
else:
    print('WIN X LINE => ERROR')


### WIN COLUMN
env_win_x = Environment()
env_win_x.board = np.array([
        [X, _, _],
        [X, _, _],
        [X, _, _]
        ])    

result = env_win_x.check_win(X)

if result and env_win_x.winner == X:
    print('WIN X COLUMN => OK')
else:
    print('WIN X COLUMN => ERROR')


### WIN MAIN DIAGONAL
env_win_x = Environment()
env_win_x.board = np.array([
        [X, _, _],
        [_, X, _],
        [_, _, X]
        ])    

result = env_win_x.check_win(X)

if result and env_win_x.winner == X:
    print('WIN X MAIN DIAGONAL => OK')
else:
    print('WIN X MAIN DIAGONAL => ERROR')


### WIN SECOND DIAGONAL
env_win_x = Environment()
env_win_x.board = np.array([
        [_, _, X],
        [_, X, _],
        [X, _, _]
        ])    

result = env_win_x.check_win(X)

if result and env_win_x.winner == X:
    print('WIN X SECOND DIAGONAL => OK')
else:
    print('WIN X SECOND DIAGONAL => ERROR')


#TESTS FOR DRAW
print()
print('------')
print('CHECK DRAW')
print('------')

#DRAW
env_draw = Environment()
env_draw.board = np.array([
        [X, O, X],
        [X, O, O],
        [O, X, O]
        ])    

result = env_draw.check_draw()

if result and env_draw.winner == _:
    print('DRAW => OK')
else:
    print('DRAW => ERROR')


#TESTS FOR END GAME
print()
print('------')
print('CHECK END GAME')
print('------')

#WIN NOT FULL
env_end_game = Environment()
env_end_game.board = np.array([
        [X, O, _],
        [X, _, O],
        [X, _, _]
        ])    

result = env_end_game.check_end_game()

if result and env_end_game.winner == X:
    print('END GAME => OK')
else:
    print('END GAME => ERROR')