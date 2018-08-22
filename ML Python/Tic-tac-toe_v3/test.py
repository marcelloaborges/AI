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


#TESTS FOR BLOCK WIN
print()
print('------')
print('CHECK BLOCK WIN')
print('------')

#WIN NOT FULL
env_rejected = Environment()
env_rejected.board = np.array([
        [O, _, O],
        [_, O, _],
        [X, O, _]
        ])    

result = env_rejected.block_win(7, X)

if result:
    print('BLOCK WIN => OK')
else:
    print('BLOCK WIN  => ERROR')