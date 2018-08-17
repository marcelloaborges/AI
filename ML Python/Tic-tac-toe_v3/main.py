from environment import Environment
from player import Player

env = Environment()

p1 = Player('X')
p2 = Player('O')

games = 1000
cp = p1

for i in range(games):        
    s = env.reset()

    while True:                
        actions = cp.play(s)
        a, s_, r, done = env.step(cp, actions)
        cp.observe(i, s, a, r, s_, done)

        if cp == p1:
            cp = p2
        else:
            cp = p1

        if done:
            cp.observe(i, s, a, r, s_, done)
            break

        s = s_         
    
    # env.print_board()

    p1.learn()    
    p2.learn()
    
# GAME TEST

cp = p1
s = env.reset()
while True:                
    actions = cp.play(s)        
    a, s_, r, done = env.step(cp, actions)    
    print(actions[0][0], a + 1)
    env.print_board()            

    if cp == p1:
        cp = p2
    else:
        cp = p1

    if done:        
        break

    s = s_

cp = p2
s = env.reset()
while True:                
    actions = cp.play(s)        
    a, s_, r, done = env.step(cp, actions)    
    print(actions[0][0], a + 1)
    env.print_board()

    if cp == p1:
        cp = p2
    else:
        cp = p1

    if done:        
        break

    s = s_