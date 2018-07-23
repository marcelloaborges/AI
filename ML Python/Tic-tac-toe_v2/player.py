class Player:

    action = None

    moves = {
        1 : 0.5,
        2 : 0.5,
        3 : 0.5,
        4 : 0.5,
        5 : 0.5,
        6 : 0.5,
        7 : 0.5,
        8 : 0.5,
        9 : 0.5
    }

    moves_history = []

    def __init__(self, action):
        self.action = action

    def print_moves(self):
        print('PLAYER: ', self.action)
        for key in self.moves:
            print(key, self.moves[key])
        print()

    def play(self, environment):
        available_moves = environment.available_moves()

        best_move = None
        best_move_value = 0
        
        for move_key in available_moves:
            move_value = self.moves[move_key]

            if move_value > best_move_value:
                best_move = move_key
                best_move_value = move_value
        
        environment.apply_move(best_move, self)
        self.moves_history.append(best_move)

    def update_moves_value(self, environment):
        target = 0
        if environment.winner == self.action:
            target = 1

        for move_key in reversed(self.moves_history):
            value = self.moves[move_key] + 0.5 * (target - self.moves[move_key])
            self.moves[move_key] = value
            target = value

        self.moves_history = []
