from Game import Game


class UI:
    def __init__(self, game: Game):
        self.game = game

    def display_game_board(self):
        for line in self.game.board:
            print(line)
        print('\n')

    def display_shapes(self):
        for shape in self.game.shapes:
            print(shape.shape)
        print('\n')

    def display_points(self):
        print(self.game.points)
