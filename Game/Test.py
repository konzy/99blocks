from Game import Game
from Metrics import Metrics
from UI import UI


def step_one():
    game.play_move(0, 0, 0)
    game.rotate_shape(0)
    game.play_move(0, 0, 0)
    game.play_move(2, 2, 0)
    game.rotate_shape(0)


def step_two():
    game.play_move(2, 0, 3)
    game.play_move(2, 2, 3)
    game.play_move(1, 2, 4)


def step_three():
    game.play_move(2, 0, 6)
    game.play_move(2, 5, 2)


def step_four():
    game.rotate_shape(0)


def complete_step_one():
    game.play_move(0, 1, 2)


def complete_step_two():
    game.play_move(0, 1, 2)


def complete_step_three():
    game.play_move(1, 4, 4)


def complete_step_four():
    game.play_move(0, 0, 6)


def simulation():
    step_one()
    step_two()
    step_three()
    complete_step_three()
    complete_step_two()
    step_four()
    complete_step_four()


def single_hole():
    step_one()
    game.rotate_shape(0)
    game.play_move(0, 0, 3)


def double_hole():
    step_one()
    game.play_move(0, 0, 3)
    game.rotate_shape(1)
    game.play_move(1, 1, 3)


if __name__ == '__main__':
    game = Game()
    ui = UI(game)
    double_hole()

    Metrics.get_board_holes_rating_v2(game.board)

    ui.display_game_board()
    ui.display_shapes()
    ui.display_points()
