from Game import Game
from UI import UI

if __name__ == '__main__':
    print("hello")
    game = Game()
    ui = UI(game)

    for i in range(0, 20):
        ui.display_game_board()
        ui.display_shapes()
        response = input('"rot n" or "place id y x": ')
        if 'rot' in response:
            _, shape_idx = response.split(' ')
            shape_idx = int(shape_idx)
            game.rotate_shape(shape_idx)

        elif 'place' in response:
            _, shape_idx, place_y, place_x = response.split(' ')
            shape_idx = int(shape_idx)
            place_y = int(place_y)
            place_x = int(place_x)
            if not game.play_move(shape_idx, place_x, place_y):
                print('could not place shape there')
