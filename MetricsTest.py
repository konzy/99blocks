import numpy as np

from Metrics.Metrics import get_board_holes_rating_v2, surface_area

if __name__ == '__main__':
    board = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 2, 1, 0, 1, 0, 0, 0, 0],
                        [1, 0, 3, 1, 1, 0, 0, 0, 0],
                        [1, 0, 3, 0, 0, 1, 0, 0, 0],
                        [3, 2, 6, 2, 1, 0, 0, 0, 0],
                        [1, 1, 0, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 4, 0, 0, 0, 0, 0],
                        [5, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 0, 0, 0, 0, 0]])

    print(board)
    get_board_holes_rating_v2(board)
    print(surface_area(board))
