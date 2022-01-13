from copy import deepcopy

import numpy as np

from scipy import signal


def get_surrounding_squares(board: np.ndarray, x, y):
    squares = []
    if x > 0:
        squares.append((x - 1, y))
    if x < board.shape[1]:
        squares.append((x + 1, y))
    if y > 0:
        squares.append((x, y - 1))
    if x < board.shape[0]:
        squares.append((x, y + 1))
    return squares


def detect(board, conv, value):
    detector = signal.convolve2d(board, conv, mode='same', boundary='fill', fillvalue=1)
    unique, counts = np.unique(detector, return_counts=True)
    cnt = dict(zip(unique, counts)).get(value)
    if cnt is None:
        return 0
    return cnt


def rot_detect(board, conv, value, rot_num):
    cnt = 0
    for i in range(0, rot_num):
        cnt += detect(board, conv, value)
        conv = np.rot90(conv)
    return cnt


def surface_area(board: np.ndarray):
    board = board > 0
    board_mask = np.invert(deepcopy(board))

    conv = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]])

    detector = signal.convolve2d(board, conv, mode='same', boundary='fill', fillvalue=1)
    res = detector * board_mask
    return np.sum(res)


def get_board_holes_rating_v2(board: np.ndarray):
    board = board > 0
    single_hole = np.array([[0, 1, 0],
                            [1, 5, 1],
                            [0, 1, 0]])

    single_count = detect(board, single_hole, 4)

    vertical_double_hole = np.array([[0, 1, 0, 0],
                                     [1, 7, 1, 0],
                                     [1, 7, 1, 0],
                                     [0, 1, 0, 0]])

    double_count = detect(board, vertical_double_hole, 6)

    horizontal_double_hole = np.rot90(vertical_double_hole)

    double_count += detect(board, horizontal_double_hole, 6)

    triple_angle_hole = np.array([[0, 1, 0, 0],
                                  [1, 8, 1, 0],
                                  [1, 8, 8, 1],
                                  [0, 1, 1, 0]])
    triple_count = rot_detect(board, triple_angle_hole, 7, 4)

    triple_straight_hole = np.array([[0, 1, 0],
                                     [1, 9, 1],
                                     [1, 9, 1],
                                     [1, 9, 1],
                                     [0, 1, 0]])

    triple_count += rot_detect(board, triple_straight_hole, 8, 2)

    total_count = single_count + double_count + triple_count

    return single_count, double_count, triple_count, total_count


def get_spaces_used(board: np.ndarray):
    return np.count_nonzero(board)


def get_board_holes_rating(board: np.ndarray):
    single_holes = 0
    double_holes = 0
    for y in range(0, board.shape[1]):
        for x in range(0, board.shape[0]):
            cnt = 0
            adjacent_gap = None
            surrounding_squares = get_surrounding_squares(board, x, y)
            for square in surrounding_squares:
                x, y = square
                if board[x][y] > 0:
                    cnt += 1
                else:
                    adjacent_gap = (x, y)
            if cnt == len(surrounding_squares):
                single_holes += 1
            if cnt == 1 - len(surrounding_squares) and adjacent_gap:
                x, y = adjacent_gap
                surrounding_squares = get_surrounding_squares(board, x, y)
                for square in surrounding_squares:
                    x, y = square
                    if board[x][y] > 0:
                        cnt += 1
                if cnt == 1 - len(surrounding_squares):
                    double_holes += 1
