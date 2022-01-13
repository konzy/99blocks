import logging
import random
from copy import deepcopy

import numpy as np


class Game:
    max_invalid_move_count = 3
    max_rotation_count = 4

    @staticmethod
    def shapes_on_blank_board(game):
        result = game.blank_board()
        result = game.add_shape_to_array_at(game.shapes[0], result, (0, 0))
        result = game.add_shape_to_array_at(game.shapes[1], result, (5, 5))
        result = game.add_shape_to_array_at(game.shapes[2], result, (5, 0))
        return result

    @staticmethod
    def add_shape_to_array_at(shape, array, loc=(0, 0)):
        y1, x1 = shape.shape.shape
        array[loc[0]:y1 + loc[0], loc[1]:x1 + loc[1]] = shape.shape
        return array

    @staticmethod
    def blank_board():
        x = np.empty(shape=(9, 9))
        x.fill(0)
        return x

    def __init__(self):
        log = logging.getLogger('99BlockGame')
        self.seed = 1234
        self.rand = random.Random(self.seed)
        self.multi = 0
        self.combo = 0
        self.points = 0
        self.invalid_move_count = 0
        self.rotation_count = 0

        self.board = self.blank_board()
        self.shapes = [Shape(self.rand, 1), Shape(self.rand, 1), Shape(self.rand, 1)]

    def end_score(self):
        return self.points + sum(self.board)

    def rotate_shape(self, idx):
        if self.end_game():
            return self.end_score()
        self.shapes[idx].turn(1)

    def _invalid_move(self):
        self.invalid_move_count += 1
        return -1

    def end_game(self):
        return self.invalid_move_count >= self.max_invalid_move_count or self.rotation_count > self.max_rotation_count

    def play_move(self, game_shape_index, x_loc, y_loc):
        if self.end_game():
            return self.end_score()
        game_shape = self.shapes[game_shape_index]
        y_len, x_len = game_shape.shape.shape
        if x_len + x_loc > self.board.shape[1]:
            return self._invalid_move()
        if y_len + y_loc > self.board.shape[0]:
            return self._invalid_move()

        blank_b = self.blank_board()
        neg_shape = game_shape.shape * -1
        self.add_shape_to_array(neg_shape, blank_b)

        temp_board = deepcopy(self.board) * blank_b

        if np.min(temp_board) < 0:
            return self._invalid_move()
        else:
            new_board = self.board + (blank_b * -1)
            self.board = new_board
            self._check_point_condition()
            self.shapes[game_shape_index] = Shape(self.rand, self.combo)
            self.invalid_move_count = 0
            self.rotation_count = 0
        return -2

    @staticmethod
    def _get_quadrant_loc(loc):
        x1 = (loc % 3) * 3
        x2 = x1 + 3
        y1 = int(loc / 3) * 3
        y2 = y1 + 3

        return x1, x2, y1, y2

    def _get_quadrant(self, loc):
        x1, x2, y1, y2 = self._get_quadrant_loc(loc)

        return self.board[y1:y2, x1:x2]

    def _clear_quadrant(self, loc):
        x1, x2, y1, y2 = self._get_quadrant_loc(loc)

        self.board[y1:y2, x1:x2] = 0

    def _check_point_condition(self):
        quadrants_finished = 0
        point_sum = 0

        for i in range(0, 9):
            quad = self._get_quadrant(i)
            if np.min(quad) > 0:
                point_sum += np.sum(self._get_quadrant(i))
                self._clear_quadrant(i)
                quadrants_finished += 1

        if quadrants_finished > 0:
            self.combo += quadrants_finished
            self.combo = min(self.combo, 5)
        else:
            self.combo = 1

        self.multi = quadrants_finished

        self.points += point_sum * self.multi


class Shape:
    def __init__(self, rand: random.Random, multi):
        all_shapes = \
            [
                [[1, 0, 0],
                 [1, 0, 0],
                 [1, 1, 1]],

                [[1, 1, 0],
                 [0, 1, 1]],

                [[0, 1, 1],
                 [1, 1, 0]],

                [[1, 0],
                 [1, 0],
                 [1, 1]],

                [[0, 1],
                 [0, 1],
                 [1, 1]],

                [[1, 0],
                 [1, 1],
                 [1, 0]],

                [[1, 0],
                 [1, 1]],

                [[1]],

                [[1, 1]],

                [[1, 1, 1]],

                [[1, 1, 1, 1]],

                [[1, 1],
                 [1, 1]]
            ]

        idx = rand.randint(0, len(all_shapes) - 1)
        self.shape = np.array(all_shapes[idx]) * multi

    def turn(self, times=1):
        self.shape = np.rot90(self.shape, times)
