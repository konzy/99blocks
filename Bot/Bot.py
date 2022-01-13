import numpy as np
import torch
import torch.nn as nn

import AutoResNet
from AutoResNet.Conv2dAuto import Conv2dAuto
from AutoResNet.ResNet import resnet18
from Bot import Head
from Game.Game import Game


def my_loss(output):
    # 1000/ 1 + points
    loss = torch.div(1000, 1 + output)
    return loss


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = resnet18(9*9, 9*9)
        self.policy_head = Head.Policy(9 * 9, 4)
        self.value_head = Head.Value(9 * 9, 1)

    def forward(self, x):
        #
        # with open(file, 'w') as f:
        loss_fn = nn.MSELoss(reduction='min')
        game = Game()
        move_string = ''
        result = -1
        t = 0
        while t < 100 or result >= 0:
            t += 1
            training_base = np.append(game.board, Game.add_shape_to_array_at(game.shapes[0], Game.blank_board()))
            training_base = np.append(training_base, Game.add_shape_to_array_at(game.shapes[1], Game.blank_board()))
            training_base = np.append(training_base, Game.add_shape_to_array_at(game.shapes[2], Game.blank_board()))

            all_training = np.empty(game.board.shape)
            for i in range(0, 8):
                all_training = np.append(all_training, training_base)

            y_pred = self.net()
            game_shape_index = int(y_pred[0] * 3)
            rotate = int(y_pred[1] * 2)
            x_loc = int(y_pred[2] * 9)
            y_loc = int(y_pred[3] * 9)
            if rotate:
                result = game.rotate_shape(game_shape_index)
            else:
                result = game.play_move(game_shape_index, x_loc, y_loc)
