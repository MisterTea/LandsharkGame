#!/usr/bin/env python3

import copy
import random
from enum import IntEnum
from typing import Dict, List, Tuple
from uuid import UUID, uuid4

import numpy as np
import torch

from engine.game_interface import GameInterface


class RpsGame(GameInterface):
    class ActionName(IntEnum):
        ROCK = 0
        PAPER = 1
        SCISSORS = 2
        LOSE = 3

    def __init__(self):
        NUM_ACTIONS = len(RpsGame.ActionName)

        self.payoff_matrix = torch.zeros(
            (NUM_ACTIONS, NUM_ACTIONS, 2), dtype=torch.float
        )

        for a1 in RpsGame.ActionName:
            for a2 in RpsGame.ActionName:
                if (int(a1) + 1) % 3 == int(a2):
                    self.payoff_matrix[a1, a2, 0] = -1
                    self.payoff_matrix[a1, a2, 1] = 1
                    # print(a2, "beats", a1)
                elif (int(a2) + 1) % 3 == int(a1):
                    self.payoff_matrix[a1, a2, 0] = 1
                    self.payoff_matrix[a1, a2, 1] = -1
                    # print(a1, "beats", a2)
                else:
                    self.payoff_matrix[a1, a2, 0] = 0
                    self.payoff_matrix[a1, a2, 1] = 0
                    # print(a2, "ties", a1)

                if a1 == RpsGame.ActionName.LOSE and a2 == RpsGame.ActionName.LOSE:
                    self.payoff_matrix[a1, a2, :] = -10
                elif a1 == RpsGame.ActionName.LOSE:
                    self.payoff_matrix[a1, a2, 0] = -10
                    self.payoff_matrix[a1, a2, 1] = 1
                elif a2 == RpsGame.ActionName.LOSE:
                    self.payoff_matrix[a1, a2, 0] = 1
                    self.payoff_matrix[a1, a2, 1] = -10

        self.reset()

    @property
    def num_players(self):
        return 2

    def clone(self):
        return deepcopy(self)

    def reset(self):
        self.actions = [-1] * self.num_players

    def action_dim(self):
        return len(RpsGame.ActionName)

    def terminal(self):
        return self.get_player_to_act() == -1

    def payoffs(self):
        return self.payoff_matrix[self.actions[0], self.actions[1]]

    def get_one_hot_actions(self, hacks):
        return torch.ones((len(RpsGame.ActionName),), dtype=torch.float)

    def populate_features(self, features: torch.Tensor):
        features.fill_(1.0)

    def feature_dim(self):
        return 1

    def get_features(self):
        return torch.ones((self.feature_dim(),), dtype=torch.float)

    def get_player_to_act(self):
        for i, a in enumerate(self.actions):
            if a == -1:
                return i
        return -1

    def act(self, player: int, action_index: int):
        assert player == self.get_player_to_act()
        self.actions[player] = action_index
