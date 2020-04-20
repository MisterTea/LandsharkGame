#!/usr/bin/env python3

import torch


class GameInterface:
    @property
    def num_players(self):
        raise NotImplementedError()

    def feature_dim(self):
        raise NotImplementedError()

    def action_dim(self):
        raise NotImplementedError()

    def terminal(self):
        raise NotImplementedError()

    def payoffs(self):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()

    def act(self, player: int, action: int):
        raise NotImplementedError()

    def populate_features(self, features: torch.Tensor):
        raise NotImplementedError()

    def get_player_to_act(self) -> int:
        raise NotImplementedError()

    def get_one_hot_actions(self, hacks) -> torch.Tensor:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
