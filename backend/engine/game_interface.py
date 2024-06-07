#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class PlayerFeatureIndices:
    dense_start: int
    dense_end: int
    embedding_start: int
    embedding_end: int

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
    
    def get_player_cursors(self) -> List[PlayerFeatureIndices]:
        raise NotImplementedError()

    def populate_features(
        self, dense_features: torch.Tensor, embedding_features: torch.Tensor
    ):
        raise NotImplementedError()

    def get_player_to_act(self) -> int:
        raise NotImplementedError()

    def getPossibleActions(self) -> List[int]:
        raise NotImplementedError()

    def get_one_hot_actions(self, hacks) -> torch.Tensor:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
