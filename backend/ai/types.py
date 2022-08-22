from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class GameEmbedding:
    cardinality: int
    ranges: List[Tuple[int, int]]


@dataclass
class GameRollout:
    dense_state_features: torch.Tensor
    embedding_state_features: torch.Tensor
    actions: torch.Tensor
    possible_actions: torch.Tensor
    player_to_act: torch.Tensor
    payoffs: torch.Tensor
    distance_to_payoff: torch.Tensor
    policy: torch.Tensor
