from dataclasses import dataclass

import torch


@dataclass
class GameRollout:
    states: torch.Tensor
    actions: torch.Tensor
    possible_actions: torch.Tensor
    player_to_act: torch.Tensor
    payoffs: torch.Tensor
    distance_to_payoff: torch.Tensor
