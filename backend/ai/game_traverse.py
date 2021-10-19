#!/usr/bin/env python3

import copy
import multiprocessing
import random
import time
from collections import Counter
from threading import Thread
from typing import List

import cython
import torch
from engine.game_interface import GameInterface
from torch.utils.data import IterableDataset
from utils.priority import lowpriority
from utils.profiler import Profiler

from ai.types import GameRollout


# @cython.cfunc
@torch.no_grad()
def traverse(
    game: GameInterface,
    policy_network,
    metrics: Counter,
    level: cython.int,
):
    if game.terminal():
        gr = GameRollout(
            torch.zeros((level, game.feature_dim()), dtype=torch.float),  # States
            torch.zeros((level, 1), dtype=torch.long),  # Actions
            torch.zeros(
                (level, game.action_dim()), dtype=torch.int
            ),  # Possible Actions
            torch.zeros((level, 1), dtype=torch.long),  # Player to act
            game.payoffs().float().repeat((level, 1)),  # Payoffs
            torch.arange(level - 1, -1, -1, dtype=torch.float).unsqueeze(
                1
            ),  # Distance to payoff
            torch.zeros((level, game.action_dim()), dtype=torch.float),  # Policy
        )
        return gr

    features = torch.zeros((game.feature_dim(),), dtype=torch.float)
    game.populate_features(features)
    player_to_act = game.get_player_to_act()
    possible_actions = game.get_one_hot_actions(False)
    num_choices = possible_actions.sum()
    metrics.update({"possible_actions_" + str(possible_actions.sum()): 1})
    has_a_choice = num_choices > 1
    if policy_network is None or not has_a_choice:
        strategy = strategy_without_exploration = (
            possible_actions.float() / possible_actions.sum()
        )
        active_sampling_chances = None
    else:
        strategy = policy_network(
            features.unsqueeze(0), possible_actions.unsqueeze(0), True
        )[0][0]
        assert (strategy * (1 - possible_actions)).sum() == 0
        strategy_without_exploration = policy_network(
            features.unsqueeze(0), possible_actions.unsqueeze(0), False
        )[0][0]

    action_dist = torch.distributions.Categorical(strategy)
    if action_dist.probs.min() < 0 or action_dist.probs.max() == 0:
        print("Invalid action dist:", action_dist.probs)
    if strategy.min() < 0 or strategy.max() == 0:
        print("Invalid strategy:", strategy)

    metrics.update({"visit_level_" + str(level): 1})
    metrics["visit"] += 1
    if metrics["visit"] % 100000 == 0:
        print("Visits", metrics["visit"])

    action_taken = int(action_dist.sample().item())
    game.act(player_to_act, action_taken)
    if has_a_choice:
        result = traverse(
            game,
            policy_network,
            metrics,
            level + 1,
        )
        payoff = result.payoffs[player_to_act]
        result.states[level] = features
        result.actions[level] = action_taken
        result.player_to_act[level] = player_to_act
        result.possible_actions[level] = possible_actions
        result.policy[level] = strategy_without_exploration
    else:
        # Don't advance the level, skip this non-choice
        result = traverse(
            game,
            policy_network,
            metrics,
            level,
        )
    return result


@torch.no_grad()
def start_traverse(
    game: GameInterface,
    policy_network,
    metrics: Counter,
    level: cython.int,
) -> GameRollout:
    # lowpriority()
    with Profiler(False):
        return traverse(game, policy_network, metrics, level)
