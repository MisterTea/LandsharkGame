#!/usr/bin/env python3

import copy
import multiprocessing
import random
import threading
import time
from collections import Counter
from threading import Thread
from typing import List, Optional

import cython
import torch
from engine.game_interface import GameInterface
from torch.utils.data import IterableDataset
from utils.priority import lowpriority
from utils.profiler import Profiler

from ai.mean_actor_critic import (
    ImitationLearningModel,
    StateValueModel,
    get_action_from_imitator,
)
from ai.types import GameRollout


def one_step_best_response(
    game: GameInterface,
    value_network: StateValueModel,
    policy_network: ImitationLearningModel,
) -> int:
    dense_features = torch.zeros((game.feature_dim(),), dtype=torch.float)
    embedding_features = torch.zeros((game.embedding_dim(),), dtype=torch.int)
    player_to_act = game.get_player_to_act()
    actions = game.get_one_hot_actions()

    best_score = -1.0
    best_action = -1

    for i in range(game.action_dim()):
        if actions[i] == 0:
            continue
        action = i
        g = copy.deepcopy(game)
        g.act(player_to_act, action)

        score = None

        while not g.terminal() and g.get_player_to_act() != player_to_act:
            # Make opponent moves based on policy network
            action = get_action_from_imitator(
                model=policy_network,
                game=g,
                dense_features=dense_features,
                embedding_features=embedding_features,
            )
            g.act(g.get_player_to_act(), action)

        if g.terminal():
            # print("G IS TERMINAL")
            score = float((torch.argmax(g.payoffs()) == player_to_act).item())
            # print("SCORE", score, g.payoffs(), player_to_act)
        else:
            g.populate_features(dense_features, embedding_features)
            score = value_network.get_value_logits(
                dense_features.unsqueeze(0), embedding_features.unsqueeze(0)
            )[0][0]

        if best_action == -1 or best_score < score:
            best_score = score
            best_action = i

    return best_action


# @cython.cfunc
@torch.no_grad()
def traverse(
    game: GameInterface,
    value_network: Optional[StateValueModel],
    policy_network: Optional[ImitationLearningModel],
    metrics: Counter,
    level: cython.int,
) -> GameRollout:
    if game.terminal():
        gr = GameRollout(
            torch.zeros((level, game.feature_dim()), dtype=torch.float),  # States
            torch.zeros((level, game.embedding_dim()), dtype=torch.int),  # States
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

    metrics.update({"visit_level_" + str(level): 1})
    metrics["visit"] += 1
    if metrics["visit"] % 100000 == 0:
        print("Visits", metrics["visit"])

    dense_features = torch.zeros((game.feature_dim(),), dtype=torch.float)
    embedding_features = torch.zeros((game.embedding_dim(),), dtype=torch.int)
    game.populate_features(dense_features, embedding_features)
    player_to_act = game.get_player_to_act()
    possible_actions = game.get_one_hot_actions(False)
    num_choices = possible_actions.sum()
    metrics.update({"possible_actions_" + str(possible_actions.sum()): 1})
    has_a_choice = num_choices > 1
    if value_network is None or not has_a_choice:
        strategy = strategy_without_exploration = (
            possible_actions.float() / possible_actions.sum()
        )
        active_sampling_chances = None

        action_dist = torch.distributions.Categorical(strategy)
        if action_dist.probs.min() < 0 or action_dist.probs.max() == 0:
            print("Invalid action dist:", action_dist.probs)
        if strategy.min() < 0 or strategy.max() == 0:
            print("Invalid strategy:", strategy)

        action_taken = int(action_dist.sample().item())

        # Deterministic player to test imitation learning
        # action_taken = int(possible_actions.argmax().item())
    else:
        strategy_without_exploration = None
        action_taken = one_step_best_response(game, value_network, policy_network)

    game.act(player_to_act, action_taken)
    if has_a_choice:
        result = traverse(
            game,
            value_network,
            policy_network,
            metrics,
            level + 1,
        )

        result.payoffs[level] = torch.roll(
            result.payoffs[level], shifts=-player_to_act, dims=0
        )

        result.dense_state_features[level] = dense_features
        result.embedding_state_features[level] = embedding_features
        result.actions[level] = action_taken
        result.player_to_act[level] = player_to_act
        result.possible_actions[level] = possible_actions
        if strategy_without_exploration is not None:
            result.policy[level] = strategy_without_exploration
    else:
        assert False, "Skipped actions should be forced in the game engine"
        # Don't advance the level, skip this non-choice
        result = traverse(
            game,
            value_network,
            policy_network,
            metrics,
            level,
        )
    return result


@torch.no_grad()
def start_traverse(
    game: GameInterface,
    value_network: Optional[StateValueModel],
    policy_network: Optional[ImitationLearningModel],
    metrics: Counter,
    level: cython.int,
) -> GameRollout:
    # lowpriority()
    with Profiler(False):
        return traverse(game, value_network, policy_network, metrics, level)
