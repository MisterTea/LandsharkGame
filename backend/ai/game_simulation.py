#!/usr/bin/env python3

import copy
import multiprocessing
import time
from collections import Counter
from threading import Thread
from typing import List

import torch
from torch.utils.data import IterableDataset

from ai.types import GameRollout
from engine.game_interface import GameInterface
from utils.profiler import Profiler


def traverse(
    game: GameInterface, policy_network, metrics: Counter, level: int,
) -> GameRollout:
    if game.terminal():
        gr = GameRollout(
            torch.zeros((level, game.feature_dim()), dtype=torch.float),  # States
            torch.zeros((level, 1), dtype=torch.long),  # Actions
            torch.zeros(
                (level, game.action_dim()), dtype=torch.long
            ),  # Possible Actions
            torch.zeros((level, 1), dtype=torch.long),  # Player to act
            game.payoffs().float().repeat((level, 1)),  # Payoffs
            torch.arange(level - 1, -1, -1, dtype=torch.float).unsqueeze(
                1
            ),  # Distance to payoff
        )
        return gr

    features = torch.zeros((game.feature_dim(),), dtype=torch.float)
    game.populate_features(features)
    player_to_act = game.get_player_to_act()
    possible_actions = game.get_one_hot_actions(True)
    num_choices = possible_actions.sum()
    metrics.update({"possible_actions_" + str(possible_actions.sum()): 1})
    has_a_choice = num_choices > 1
    if policy_network is None or not has_a_choice:
        strategy = possible_actions.float()
        active_sampling_chances = None
    else:
        strategy = policy_network(features.unsqueeze(0), possible_actions.unsqueeze(0))[
            0
        ][0]
        assert (strategy * (1 - possible_actions)).sum() == 0

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
        result = traverse(game, policy_network, metrics, level + 1,)
        payoff = result.payoffs[player_to_act]
        result.states[level] = features
        result.actions[level] = action_taken
        result.player_to_act[level] = player_to_act
        result.possible_actions[level] = possible_actions
    else:
        # Don't advance the level, skip this non-choice
        result = traverse(game, policy_network, metrics, level,)
    return result


NUM_PARALLEL_GAMES = 8


class GameSimulationIterator(Thread):
    def __init__(self, game: GameInterface, max_games: int, policy_network, pool):
        super().__init__()
        self.on_game = 0
        self.game = game.clone()
        self.policy_network = policy_network
        self.max_games = max_games
        self.pool = pool
        self.results = []
        self.futures = []
        self.on_iter = 0

        # Spin up some games
        for _ in range(NUM_PARALLEL_GAMES):
            self.start_game()

    def __iter__(self):
        self.on_game = 0
        self.on_iter = 0
        return self

    def start_game(self):
        if self.on_iter == self.max_games:
            return
        self.on_game += 1
        # print("Starting", self.on_game)
        ng = self.game.clone()
        ng.reset()
        with Profiler(False):
            metrics = Counter()
            if self.policy_network.num_steps >= 100:
                eval_net = copy.deepcopy(self.policy_network).cpu().eval()
            else:
                eval_net = None
            self.pool.apply_async(
                traverse, args=(ng, eval_net, metrics, 0,), callback=self.finish_game
            )

    def finish_game(self, gr):
        self.results.append(gr)
        self.start_game()

    def run(self):
        while self.on_game + len(self.results) + len(self.futures) < self.max_games:
            time.sleep(1.0)

            # Check for finished futures

    def __next__(self):
        if self.on_iter == self.max_games:
            raise StopIteration
        while len(self.results) == 0:
            time.sleep(1.0)
        r = self.results
        self.results = []
        self.on_iter += 1
        # print("Processing", len(r))
        states = torch.cat([gr.states for gr in r])
        actions = torch.cat([gr.actions for gr in r])
        possible_actions = torch.cat([gr.possible_actions for gr in r])
        player_to_act = torch.cat([gr.player_to_act for gr in r])
        payoffs = torch.cat([gr.payoffs for gr in r])
        distance_to_payoff = torch.cat([gr.distance_to_payoff for gr in r])

        return (
            states,
            actions,
            possible_actions,
            player_to_act,
            payoffs,
            distance_to_payoff,
        )


class GameSimulationDataset(IterableDataset):
    def __init__(self, game: GameInterface, max_games: int, policy_network):
        self.max_games = max_games
        self.game = game.clone()
        self.policy_network = policy_network
        self.pool = multiprocessing.Pool(NUM_PARALLEL_GAMES)

    def __iter__(self) -> GameSimulationIterator:
        gsi = GameSimulationIterator(
            self.game, self.max_games, self.policy_network, self.pool
        )
        return gsi

    def __len__(self):
        return self.max_games
