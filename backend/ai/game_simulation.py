#!/usr/bin/env python3

import copy
import random
import time
from collections import Counter
from multiprocessing.pool import ThreadPool
from threading import Thread
from typing import List

import torch
from engine.game_interface import GameInterface
from torch import multiprocessing
from torch.utils.data import IterableDataset
from utils.profiler import Profiler

from ai.game_simulation_nopool import GAMES_PER_MINIBATCH
from ai.game_traverse import start_traverse
from ai.types import GameRollout


class GameSimulationIterator:
    def __init__(
        self, game: GameInterface, minibatches_per_epoch: int, policy_networks
    ):
        super().__init__()
        self.game = game.clone()
        self.policy_networks = policy_networks
        self.minibatches_per_epoch = minibatches_per_epoch
        self.pool = ThreadPool(4)
        self.games_in_progress = []
        self.on_iter = 0
        self.eval_net = None
        self.eval_net_age = -1

        # Spin up some games
        for _ in range(GAMES_PER_MINIBATCH):
            # print("SPINNING UP GAMES")
            self.start_game()

    def __iter__(self):
        self.on_iter = 0
        return self

    def start_game(self):
        # print("Starting", id(self), self.minibatches_per_epoch)
        ng = self.game.clone()
        ng.reset()
        metrics = Counter()
        self.eval_net = None

        if self.policy_networks is not None:
            policy_network = random.choice(self.policy_networks)
            if policy_network.num_steps != self.eval_net_age:
                # print("BUMPING POLICY NET")
                self.eval_net_age = policy_network.num_steps
                self.eval_net = copy.deepcopy(policy_network).cpu().eval()

        self.games_in_progress.append(
            self.pool.apply_async(
                start_traverse,
                args=(
                    ng,
                    self.eval_net,
                    metrics,
                    0,
                ),
            )
        )

    def __next__(self):
        self.on_iter += 1
        if self.on_iter == self.minibatches_per_epoch:
            # self.pool.terminate()
            raise StopIteration
        results = [
            game_in_progress.get() for game_in_progress in self.games_in_progress
        ]
        self.games_in_progress = []
        if self.on_iter + 1 < self.minibatches_per_epoch:
            # Spin up some games
            for _ in range(GAMES_PER_MINIBATCH):
                # print("SPINNING UP GAMES")
                self.start_game()

        with torch.no_grad():
            r = results
            states = torch.cat([gr.states for gr in r])
            actions = torch.cat([gr.actions for gr in r])
            possible_actions = torch.cat([gr.possible_actions for gr in r])
            player_to_act = torch.cat([gr.player_to_act for gr in r])
            payoffs = torch.cat([gr.payoffs for gr in r])
            distance_to_payoff = torch.cat([gr.distance_to_payoff for gr in r])
            policy = torch.cat([gr.policy for gr in r])

            return (
                states,
                actions,
                possible_actions,
                player_to_act,
                payoffs,
                distance_to_payoff,
                policy,
            )


class GameSimulationDataset(IterableDataset):
    def __init__(
        self, game: GameInterface, minibatches_per_epoch: int, policy_networks
    ):
        self.minibatches_per_epoch = minibatches_per_epoch
        self.game = game.clone()
        self.policy_networks = policy_networks

    def __iter__(self):
        # print("Getting iterator")
        gsi = GameSimulationIterator(
            self.game, self.minibatches_per_epoch, self.policy_networks
        )
        return gsi

    # def __len__(self):
    # return self.minibatches_per_epoch
