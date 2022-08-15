#!/usr/bin/env python3

import copy
import random
import time
from collections import Counter
from threading import Thread
from typing import List, Optional

import torch
from engine.game_interface import GameInterface
from torch import multiprocessing
from torch.utils.data import IterableDataset
from utils.priority import lowpriority
from utils.profiler import Profiler

from ai.game_traverse import start_traverse
from ai.types import GameRollout

from .mean_actor_critic import ImitationLearningModel, StateValueModel

GAMES_PER_MINIBATCH = 64


class GameSimulationIterator:
    def __init__(
        self,
        game: GameInterface,
        minibatches_per_epoch: int,
        value_network: Optional[StateValueModel],
        policy_networks: List[ImitationLearningModel],
    ):
        super().__init__()
        self.on_game = 0
        self.game = game.clone()
        self.value_network = value_network
        self.policy_networks = policy_networks
        self.minibatches_per_epoch = minibatches_per_epoch
        self.results = []
        self.games_in_progress = []
        self.on_iter = 0
        self.eval_net = None
        self.eval_net_age = -1
        lowpriority()

    def __iter__(self):
        self.on_game = 0
        self.on_iter = 0
        return self

    def __next__(self):
        if self.on_iter == self.minibatches_per_epoch:
            raise StopIteration
        self.on_iter += 1
        for x in range(GAMES_PER_MINIBATCH):
            self.on_game += 1
            # print("Starting", self.on_game)
            ng = self.game.clone()
            ng.reset()
            metrics = Counter()
            gr = start_traverse(
                ng,
                self.value_network,
                None
                if self.value_network is None
                else random.choice(self.policy_networks),
                metrics,
                0,
            )
            self.results.append(gr)
        with torch.no_grad():
            r, self.results = self.results, []
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
        self,
        game: GameInterface,
        minibatches_per_epoch: int,
        value_network: Optional[StateValueModel],
        policy_networks: List[ImitationLearningModel],
    ):
        self.minibatches_per_epoch = minibatches_per_epoch
        self.game = game.clone()
        self.value_network = value_network
        self.policy_networks = policy_networks

    def __iter__(self):
        gsi = GameSimulationIterator(
            self.game,
            self.minibatches_per_epoch,
            self.value_network,
            self.policy_networks,
        )
        return gsi

    # def __len__(self):
    # return self.minibatches_per_epoch
