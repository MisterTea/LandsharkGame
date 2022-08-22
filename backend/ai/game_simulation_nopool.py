#!/usr/bin/env python3

import copy
import os
import pickle
import random
import time
from collections import Counter
from threading import Thread
from typing import List, Optional

import numpy as np
import torch
from engine.game_interface import GameInterface
from torch import multiprocessing
from torch.utils.data import IterableDataset
from utils.model_serialization import load
from utils.priority import lowpriority
from utils.profiler import Profiler

from ai.game_traverse import start_traverse
from ai.types import GameRollout

from .mean_actor_critic import ImitationLearningModel, StateValueModel


class GameSimulationIterator:
    def __init__(
        self,
        name: str,
        game: GameInterface,
        minibatches_per_epoch: int,
        games_per_minibatch: int,
        value_network_bytes: Optional[bytes],
        policy_network_bytes: List[bytes],
    ):
        super().__init__()
        self.name = name
        self.iter_count = torch.utils.data.get_worker_info().id
        self.on_game = 0
        self.game = game.clone()
        if value_network_bytes is not None:
            self.value_network = load(value_network_bytes)
        else:
            self.value_network = None
        self.policy_networks = [load(p) for p in policy_network_bytes]
        self.minibatches_per_epoch = minibatches_per_epoch
        self.games_per_minibatch = games_per_minibatch
        self.results: List[GameRollout] = []
        self.games_in_progress = []
        self.on_iter = 0
        self.eval_net = None
        self.eval_net_age = -1
        lowpriority()

        name_mod_hash = self.name.__hash__() % (1024 * 1024)
        torch.manual_seed(name_mod_hash + self.iter_count)
        random.seed(name_mod_hash + self.iter_count)
        np.random.seed(name_mod_hash + self.iter_count)

        # print("ON ITERATOR", self.iter_count, self.cache_pathname)
        self.cached_results = []
        if os.path.exists(self.cache_pathname):
            with open(self.cache_pathname, "rb") as handle:
                self.cached_results = pickle.load(handle)
                assert len(self.cached_results) > 0
            # print(f"LOADING FROM CACHE: {len(self.cached_results)}")
            self.got_cache = True
        else:
            self.got_cache = False

    @property
    def cache_pathname(self):
        return f"game_cache/games_{self.name}_{self.iter_count}.pkl"

    @property
    def game_metrics_pathname(self):
        return f"game_metrics/games_{self.name}_{self.iter_count}.pkl"

    def __iter__(self):
        self.on_game = 0
        self.on_iter = 0
        return self

    def __next__(self):
        if self.on_iter == self.minibatches_per_epoch:
            if self.got_cache == False:
                with open(self.cache_pathname, "wb") as handle:
                    pickle.dump(self.cached_results, handle)
            raise StopIteration
        self.on_iter += 1
        if self.got_cache:
            if len(self.cached_results) == 0:
                raise StopIteration
            retval = self.cached_results.pop(0)
            return retval
        for x in range(self.games_per_minibatch):
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
            dense_state_features = torch.cat([gr.dense_state_features for gr in r])
            embedding_state_features = torch.cat(
                [gr.embedding_state_features for gr in r]
            )
            actions = torch.cat([gr.actions for gr in r])
            possible_actions = torch.cat([gr.possible_actions for gr in r])
            player_to_act = torch.cat([gr.player_to_act for gr in r])
            payoffs = torch.cat([gr.payoffs for gr in r])
            distance_to_payoff = torch.cat([gr.distance_to_payoff for gr in r])
            policy = torch.cat([gr.policy for gr in r])

            result_batch = (
                dense_state_features,
                embedding_state_features,
                actions,
                possible_actions,
                player_to_act,
                payoffs,
                distance_to_payoff,
                policy,
            )

            self.cached_results.append(result_batch)

            return result_batch


class GameSimulationDataset(IterableDataset):
    def __init__(
        self,
        name: str,
        game: GameInterface,
        minibatches_per_epoch: int,
        games_per_minibatch: int,
        value_network: Optional[StateValueModel],
        policy_networks: List[ImitationLearningModel],
    ):
        assert minibatches_per_epoch > 0
        self.minibatches_per_epoch = minibatches_per_epoch
        self.name = name
        self.game = game.clone()
        self.games_per_minibatch = games_per_minibatch
        self.value_network = value_network
        self.policy_networks = policy_networks

    def __iter__(self):
        gsi = GameSimulationIterator(
            self.name,
            self.game,
            self.minibatches_per_epoch,
            self.games_per_minibatch,
            self.value_network,
            self.policy_networks,
        )
        return gsi

    # def __len__(self):
    # return self.minibatches_per_epoch
