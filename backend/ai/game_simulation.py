#!/usr/bin/env python3

import copy
import random
import threading
import time
from collections import Counter
from threading import Thread
from typing import List

import torch
from engine.game_interface import GameInterface
from torch import multiprocessing
from torch.utils.data import IterableDataset
from utils.profiler import Profiler

from ai.game_traverse import start_traverse
from ai.types import GameRollout

NUM_PARALLEL_GAMES = max(1, multiprocessing.cpu_count() - 2)
GAMES_PER_MINIBATCH = 32


class GameSimulationIterator:
    def __init__(self, game: GameInterface, max_games: int, policy_networks, pool):
        super().__init__()
        self.on_game = 0
        self.game = game.clone()
        self.policy_networks = policy_networks
        self.max_games = max_games
        self.pool = multiprocessing.Pool(NUM_PARALLEL_GAMES)
        self.results = []
        self.games_in_progress = []
        self.on_iter = 0
        self.eval_net = None
        self.eval_net_age = -1

        # Spin up some games
        for _ in range(NUM_PARALLEL_GAMES):
            # print("SPINNING UP GAMES")
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
        metrics = Counter()
        # self.eval_net = None

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
                callback=self.finish_game,
            )
        )

    def finish_game(self, gr):
        # print("game ended")
        self.start_game()
        pass

    def __next__(self):
        self.on_iter += 1
        if self.on_iter == self.max_games:
            self.pool.terminate()
            raise StopIteration
        while len(self.games_in_progress) > 0:
            if (
                not self.games_in_progress[0].ready()
                and len(self.results) >= GAMES_PER_MINIBATCH
            ):
                # We are caught up and have enough games
                # print(
                #     f"Stopping collection with {len(self.games_in_progress)} in the queue"
                # )
                break
            game_in_progress = self.games_in_progress.pop(0)
            gr = game_in_progress.get()
            self.results.append(gr)
        while len(self.results) < GAMES_PER_MINIBATCH:
            print(
                f"WAITING FOR RESULTS {len(self.results)} {len(self.games_in_progress)}"
            )
            time.sleep(0.001)
            while (
                len(self.games_in_progress) > 0
                and len(self.results) < GAMES_PER_MINIBATCH
            ):
                game_in_progress = self.games_in_progress.pop()
                game_in_progress.get()
        # r = random.sample(self.results, min(GAMES_PER_MINIBATCH, len(self.results)))
        # self.results.clear()
        with torch.no_grad():
            # print(f"GOT {len(self.results)} GAMES")
            r, self.results = self.results[0:GAMES_PER_MINIBATCH], []
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
    def __init__(self, game: GameInterface, max_games: int, policy_networks):
        self.max_games = max_games
        self.game = game.clone()
        self.policy_networks = policy_networks
        self.pool = None

    def __iter__(self):
        print("Getting iterator")
        gsi = GameSimulationIterator(
            self.game, self.max_games, self.policy_networks, self.pool
        )
        return gsi

    def __len__(self):
        return self.max_games
