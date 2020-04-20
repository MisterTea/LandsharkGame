#%%
import hashlib
import os
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing import Pool
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import IterableDataset

from ai.expandable_tensor_set import ExpandableTensorSet
from ai.regret_matching import RegretMatching, RegretMatchingForward
from ai.reservoir_buffer import ReservoirBuffer
from ai.types import GameRollout
from engine.game import GamePhase, GameState
from engine.game_interface import GameInterface
from utils.priority import lowpriority
from utils.profiler import Profiler


def train(iterations: int, game: GameInterface, output_file: str):
    NUM_GAME_ITERATIONS = 1000

    model = MeanActorCritic(game.feature_dim(), game.action_dim())

    with Pool(os.cpu_count()) as gamePool:
        for iteration in range(iterations):
            print("ON ITERATION", iteration)
            if iteration > 0:
                policy = model.get_policy()
            else:
                policy = None

            with torch.no_grad():
                starts = []
                for player_to_train in range(game.num_players):
                    for game_iteration in range(NUM_GAME_ITERATIONS):
                        # print("Queueing Game", player_to_train, game_iteration)
                        new_game = game.clone()
                        new_game.reset()
                        starts.append((new_game, policy))

                if True:
                    results = gamePool.starmap(start_traverse, starts)
                else:
                    results = []
                    for start in starts:
                        results.append(start_traverse(*start))
                print("Finished playing games")

                metrics: Counter = Counter()
                for result in results:
                    (
                        player_to_train,
                        new_player_regret,
                        new_strategy_data,
                        new_metrics,
                    ) = result
                    playerRegrets[player_to_train].cat(new_player_regret)
                    strategyData.cat(new_strategy_data)
                    metrics.update(new_metrics)
                print(metrics)

            if (iteration + 1) % 1 == 0:
                stratModel = RegretMatching(game.feature_dim(), game.action_dim())
                features, active_labels, labels = strategyData.getFilled()
                stratModel.train_model(
                    features, active_labels, labels, "strategy", None
                )
                bestStrategy = RegretMatchingForward(stratModel)
                # print(
                #     "Learned Strategy at " + str(iteration) + ": ", str(bestStrategy),
                # )
                # print("***")
                torch.save(bestStrategy, output_file)

                with torch.no_grad():
                    # Check winrate against random player
                    scoreCounter: Counter = Counter()
                    NUM_RANDOM_GAMES = 1000
                    num_decisions = 0
                    average_decision = torch.zeros(
                        (game.action_dim(),), dtype=torch.float
                    )
                    for on_game in range(NUM_RANDOM_GAMES):
                        gameState = game.clone()
                        gameState.reset()
                        features = torch.zeros(
                            (1, gameState.feature_dim()), dtype=torch.float
                        )
                        while (
                            not gameState.terminal()
                        ):  # gameState.phase != GamePhase.GAME_OVER:
                            seatToAct = gameState.get_player_to_act()
                            if seatToAct == 0:
                                possible_action_mask = gameState.get_one_hot_actions(
                                    True
                                )
                                gameState.populate_features(features[0])
                                action_probs = (
                                    bestStrategy(features).detach()[0].clamp(min=1e-6)
                                )
                            else:
                                possible_action_mask = gameState.get_one_hot_actions(
                                    True
                                )
                                action_probs = possible_action_mask.float()
                            action_prob_dist = torch.distributions.Categorical(
                                action_probs * possible_action_mask
                            )
                            if on_game == 0:
                                print("ACTION", action_prob_dist.probs)
                            action_index = int(action_prob_dist.sample().item())
                            average_decision[action_index] += 1.0
                            num_decisions += 1
                            gameState.act(seatToAct, action_index)
                        payoffs = gameState.payoffs()
                        for i, p in enumerate(payoffs):
                            scoreCounter[str(i)] += p
                    print("DECISION HISTOGRAM")
                    print(average_decision / num_decisions)
                    print("SCORE AGAINST RANDOM")
                    for x in range(gameState.num_players):
                        print(x, scoreCounter[str(x)] / float(NUM_RANDOM_GAMES))

    stratModel = RegretMatching(game.feature_dim(), game.action_dim())
    stratModel.train_model(*strategyData.getFilled())
    return stratModel


# bestStrategy = train(100, RpsGame())
# print("FINAL STRAT: ", bestStrategy)


# %%
