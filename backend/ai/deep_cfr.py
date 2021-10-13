#%%
import hashlib
import os
import random
from collections import Counter
from copy import deepcopy
from enum import IntEnum
from multiprocessing import Pool
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from engine.game import GamePhase, GameState
from engine.game_interface import GameInterface
from utils.priority import lowpriority
from utils.profiler import Profiler

from ai.expandable_tensor_set import ExpandableTensorSet
from ai.fully_connected_forward import FullyConnectedForward
from ai.regret_matching import RegretMatching
from ai.reservoir_buffer import ReservoirBuffer


def traverse(
    game: GameInterface,
    player_to_train: int,
    regretModels: List[Optional[FullyConnectedForward]],
    playerRegret: ExpandableTensorSet,
    strategyModels: List[Optional[FullyConnectedForward]],
    strategyData: ExpandableTensorSet,
    metrics: Counter,
    level: int,
    first_pass: bool,
    branch_factor_estimate: float,
) -> torch.Tensor:
    if game.terminal():
        return game.payoffs()

    features = torch.zeros((game.feature_dim(),), dtype=torch.float)
    game.populate_features(features)

    player_to_act = game.get_player_to_act()
    model = regretModels[player_to_act]
    possible_actions = game.get_one_hot_actions(True)
    num_choices = possible_actions.sum()
    branch_factor_estimate = float((branch_factor_estimate * level) + (num_choices)) / (
        level + 1.0
    )
    metrics.update({"possible_actions_" + str(possible_actions.sum()): 1})
    has_a_choice = num_choices > 1
    if model is None:
        strategy = possible_actions.float()
        active_sampling_chances = None
    else:
        assert strategyModels[player_to_act] is not None
        model_regrets = model.forward_cache(features.unsqueeze(0))[0]
        model_probs = model_regrets.clamp(min=1e-3) * possible_actions.float()
        strategy = model_probs
        active_sampling_chances = (
            strategyModels[player_to_act]  # type: ignore
            .forward(features.unsqueeze(0))[0]
            .clamp(min=1e-3)
            * possible_actions.float()
        )
        active_sampling_chances_sum = float(active_sampling_chances.sum().item())

    action_dist = torch.distributions.Categorical(strategy)
    if action_dist.probs.min() < 0 or action_dist.probs.max() == 0:
        print("Invalid action dist:", action_dist.probs)
    if strategy.min() < 0 or strategy.max() == 0:
        print("Invalid strategy:", strategy)

    chance_to_sample = (
        1.0
        if level < 2
        else 1.0 - (1.0 / (100.0 ** (1.0 / ((level) ** branch_factor_estimate))))
    )
    do_sample = random.random() < chance_to_sample

    if has_a_choice and first_pass:
        strategyData.append(
            (
                features.unsqueeze(0),
                possible_actions.unsqueeze(0),
                action_dist.probs.unsqueeze(0),
            )
        )

    metrics.update({"visit_level_" + str(level): 1})
    metrics["visit"] += 1
    if metrics["visit"] % 100000 == 0:
        print("Visits", metrics["visit"])

    can_traverse = player_to_train == player_to_act
    if can_traverse and has_a_choice and do_sample:
        # print("PASSED",level,chance_to_sample)
        metrics.update({"sample_level_" + str(level): 1})
        metrics["sample"] += 1
        if metrics["sample_level_" + str(level)] % 10000 == 0:
            print(
                "Samples",
                metrics["sample"],
                metrics["sample_level_" + str(level)],
                level,
                chance_to_sample,
            )

        payoff_for_action = torch.zeros(
            (possible_actions.size()[0], game.num_players), dtype=torch.float
        )
        chosen_actions = torch.zeros_like(possible_actions)
        enum_actions = list(enumerate(possible_actions))
        random.shuffle(enum_actions)
        num_chosen = 0
        for i, a in enum_actions:
            if a == 0:
                continue
            g = game.clone()
            g.act(player_to_act, i)

            # Active sampling: https://papers.nips.cc/paper/4569-efficient-monte-carlo-counterfactual-regret-minimization-in-games-with-many-player-actions.pdf
            EPSILON = 0.05
            BONUS = 1e-6
            THRESHOLD = 1
            if active_sampling_chances is None:
                # Do Outcome sampling for the first iteration
                as_pass = num_chosen == 0
            else:
                as_pass = random.random() < float(
                    (
                        (BONUS + THRESHOLD * active_sampling_chances[i])
                        / (BONUS + active_sampling_chances_sum)
                    ).item()
                )
            if level == 0:
                # Do external sampling for the game tree root
                as_pass = True
            if True or i == 0 or random.random() < EPSILON or as_pass:
                value = traverse(
                    g,
                    player_to_train,
                    regretModels,
                    playerRegret,
                    strategyModels,
                    strategyData,
                    metrics,
                    level + 1,
                    True if first_pass and num_chosen == 0 else False,
                    branch_factor_estimate,
                )
                payoff_for_action[i] = value
                chosen_actions[i] = 1.0
                num_chosen += 1
        weighted_action_dist = torch.distributions.Categorical(
            action_dist.probs * chosen_actions.float()
        )
        assert payoff_for_action.size()[0] == weighted_action_dist.probs.size()[0]
        expected_utility = payoff_for_action * weighted_action_dist.probs.unsqueeze(1)
        assert expected_utility.size() == payoff_for_action.size()
        expected_utility_over_all_actions = expected_utility.sum(dim=0)
        playerRegret.append(
            (
                features.unsqueeze(0),
                chosen_actions.unsqueeze(0),
                (
                    payoff_for_action[:, player_to_act]
                    - expected_utility_over_all_actions[player_to_act]
                ).unsqueeze(0),
            )
        )
        assert expected_utility_over_all_actions.size() == (game.num_players,), str(
            expected_utility_over_all_actions.size()
        )
        return expected_utility_over_all_actions
    else:
        game.act(player_to_act, int(action_dist.sample().item()))
        return traverse(
            game,
            player_to_train,
            regretModels,
            playerRegret,
            strategyModels,
            strategyData,
            metrics,
            level + int(can_traverse),
            True if first_pass else False,
            branch_factor_estimate,
        )


def start_traverse(
    game: GameInterface,
    player_to_train: int,
    regretModels: List[Optional[RegretMatching]],
    strategyModels: List[Optional[RegretMatching]],
) -> Tuple[int, ExpandableTensorSet, ExpandableTensorSet, Counter]:
    NUM_INNER_GAME_ITERATIONS = 100
    with torch.no_grad():
        playerRegret = ExpandableTensorSet(
            16 * 1024, (game.feature_dim(), game.action_dim(), game.action_dim())
        )
        strategyData = ExpandableTensorSet(
            16 * 1024, (game.feature_dim(), game.action_dim(), game.action_dim())
        )
        metrics: Counter = Counter()
        for _ in range(NUM_INNER_GAME_ITERATIONS):
            ng = game.clone()
            ng.reset()
            with Profiler(False):
                traverse(
                    ng,
                    player_to_train,
                    regretModels,
                    playerRegret,
                    strategyModels,
                    strategyData,
                    metrics,
                    0,
                    True,
                    1,
                )
        # print(metrics)

    return player_to_train, playerRegret, strategyData, metrics


def train(iterations: int, game: GameInterface, output_file: str):
    NUM_GAME_ITERATIONS = 10
    playerRegrets = []
    for _ in range(game.num_players):
        playerRegrets.append(
            ReservoirBuffer(
                1024 * 1024, (game.feature_dim(), game.action_dim(), game.action_dim()),
            )
        )

    strategyData = ReservoirBuffer(
        1024 * 1024, (game.feature_dim(), game.action_dim(), game.action_dim()),
    )

    with Pool(os.cpu_count()) as gamePool:
        for iteration in range(iterations):
            print("ON ITERATION", iteration)
            regretModels: List[Optional[RegretMatching]] = []
            stratModels: List[Optional[RegretMatching]] = []
            if iteration > 0:
                for player in range(game.num_players):
                    regretModel = RegretMatching(game.feature_dim(), game.action_dim())
                    # Train the regret model
                    regretModel.train_model(
                        *playerRegrets[player].getFilled(),
                        "regret_" + str(player),
                        None
                    )

                    regretForwardModel = FullyConnectedForward(regretModel)
                    torch.save(regretForwardModel, "test.model")
                    # Throw away the data from the last iteration
                    # playerRegrets[player].reset()
                    regretModels.append(regretForwardModel)

                    stratModel = RegretMatching(game.feature_dim(), game.action_dim())
                    features, active_labels, labels = strategyData.getFilled()
                    stratModel.train_model(
                        features, active_labels, labels, "strategy_" + str(player), None
                    )
                    stratModels.append(FullyConnectedForward(stratModel))
            else:
                # Begin with random strategies
                for _ in range(game.num_players):
                    regretModels.append(None)
                    stratModels.append(None)

            with torch.no_grad():
                starts = []
                for player_to_train in range(game.num_players):
                    for game_iteration in range(NUM_GAME_ITERATIONS):
                        # print("Queueing Game", player_to_train, game_iteration)
                        new_game = game.clone()
                        new_game.reset()
                        starts.append(
                            (new_game, player_to_train, regretModels, stratModels)
                        )

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
                bestStrategy = FullyConnectedForward(stratModel)
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
