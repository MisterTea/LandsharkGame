#%%
import random
from collections import Counter
from copy import deepcopy
from enum import IntEnum
from multiprocessing import Pool
from typing import List, Optional, Tuple

import torch

from ai.reservoir_buffer import ReservoirBuffer
from engine.game import GamePhase, GameState
from utils.profiler import Profiler


class GameInterface:
    @property
    def num_players(self):
        raise NotImplementedError()

    def feature_dim(self):
        raise NotImplementedError()

    def action_dim(self):
        raise NotImplementedError()

    def terminal(self):
        raise NotImplementedError()

    def payoff(self, player):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()

    def act(self, player: int, action: int):
        raise NotImplementedError()

    def populate_features(self, features: torch.Tensor):
        raise NotImplementedError()

    def get_player_to_act(self) -> int:
        raise NotImplementedError()

    def get_one_hot_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class RpsGame(GameInterface):
    class ActionName(IntEnum):
        ROCK = 0
        PAPER = 1
        SCISSORS = 2

    def __init__(self):
        self.actions = [-1, -1]
        self.num_players = 2
        NUM_ACTIONS = len(RpsGame.ActionName)

        self.payoff_matrix = torch.zeros(
            (NUM_ACTIONS, NUM_ACTIONS, 2), dtype=torch.float
        )

        for a1 in RpsGame.ActionName:
            for a2 in RpsGame.ActionName:
                if (int(a1) + 1) % NUM_ACTIONS == int(a2):
                    self.payoff_matrix[a1, a2, 0] = -1
                    self.payoff_matrix[a1, a2, 1] = 1
                    # print(a2, "beats", a1)
                elif (int(a2) + 1) % NUM_ACTIONS == int(a1):
                    self.payoff_matrix[a1, a2, 0] = 1
                    self.payoff_matrix[a1, a2, 1] = -1
                    # print(a1, "beats", a2)
                else:
                    self.payoff_matrix[a1, a2, 0] = 0
                    self.payoff_matrix[a1, a2, 1] = 0
                    # print(a2, "ties", a1)

    def clone(self):
        return deepcopy(self)

    def action_dim(self):
        return len(RpsGame.ActionName)

    def terminal(self):
        return self.get_player_to_act() == -1

    def payoff(self, player: int):
        return self.payoff_matrix[self.actions[0], self.actions[1]][player]

    def get_one_hot_actions(self):
        return torch.ones((len(RpsGame.ActionName),), dtype=torch.float)

    def feature_dim(self):
        return 1

    def get_features(self):
        return torch.ones((self.feature_dim(),), dtype=torch.float)

    def get_player_to_act(self):
        for i, a in enumerate(self.actions):
            if a == -1:
                return i
        return -1

    def act(self, player: int, action_index: int):
        assert player == self.get_player_to_act()
        self.actions[player] = action_index


class RegretMatching(torch.nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(feature_dim, action_dim, bias=False)

    def forward(self, inputs):
        expectedRegrets = self.linear1(inputs)
        return expectedRegrets

    def train_model(self, features, active_labels, labels):
        self.train()
        regretOptimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        criterion = torch.nn.SmoothL1Loss()
        MAX_TRAIN_ITERATIONS = 1000
        last_loss = 1e10
        for train_iteration in range(MAX_TRAIN_ITERATIONS):
            regretOptimizer.zero_grad()
            train_outputs = self(features) * active_labels
            loss = criterion(train_outputs, labels)
            if train_iteration % (MAX_TRAIN_ITERATIONS // 10) == 0:
                pass
                # print("Train loss", loss.item())
            loss.backward()
            regretOptimizer.step()
            if abs(last_loss - loss.item()) < 1e-6:
                print("Stopped after", train_iteration, "iterations")
                break
            last_loss = loss.item()
        print("Train loss", loss.item())
        self.eval()


class TensorData:
    def __init__(self, capacity: int, feature_dim: int, label_dim: int):
        self.filled = 0
        self.features = torch.zeros((capacity, feature_dim), dtype=torch.float)
        self.active_labels = torch.zeros((capacity, label_dim), dtype=torch.float)
        self.labels = torch.zeros((capacity, label_dim), dtype=torch.float)

    @property
    def capacity(self):
        return self.features.size()[0]

    def append(
        self, features: torch.Tensor, active_labels: torch.Tensor, labels: torch.Tensor,
    ):
        assert len(features.size()) == 2
        assert len(active_labels.size()) == 2
        assert len(labels.size()) == 2
        num_new_features = features.size()[0]
        while (self.filled + num_new_features) > self.features.size()[0]:
            self.features = torch.cat(
                (self.features, torch.zeros_like(self.features)), dim=0
            )
            self.active_labels = torch.cat(
                (self.active_labels, torch.zeros_like(self.active_labels)), dim=0
            )
            self.labels = torch.cat((self.labels, torch.zeros_like(self.labels)), dim=0)
            print("Expanding training data to", self.features.size()[0])
        self.features[self.filled : self.filled + num_new_features] = features
        if active_labels is not None:
            self.active_labels[
                self.filled : self.filled + num_new_features
            ] = active_labels
        self.labels[self.filled : self.filled + num_new_features] = labels
        self.filled += num_new_features

    def cat(self, other):
        self.append(other.features, other.active_labels, other.labels)

    def getFilled(self):
        return (
            self.features[0 : self.filled],
            self.active_labels[0 : self.filled],
            self.labels[0 : self.filled],
        )

    def reset(self):
        self.filled = 0
        self.features.fill_(0.0)
        self.labels.fill_(0.0)
        self.active_labels.fill_(0.0)


def traverse(
    game: GameInterface,
    player_to_train: int,
    regretModels: List[Optional[RegretMatching]],
    playerRegret: ReservoirBuffer,
    strategyData: ReservoirBuffer,
    metrics: Counter,
    level: int,
) -> float:
    if game.terminal():
        return game.payoff(player_to_train)

    metrics.update({"visit_level_" + str(level): 1})

    features = torch.zeros((game.feature_dim(),), dtype=torch.float)
    game.populate_features(features)

    player_to_act = game.get_player_to_act()
    model = regretModels[player_to_act]
    possible_actions = game.get_one_hot_actions()
    metrics.update({"possible_actions_" + str(possible_actions.sum()): 1})
    has_a_choice = possible_actions.sum() > 1
    if model is None:
        strategy = possible_actions.float()
    else:
        model_regrets = model(features.unsqueeze(0)).detach()[0]
        model_probs = model_regrets.clamp(min=1e-6) * possible_actions.float()
        strategy = model_probs

    action_dist = torch.distributions.Categorical(strategy)
    if (
        player_to_act == player_to_train
        and has_a_choice
        # and random.random() > 0.8
        # and playerRegret.filled * 1.1 < playerRegret.capacity
    ):
        payoff_for_action = torch.zeros_like(possible_actions, dtype=torch.float)
        chosen_actions = torch.zeros_like(possible_actions)
        for i, a in enumerate(possible_actions):
            if a == 0:
                continue
            g = game.clone()
            g.act(player_to_act, i)
            if random.random() < 0.1 or random.random() < action_dist.probs[i]:
                payoff_for_action[i] = traverse(
                    g,
                    player_to_train,
                    regretModels,
                    playerRegret,
                    strategyData,
                    metrics,
                    level + 1,
                )
                chosen_actions[i] = 1.0
        expected_utility = (payoff_for_action * action_dist.probs).sum().item()
        playerRegret.append(
            (
                features.unsqueeze(0),
                chosen_actions.unsqueeze(0),
                (payoff_for_action - expected_utility).unsqueeze(0),
            )
        )
        return expected_utility
    else:
        if has_a_choice and player_to_act != player_to_train:
            strategyData.append(
                (
                    features.unsqueeze(0),
                    possible_actions.unsqueeze(0),
                    action_dist.probs.unsqueeze(0),
                )
            )
        game.act(player_to_act, int(action_dist.sample().item()))
        return traverse(
            game,
            player_to_train,
            regretModels,
            playerRegret,
            strategyData,
            metrics,
            level + (1 if player_to_act == player_to_train else 0),
        )


def start_traverse(
    game: GameInterface,
    player_to_train: int,
    regretModels: List[Optional[RegretMatching]],
) -> Tuple[int, ReservoirBuffer, ReservoirBuffer, Counter]:
    game.reset()
    playerRegret = ReservoirBuffer(
        16 * 1024, (game.feature_dim(), game.action_dim(), game.action_dim())
    )
    strategyData = ReservoirBuffer(
        16 * 1024, (game.feature_dim(), game.action_dim(), game.action_dim())
    )
    metrics: Counter = Counter()
    with Profiler(False):
        traverse(
            game, player_to_train, regretModels, playerRegret, strategyData, metrics, 0
        )
    # print(metrics)

    return player_to_train, playerRegret, strategyData, metrics


def train(iterations: int, game: GameInterface, output_file: str):
    NUM_GAME_ITERATIONS = 100
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

    with Pool() as gamePool:
        for iteration in range(iterations):
            print("ON ITERATION", iteration)
            regretModels: List[Optional[RegretMatching]] = []
            if iteration > 0:
                for player in range(game.num_players):
                    regretModel = RegretMatching(game.feature_dim(), game.action_dim())
                    # Train the regret model
                    regretModel.train_model(*playerRegrets[player].getFilled())
                    # Throw away the data from the last iteration
                    # playerRegrets[player].reset()
                    regretModels.append(regretModel)
            else:
                # Begin with random strategies
                for _ in range(game.num_players):
                    regretModels.append(None)

            starts = []
            for player_to_train in range(game.num_players):
                for game_iteration in range(NUM_GAME_ITERATIONS):
                    # print("Queueing Game", player_to_train, game_iteration)
                    new_game = game.clone()
                    starts.append((new_game, player_to_train, regretModels))

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
                    new_player_regrets,
                    new_strategy_data,
                    new_metrics,
                ) = result
                playerRegrets[player_to_train].cat(new_player_regrets)
                strategyData.cat(new_strategy_data)
                metrics.update(new_metrics)
            print(metrics)

            if (iteration + 1) % 1 == 0:
                stratModel = RegretMatching(game.feature_dim(), game.action_dim())
                stratModel.train_model(*strategyData.getFilled())
                bestStrategy = stratModel
                # torch.nn.Softmax(dim=0)(
                #     stratModel(strategyData.features[0:1])[0].detach()
                # )
                # print(
                #     "Learned Strategy at " + str(iteration) + ": ", str(bestStrategy),
                # )
                # print("***")
                torch.save(bestStrategy, output_file)

                # Check winrate against random player
                scoreCounter: Counter = Counter()
                for _ in range(1000):
                    gameState = GameState(4)
                    features = torch.zeros(
                        (1, gameState.feature_dim()), dtype=torch.float
                    )
                    while gameState.phase != GamePhase.GAME_OVER:
                        seatToAct = gameState.get_players_to_act()[0]
                        possible_actions = gameState.getPossibleActions(seatToAct)
                        if seatToAct == 0:
                            gameState.populate_features(features[0])
                            action_probs = bestStrategy(features).detach()[0]
                            possible_action_mask = gameState.get_one_hot_actions(False)
                            action_index = int(
                                torch.distributions.Categorical(
                                    torch.nn.Softmax(dim=0)(action_probs)
                                    * possible_action_mask
                                )
                                .sample()
                                .item()
                            )
                            gameState.act(seatToAct, action_index)
                        else:
                            action = random.choice(possible_actions)
                            gameState.playerAction(seatToAct, action)
                    aiScore = gameState.playerStates[0].getScore()
                    place = 1
                    for i, playerState in enumerate(gameState.playerStates):
                        if i == 0:
                            continue
                        if playerState.getScore() > aiScore:
                            place += 1
                    scoreCounter[place] += 1
                print("SCORE AGAINST RANDOM")
                print(scoreCounter)

    stratModel = RegretMatching(game.feature_dim(), game.action_dim())
    stratModel.train_model(*strategyData.getFilled())
    return stratModel


# bestStrategy = train(100, RpsGame())
# print("FINAL STRAT: ", bestStrategy)
