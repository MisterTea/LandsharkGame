#%%
import random
from collections import Counter
from copy import deepcopy
from enum import IntEnum
from multiprocessing import Pool
from typing import List, Optional, Tuple
import hashlib
import os

import torch
import pytorch_lightning as pl
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

    def get_one_hot_actions(self, hacks=True) -> torch.Tensor:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class RpsGame(GameInterface):
    class ActionName(IntEnum):
        ROCK = 0
        PAPER = 1
        SCISSORS = 2
        LOSE = 3

    def __init__(self):
        NUM_ACTIONS = len(RpsGame.ActionName)

        self.payoff_matrix = torch.zeros(
            (NUM_ACTIONS, NUM_ACTIONS, 2), dtype=torch.float
        )

        for a1 in RpsGame.ActionName:
            for a2 in RpsGame.ActionName:
                if (int(a1) + 1) % 3 == int(a2):
                    self.payoff_matrix[a1, a2, 0] = -1
                    self.payoff_matrix[a1, a2, 1] = 1
                    # print(a2, "beats", a1)
                elif (int(a2) + 1) % 3 == int(a1):
                    self.payoff_matrix[a1, a2, 0] = 1
                    self.payoff_matrix[a1, a2, 1] = -1
                    # print(a1, "beats", a2)
                else:
                    self.payoff_matrix[a1, a2, 0] = 0
                    self.payoff_matrix[a1, a2, 1] = 0
                    # print(a2, "ties", a1)

                if a1 == RpsGame.ActionName.LOSE and a2 == RpsGame.ActionName.LOSE:
                    self.payoff_matrix[a1,a2,:] = -10
                elif a1 == RpsGame.ActionName.LOSE:
                    self.payoff_matrix[a1,a2,0] = -10
                    self.payoff_matrix[a1,a2,1] = 1
                elif a2 == RpsGame.ActionName.LOSE:
                    self.payoff_matrix[a1,a2,0] = 1
                    self.payoff_matrix[a1,a2,1] = -10

        self.reset()

    @property
    def num_players(self):
        return 2

    def clone(self):
        return deepcopy(self)

    def reset(self):
        self.actions = [-1] * self.num_players

    def action_dim(self):
        return len(RpsGame.ActionName)

    def terminal(self):
        return self.get_player_to_act() == -1

    def payoffs(self):
        return self.payoff_matrix[self.actions[0], self.actions[1]]

    def get_one_hot_actions(self, hacks=True):
        return torch.ones((len(RpsGame.ActionName),), dtype=torch.float)

    def populate_features(self, features:torch.Tensor):
        features.fill_(1.0)

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


class RegretMatchingForward(torch.nn.Module):
    def __init__(self, backModule):
        super().__init__()
        self.layers = backModule.layers
        self.activations = backModule.activations
        #self.forward = backModule.forward

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if self.activations[i] is not None:
                x = self.activations[i](x)
        return x

class RegretMatching(pl.LightningModule):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        #self.layers = torch.nn.ModuleList([
        #    torch.nn.Linear(feature_dim, 128),
        #    torch.nn.Linear(128, 64),
        #    torch.nn.Linear(64, action_dim),
        #    ])
        #self.activations = torch.nn.ModuleList([
        #    torch.nn.ReLU(),
        #    torch.nn.ReLU(),
        #    None,
        #])
        self.layers = torch.nn.ModuleList([torch.nn.Linear(feature_dim, action_dim),])
        self.activations = torch.nn.ModuleList([None])
        #self.forward_cache_dict = {}

    def forward_cache(self, inputs):
        input_key = hashlib.blake2b(inputs.numpy().tobytes()).digest()
        labels = self.forward_cache_dict.get(input_key, None)
        if labels is None:
            x = self.forward(inputs)
            self.forward_cache_dict[input_key] = x
            return x
        else:
            return labels

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if self.activations[i] is not None:
                x = self.activations[i](x)
        return x

    def train_model(self, features, active_labels, labels, model_name, output_file=None):
        full_dataset = torch.utils.data.TensorDataset(features, active_labels, labels)
        dataset_size = len(full_dataset)
        test_size = dataset_size // 5
        train_size = dataset_size - test_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        print("TRAINING ON",dataset_size, len(self.train_dataset), len(self.val_dataset))
        trainer = pl.Trainer(early_stop_callback=True, max_epochs=1000, default_save_path=os.path.join(os.getcwd(), 'models', model_name))
        trainer.fit(self)
        if output_file is not None:
            trainer.save_checkpoint(output_file)
        self.train_dataset = self.val_dataset = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=1024, shuffle=True, drop_last=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=1024*1024*1024, drop_last=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        features, active_labels, labels = batch
        outputs = self(features) * active_labels.float()
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(outputs, labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        train_results = self.training_step(batch, batch_idx)
        return {'val_loss': train_results["loss"]}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_model_(self, features, active_labels, labels):
        self.forward_cache_dict.clear()
        self.train()
        regretOptimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = torch.nn.SmoothL1Loss()
        MAX_TRAIN_ITERATIONS = 10000
        last_loss = 1e10
        for train_iteration in range(MAX_TRAIN_ITERATIONS):
            regretOptimizer.zero_grad()
            train_outputs = self(features.detach().requires_grad_()) * active_labels.detach().requires_grad_()
            #assert (labels[active_labels == 0]).sum() < 1e-6
            loss = criterion(train_outputs, labels)
            if train_iteration % (MAX_TRAIN_ITERATIONS // 10) == 0:
                pass
                # print("Train loss", loss.item())
            loss.backward()
            regretOptimizer.step()
            if abs(last_loss - loss.item()) < 1e-7 and train_iteration > 1000:
                print("Stopped after", train_iteration, "iterations")
                break
            last_loss = loss.item()
        print("Train loss", loss.item())
        self.eval()


class TensorData:
    def __init__(self, capacity: int, dims:List[int]):
        self.filled = 0
        self.features = torch.zeros((capacity, dims[0]), dtype=torch.float)
        self.active_labels = torch.zeros((capacity, dims[1]), dtype=torch.float)
        self.labels = torch.zeros((capacity, dims[2]), dtype=torch.float)

    @property
    def capacity(self):
        return self.features.size()[0]

    def append(
        self, tensors:List[torch.Tensor]
    ):
        features, active_labels, labels = tensors
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
    playerRegret: TensorData,
    strategyData: TensorData,
    metrics: Counter,
    level: int,
    first_pass: bool
) -> torch.Tensor:
    if game.terminal():
        return game.payoffs()

    metrics.update({"visit_level_" + str(level): 1})
    metrics["visit"] += 1
    if metrics["visit"] % 10000 == 0:
        print("Visits",metrics["visit"])

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
        model_regrets = model.forward(features.unsqueeze(0))[0]
        model_probs = model_regrets.clamp(min=1e-6) * possible_actions.float()
        strategy = model_probs

    action_dist = torch.distributions.Categorical(strategy)

    if has_a_choice and first_pass:
        strategyData.append(
            (
                features.unsqueeze(0),
                possible_actions.unsqueeze(0),
                action_dist.probs.unsqueeze(0),
            )
        )

    if player_to_train == player_to_act and has_a_choice:
        payoff_for_action = torch.zeros_like(possible_actions, dtype=torch.float)
        #chosen_actions = torch.zeros_like(possible_actions)
        enum_actions = list(enumerate(possible_actions))
        random.shuffle(enum_actions)
        num_chosen = 0
        expected_value = torch.zeros((game.num_players,), dtype=torch.float)
        for i, a in enum_actions:
            if a == 0:
                continue
            g = game.clone()
            g.act(player_to_act, i)
            if True:
                value = traverse(
                    g,
                    player_to_train,
                    regretModels,
                    playerRegret,
                    strategyData,
                    metrics,
                    level + 1,
                    True if first_pass and num_chosen == 0 else False
                )
                expected_value += value * action_dist.probs[i]
                payoff_for_action[i] = value[player_to_act]
                #chosen_actions[i] = 1.0
                num_chosen += 1
        #weighted_action_dist = torch.distributions.Categorical(action_dist.probs * chosen_actions.float())
        expected_utility = (payoff_for_action * action_dist.probs).sum().item()
        playerRegret.append(
            (
                features.unsqueeze(0),
                #chosen_actions.unsqueeze(0),
                possible_actions.unsqueeze(0),
                (payoff_for_action - expected_utility).unsqueeze(0),
            )
        )
        return expected_value
    else:
        game.act(player_to_act, int(action_dist.sample().item()))
        return traverse(
            game,
            player_to_train,
            regretModels,
            playerRegret,
            strategyData,
            metrics,
            level + 1,
            True if first_pass else False
        )


def start_traverse(
    game: GameInterface,
    player_to_train:int,
    regretModels: List[Optional[RegretMatching]],
) -> Tuple[int, TensorData, TensorData, Counter]:
    NUM_INNER_GAME_ITERATIONS=100
    with torch.no_grad():
        playerRegret = TensorData(
                16 * 1024, (game.feature_dim(), game.action_dim(), game.action_dim())
            )
        strategyData = TensorData(
            1024, (game.feature_dim(), game.action_dim(), game.action_dim())
        )
        metrics: Counter = Counter()
        for _ in range(NUM_INNER_GAME_ITERATIONS):
            ng = game.clone()
            ng.reset()
            with Profiler(False):
                traverse(
                    ng, player_to_train, regretModels, playerRegret, strategyData, metrics, 0, True
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

    with Pool(os.cpu_count()//2) as gamePool:
        for iteration in range(iterations):
            print("ON ITERATION", iteration)
            regretModels: List[Optional[RegretMatching]] = []
            if iteration > 0:
                for player in range(game.num_players):
                    regretModel = RegretMatching(game.feature_dim(), game.action_dim())
                    # Train the regret model
                    regretModel.train_model(*playerRegrets[player].getFilled(), "regret_"+str(player), None)

                    regretForwardModel = RegretMatchingForward(regretModel)
                    torch.save(regretForwardModel, "test.model")
                    # Throw away the data from the last iteration
                    # playerRegrets[player].reset()
                    regretModels.append(regretForwardModel)
            else:
                # Begin with random strategies
                for _ in range(game.num_players):
                    regretModels.append(None)

            with torch.no_grad():
                starts = []
                for player_to_train in range(game.num_players):
                    for game_iteration in range(NUM_GAME_ITERATIONS):
                        # print("Queueing Game", player_to_train, game_iteration)
                        new_game = game.clone()
                        new_game.reset()
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
                stratModel.train_model(*strategyData.getFilled(), "strategy", output_file)
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
                    average_decision = torch.zeros((game.action_dim(),), dtype=torch.float)
                    for on_game in range(NUM_RANDOM_GAMES):
                        gameState = game.clone()
                        gameState.reset()
                        features = torch.zeros(
                            (1, gameState.feature_dim()), dtype=torch.float
                        )
                        while not gameState.terminal():#gameState.phase != GamePhase.GAME_OVER:
                            seatToAct = gameState.get_player_to_act()
                            possible_action_mask = gameState.get_one_hot_actions(False)
                            if seatToAct == 0:
                                gameState.populate_features(features[0])
                                action_probs = bestStrategy(features).detach()[0].clamp(min=1e-6)
                            else:
                                action_probs = possible_action_mask.float()
                            action_prob_dist = torch.distributions.Categorical(
                                    action_probs
                                    * possible_action_mask
                                )
                            if on_game == 0:
                                print("ACTION",action_prob_dist.probs)
                            action_index = int(
                                action_prob_dist
                                .sample()
                                .item()
                            )
                            average_decision[action_index] += 1.0
                            num_decisions += 1
                            gameState.act(seatToAct, action_index)
                        payoffs = gameState.payoffs()
                        for i,p in enumerate(payoffs):
                            scoreCounter[str(i)] += p
                    print("DECISION HISTOGRAM")
                    print(average_decision / num_decisions)
                    print("SCORE AGAINST RANDOM")
                    for x in range(gameState.num_players):
                        print(x,scoreCounter[str(x)] / float(NUM_RANDOM_GAMES))

    stratModel = RegretMatching(game.feature_dim(), game.action_dim())
    stratModel.train_model(*strategyData.getFilled())
    return stratModel


# bestStrategy = train(100, RpsGame())
# print("FINAL STRAT: ", bestStrategy)
