#!/usr/bin/env python3

import copy
import os
import random
from collections import Counter
from enum import IntEnum
from typing import Dict, List, Tuple
from uuid import UUID, uuid4

import numpy as np
import pytorch_lightning as pl
import torch
import torch_optimizer
from engine.game_interface import GameInterface
from torch import multiprocessing
from utils.profiler import Profiler

from ai.types import GameRollout


def masked_softmax(x, mask, temperature):
    x = x / temperature
    x[mask < 1] = -1e20
    x = x - torch.max(x, dim=1, keepdim=True)[0]
    e_x = torch.exp(x)
    e_x = e_x * mask
    softmax_sum = e_x.sum(dim=1, keepdim=True)
    softmax_sum[softmax_sum == 0.0] = 1.0
    out = e_x / softmax_sum
    return out


class ActorCritic(torch.nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.player_property_embedding = torch.nn.Embedding(31, 16)

        self.player_embedding = torch.nn.Sequential(
            torch.nn.Linear(18, 16), torch.nn.LeakyReLU()
        )

        self.bidding_property_embedding = torch.nn.Embedding(31, 16)
        self.property_consumed_embedding = torch.nn.Embedding(31, 16)

        self.bidding_dollar_embedding = torch.nn.Embedding(17, 16)
        self.dollar_consumed_embedding = torch.nn.Embedding(17, 16)

        self.shared = torch.nn.Sequential(
            (torch.nn.Linear(98, 64)),
            torch.nn.LeakyReLU(),
        )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(64, action_dim),
            torch.nn.Identity(),
        )

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(64, action_dim),
            torch.nn.Identity(),
        )

        self.num_steps = 0

    def forward(self, inputs, possible_actions, do_epsilon_greedy: bool):
        inputs = inputs.detach()

        # Embed player properties
        opponent_vector = None
        for player_index in range(0, 4):
            cursor = 1 + 4 + 30 + 30 + 1 + 4 + (9 * player_index)
            property_indices = inputs[:, cursor + 1 : cursor + 8].long()
            e = (self.player_property_embedding(property_indices)).mean(dim=1)

            player_embedding = self.player_embedding(
                torch.cat(
                    (
                        e,
                        inputs[
                            :,
                            (cursor, cursor + 8),
                        ],
                    ),
                    dim=1,
                )
            )
            if player_index == 0:
                self_vector = player_embedding
            elif opponent_vector is None:
                opponent_vector = player_embedding
            else:
                opponent_vector = opponent_vector + player_embedding

        # Embed houses to buy
        cursor = 1
        property_indices = inputs[:, cursor : cursor + 4].long()
        houses_to_buy = (self.bidding_property_embedding(property_indices)).mean(dim=1)

        # Embed properties consumed
        cursor = 1 + 4
        property_indices = inputs[:, cursor : cursor + 30].long()
        property_consumed = (self.property_consumed_embedding(property_indices)).mean(
            dim=1
        )

        # Embed dollar cards to buy
        cursor = 1 + 4 + 30 + 30 + 1
        dollar_indices = inputs[:, cursor : cursor + 4].long()
        dollars_to_buy = (self.bidding_dollar_embedding(dollar_indices)).mean(dim=1)

        # Embed dollars consumed
        cursor = 1 + 4 + 30
        dollar_indices = inputs[:, cursor : cursor + 30].long()
        dollars_consumed = (self.dollar_consumed_embedding(dollar_indices)).mean(dim=1)

        x = torch.cat(
            (
                inputs[:, 0:1],
                houses_to_buy,
                property_consumed,
                inputs[:, 1 + 4 + 30 + 30 : 1 + 4 + 30 + 30 + 1],
                dollars_to_buy,
                dollars_consumed,
                self_vector,
                opponent_vector,
            ),
            dim=1,
        )

        x = self.shared(x)

        shared_result = x
        # Critic
        critic_action_values = self.critic(x)

        # Actor
        x = self.actor(shared_result)
        # x_before_activation = x.cpu()
        # x = torch.clamp(torch.nn.Sigmoid()(x), min=1e-3, max=(1.0 - 1e-3))
        # assert torch.min(x) > 0.0

        x = masked_softmax(x, possible_actions, 1.0)
        # Replace softmax with linear scale
        # x = x * possible_actions.float()
        # original_x = x.cpu()
        # x = x / x.sum(dim=1, keepdim=True)
        assert torch.allclose(
            torch.max(possible_actions, dim=1).values.cpu(), torch.IntTensor([1])
        ), f"{torch.max(possible_actions, dim=1).values.cpu()}"
        assert torch.allclose(x.sum(dim=1).cpu(), torch.Tensor([1.0]))
        assert torch.min(x) >= 0.0

        if do_epsilon_greedy:
            epsilon_prob = possible_actions * (
                0.1 / possible_actions.sum(dim=1, keepdim=True).float()
            )
            assert epsilon_prob.size() == x.size()
            for _ in range(5):
                x = torch.max(x, epsilon_prob)
                x = x / x.sum(dim=1, keepdim=True)
        actor_probs = x

        return actor_probs, critic_action_values

    def training_step(self, batch_list: List[torch.Tensor], batch_idx):
        self.num_steps += 1
        batch = GameRollout(*[x[0] for x in batch_list])
        actor_probs, critic_action_values = self(
            batch.states, batch.possible_actions, True
        )

        action_one_hot = torch.nn.functional.one_hot(
            batch.actions.squeeze(1), num_classes=self.action_dim
        ).type_as(batch.states)
        payoff_for_player_to_act = (
            (
                batch.payoffs
                * torch.nn.functional.one_hot(batch.player_to_act.squeeze(1))
            )
            .sum(dim=1, keepdim=True)
            .type_as(batch.states)
        )
        batch_size = batch.actions.size()[0]
        # labels = (0.99 ** batch.distance_to_payoff) * payoff_for_player_to_act
        labels = payoff_for_player_to_act
        assert labels.size() == (batch_size, 1)
        labels = (labels * action_one_hot).sum(dim=1, keepdim=True)
        assert action_one_hot.size() == batch.possible_actions.size()

        total_prob = actor_probs.sum(dim=1)

        outputs = (critic_action_values * action_one_hot).sum(dim=1, keepdim=True)
        criterion = torch.nn.L1Loss(reduction="none")

        # Some actions are impossible, these will get an importance weight of 0
        importance_weight = (actor_probs / batch.policy.clamp(min=1e-6)).clamp(max=10.0)
        assert (
            torch.isnan(importance_weight).sum() == 0
        ), f"Invalid importance weight {actor_probs}"

        importance_weight_for_action_taken = (
            (importance_weight * action_one_hot).sum(dim=1, keepdim=True).detach()
        )
        loss_before_importance = criterion(outputs, labels)
        assert (
            importance_weight_for_action_taken.size() == loss_before_importance.size()
        )
        critic_loss = (
            loss_before_importance * importance_weight_for_action_taken
        ).mean()

        # print("Critic loss", critic_loss)

        if True or self.num_steps >= 100:
            detached_critic = critic_action_values.detach()
            assert critic_action_values.size() == batch.possible_actions.size()

            baseline = (actor_probs * detached_critic).sum(dim=1, keepdim=True)
            advantage = detached_critic - baseline
            advantage_loss = -1 * (
                importance_weight.detach() * actor_probs * advantage.detach()
            ).sum(dim=1, keepdim=True)

            # advantage_loss = -1 * (
            #     importance_weight.detach() * actor_probs * detached_critic
            # ).sum(dim=1, keepdim=True)
            # advantage_loss = torch.nn.LeakyReLU()(advantage).sum(dim=1, keepdim=True)
            advantage_loss = advantage_loss.mean()

            entropy = (
                torch.sum(
                    -actor_probs * torch.log(actor_probs.clamp(min=1e-6)),
                    dim=1,
                    keepdim=True,
                )
                / torch.log((actor_probs > 0).sum(dim=1, keepdim=True).float()).clamp(
                    min=1e-6
                )
            )
            assert (
                (entropy > 1.01).sum() + (entropy < -0.01).sum()
            ) == 0, f"Invalid entropy {torch.min(torch.sum(actor_probs, dim=1))}, {torch.max(torch.sum(actor_probs, dim=1))}, {entropy[entropy > 1.0]}, {entropy[entropy < 0.0]}"
            entropy_loss = (
                torch.nn.L1Loss()(entropy, torch.ones_like(entropy).float()) * 0.1
            )

            actor_loss = advantage_loss + entropy_loss
            # print("Actor losses", loss, entropy_loss)

            advantage_loss = advantage_loss.detach()
            entropy_loss = entropy_loss.detach()
        else:
            # Don't bother training actor while critic is so wrong
            actor_loss = advantage_loss = entropy_loss = 0

        return {
            "progress_bar": {
                "advantage_loss": advantage_loss,
                "entropy_loss": entropy_loss,
                "critic_loss": critic_loss.detach(),
                "actor_loss": actor_loss.detach(),
            },
            "loss": actor_loss + critic_loss,
        }


@torch.no_grad()
def test_actor(game, actor, opponent_policy, current_epoch):
    scoreCounter: Counter = Counter()
    NUM_RANDOM_GAMES = 1000
    num_decisions = 0
    average_decision = torch.zeros((game.action_dim(),), dtype=torch.float)
    for on_game in range(NUM_RANDOM_GAMES):
        gameState = game.clone()
        gameState.reset()
        features = torch.zeros((1, gameState.feature_dim()), dtype=torch.float)
        while not gameState.terminal():  # gameState.phase != GamePhase.GAME_OVER:
            seatToAct = gameState.get_player_to_act()
            possible_action_mask = gameState.get_one_hot_actions(True)
            if seatToAct == 0:
                gameState.populate_features(features[0])
                action_probs = actor(
                    features, possible_action_mask.unsqueeze(0), False
                )[0]
            elif opponent_policy is not None:
                gameState.populate_features(features[0])
                action_probs = opponent_policy(
                    features, possible_action_mask.unsqueeze(0), False
                )[0]
            else:
                action_probs = possible_action_mask.float()
            action_prob_dist = torch.distributions.Categorical(
                action_probs * possible_action_mask
            )
            action_index = int(action_prob_dist.sample().item())
            if seatToAct == 0:
                average_decision[action_index] += 1.0
                num_decisions += 1
            gameState.act(seatToAct, action_index)
        payoffs = gameState.payoffs()
        for i, p in enumerate(payoffs):
            scoreCounter[str(i)] += p
    print(f"TESTING EPOCH {current_epoch}")
    print("DECISION HISTOGRAM")
    print(average_decision / num_decisions)
    print(f"SCORE AGAINST PLAYER")
    for x in range(gameState.num_players):
        print(x, scoreCounter[str(x)] / float(NUM_RANDOM_GAMES))


def test_actors(game, epoch: int, actor_critics):
    p = multiprocessing.Pool(8)
    for i, actor_critic in enumerate(actor_critics):
        actor = copy.deepcopy(actor_critic).cpu().eval()

        with torch.no_grad():
            test_actor_params = []
            for current_epoch in range(-1, epoch + 1, max(1, epoch // 5)):
                # Check winrate against random or past player
                opponent_policy = None
                if current_epoch >= 0:
                    opponent_policy = (
                        torch.load(f"models/MAC_ActorCritic_{current_epoch}_{0}.torch")
                        .cpu()
                        .eval()
                    )
                test_actor_params.append([game, actor, opponent_policy, current_epoch])
            p.starmap(test_actor, test_actor_params, 1)
    p.terminate()


class TorchSaveCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # for i, actor_critic in enumerate(pl_module.actor_critics):
        # torch.save(actor_critic, f"models/MAC_ActorCritic_{epoch}_{i}.torch")
        # test_actors(pl_module.game, epoch, pl_module.actor_critics)
        os.makedirs(f"{trainer.logger.log_dir}/models", exist_ok=True)
        torch.save(
            pl_module.value, f"{trainer.logger.log_dir}/models/StateValue_{epoch}.torch"
        )
        torch.save(
            pl_module.policy, f"{trainer.logger.log_dir}/models/Policy_{epoch}.torch"
        )


GotException: bool = False


def has_exception():
    global GotException
    return GotException


class ExitOnExceptionCallback(pl.Callback):
    def on_exception(self, trainer, pl_module, exception):
        print("GOT EXCEPTION")
        print(exception)
        global GotException
        GotException = True


NUM_PARALLEL_MODELS = 1


class MeanActorCritic(pl.LightningModule):
    def __init__(self, game: GameInterface):
        super().__init__()
        self.game = game
        self.actor_critics = torch.nn.ModuleList(
            [
                ActorCritic(game.feature_dim(), game.action_dim())
                for _ in range(NUM_PARALLEL_MODELS)
            ]
        )
        self.learning_rate = 1e-3
        # test_actors(self.game, self.current_epoch, self.actor_critics)

    def forward(self, inputs):
        self.actor_critics[0].forward(inputs)

    def train_model(self, train_dataset, output_file=None):
        self.train_dataset = train_dataset
        trainer = pl.Trainer(
            gpus=1,
            # show_progress_bar=False,
            max_epochs=1000,
            # default_save_path=os.path.join(os.getcwd(), "models", "MAC"),
            val_check_interval=train_dataset.max_games,
            callbacks=[TorchSaveCallback()],
            # auto_lr_find=True,
            num_sanity_val_steps=0,
        )
        with Profiler(True):
            trainer.fit(self)
        if output_file is not None:
            trainer.save_checkpoint(output_file)
        self.train_dataset = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, pin_memory=True)

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(x.parameters(), lr=(self.learning_rate), weight_decay=1e-5)
            for x in self.actor_critics
        ]
        return optimizers

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        retval = self.actor_critics[optimizer_idx].training_step(batch, batch_idx)
        if "progress_bar" in retval:
            self.log(
                "critic_loss",
                retval["progress_bar"]["critic_loss"],
                prog_bar=True,
                on_step=True,
            )
            self.log(
                "advantage_loss",
                retval["progress_bar"]["advantage_loss"],
                prog_bar=True,
                on_step=True,
            )
            self.log(
                "entropy_loss",
                retval["progress_bar"]["entropy_loss"],
                prog_bar=True,
                on_step=True,
            )
        return retval


class StateValueModel(torch.nn.Module):
    def __init__(self, feature_dim: int, num_players: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_players = num_players

        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Identity(),
        )

    def get_value_logits(self, inputs):
        inputs = inputs.detach()

        x = self.trunk(inputs)
        return x

    def forward(self, inputs):
        torch.nn.Softmax(dim=1)(self.get_value_logits(inputs))

    def get_action(self, game: GameInterface, features: torch.Tensor) -> int:
        player_to_act = game.get_player_to_act()
        actions = game.getPossibleActions()
        scores = torch.zeros((len(actions)), dtype=torch.float)
        for i, action in enumerate(actions):
            g = copy.deepcopy(game)
            g.act(player_to_act, action)
            g.populate_features(features)
            scores[i] = self.get_value_logits(features)[0][0]
        # probs = torch.nn.Softmax()(scores)
        action_to_play = actions[
            int(torch.distributions.Categorical(logits=scores).sample().item())
        ]
        return action_to_play

    def get_loss(self, batch_list, batch_idx):
        batch = GameRollout(*[x[0] for x in batch_list])
        state_values = self.get_value_logits(batch.states)

        player_to_act_wins = (torch.argmax(batch.payoffs, 1, keepdim=True) == 0).float()
        # winners = torch.argmax(batch.payoffs, 1, keepdim=True)
        # loss = torch.nn.CrossEntropyLoss()(state_values, winners)
        loss = torch.nn.BCEWithLogitsLoss()(state_values, player_to_act_wins)

        return state_values, loss

    def training_step(self, batch_list: List[torch.Tensor], batch_idx):
        loss = self.get_loss(batch_list, batch_idx)[1]
        return {
            "loss": loss,
        }

    def validation_step(self, batch_list: List[torch.Tensor], batch_idx):
        batch = GameRollout(*[x[0] for x in batch_list])
        state_values, loss = self.get_loss(batch_list, batch_idx)
        player_to_act_wins = (torch.argmax(batch.payoffs, 1, keepdim=True) == 0).bool()
        win_probs = torch.nn.Sigmoid()(state_values)

        model_targets = (win_probs > 0.5).bool()

        true_positive = (
            torch.logical_and(model_targets, player_to_act_wins).long().sum().item()
        )
        # true_negative = torch.logical_and(torch.logical_not(model_targets), torch.logical_not((player_to_act_wins))).long().sum().item()

        false_positive = (
            torch.logical_and(model_targets, torch.logical_not((player_to_act_wins)))
            .long()
            .sum()
            .item()
        )
        false_negative = (
            torch.logical_and(torch.logical_not(model_targets), player_to_act_wins)
            .long()
            .sum()
            .item()
        )
        # print(
        #     f"Precision: {float(true_positive) / (float(true_positive) + float(false_positive))} Recall: {float(true_positive) / (float(true_positive) + float(false_negative))}"
        # )
        # print(
        #     f"Precision: {float(true_positive) / (float(true_positive) + float(false_positive))} Recall: {float(true_positive) / (float(true_positive) + float(false_negative))}"
        # )

        return {
            "val_loss": loss,
            "tp": float(true_positive),
            "precision": float(true_positive)
            / (float(true_positive) + float(false_positive)),
            "recall": float(true_positive)
            / (float(true_positive) + float(false_negative)),
        }


class ImitationLearningModel(torch.nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, action_dim),
            torch.nn.Identity(),
        )

    def get_logits(self, inputs, possible_action_mask):
        inputs = inputs.detach()
        assert len(inputs.shape) == 2

        x = self.trunk(inputs)

        x[torch.logical_not(possible_action_mask)] = float("-inf")
        return x

    def forward(self, inputs, possible_action_mask):
        torch.nn.Softmax(dim=1)(self.get_logits(inputs, possible_action_mask))

    def get_action(self, game: GameInterface, features: torch.Tensor) -> int:
        possible_actions = game.get_one_hot_actions().unsqueeze(0)
        game.populate_features(features)
        features = features.unsqueeze(0)
        logits = self.get_logits(features, possible_actions)
        action_taken = torch.argmax(logits, dim=1, keepdim=False)[0].item()
        assert possible_actions[0][
            action_taken
        ], f"{possible_actions} doesn't have {action_taken}"
        return action_taken

    def get_loss(self, batch_list, batch_idx):
        batch = GameRollout(*[x[0] for x in batch_list])
        action_logits = self.get_logits(batch.states, batch.possible_actions)

        action_taken = batch.actions.squeeze(dim=1)
        loss = torch.nn.CrossEntropyLoss()(action_logits, action_taken)

        return action_logits, loss

    def training_step(self, batch_list: List[torch.Tensor], batch_idx):
        loss = self.get_loss(batch_list, batch_idx)[1]
        return {
            "loss": loss,
        }

    def validation_step(self, batch_list: List[torch.Tensor], batch_idx):
        _, loss = self.get_loss(batch_list, batch_idx)

        return {
            "val_loss": loss,
        }


from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class StateValueLightning(pl.LightningModule):
    def __init__(self, game: GameInterface):
        super().__init__()
        self.game = game
        self.value = StateValueModel(game.feature_dim(), game.num_players)
        self.policy = ImitationLearningModel(game.feature_dim(), game.action_dim())

    def train_model(
        self, train_dataset, val_dataset, num_workers: int, output_file=None
    ):
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            # show_progress_bar=False,
            max_epochs=1,
            # default_save_path=os.path.join(os.getcwd(), "models", "MAC"),
            # val_check_interval=train_dataset.max_games,
            callbacks=[
                ExitOnExceptionCallback(),
                TorchSaveCallback(),
                EarlyStopping(
                    monitor="val_loss", mode="min", patience=10, verbose=True
                ),
            ],
            # auto_lr_find=True,
            # num_sanity_val_steps=0,
            # resume_from_checkpoint="lightning_logs/version_125/"
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
        with Profiler(False):
            trainer.fit(
                self, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
        if output_file is not None:
            trainer.save_checkpoint(output_file)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)

        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

        # optimizer = torch_optimizer.Shampoo(self.parameters(), lr=0.01)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.5, patience=2, verbose=True
                ),
                "monitor": "val_loss",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def training_step(self, batch, batch_idx):
        value_loss_dict = self.value.training_step(batch, batch_idx)
        self.log("value_loss", value_loss_dict["loss"], prog_bar=True)
        policy_loss_dict = self.policy.training_step(batch, batch_idx)
        self.log("policy_loss", policy_loss_dict["loss"], prog_bar=True)
        return {"loss": value_loss_dict["loss"] + policy_loss_dict["loss"]}

    def validation_step(self, batch, batch_idx):
        value_loss_dict = self.value.validation_step(batch, batch_idx)
        self.log(
            "value_val_loss", value_loss_dict["val_loss"], prog_bar=True, on_epoch=True
        )
        self.log("value_tp", value_loss_dict["tp"], prog_bar=True, on_epoch=True)
        self.log(
            "value_precision",
            value_loss_dict["precision"],
            prog_bar=True,
            on_epoch=True,
        )
        self.log(
            "value_recall", value_loss_dict["recall"], prog_bar=True, on_epoch=True
        )

        policy_loss_dict = self.policy.validation_step(batch, batch_idx)
        self.log(
            "policy_val_loss",
            policy_loss_dict["val_loss"],
            prog_bar=True,
            on_epoch=True,
        )

        self.log(
            "val_loss",
            value_loss_dict["val_loss"] + policy_loss_dict["val_loss"],
            on_epoch=True,
        )
        return {"val_loss": value_loss_dict["val_loss"] + policy_loss_dict["val_loss"]}
