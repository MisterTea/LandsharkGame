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

from ai.types import GameRollout
from engine.game_interface import GameInterface


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
        self.shared_layers = torch.nn.ModuleList(
            [torch.nn.Linear(feature_dim, 128), torch.nn.Linear(128, 64),]
        )
        self.shared_activations = torch.nn.ModuleList(
            [torch.nn.ReLU(), torch.nn.ReLU(),]
        )

        self.critic_layers = torch.nn.ModuleList([torch.nn.Linear(64, action_dim),])
        self.critic_activations = torch.nn.ModuleList([torch.nn.Identity(),])

        self.actor_layers = torch.nn.ModuleList([torch.nn.Linear(64, action_dim),])
        self.actor_activations = torch.nn.ModuleList([torch.nn.Identity(),])

        self.num_steps = 0

    def critic_forward(self, inputs, possible_actions):
        x = inputs
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x

    def forward(self, inputs, possible_actions):
        x = inputs
        for i in range(len(self.shared_layers)):
            x = self.shared_layers[i](x)
            x = self.shared_activations[i](x)

        shared_result = x
        # Critic
        for i in range(len(self.critic_layers)):
            x = self.critic_layers[i](x)
            x = self.critic_activations[i](x)
        critic_action_values = x

        # Actor
        x = shared_result
        for i in range(len(self.actor_layers)):
            x = self.actor_layers[i](x)
            x = self.actor_activations[i](x)
        x = masked_softmax(x, possible_actions, 0.1)
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
        batch = GameRollout(*[x[0] for x in batch_list])
        action_one_hot = torch.nn.functional.one_hot(
            batch.actions.squeeze(1), num_classes=self.action_dim
        ).type_as(batch.states)
        player_to_act_one_hot = (
            (
                batch.payoffs
                * torch.nn.functional.one_hot(batch.player_to_act.squeeze(1))
            )
            .sum(dim=1, keepdim=True)
            .type_as(batch.states)
        )
        batch_size = batch.actions.size()[0]
        labels = (0.99 ** batch.distance_to_payoff) * player_to_act_one_hot
        assert labels.size() == (batch_size, 1)
        labels = (labels * action_one_hot).sum(dim=1, keepdim=True)
        assert action_one_hot.size() == batch.possible_actions.size()

        actor_probs, critic_action_values = self(batch.states, batch.possible_actions)
        total_prob = actor_probs.sum(dim=1)

        outputs = (critic_action_values * action_one_hot).sum(dim=1, keepdim=True)
        criterion = torch.nn.SmoothL1Loss(reduction="mean")
        critic_loss = criterion(outputs, labels)
        # print("Critic loss", critic_loss)

        if self.num_steps >= 100:
            detached_critic = critic_action_values.detach()
            assert critic_action_values.size() == batch.possible_actions.size()
            # baseline = (actor_probs * detached_critic).sum(dim=1, keepdim=True)
            # advantage = detached_critic - baseline
            # advantage_loss = -1 * (actor_probs * advantage.detach()).sum(
            #     dim=1, keepdim=True
            # )
            advantage_loss = -1 * (actor_probs * detached_critic).sum(
                dim=1, keepdim=True
            )
            # advantage_loss = torch.nn.LeakyReLU()(advantage).sum(dim=1, keepdim=True)
            advantage_loss = advantage_loss.mean()

            entropy_loss = 1.0 * torch.mean(
                torch.sum(
                    -actor_probs * torch.log(actor_probs.clamp(min=1e-6)),
                    dim=1,
                    keepdim=True,
                )
            )

            actor_loss = advantage_loss + entropy_loss
            # print("Actor losses", loss, entropy_loss)
        else:
            # Don't bother training actor while critic is so wrong
            actor_loss = advantage_loss = entropy_loss = 0

        self.num_steps += 1
        return {
            "progress_bar": {
                "advantage_loss": advantage_loss,
                "entropy_loss": entropy_loss,
                "critic_loss": critic_loss,
            },
            "loss": actor_loss + critic_loss,
        }


class TorchSaveCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        torch.save(pl_module.actor_critic, "MAC_ActorCritic.torch")
        game = pl_module.game
        actor = copy.deepcopy(pl_module.actor_critic).cpu().eval()

        with torch.no_grad():
            # Check winrate against random player
            scoreCounter: Counter = Counter()
            NUM_RANDOM_GAMES = 1000
            num_decisions = 0
            average_decision = torch.zeros((game.action_dim(),), dtype=torch.float)
            for on_game in range(NUM_RANDOM_GAMES):
                gameState = game.clone()
                gameState.reset()
                features = torch.zeros((1, gameState.feature_dim()), dtype=torch.float)
                while (
                    not gameState.terminal()
                ):  # gameState.phase != GamePhase.GAME_OVER:
                    seatToAct = gameState.get_player_to_act()
                    if seatToAct == 0:
                        possible_action_mask = gameState.get_one_hot_actions(True)
                        gameState.populate_features(features[0])
                        action_probs = actor(
                            features, possible_action_mask.unsqueeze(0)
                        )[0]
                    else:
                        possible_action_mask = gameState.get_one_hot_actions(True)
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
            print("DECISION HISTOGRAM")
            print(average_decision / num_decisions)
            print("SCORE AGAINST RANDOM")
            for x in range(gameState.num_players):
                print(x, scoreCounter[str(x)] / float(NUM_RANDOM_GAMES))


class MeanActorCritic(pl.LightningModule):
    def __init__(self, game: GameInterface):
        super().__init__()
        self.game = game
        self.actor_critic = ActorCritic(game.feature_dim(), game.action_dim())

    def forward(self, inputs):
        self.actor_critic.forward(inputs)

    def train_model(self, train_dataset, output_file=None):
        self.train_dataset = train_dataset
        trainer = pl.Trainer(
            gpus=1,
            # show_progress_bar=False,
            max_epochs=10000,
            default_save_path=os.path.join(os.getcwd(), "models", "MAC"),
            val_check_interval=1000,
            callbacks=[TorchSaveCallback()],
        )
        trainer.disable_validation = True
        trainer.fit(self)
        if output_file is not None:
            trainer.save_checkpoint(output_file)
        self.train_dataset = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, pin_memory=True,)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.actor_critic.parameters(), lr=0.001, weight_decay=0.01
        )

    def training_step(self, batch, batch_idx):
        return self.actor_critic.training_step(batch, batch_idx)
