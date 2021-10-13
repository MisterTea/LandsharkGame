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
from engine.game_interface import GameInterface
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
        self.shared = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(feature_dim),
                (torch.nn.Linear(feature_dim, 128)),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm1d(128),
                (torch.nn.Linear(128, 64)),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm1d(64),
            ]
        )

        self.critic_layers = torch.nn.ModuleList([(torch.nn.Linear(64, action_dim)),])
        self.critic_activations = torch.nn.ModuleList([torch.nn.Identity(),])

        self.actor_layers = torch.nn.ModuleList([(torch.nn.Linear(64, action_dim)),])
        self.actor_activations = torch.nn.ModuleList([torch.nn.Identity(),])

        self.num_steps = 0

    def critic_forward(self, inputs, possible_actions):
        x = inputs
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x

    def forward(self, inputs, possible_actions, do_epsilon_greedy:bool):
        x = inputs
        for i in range(len(self.shared)):
            x = self.shared[i](x)

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
        x_before_activation = x.cpu()
        x = torch.clamp(torch.nn.Sigmoid()(x), min=1e-3, max=(1.0 - 1e-3))
        assert torch.min(x) > 0.0
        
        #x = masked_softmax(x, possible_actions, 0.1)
        # Replace softmax with linear scale
        x = x * possible_actions.float()
        original_x = x.cpu()
        x = x / x.sum(dim=1, keepdim=True)
        assert torch.allclose(torch.max(possible_actions, dim=1).values.cpu(), torch.IntTensor([1])), f"{torch.max(possible_actions, dim=1).values.cpu()}"
        assert torch.allclose(x.sum(dim=1).cpu(), torch.Tensor([1.0])), f"{original_x.sum(dim=1).cpu()} {x_before_activation[torch.logical_not(torch.isclose(x.sum(dim=1).cpu(), torch.Tensor([1.0])))]} {original_x[torch.logical_not(torch.isclose(x.sum(dim=1).cpu(), torch.Tensor([1.0])))]}"
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
        batch = GameRollout(*[x[0] for x in batch_list])
        actor_probs, critic_action_values = self(batch.states, batch.possible_actions, True)

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

            # Some actions are impossible, these will get an importance weight of 0
            importance_weight = (actor_probs / batch.policy.clamp(min=1e-6)).clamp(
                max=10.0
            )
            assert (
                torch.isnan(importance_weight).sum() == 0
            ), f"Invalid importance weight {actor_probs}"
            advantage_loss = -1 * (
                importance_weight.detach() * actor_probs * detached_critic
            ).sum(dim=1, keepdim=True)
            # advantage_loss = torch.nn.LeakyReLU()(advantage).sum(dim=1, keepdim=True)
            advantage_loss = advantage_loss.mean()

            entropy = torch.sum(
                -actor_probs * torch.log(actor_probs.clamp(min=1e-6)),
                dim=1,
                keepdim=True,
            ) / torch.log((actor_probs > 0).sum(dim=1, keepdim=True).float()).clamp(
                min=1e-6
            )
            assert ((entropy > 1.01).sum() + (entropy < -0.01).sum()) == 0, f"Invalid entropy {torch.min(torch.sum(actor_probs, dim=1))}, {torch.max(torch.sum(actor_probs, dim=1))}, {entropy[entropy > 1.0]}, {entropy[entropy < 0.0]}"
            entropy_loss = torch.nn.L1Loss()(entropy, torch.ones_like(entropy).float()) * 0.1

            actor_loss = advantage_loss + entropy_loss
            # print("Actor losses", loss, entropy_loss)

            advantage_loss = advantage_loss.detach()
            entropy_loss = entropy_loss.detach()
        else:
            # Don't bother training actor while critic is so wrong
            actor_loss = advantage_loss = entropy_loss = 0

        self.num_steps += 1
        return {
            "progress_bar": {
                "advantage_loss": advantage_loss,
                "entropy_loss": entropy_loss,
                "critic_loss": critic_loss.detach(),
            },
            "loss": actor_loss + critic_loss,
        }


def test_actors(game, epoch: int, actor_critics):
    for i, actor_critic in enumerate(actor_critics):
        actor = copy.deepcopy(actor_critic).cpu().eval()

        with torch.no_grad():
            for current_epoch in range(-1, epoch + 1, max(1, epoch // 5)):
                # Check winrate against random or past player
                opponent_policy = None
                if current_epoch >= 0:
                    opponent_policy = (
                        torch.load(f"models/MAC_ActorCritic_{current_epoch}_{0}.torch")
                        .cpu()
                        .eval()
                    )
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
                    while (
                        not gameState.terminal()
                    ):  # gameState.phase != GamePhase.GAME_OVER:
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


class TorchSaveCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        for i, actor_critic in enumerate(pl_module.actor_critics):
            torch.save(actor_critic, f"models/MAC_ActorCritic_{epoch}_{i}.torch")
        test_actors(pl_module.game, epoch, pl_module.actor_critics)


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
        self.learning_rate = 0.01
        # test_actors(self.game, self.current_epoch, self.actor_critics)

    def forward(self, inputs):
        self.actor_critics[0].forward(inputs)

    def train_model(self, train_dataset, output_file=None):
        self.train_dataset = train_dataset
        trainer = pl.Trainer(
            #gpus=1,
            
            # show_progress_bar=False,
            max_epochs=1000,
            # default_save_path=os.path.join(os.getcwd(), "models", "MAC"),
            val_check_interval=train_dataset.max_games,
            callbacks=[TorchSaveCallback()],
            # auto_lr_find=True,
            num_sanity_val_steps=0
        )
        with Profiler(True):
            trainer.fit(self)
        if output_file is not None:
            trainer.save_checkpoint(output_file)
        self.train_dataset = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, pin_memory=True, num_workers=10)

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(x.parameters(), lr=(self.learning_rate))
            for x in self.actor_critics
        ]
        schedulers = [
            torch.optim.lr_scheduler.MultiplicativeLR(o, lr_lambda=lambda epoch: 0.95)
            for o in optimizers
        ]
        return optimizers, schedulers

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return self.actor_critics[optimizer_idx].training_step(batch, batch_idx)
