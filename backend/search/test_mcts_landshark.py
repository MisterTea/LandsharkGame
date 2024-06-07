import random
from typing import Any, List, Tuple

import numpy as np
import torch
from torch.multiprocessing import Pool

from ai.mean_actor_critic import ImitationLearningModel
from engine.landshark_game import Game
from search.mcts import MCTS

np.random.seed(1)
torch.manual_seed(1)

torch.multiprocessing.set_sharing_strategy("file_system")


class Policy(torch.nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, num_players: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.num_players = num_players
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

        self.critic = torch.nn.ModuleList(
            [torch.nn.Linear(64, num_players), torch.nn.Softmax(dim=1)]
        )

        self.actor = torch.nn.ModuleList(
            [torch.nn.Linear(64, action_dim), torch.nn.Softmax(dim=1)]
        )

        self.critic_criterion = torch.nn.NLLLoss()

        self.optimizer = torch.optim.Adam(
            list(self.shared.parameters())
            + list(self.critic.parameters())
            + list(self.actor.parameters()),
            lr=0.001,
        )

    def forward(self, features: torch.Tensor):
        x = features
        for m in self.shared:
            x = m(x)
        head = x
        for m in self.critic:
            x = m(x)
        critic_output = x
        x = head
        for m in self.actor:
            x = m(x)
        actor_output = x
        return critic_output, actor_output

    def fit(self, features, payoffs, probabilities):
        previous_loss = None
        stale_count = 0
        for x in range(100000):
            self.optimizer.zero_grad()
            critic_output, actor_output = self(features.detach())

            critic_loss = self.critic_criterion(
                torch.log(critic_output.clamp(min=1e-3)), payoffs.flatten()
            )
            actor_loss = (
                -(probabilities * torch.log(actor_output.clamp(min=1e-3)))
                .sum(dim=1)
                .mean()
            )
            print(actor_output)
            print(probabilities)

            total_loss = critic_loss + actor_loss
            if x % 1 == 0:
                print(
                    f"Critic: {critic_loss} Actor: {actor_loss} Total Loss: {total_loss} Previous loss: {previous_loss}"
                )
                assert torch.isnan(actor_loss).sum() == 0
                assert torch.isnan(critic_loss).sum() == 0
            total_loss.backward()
            if x > 0 and total_loss + 1e-4 > previous_loss:
                stale_count += 1
                if stale_count >= 3:
                    break
            else:
                stale_count = 0
            previous_loss = float(total_loss)
            self.optimizer.step()
            self.optimizer.zero_grad()


def one_mcts(game, policy):
    np.random.seed(random.SystemRandom().randint(0, 1000000))
    torch.manual_seed(random.SystemRandom().randint(0, 1000000))
    dense_features = []
    embedding_features = []
    payoffs = []
    probabilities = []
    mcts = MCTS(policy)
    game.reset()

    partial_training_examples = []
    episodeStep = 0

    with torch.no_grad():
        while True:
            episodeStep += 1

            # game.print()
            if game.get_player_to_act() == 0 and False:
                print("Possible Actions: " + str(game.getPossibleActions()))
                action_str = input(
                    "Please give an action for seat "
                    + str(game.get_player_to_act())
                    + ": "
                )
                action = int(action_str)
                game.playerAction(game.get_player_to_act(), action)
            else:
                pi = mcts.getActionProb(game, num_sims=1000, temp=1)
                dense_features = torch.zeros((game.feature_dim(),), dtype=torch.float)
                embedding_features = torch.zeros((game.embedding_dim(),), dtype=torch.int)
                game.populate_features(dense_features, embedding_features)
                partial_training_examples.append(
                    [dense_features, embedding_features, game.get_player_to_act(), pi, None]
                )

                action = np.random.choice(len(pi), p=pi)
                board = game.act(game.get_player_to_act(), action)

            if game.terminal():
                game_payoffs = game.payoffs()
                game_payoffs = (game_payoffs == int(game_payoffs.max().item())).long()
                for pte in partial_training_examples:
                    dense_features.append(pte[0])
                    embedding_features.append(pte[1])
                    ego_centric_payoffs = torch.roll(game_payoffs, pte[2], dims=[0])
                    winner_index = torch.nonzero(ego_centric_payoffs, as_tuple=False)[0]
                    payoffs.append(winner_index)
                    probabilities.append(torch.tensor(pte[3]))
                # print(payoffs)
                # print(probabilities)
                break
    return (dense_features, embedding_features, payoffs, probabilities)

def main():
    game = Game(4)
    dense_features = []
    embedding_features = []
    payoffs = []
    probabilities = []
    policy = None
    step = 128
    milestone = step
    for x in range(100):
        # results: List[Tuple[Any, Any, Any]] = mcts_pool.starmap(
        #     one_mcts, [(game, policy)] * 64
        # )
        results: List[Tuple[Any, Any, Any]] = [one_mcts(game, policy) for _ in range(64)]

        # f, pa, pr = one_mcts(game, policy)
        for df, ef, pa, pr in results:
            dense_features.extend(df)
            embedding_features.extend(ef)
            payoffs.extend(pa)
            probabilities.extend(pr)

        while len(features) >= milestone:
            milestone += step
            if policy is None:
                policy = Policy(game)
                policy.eval()
            policy.train()
            policy.fit(
                torch.stack(features), torch.stack(payoffs), torch.stack(probabilities)
            )
            features.clear()
            payoffs.clear()
            probabilities.clear()
            policy.eval()
            torch.save(policy, "MCTS_AC.torch")

if __name__ == '__main__':
    main()
