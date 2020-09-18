import copy
import logging
import math
import random

import numpy as np
import torch

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, policy):
        self.policy = policy

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.player_to_act = {}
        self.cpuct = 1.0

    def getActionProb(self, game, num_sims, temp):
        for i in range(num_sims):
            g = copy.deepcopy(game)
            self.search(g)

        state = hash(game)
        counts = [
            self.Nsa.get((state, action), 0) if is_valid else 0
            for action, is_valid in enumerate(game.get_one_hot_actions())
        ]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, game):
        if game.terminal():
            payoffs = game.payoffs()
            return payoffs

        state = hash(game)
        player_to_act = game.get_player_to_act()
        valids = game.get_one_hot_actions()
        if state not in self.Ps:
            # leaf node
            if self.policy is None:
                self.Ps[state] = game.get_one_hot_actions()
                # Without a value function, we need a random simulation to get the value
                sim_game = game.clone()
                while not sim_game.terminal():
                    possible_action_mask = sim_game.get_one_hot_actions(hacks=True)
                    action_probs = possible_action_mask.float()
                    action_index = int(
                        torch.distributions.Categorical(action_probs).sample().item()
                    )
                    sim_game.act(sim_game.get_player_to_act(), action_index)
                v = sim_game.payoffs()
            else:
                features = torch.zeros((1, game.feature_dim()))
                game.populate_features(features[0])
                neural_net_output = self.policy(features)
                v, self.Ps[state] = neural_net_output[0][0], neural_net_output[1][0]
                v = torch.roll(v, -1 * game.get_player_to_act(), dims=[0])
                v = (v * 3) + ((1 - v) * -1)
                self.Ps[state] = self.Ps[state] * valids  # masking invalid moves
                sum_Ps_s = torch.sum(self.Ps[state])
                if sum_Ps_s > 0:
                    self.Ps[state] /= sum_Ps_s  # renormalize
                    self.Ps[state] *= len(game.getPossibleActions())
                else:
                    # if all valid moves were masked make all valid moves equally probable

                    # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                    # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                    log.error("All valid moves were masked, doing a workaround.")
                    self.Ps[state] = valids

            self.Ns[state] = 0
            return v

        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for action, is_valid in enumerate(valids):
            if is_valid == 0:
                continue
            if (state, action) in self.Qsa:
                u = self.Qsa[(state, action)] + self.cpuct * self.Ps[state][
                    action
                ] * math.sqrt(self.Ns[state]) / (1 + self.Nsa[(state, action)])
            else:
                u = (
                    self.cpuct
                    * self.Ps[state][action]
                    * math.sqrt(self.Ns[state] + EPS)
                )  # Q = 0 ?

            # Tie breaking
            u += random.uniform(1e-6, 2e-6)

            if u > cur_best:
                cur_best = u
                best_act = action

        assert best_act != -1
        action = best_act
        game.act(game.get_player_to_act(), action)

        v = self.search(game)

        if (state, action) in self.Qsa:
            self.Qsa[(state, action)] = (
                self.Nsa[(state, action)] * self.Qsa[(state, action)]
                + float(v[player_to_act].item())
            ) / (self.Nsa[(state, action)] + 1)
            self.Nsa[(state, action)] += 1

        else:
            self.Qsa[(state, action)] = float(v[player_to_act].item())
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1
        return v
