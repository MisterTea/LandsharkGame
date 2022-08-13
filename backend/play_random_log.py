import random

import torch

from engine.landshark_game import Game, GamePhase
from utils.profiler import Profiler

PROFILE = True
if __name__ == "__main__":
    dummyState = Game(4)
    payoff_sum = torch.zeros((4,), dtype=torch.float)
    action_hist = torch.zeros(
        (
            4,
            dummyState.action_dim(),
        ),
        dtype=torch.float,
    )
    with Profiler(PROFILE):
        random.seed(1)
        NUM_GAMES = 1024
        NUM_PLAYERS = 4

        game = Game(NUM_PLAYERS)
        features = torch.zeros((1, game.feature_dim()), dtype=torch.float)

        MAX_MOVES = 32
        train_data = torch.zeros(
            (NUM_GAMES * NUM_PLAYERS, MAX_MOVES, game.feature_dim())
        )
        train_data.fill_(-1)
        train_output = torch.zeros((NUM_GAMES * NUM_PLAYERS, MAX_MOVES, 1))
        train_output.fill_(-1)

        for x in range(NUM_GAMES):
            game = Game(NUM_PLAYERS)
            move_count = [0] * NUM_PLAYERS
            first = True
            while not game.terminal():
                seatToAct = game.get_players_to_act()[0]
                if not PROFILE:
                    game.print()
                if False and seatToAct == 0:
                    possible_actions = game.getPossibleActions(seatToAct)
                    print("Possible Actions: " + str(possible_actions))
                    action = input(
                        "Please give an action for seat " + str(seatToAct) + ": "
                    )
                    action = int(action)
                    game.playerAction(seatToAct, action)
                else:
                    game.populate_features(features[0])
                    assert move_count[seatToAct] < MAX_MOVES
                    train_data[x * NUM_PLAYERS + seatToAct][
                        move_count[seatToAct]
                    ] = features[0]

                    possible_action_mask = game.get_one_hot_actions(False)
                    action_hist[seatToAct] += possible_action_mask
                    action_probs = possible_action_mask.float()
                    # action_index = int(
                    # torch.distributions.Categorical(action_probs).sample().item()
                    # )
                    action_index = torch.argmax(action_probs).item()

                    train_output[x * NUM_PLAYERS + seatToAct][
                        move_count[seatToAct]
                    ] = action_index

                    game.act(seatToAct, action_index)

                    move_count[seatToAct] += 1
            payoff_sum += game.payoffs()
            for i, player in enumerate(game.playerStates):
                playerScore = game.getScore(i)
                if not PROFILE:
                    print(
                        "Player",
                        i,
                        "has a score: ",
                        playerScore,
                    )
    print("PAYOFF SUM", payoff_sum)
    print(train_data.shape)
    print(train_output.shape)
    torch.save(
        {"train_data": train_data, "train_output": train_output}, "training_data.pt"
    )
