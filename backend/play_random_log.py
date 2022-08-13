import random

import torch

from engine.landshark_game import GamePhase, GameState
from utils.profiler import Profiler

PROFILE = True
if __name__ == "__main__":
    dummyState = GameState(4)
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

        gameState = GameState(NUM_PLAYERS)
        features = torch.zeros((1, gameState.feature_dim()), dtype=torch.float)

        MAX_MOVES = 32
        train_data = torch.zeros(
            (NUM_GAMES * NUM_PLAYERS, MAX_MOVES, gameState.feature_dim())
        )
        train_data.fill_(-1)
        train_output = torch.zeros((NUM_GAMES * NUM_PLAYERS, MAX_MOVES, 1))
        train_output.fill_(-1)

        for x in range(NUM_GAMES):
            gameState = GameState(NUM_PLAYERS)
            move_count = [0] * NUM_PLAYERS
            first = True
            while not gameState.terminal():
                seatToAct = gameState.get_players_to_act()[0]
                if not PROFILE:
                    gameState.print()
                if False and seatToAct == 0:
                    possible_actions = gameState.getPossibleActions(seatToAct)
                    print("Possible Actions: " + str(possible_actions))
                    action = input(
                        "Please give an action for seat " + str(seatToAct) + ": "
                    )
                    action = int(action)
                    gameState.playerAction(seatToAct, action)
                else:
                    gameState.populate_features(features[0])
                    assert move_count[seatToAct] < MAX_MOVES
                    train_data[x * NUM_PLAYERS + seatToAct][
                        move_count[seatToAct]
                    ] = features[0]

                    possible_action_mask = gameState.get_one_hot_actions(False)
                    action_hist[seatToAct] += possible_action_mask
                    action_probs = possible_action_mask.float()
                    # action_index = int(
                    # torch.distributions.Categorical(action_probs).sample().item()
                    # )
                    action_index = torch.argmax(action_probs).item()

                    train_output[x * NUM_PLAYERS + seatToAct][
                        move_count[seatToAct]
                    ] = action_index

                    gameState.act(seatToAct, action_index)

                    move_count[seatToAct] += 1
            payoff_sum += gameState.payoffs()
            for i, player in enumerate(gameState.playerStates):
                playerScore = gameState.getScore(i)
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
