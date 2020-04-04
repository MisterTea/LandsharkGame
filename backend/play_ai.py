import random

import torch

from ai.deep_cfr import RegretMatching
from engine.game import GamePhase, GameState

if __name__ == "__main__":
    tmpgs = GameState(4)
    #policy = RegretMatching.load_from_checkpoint("LandsharkAi.torch", feature_dim=tmpgs.feature_dim(), action_dim=tmpgs.action_dim())
    policy = torch.load("LandsharkAi.torch")
    print(policy)
    random.seed(1)
    for x in range(1000):
        gameState = GameState(4)
        features = torch.zeros((1, gameState.feature_dim()), dtype=torch.float)
        while gameState.phase != GamePhase.GAME_OVER:
            seatToAct = gameState.get_players_to_act()[0]
            possible_actions = gameState.getPossibleActions(seatToAct)
            gameState.print()
            if seatToAct == 0:
                print("Possible Actions: " + str(possible_actions))
                action_str = input(
                    "Please give an action for seat " + str(seatToAct) + ": "
                )
                action = int(action_str)
                gameState.playerAction(seatToAct, action)
            else:
                gameState.populate_features(features[0])
                action_probs = policy(features).detach()[0]
                possible_action_mask = gameState.get_one_hot_actions()
                action_index = int(
                    torch.distributions.Categorical(
                        torch.nn.Softmax(dim=0)(action_probs) * possible_action_mask
                    )
                    .sample()
                    .item()
                )
                gameState.act(seatToAct, action_index)
        for i, player in enumerate(gameState.playerStates):
            playerScore = player.getScore()
            print(
                "Player", i, "has a score: ", playerScore,
            )
