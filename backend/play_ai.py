import random

import torch

from engine.landshark_game import GamePhase, GameState

if __name__ == "__main__":
    tmpgs = GameState(4)
    # policy = RegretMatching.load_from_checkpoint("LandsharkAi.torch", feature_dim=tmpgs.feature_dim(), action_dim=tmpgs.action_dim())
    policy = torch.load("MAC_ActorCritic.torch").cpu().eval()
    print(policy)
    random.seed(1)
    for x in range(1000):
        gameState = GameState(4)
        gameState.reset()
        features = torch.zeros((1, gameState.feature_dim()), dtype=torch.float)
        while not gameState.terminal():
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
                possible_action_mask = gameState.get_one_hot_actions(True)
                action_probs = policy(features, possible_action_mask.unsqueeze(0))[0]
                action_index = int(
                    torch.distributions.Categorical(action_probs).sample().item()
                )
                gameState.act(seatToAct, action_index)
        for i, player in enumerate(gameState.playerStates):
            playerScore = gameState.getScore(i)
            print(
                "Player", i, "has a score: ", playerScore,
            )
