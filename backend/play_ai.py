import random

import torch

from engine.landshark_game import Game, GamePhase

if __name__ == "__main__":
    tmpgs = Game(4)
    # policy = RegretMatching.load_from_checkpoint("LandsharkAi.torch", feature_dim=tmpgs.feature_dim(), action_dim=tmpgs.action_dim())
    policy_network = (
        torch.load("lightning_logs/version_141/models/Policy_0.torch").cpu().eval()
    )
    print(policy_network)
    random.seed(1)
    for x in range(1000):
        gameState = Game(4)
        gameState.reset()
        features = torch.zeros((gameState.feature_dim(),), dtype=torch.float)
        while not gameState.terminal():
            seatToAct = gameState.get_players_to_act()[0]
            possible_actions = gameState.getPossibleActions()
            gameState.print()
            if seatToAct == 0:
                print("Possible Actions: " + str(possible_actions))
                action_str = input(
                    "Please give an action for seat " + str(seatToAct) + ": "
                )
                action = int(action_str)
                gameState.playerAction(seatToAct, action)
            else:
                action = policy_network.get_action(game=gameState, features=features)
                gameState.act(gameState.get_player_to_act(), action)
        for i, player in enumerate(gameState.playerStates):
            playerScore = gameState.getScore(i)
            print(
                "Player",
                i,
                "has a score: ",
                playerScore,
            )
