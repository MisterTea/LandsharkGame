import random

import torch

from ai.game_traverse import one_step_best_response_mixed
from ai.mean_actor_critic import get_action_from_imitator
from engine.landshark_game import Game, GamePhase

if __name__ == "__main__":
    tmpgs = Game(4)
    # policy = RegretMatching.load_from_checkpoint("LandsharkAi.torch", feature_dim=tmpgs.feature_dim(), action_dim=tmpgs.action_dim())
    policy_network = (
        torch.load("lightning_logs/version_145/models/Policy_43.torch").cpu().eval()
    )
    value_network = (
        torch.load("lightning_logs/version_145/models/StateValue_43.torch").cpu().eval()
    )
    print(policy_network)
    random.seed(1)
    for x in range(1000):
        gameState = Game(4)
        gameState.reset()
        dense_features = torch.zeros((gameState.feature_dim(),), dtype=torch.float)
        embedding_features = torch.zeros((gameState.embedding_dim(),), dtype=torch.int)
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
                print("Playing mixed BR")
                action = one_step_best_response_mixed(
                    game=gameState,
                    value_network=value_network,
                    policy_network=policy_network,
                )
                gameState.act(gameState.get_player_to_act(), action)
        for i, player in enumerate(gameState.playerStates):
            playerScore = gameState.getScore(i)
            print(
                "Player",
                i,
                "has a score: ",
                playerScore,
            )
