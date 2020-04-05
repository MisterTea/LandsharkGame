import random

from engine.game import GamePhase, GameState
from utils.profiler import Profiler
import torch

PROFILE = True
if __name__ == "__main__":
    dummyState = GameState(4)
    payoff_sum = torch.zeros((4,), dtype=torch.float)
    action_hist = torch.zeros((4,dummyState.action_dim(),), dtype=torch.float)
    with Profiler(PROFILE):
        random.seed(1)
        for x in range(1000):
            gameState = GameState(4)
            first = True
            while not gameState.terminal():
                seatToAct = gameState.get_players_to_act()[0]
                if not PROFILE: gameState.print()
                if False and seatToAct == 0:
                    possible_actions = gameState.getPossibleActions(seatToAct)
                    print("Possible Actions: " + str(possible_actions))
                    action = input(
                        "Please give an action for seat " + str(seatToAct) + ": "
                    )
                    action = int(action)
                    gameState.playerAction(seatToAct, action)
                else:
                    possible_action_mask = gameState.get_one_hot_actions()
                    action_hist[seatToAct] += possible_action_mask
                    action_probs = possible_action_mask.float()
                    action_index = int(torch.distributions.Categorical(
                            action_probs
                        ).sample().item())
                    gameState.act(seatToAct, action_index)
            payoff_sum += gameState.payoffs()
            for i, player in enumerate(gameState.playerStates):
                playerScore = gameState.getScore(i)
                if not PROFILE:
                    print(
                        "Player", i, "has a score: ", playerScore,
                    )
    print("PAYOFF SUM",payoff_sum)
