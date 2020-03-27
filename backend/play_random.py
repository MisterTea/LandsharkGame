import random

from engine.game import GamePhase, GameState

if __name__ == "__main__":
    random.seed(1)
    for x in range(1000):
        gameState = GameState(4)
        while gameState.phase != GamePhase.GAME_OVER:
            seatToAct = gameState.getSeatToAct()
            playerToAct = gameState.playerStates[seatToAct].playerId
            possible_actions = gameState.getPossibleActions(playerToAct)
            gameState.print()
            if seatToAct == 0:
                print("Possible Actions: " + str(possible_actions))
                action = input(
                    "Please give an action for seat " + str(seatToAct) + ": "
                )
                action = int(action)
            else:
                action = random.choice(possible_actions)
            gameState.playerAction(playerToAct, action)
