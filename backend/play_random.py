import random

from engine.game import GamePhase, GameState

if __name__ == "__main__":
    random.seed(1)
    for x in range(1000):
        gameState = GameState(4)
        while gameState.phase != GamePhase.GAME_OVER:
            seatToAct = gameState.get_players_to_act()[0]
            possible_actions = gameState.getPossibleActions(seatToAct)
            gameState.print()
            if False and seatToAct == 0:
                print("Possible Actions: " + str(possible_actions))
                action = input(
                    "Please give an action for seat " + str(seatToAct) + ": "
                )
                action = int(action)
            else:
                action = random.choice(possible_actions)
            gameState.playerAction(seatToAct, action)
        for i, player in enumerate(gameState.playerStates):
            playerScore = player.getScore()
            print(
                "Player", i, "has a score: ", playerScore,
            )
