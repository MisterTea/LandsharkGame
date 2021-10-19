#
import pyximport

pyximport.install(language_level=3)
#

import torch
from torch import autograd

from ai.game_simulation import GameSimulationDataset
from ai.mean_actor_critic import MeanActorCritic
from engine.landshark_game import GameState

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.set_printoptions(profile="full")
    # policy = train(100, GameState(4), "LandsharkAi.torch")
    game = GameState(4)
    mac = MeanActorCritic(game)
    train_dataset = GameSimulationDataset(game, 1000, mac.actor_critics)
    mac.train_model(train_dataset)
