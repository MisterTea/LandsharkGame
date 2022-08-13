#
import pyximport

pyximport.install(language_level=3)
#

import torch
from torch import autograd, multiprocessing

from ai.game_simulation import GameSimulationDataset
from ai.mean_actor_critic import MeanActorCritic, StateValueLightning
from engine.rps_game import Game

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.set_printoptions(profile="full")
    # policy = train(100, Game(), "LandsharkAi.torch")
    game = Game()
    # lit = MeanActorCritic(game)
    lit = StateValueLightning(game)

    # Train a value model on a random policy
    NUM_PARALLEL_GAMES = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(NUM_PARALLEL_GAMES) as pool:

        train_dataset = GameSimulationDataset(game, 100, None, pool)
        val_dataset = GameSimulationDataset(game, 10, None, pool)

        lit.train_model(train_dataset, val_dataset)
