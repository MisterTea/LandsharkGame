#
import pyximport

pyximport.install(language_level=3)
#

import torch
from torch import autograd, multiprocessing

from ai.game_simulation_nopool import GameSimulationDataset
from ai.mean_actor_critic import MeanActorCritic, StateValueLightning
from engine.landshark_game import Game


def main():
    torch.multiprocessing.set_start_method("spawn")
    torch.set_printoptions(profile="full")
    # policy = train(100, Game(), "LandsharkAi.torch")
    game = Game(4)
    # lit = MeanActorCritic(game)
    lit = StateValueLightning(game)

    # Train a value model on a random policy
    NUM_PARALLEL_GAMES = max(1, multiprocessing.cpu_count() - 2)

    train_dataset = GameSimulationDataset(game, 25, None)
    val_dataset = GameSimulationDataset(game, 2, None)

    lit.train_model(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
