#
import copy
import os
import shutil

import pyximport

from utils.model_serialization import save

pyximport.install(language_level=3)
#

import torch
from torch import autograd, multiprocessing

from ai.game_simulation_nopool import GameSimulationDataset
from ai.mean_actor_critic import MeanActorCritic, StateValueLightning, has_exception
from engine.landshark_game import Game


def main():
    torch.multiprocessing.set_start_method("spawn")
    torch.set_printoptions(profile="full")
    # policy = train(100, Game(), "LandsharkAi.torch")
    game = Game(4)

    NUM_TRAIN_BATCHES = 800
    NUM_VAL_BATCHES = 32
    NUM_WORKERS = 32
    GAMES_PER_MINIBATCH = 64

    current_value_network = None
    historical_policy_networks = []

    # current_value_network = (
    #     torch.load("lightning_logs/version_141/models/StateValue_0.torch").cpu().eval()
    # )
    # historical_policy_networks = [
    #     torch.load("lightning_logs/version_141/models/Policy_0.torch").cpu().eval()
    # ]

    for epoch in range(10):
        lit = StateValueLightning(game)

        if os.path.exists("game_cache"):
            shutil.rmtree("game_cache")
        os.mkdir("game_cache")

        train_dataset = GameSimulationDataset(
            "train",
            game,
            NUM_TRAIN_BATCHES // NUM_WORKERS,
            GAMES_PER_MINIBATCH,
            current_value_network,
            historical_policy_networks,
        )
        val_dataset = GameSimulationDataset(
            "val",
            game,
            NUM_VAL_BATCHES // NUM_WORKERS,
            GAMES_PER_MINIBATCH,
            current_value_network,
            historical_policy_networks,
        )

        lit.train_model(train_dataset, val_dataset, NUM_WORKERS)

        if has_exception():
            break

        current_value_network = save(
            torch.jit.script(copy.deepcopy(lit.value).cpu().eval())
        )
        historical_policy_networks.append(
            save(torch.jit.script(copy.deepcopy(lit.policy).cpu().eval()))
        )

        del lit
        del train_dataset
        del val_dataset


if __name__ == "__main__":
    main()
