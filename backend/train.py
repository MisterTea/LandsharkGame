#
import pyximport

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

    # Train a value model on a random policy
    NUM_PARALLEL_GAMES = max(1, multiprocessing.cpu_count() - 2)

    NUM_TRAIN_BATCHES = 400
    NUM_VAL_BATCHES = 32
    NUM_WORKERS = 8

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

        train_dataset = GameSimulationDataset(
            game,
            NUM_TRAIN_BATCHES / NUM_WORKERS,
            current_value_network,
            historical_policy_networks,
        )
        val_dataset = GameSimulationDataset(
            game,
            NUM_VAL_BATCHES / NUM_WORKERS,
            current_value_network,
            historical_policy_networks,
        )

        lit.train_model(train_dataset, val_dataset, NUM_WORKERS)

        if has_exception():
            break

        current_value_network = lit.value.cpu().eval()
        historical_policy_networks.append(lit.policy.cpu().eval())


if __name__ == "__main__":
    main()
