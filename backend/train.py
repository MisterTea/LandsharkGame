import torch
from torch import autograd

from ai.game_simulation import GameSimulationDataset
from ai.mean_actor_critic import MeanActorCritic
from engine.landshark_game import GameState

if __name__ == "__main__":
    # policy = train(100, GameState(4), "LandsharkAi.torch")
    game = GameState(4)
    mac = MeanActorCritic(game)
    train_dataset = GameSimulationDataset(game, 1000, mac.actor_critic)
    torch.save(mac.actor_critic, "MAC_ActorCritic.torch")
    mac.train_model(train_dataset)
