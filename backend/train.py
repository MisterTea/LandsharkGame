import torch
from torch import autograd

from ai.deep_cfr import train, RpsGame
from engine.game import GameState

if __name__ == "__main__":
    policy = train(100, GameState(4), "LandsharkAi.torch")
