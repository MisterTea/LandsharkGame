#!/usr/bin/env python3

from typing import Dict, List, Tuple
from uuid import UUID, uuid4


class PlayerState:
    def __init__(self, playerId: UUID):
        self.money = 18
        self.propertyCards = []
        self.dollarCards = []
        self.playerId = playerId
        self.canBid = True
        self.moneyBid = 0
        self.propertyBid = 0

    def getLargestHouse(self):
        if len(self.propertyCards) == 0:
            return 0
        return max(self.propertyCards)

    def removeProperty(self, property: int):
        for i, p in enumerate(self.propertyCards):
            if p == property:
                del self.propertyCards[i]
                return
        assert False

    def print(self):
        print(
            "Player " + str(self.playerId)[-4:],
            self.money,
            self.propertyCards,
            self.dollarCards,
        )

    def getScore(self) -> Tuple[int, int]:
        totalMoney = self.money + sum(self.dollarCards)
        cashMoney = self.money
        return (totalMoney, cashMoney)
