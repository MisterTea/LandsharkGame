#!/usr/bin/env python3

import copy
import random
from enum import IntEnum
from typing import Dict, List, Tuple
from uuid import UUID, uuid4

import numpy as np
import torch

from engine.player import PlayerState


class GamePhase(IntEnum):
    BUYING_HOUSES = 1
    SELLING_HOUSES = 2
    GAME_OVER = 3


class GameInterface:
    def action_dim(self):
        raise NotImplementedError()


class GameState:
    __slots__ = [
        "propertyCardsToDraw",
        "dollarCardsToDraw",
        "playerStates",
        "biddingPlayer",
        "highestBid",
        "num_players",
        "phase",
        "onPropertyCard",
        "onDollarCard",
        "money",
        "moneyBid",
        "canBid",
        "propertyBid",
    ]

    def __init__(self, num_players: int):
        self.propertyCardsToDraw = np.array(list(range(1, 31)))
        self.dollarCardsToDraw = np.array([0, 0] + (list(range(2, 16)) * 2))
        self.biddingPlayer = 0
        self.highestBid = -1
        self.num_players = num_players
        assert num_players <= 4 and num_players > 0

        playerStates = []
        for x in range(self.num_players):
            playerState = PlayerState()
            playerStates.append(playerState)
        self.playerStates = tuple(playerStates)

        # HACK: Reduce rounds to make training faster
        # self.money = np.array([5] * self.num_players, dtype=np.int)
        self.money = np.array([18] * self.num_players, dtype=np.int)
        self.moneyBid = np.zeros((self.num_players,), dtype=np.int)
        self.canBid = np.zeros((self.num_players,), dtype=np.bool)
        self.canBid[:] = True
        self.propertyBid = np.zeros((self.num_players,), dtype=np.int)

        self.phase = GamePhase.BUYING_HOUSES

        self.reset()

    def __hash__(self):
        items_to_hash = [self.phase]
        if self.phase == GamePhase.BUYING_HOUSES:
            for p in self.playerStates:
                items_to_hash.append(p.money)
                items_to_hash.append(p.canBid)
                items_to_hash.append(p.moneyBid)
                items_to_hash.append(tuple(p.propertyCards))
        elif self.phase == GamePhase.SELLING_HOUSES:
            for p in self.playerStates:
                items_to_hash.append(p.money)
                items_to_hash.append(p.propertyBid)
                items_to_hash.append(tuple(p.propertyCards))
                items_to_hash.append(tuple(p.dollarCards))
        else:
            raise NotImplementedError()
        return hash(tuple(items_to_hash))

    def clone(self):
        cloned_game = copy.deepcopy(self)
        return cloned_game

    def reset(self):
        if self.num_players == 3:
            self.onPropertyCard = 6
            self.onDollarCard = 6
        elif self.num_players == 4:
            # HACK: Reduce rounds to make training faster
            self.onPropertyCard = 2 + (0 * 4)
            self.onDollarCard = self.onPropertyCard

        self.biddingPlayer = random.randint(0, self.num_players - 1)

        # Shuffle all the cards
        np.random.shuffle(self.propertyCardsToDraw)
        np.random.shuffle(self.dollarCardsToDraw)

        # Call shuffleNext to ensure the first batch is sorted
        self.shuffleNext()

    def shuffleNext(self):
        if self.phase == GamePhase.BUYING_HOUSES:
            np.random.shuffle(self.propertyCardsToDraw[self.onPropertyCard :])
            startSort = self.onPropertyCard
            # self.propertyCardsToDraw[startSort : startSort + self.num_players] = sorted(
            #    self.propertyCardsToDraw[startSort : startSort + self.num_players],
            # )
            self.propertyCardsToDraw[startSort : startSort + self.num_players].sort()
        elif self.phase == GamePhase.SELLING_HOUSES:
            np.random.shuffle(self.dollarCardsToDraw[self.onDollarCard :])
            startSort = self.onDollarCard
            # self.dollarCardsToDraw[startSort : startSort + self.num_players] = sorted(
            #    self.dollarCardsToDraw[startSort : startSort + self.num_players],
            #    reverse=True,
            # )
            self.dollarCardsToDraw[startSort : startSort + self.num_players].sort()
        else:
            assert False

    def print(self):
        for player in range(self.num_players):
            print(
                "Player",
                player,
                self.money[player],
                self.playerStates[player].propertyCards,
                self.playerStates[player].dollarCards,
            )
        if self.phase == GamePhase.BUYING_HOUSES:
            print("Houses on auction: " + str(self.getPropertyOnAuction()))
            for i, player in enumerate(self.playerStates):
                print("Player", i, "bid", self.moneyBid[i], "canbid", self.canBid[i])
        elif self.phase == GamePhase.SELLING_HOUSES:
            print("Dollar Cards on auction: " + str(self.getDollarsOnAuction()))
            for i, player in enumerate(self.playerStates):
                print("Player", i, "bid", self.propertyBid[i])

    def action_dim(self):
        return 1 + 19 + 30  # fold, bet 0-18, bet 1-30 properties

    def terminal(self):
        return self.phase == GamePhase.GAME_OVER

    def payoffs(self):
        absolute_payoff = torch.tensor(
            [float(self.getScore(i)) for i in range(self.num_players)]
        )
        places = torch.sort(absolute_payoff)[1]
        payoff = torch.zeros_like(absolute_payoff)
        scores = torch.tensor([-8.0, -2.0, 2.0, 8.0])
        for index, place in enumerate(places):
            payoff[place] = scores[index]
        return payoff

    def get_one_hot_actions(self, hacks):
        player_index = self.get_player_to_act()
        player = self.playerStates[player_index]
        actions = self.getPossibleActions(player_index)
        actions_one_hot = torch.zeros((self.action_dim(),), dtype=torch.int)
        if (
            False
            and self.phase == GamePhase.BUYING_HOUSES
            and player.moneyBid > 0
            and hacks
        ):
            # HACK: Let's fold after 1 round of bidding to speed up training
            actions_one_hot[0] = 1
        else:
            for a in actions:
                if self.phase == GamePhase.BUYING_HOUSES:
                    # HACK: Do not ever bid more than the spread/2
                    if (a * 2) <= self.getPropertySpread() or hacks == False:
                        actions_one_hot[1 + a] = 1
                elif self.phase == GamePhase.SELLING_HOUSES:
                    actions_one_hot[20 + (a - 1)] = 1
                else:
                    assert False, "Oops"
        return actions_one_hot

    def feature_dim(self):
        return 77

    def getPropertySpread(self):
        a = self.getPropertyOnAuction()
        return int(a.max() - a.min())

    def populate_features(self, features: torch.Tensor):
        player_index = self.get_player_to_act()
        player = self.playerStates[player_index]
        cursor = 0
        if self.phase == GamePhase.BUYING_HOUSES:
            features[cursor] = int(self.money[player_index])
            cursor += 1
            t = torch.tensor(self.getPropertyOnAuction(), dtype=torch.long)
            features[cursor + (t - 1)] = 1.0
            cursor += 30
        else:
            cursor += 31
        if self.phase == GamePhase.SELLING_HOUSES:
            t = torch.tensor(self.getDollarsOnAuction())
            dollars, counts = torch.unique(t, return_counts=True)
            features[dollars.long() + cursor] = counts.float()
            cursor += 16
            t = torch.tensor(player.propertyCards, dtype=torch.long)
            features[cursor + (t - 1)] = 1.0
            cursor += 30

        return features

    def get_player_to_act(self):
        if self.phase == GamePhase.BUYING_HOUSES:
            return self.biddingPlayer
        elif self.phase == GamePhase.SELLING_HOUSES:
            for i, player in enumerate(self.playerStates):
                if self.propertyBid[i] == 0:
                    return i
        else:
            assert False, "Oops"
        assert False, "Should never get here"

    def act(self, player: int, action_index: int):
        assert player == self.get_player_to_act()
        if self.phase == GamePhase.BUYING_HOUSES:
            self.playerAction(player, action_index - 1)
        elif self.phase == GamePhase.SELLING_HOUSES:
            self.playerAction(player, (action_index - 20) + 1)
        else:
            assert False, "Oops"

    def getPropertyOnAuction(self):
        assert self.phase == GamePhase.BUYING_HOUSES
        return self.propertyCardsToDraw[
            self.onPropertyCard : self.onPropertyCard + self.numBiddersLeft()
        ]

    def getDollarsOnAuction(self):
        assert self.phase == GamePhase.SELLING_HOUSES
        return self.dollarCardsToDraw[
            self.onDollarCard : self.onDollarCard + self.num_players
        ]

    def getPlayerSeatWithLargestHouse(self):
        bestSeat = -1
        bestHouse = 0
        for seat, player in enumerate(self.playerStates):
            if seat == 0:
                bestSeat = 0
                bestHouse = self.getLargestHouse(seat)
            else:
                newBestHouse = self.getLargestHouse(seat)
                if newBestHouse > bestHouse:
                    bestSeat = seat
                    bestHouse = newBestHouse
        return bestSeat

    def get_players_to_act(self):
        if self.phase == GamePhase.BUYING_HOUSES:
            return [self.biddingPlayer]
        elif self.phase == GamePhase.SELLING_HOUSES:
            bids_needed = []
            for i, player in enumerate(self.playerStates):
                if self.propertyBid[i] == 0:
                    bids_needed.append(i)
            assert len(bids_needed) > 0, "Should never get here"
            return bids_needed
        else:
            assert False, "Oops"

    def getPossibleActions(self, seat: int):
        if self.phase == GamePhase.BUYING_HOUSES:
            assert seat == self.biddingPlayer
            assert self.canBid[seat]
            if self.highestBid == -1:
                # Can't fold if there are no bids
                return list(
                    range(
                        self.highestBid + 1, self.money[seat] + self.moneyBid[seat] + 1
                    )
                )
            else:
                return [-1] + list(
                    range(
                        self.highestBid + 1, self.money[seat] + self.moneyBid[seat] + 1
                    )
                )
        elif self.phase == GamePhase.SELLING_HOUSES:
            assert self.propertyBid[seat] == 0
            return list(self.playerStates[seat].propertyCards)
        else:
            assert False, "Oops"

    def numBiddersLeft(self):
        biddersLeft = np.sum(self.canBid)
        assert biddersLeft > 0
        return biddersLeft

    def playerAction(self, seat: int, action: int) -> None:
        if self.phase == GamePhase.BUYING_HOUSES:
            assert seat == self.biddingPlayer
            biddingPlayer = self.playerStates[self.biddingPlayer]
            assert self.canBid[self.biddingPlayer]
            if action == -1:
                # Fold
                boughtProperty = self.propertyCardsToDraw[self.onPropertyCard]
                self.onPropertyCard += 1
                biddingPlayer.propertyCards.append(boughtProperty)
                self.money[self.biddingPlayer] += self.moneyBid[self.biddingPlayer] // 2
                self.moneyBid[self.biddingPlayer] = 0
                self.canBid[self.biddingPlayer] = False

                biddersLeft = self.numBiddersLeft()
                if biddersLeft == 1:
                    bidder_left = np.where(self.canBid)[0][0]
                    assert self.canBid[bidder_left]
                    boughtProperty = self.propertyCardsToDraw[self.onPropertyCard]
                    self.onPropertyCard += 1
                    self.playerStates[bidder_left].propertyCards.append(boughtProperty)
                    # Check for phase end
                    if len(self.propertyCardsToDraw) == self.onPropertyCard:
                        self.phase = GamePhase.SELLING_HOUSES
                    else:
                        # Reset the auction
                        self.biddingPlayer = self.getPlayerSeatWithLargestHouse()
                        self.highestBid = -1
                        self.moneyBid[:] = 0
                        self.canBid[:] = True
                    # Shuffle the next property or dollar cards
                    self.shuffleNext()
                else:
                    # Keep going
                    self.rotateBidder()
            else:
                # Raise
                raiseAmount = action - self.moneyBid[self.biddingPlayer]
                assert action > self.highestBid
                assert raiseAmount > 0 or (raiseAmount == 0 and self.highestBid == -1)
                self.moneyBid[self.biddingPlayer] = action
                self.money[self.biddingPlayer] -= raiseAmount
                assert self.money[self.biddingPlayer] >= 0
                self.highestBid = action
                # Move on to next player
                self.rotateBidder()
        elif self.phase == GamePhase.SELLING_HOUSES:
            biddingPlayer = self.playerStates[seat]
            assert self.propertyBid[seat] == 0
            assert (
                action in biddingPlayer.propertyCards
            ), "Tried to bid a property that you don't own"
            self.propertyBid[seat] = action
            self.handlePropertyBidFinish()

    def rotateBidder(self):
        self.biddingPlayer = (self.biddingPlayer + 1) % self.num_players
        if self.canBid[self.biddingPlayer] == False:
            self.rotateBidder()

    def handlePropertyBidFinish(self):
        # Check if everyone has placed a bid
        if np.any(self.propertyBid == 0):
            # Not done yet
            return
        # Execute auction
        seatBidPairs: List[Tuple[int, int]] = []
        for seat, player in enumerate(self.playerStates):
            seatBidPairs.append((seat, self.propertyBid[seat]))
        seatBidPairs = sorted(seatBidPairs, key=lambda sbp: sbp[1], reverse=True)
        dollarCardsOnTable = self.dollarCardsToDraw[
            self.onDollarCard : self.onDollarCard + self.num_players
        ]
        self.onDollarCard += self.num_players
        assert len(dollarCardsOnTable) == len(seatBidPairs)
        for x in range(len(seatBidPairs)):
            player = seatBidPairs[x][0]
            property_sold = seatBidPairs[x][1]
            playerState = self.playerStates[player]
            playerState.dollarCards.append(
                dollarCardsOnTable[(len(seatBidPairs) - 1) - x]
            )
            self.removeProperty(player, property_sold)
        if len(self.dollarCardsToDraw) == self.onDollarCard:
            # Game is over!
            self.phase = GamePhase.GAME_OVER
        else:
            self.propertyBid[:] = 0
            # Shuffle cards for the next auction
            self.shuffleNext()

    def getLargestHouse(self, player: int):
        if len(self.playerStates[player].propertyCards) == 0:
            return 0
        return max(self.playerStates[player].propertyCards)

    def removeProperty(self, player: int, property: int):
        for i, p in enumerate(self.playerStates[player].propertyCards):
            if p == property:
                del self.playerStates[player].propertyCards[i]
                return
        assert False

    def getScore(self, player: int) -> float:
        totalMoney = self.money[player] + sum(self.playerStates[player].dollarCards)
        cashMoney = self.money[player]
        return totalMoney + cashMoney / 1000.0
