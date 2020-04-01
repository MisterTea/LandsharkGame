#!/usr/bin/env python3

import copy
import random
from enum import IntEnum
from typing import Dict, List, Tuple
from uuid import UUID, uuid4

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
    ]

    def __init__(self, num_players: int):
        self.propertyCardsToDraw = list(range(1, 31))
        self.dollarCardsToDraw = [0, 0] + (list(range(2, 16)) * 2)
        self.biddingPlayer = 0
        self.highestBid = -1
        self.num_players = num_players
        assert num_players <= 4 and num_players > 0

        playerStates = []
        for x in range(self.num_players):
            playerState = PlayerState()
            playerStates.append(playerState)
        self.playerStates = tuple(playerStates)

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
        cloned_game = copy.copy(self)
        cloned_game.playerStates = []
        for ps in self.playerStates:
            cloned_game.playerStates.append(copy.deepcopy(ps))
        return cloned_game

    def reset(self):
        random.shuffle(self.propertyCardsToDraw)
        random.shuffle(self.dollarCardsToDraw)

        if self.num_players == 3:
            self.onPropertyCard = 6
            self.onDollarCard = 6
        elif self.num_players == 4:
            self.onPropertyCard = 2
            self.onDollarCard = 2

        for startSort in range(
            self.onPropertyCard, len(self.propertyCardsToDraw), self.num_players
        ):
            self.propertyCardsToDraw[startSort : startSort + self.num_players] = sorted(
                self.propertyCardsToDraw[startSort : startSort + self.num_players],
            )
            self.dollarCardsToDraw[startSort : startSort + self.num_players] = sorted(
                self.dollarCardsToDraw[startSort : startSort + self.num_players],
                reverse=True,
            )

    def print(self):
        for seat, player in enumerate(self.playerStates):
            player.print(seat)
        if self.phase == GamePhase.BUYING_HOUSES:
            print("Houses on auction: " + str(self.getPropertyOnAuction()))
            for i, player in enumerate(self.playerStates):
                print("Player", i, "bid", player.moneyBid, "canbid", player.canBid)
        elif self.phase == GamePhase.SELLING_HOUSES:
            print("Dollar Cards on auction: " + str(self.getDollarsOnAuction()))
            for i, player in enumerate(self.playerStates):
                print("Player", i, "bid", player.propertyBid)

    def action_dim(self):
        return 1 + 19 + 30  # fold, bet 0-18, bet 1-30 properties

    def terminal(self):
        return self.phase == GamePhase.GAME_OVER

    def payoff(self, player: int):
        return float(self.playerStates[player].getScore()[0])

    def get_one_hot_actions(self, hacks=True):
        player = self.playerStates[self.get_player_to_act()]
        actions = self.getPossibleActions(self.get_player_to_act())
        actions_one_hot = torch.zeros((self.action_dim(),), dtype=torch.int)
        if self.phase == GamePhase.BUYING_HOUSES and player.moneyBid > 0 and hacks:
            # HACK: Let's fold after 1 round of bidding to speed up training
            actions_one_hot[0] = 1
        else:
            for a in actions:
                if self.phase == GamePhase.BUYING_HOUSES:
                    # HACK: Do not ever bid more than 6
                    if a <= 6 or hacks == False:
                        actions_one_hot[1 + a] = 1
                elif self.phase == GamePhase.SELLING_HOUSES:
                    actions_one_hot[20 + (a - 1)] = 1
                else:
                    assert False, "Oops"
        return actions_one_hot

    def feature_dim(self):
        return 77

    def populate_features(self, features: torch.Tensor):
        player = self.playerStates[self.get_player_to_act()]
        cursor = 0
        if self.phase == GamePhase.BUYING_HOUSES:
            features[cursor] = player.money
            cursor += 1
            t = torch.tensor(self.getPropertyOnAuction())
            features[cursor + (t - 1)] = 1.0
            cursor += 30
        else:
            cursor += 31
        if self.phase == GamePhase.SELLING_HOUSES:
            for d in self.getDollarsOnAuction():
                # d==1 is not possible but just let it be 0
                features[cursor + d] += 1.0
            cursor += 16
            t = torch.tensor(player.propertyCards)
            features[cursor + (t - 1)] = 1.0
            cursor += 30

        return features

    def get_player_to_act(self):
        if self.phase == GamePhase.BUYING_HOUSES:
            return self.biddingPlayer
        elif self.phase == GamePhase.SELLING_HOUSES:
            for i, player in enumerate(self.playerStates):
                if player.propertyBid == 0:
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
                bestHouse = player.getLargestHouse()
            else:
                newBestHouse = player.getLargestHouse()
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
                if player.propertyBid == 0:
                    bids_needed.append(i)
            assert len(bids_needed) > 0, "Should never get here"
            return bids_needed
        else:
            assert False, "Oops"

    def getPossibleActions(self, seat: int):
        if self.phase == GamePhase.BUYING_HOUSES:
            assert seat == self.biddingPlayer
            assert self.playerStates[seat].canBid
            if self.highestBid == -1:
                # Can't fold if there are no bids
                return list(
                    range(self.highestBid + 1, self.playerStates[seat].money + 1)
                )
            else:
                return [-1] + list(
                    range(self.highestBid + 1, self.playerStates[seat].money + 1)
                )
        elif self.phase == GamePhase.SELLING_HOUSES:
            assert self.playerStates[seat].propertyBid == 0
            return list(self.playerStates[seat].propertyCards)
        else:
            assert False, "Oops"

    def numBiddersLeft(self):
        biddersLeft = 0
        for playerState in self.playerStates:
            if playerState.canBid:
                biddersLeft += 1
        assert biddersLeft > 0
        return biddersLeft

    def playerAction(self, seat: int, action: int) -> None:
        if self.phase == GamePhase.BUYING_HOUSES:
            assert seat == self.biddingPlayer
            biddingPlayer = self.playerStates[self.biddingPlayer]
            assert biddingPlayer.canBid
            if action == -1:
                # Fold
                boughtProperty = self.propertyCardsToDraw[self.onPropertyCard]
                self.onPropertyCard += 1
                biddingPlayer.propertyCards.append(boughtProperty)
                biddingPlayer.money += biddingPlayer.moneyBid // 2
                biddingPlayer.moneyBid = 0
                biddingPlayer.canBid = False

                biddersLeft = self.numBiddersLeft()
                if biddersLeft == 1:
                    for playerState in self.playerStates:
                        if playerState.canBid:
                            boughtProperty = self.propertyCardsToDraw[
                                self.onPropertyCard
                            ]
                            self.onPropertyCard += 1
                            playerState.propertyCards.append(boughtProperty)
                    # Check for phase end
                    if len(self.propertyCardsToDraw) == self.onPropertyCard:
                        self.phase = GamePhase.SELLING_HOUSES
                    else:
                        # Reset the auction
                        self.biddingPlayer = self.getPlayerSeatWithLargestHouse()
                        self.highestBid = -1
                        for playerState in self.playerStates:
                            playerState.moneyBid = 0
                            playerState.canBid = True
                else:
                    # Keep going
                    self.rotateBidder()
            else:
                # Raise
                raiseAmount = action - biddingPlayer.moneyBid
                assert action > self.highestBid
                assert raiseAmount > 0 or (raiseAmount == 0 and self.highestBid == -1)
                biddingPlayer.moneyBid = action
                biddingPlayer.money -= raiseAmount
                assert biddingPlayer.money >= 0
                self.highestBid = action
                # Move on to next player
                self.rotateBidder()
        elif self.phase == GamePhase.SELLING_HOUSES:
            biddingPlayer = self.playerStates[seat]
            assert biddingPlayer.propertyBid == 0
            assert (
                action in biddingPlayer.propertyCards
            ), "Tried to bid a property that you don't own"
            biddingPlayer.propertyBid = action
            self.handlePropertyBidFinish()

    def rotateBidder(self):
        self.biddingPlayer = (self.biddingPlayer + 1) % self.num_players
        if self.playerStates[self.biddingPlayer].canBid == False:
            self.rotateBidder()

    def handlePropertyBidFinish(self):
        # Check if everyone has placed a bid
        for player in self.playerStates:
            if player.propertyBid == 0:
                # Not done yet
                return
        # Execute auction
        seatBidPairs: List[Tuple[int, int]] = []
        for seat, player in enumerate(self.playerStates):
            seatBidPairs.append((seat, player.propertyBid))
        seatBidPairs = sorted(seatBidPairs, key=lambda sbp: sbp[1], reverse=True)
        dollarCardsOnTable = list(
            self.dollarCardsToDraw[
                self.onDollarCard : self.onDollarCard + self.num_players
            ]
        )
        self.onDollarCard += self.num_players
        assert len(dollarCardsOnTable) == len(seatBidPairs)
        for x in range(len(seatBidPairs)):
            playerState = self.playerStates[seatBidPairs[x][0]]
            playerState.dollarCards.append(dollarCardsOnTable[x])
            playerState.removeProperty(seatBidPairs[x][1])
        if len(self.dollarCardsToDraw) == self.onDollarCard:
            # Game is over!
            self.phase = GamePhase.GAME_OVER
        else:
            for player in self.playerStates:
                player.propertyBid = 0
