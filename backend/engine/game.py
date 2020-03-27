#!/usr/bin/env python3

import random
from enum import IntEnum
from typing import Dict, List, Tuple
from uuid import UUID, uuid4

from engine.player import PlayerState


class GamePhase(IntEnum):
    BUYING_HOUSES = 1
    SELLING_HOUSES = 2
    GAME_OVER = 3


class GameState:
    def __init__(self, numPlayers: int):
        self.propertyCardsToDraw = list(range(1, 31))
        self.dollarCardsToDraw = [0, 0] + (list(range(2, 16)) * 2)
        self.playerStates = []
        self.playerIdSeat: Dict[UUID, int] = {}
        self.biddingPlayer = 0
        self.highestBid = -1
        self.numPlayers = numPlayers
        assert numPlayers <= 4 and numPlayers > 0

        for x in range(self.numPlayers):
            playerState = PlayerState(uuid4())
            self.playerStates.append(playerState)
            self.playerIdSeat[playerState.playerId] = x

        random.shuffle(self.propertyCardsToDraw)
        random.shuffle(self.dollarCardsToDraw)

        if self.numPlayers == 3:
            self.propertyCardsToDraw = self.propertyCardsToDraw[:-6]
            self.dollarCardsToDraw = self.dollarCardsToDraw[:-6]
        elif self.numPlayers == 4:
            self.propertyCardsToDraw = self.propertyCardsToDraw[:-2]
            self.dollarCardsToDraw = self.dollarCardsToDraw[:-2]

        for startSort in range(0, len(self.propertyCardsToDraw), self.numPlayers):
            self.propertyCardsToDraw[startSort : startSort + self.numPlayers] = sorted(
                self.propertyCardsToDraw[startSort : startSort + self.numPlayers],
            )
            self.dollarCardsToDraw[startSort : startSort + self.numPlayers] = sorted(
                self.dollarCardsToDraw[startSort : startSort + self.numPlayers],
                reverse=True,
            )

        self.phase = GamePhase.BUYING_HOUSES

    def print(self):
        for player in self.playerStates:
            player.print()
        if self.phase == GamePhase.BUYING_HOUSES:
            print("Houses on auction: " + str(self.getPropertyOnAuction()))
            for i, player in enumerate(self.playerStates):
                print("Player", i, "bid", player.moneyBid, "canbid", player.canBid)
        elif self.phase == GamePhase.SELLING_HOUSES:
            print("Dollar Cards on auction: " + str(self.getDollarsOnAuction()))
            for i, player in enumerate(self.playerStates):
                print("Player", i, "bid", player.propertyBid)

    def getPropertyOnAuction(self):
        assert self.phase == GamePhase.BUYING_HOUSES
        return self.propertyCardsToDraw[0 : self.numBiddersLeft()]

    def getDollarsOnAuction(self):
        assert self.phase == GamePhase.SELLING_HOUSES
        return self.dollarCardsToDraw[0 : self.numPlayers]

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

    def getSeatToAct(self):
        if self.phase == GamePhase.BUYING_HOUSES:
            return self.biddingPlayer
        elif self.phase == GamePhase.SELLING_HOUSES:
            for i, player in enumerate(self.playerStates):
                if player.propertyBid == 0:
                    return i
            assert False, "Should never get here"
        else:
            assert False, "Oops"

    def getPossibleActions(self, playerId: UUID):
        seat = self.playerIdSeat[playerId]
        if self.phase == GamePhase.BUYING_HOUSES:
            assert seat == self.biddingPlayer
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

    def playerAction(self, playerId: UUID, action: int) -> None:
        if self.phase == GamePhase.BUYING_HOUSES:
            biddingPlayer = self.playerStates[self.biddingPlayer]
            assert playerId == biddingPlayer.playerId
            assert biddingPlayer.canBid
            if action == -1:
                # Fold
                boughtProperty = self.propertyCardsToDraw[0]
                del self.propertyCardsToDraw[0]
                biddingPlayer.propertyCards.append(boughtProperty)
                biddingPlayer.money += biddingPlayer.moneyBid // 2
                biddingPlayer.moneyBid = 0
                biddingPlayer.canBid = False

                biddersLeft = self.numBiddersLeft()
                if biddersLeft == 1:
                    for playerState in self.playerStates:
                        if playerState.canBid:
                            boughtProperty = self.propertyCardsToDraw[0]
                            del self.propertyCardsToDraw[0]
                            playerState.propertyCards.append(boughtProperty)
                    # Check for phase end
                    if len(self.propertyCardsToDraw) == 0:
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
            biddingPlayer = self.playerStates[self.playerIdSeat[playerId]]
            assert biddingPlayer.propertyBid == 0
            assert (
                action in biddingPlayer.propertyCards
            ), "Tried to bid a property that you don't own"
            biddingPlayer.propertyBid = action
            self.handlePropertyBidFinish()

    def rotateBidder(self):
        self.biddingPlayer = (self.biddingPlayer + 1) % self.numPlayers
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
        dollarCardsOnTable = list(self.dollarCardsToDraw[0 : self.numPlayers])
        del self.dollarCardsToDraw[0 : self.numPlayers]
        assert len(dollarCardsOnTable) == len(seatBidPairs)
        for x in range(len(seatBidPairs)):
            playerState = self.playerStates[seatBidPairs[x][0]]
            playerState.dollarCards.append(dollarCardsOnTable[x])
            playerState.removeProperty(seatBidPairs[x][1])
        if len(self.dollarCardsToDraw) == 0:
            # Game is over!
            self.phase = GamePhase.GAME_OVER
            for i, player in enumerate(self.playerStates):
                playerScore = player.getScore()
                print(
                    "Player",
                    str(player.playerId)[-4:],
                    "(" + str(i) + ") has a score: ",
                    playerScore,
                )
        else:
            for player in self.playerStates:
                player.propertyBid = 0
