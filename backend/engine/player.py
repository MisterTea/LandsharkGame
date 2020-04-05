#!/usr/bin/env python3

from typing import Dict, List, Tuple
from uuid import UUID, uuid4


class PlayerState:
    __slots__ = [
        "propertyCards",
        "dollarCards",
    ]

    def __init__(self):
        self.propertyCards = []
        self.dollarCards = []
