#!/usr/bin/env python3

import random
from typing import Iterable, List, Tuple

import torch


class ReservoirBuffer:
    def __init__(self, capacity: int, all_tensor_dims: List[int]):
        self.buffers = []
        for dim in all_tensor_dims:
            self.buffers.append(torch.zeros((capacity, dim), dtype=torch.float))
        self.filled = 0
        self.observed = 0

    @property
    def capacity(self) -> int:
        return self.buffers[0].size()[0]

    def append(self, new_entries: List[torch.Tensor]):
        num_new_entries = new_entries[0].size()[0]
        if self.filled + num_new_entries <= self.capacity:
            # Enough room to add all new entries
            for x, entry in enumerate(new_entries):
                self.buffers[x][self.filled : self.filled + num_new_entries] = entry
            self.filled += num_new_entries
            self.observed += num_new_entries
        elif self.filled < self.capacity:
            num_to_add = self.capacity - self.filled
            for x, entry in enumerate(new_entries):
                self.buffers[x][self.filled : self.filled + self.capacity] = entry[
                    0:num_to_add
                ]
            self.filled = self.capacity
            self.observed += num_to_add
            remaining = []
            for entry in new_entries:
                remaining.append(entry[num_to_add:])
            self.append(remaining)
        else:
            for i in range(num_new_entries):
                m = random.randint(0, self.observed)
                if m < self.capacity:
                    for j, entry in enumerate(new_entries):
                        self.buffers[j][m] = entry[i]
                self.observed += 1

    def getFilled(self) -> List[torch.Tensor]:
        retval = []
        for buffer in self.buffers:
            retval.append(buffer[0 : self.filled])
        return retval

    def cat(self, other):
        self.append(other.getFilled())
