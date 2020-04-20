#!/usr/bin/env python3

from typing import List

import torch


class ExpandableTensorSet:
    def __init__(self, capacity: int, dims: List[int]):
        self.filled = 0
        self.features = torch.zeros((capacity, dims[0]), dtype=torch.float)
        self.active_labels = torch.zeros((capacity, dims[1]), dtype=torch.float)
        self.labels = torch.zeros((capacity, dims[2]), dtype=torch.float)

    @property
    def capacity(self):
        return self.features.size()[0]

    def append(self, tensors: List[torch.Tensor]):
        features, active_labels, labels = tensors
        assert len(features.size()) == 2
        assert len(active_labels.size()) == 2
        assert len(labels.size()) == 2
        num_new_features = features.size()[0]
        while (self.filled + num_new_features) > self.features.size()[0]:
            self.features = torch.cat(
                (self.features, torch.zeros_like(self.features)), dim=0
            )
            self.active_labels = torch.cat(
                (self.active_labels, torch.zeros_like(self.active_labels)), dim=0
            )
            self.labels = torch.cat((self.labels, torch.zeros_like(self.labels)), dim=0)
            print("Expanding training data to", self.features.size()[0])
        self.features[self.filled : self.filled + num_new_features] = features
        if active_labels is not None:
            self.active_labels[
                self.filled : self.filled + num_new_features
            ] = active_labels
        self.labels[self.filled : self.filled + num_new_features] = labels
        self.filled += num_new_features

    def cat(self, other):
        self.append(other.features, other.active_labels, other.labels)

    def getFilled(self):
        return (
            self.features[0 : self.filled],
            self.active_labels[0 : self.filled],
            self.labels[0 : self.filled],
        )

    def reset(self):
        self.filled = 0
        self.features.fill_(0.0)
        self.labels.fill_(0.0)
        self.active_labels.fill_(0.0)
