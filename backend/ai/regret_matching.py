#!/usr/bin/env python3

import copy
import random
from enum import IntEnum
from typing import Dict, List, Tuple
from uuid import UUID, uuid4

import numpy as np
import pytorch_lightning as pl
import torch


class RegretMatching(pl.LightningModule):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(feature_dim, 128),
                torch.nn.Linear(128, 64),
                torch.nn.Linear(64, action_dim),
            ]
        )
        self.activations = torch.nn.ModuleList(
            [torch.nn.ReLU(), torch.nn.ReLU(), None,]
        )
        # self.layers = torch.nn.ModuleList([torch.nn.Linear(feature_dim, action_dim),])
        # self.activations = torch.nn.ModuleList([None])

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if self.activations[i] is not None:
                x = self.activations[i](x)
        return x

    def train_model(
        self, features, active_labels, labels, model_name, output_file=None
    ):
        full_dataset = torch.utils.data.TensorDataset(features, active_labels, labels)
        dataset_size = len(full_dataset)
        test_size = dataset_size // 5
        train_size = dataset_size - test_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )
        print(
            "TRAINING ON", dataset_size, len(self.train_dataset), len(self.val_dataset)
        )
        trainer = pl.Trainer(
            gpus=1,
            early_stop_callback=True,
            max_epochs=1000,
            default_save_path=os.path.join(os.getcwd(), "models", model_name),
        )
        trainer.fit(self)
        if output_file is not None:
            trainer.save_checkpoint(output_file)
        self.train_dataset = self.val_dataset = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1024,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1024 * 1024 * 1024,
            drop_last=False,
            pin_memory=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        features, active_labels, labels = batch
        outputs = self(features) * active_labels.float()
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(outputs, labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        train_results = self.training_step(batch, batch_idx)
        return {"val_loss": train_results["loss"]}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}
