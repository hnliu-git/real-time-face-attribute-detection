
from torch.optim import AdamW
from torchvision import models
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class MultiBinMobileNet(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

    def __init__(self, n_classes):
        super().__init__()

        self.save_hyperparameters()
        self.n_classes = n_classes
        mnet = models.mobilenet_v2()

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.base_model = mnet.features
        self.fcs = [nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=mnet.last_channel, out_features=2)).cuda()
                    for _ in range(n_classes)]

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return [fc(x) for fc in self.fcs]

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=1e-1, patience=2, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):
        img, attrs = train_batch
        output = self.forward(img)
        attrs = torch.unbind(attrs, 1)

        train_loss = self.get_loss(output, attrs)
        # total_loss += loss_train.item()
        return {"train_loss": train_loss}

    def validation_step(self, val_batch, batch_idx):
        img, attrs = val_batch
        output = self.forward(img)
        attrs = torch.unbind(attrs, 1)

        print(output[0].shape)
        print(attrs[0].shape)
        val_loss = self.get_loss(output, attrs)
        # total_loss += loss_train.item()
        return {"val_loss", val_loss}

    def get_loss(self, output, truth):
        losses = sum([F.cross_entropy(output[i], truth[i]) for i in range(self.n_classes)])
        return losses


class MultiTagMobileNet(pl.LightningModule):

    def __init__(self, n_classes):
        super().__init__()

        self.save_hyperparameters()
        self.n_classes = n_classes
        mnet = models.mobilenet_v2()

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.base_model = mnet.features
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=mnet.last_channel, out_features=n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return self.fc(x)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=1e-1, patience=2, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):
        # Img [bsz, w, h, c]
        img, attrs = train_batch
        # [bsz, n_classes]
        output = self.forward(img)

        train_loss = self.get_loss(output, attrs)
        return {'loss': train_loss}

    def validation_step(self, val_batch, batch_idx):
        img, attrs = val_batch
        output = self.forward(img)

        val_loss = self.get_loss(output, attrs)
        avg_acc, min_acc, max_acc = calculate_tag_metrics(output, attrs)

        return {"val_loss": val_loss, 'avg_acc': avg_acc, 'min_acc': min_acc, 'max_acc': max_acc}

    def validation_epoch_end(self, outputs):
        """"""
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = sum([x['avg_acc'] for x in outputs]) / len(outputs)
        max_acc = max([x['max_acc'] for x in outputs])
        min_acc = min([x['min_acc'] for x in outputs])
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        self.log("avg_acc", avg_acc, prog_bar=True, logger=True)
        self.log("max_acc", max_acc, prog_bar=True, logger=True)
        self.log("min_acc", min_acc, prog_bar=True, logger=True)

    def get_loss(self, output, truth):
        loss = F.binary_cross_entropy(output, truth)
        return loss
