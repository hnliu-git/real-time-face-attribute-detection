
from torch.optim import AdamW
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import calculate_metrics, calculate_tag_metrics

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class MultiBinMobileNet(pl.LightningModule):

    def __init__(self, labels, lr):
        super().__init__()

        self.save_hyperparameters()
        self.labels = labels
        self.n_classes = len(labels)
        self.lr = lr
        mnet = models.mobilenet_v2()

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.base_model = mnet.features
        self.fcs = [nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=mnet.last_channel, out_features=2)).cuda()
                    for _ in range(self.n_classes)]

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return [fc(x) for fc in self.fcs]

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=1e-1, patience=2, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):
        # Img [bsz, w, h, c]
        img, attrs = train_batch
        output = self.forward(img)
        attrs = torch.unbind(attrs, 1)

        train_loss = self.get_loss(output, attrs)
        return {'loss': train_loss}

    def validation_step(self, val_batch, batch_idx):
        img, attrs = val_batch
        output = self.forward(img)
        attrs = torch.unbind(attrs, 1)

        val_loss = self.get_loss(output, attrs)
        accs = calculate_metrics(output, attrs, agg=False)

        return {"val_loss": val_loss, 'accs': accs}

    def validation_epoch_end(self, outputs):
        """"""
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_accs = sum([x['accs'] for x in outputs]) / len(outputs)
        for i, acc in enumerate(avg_accs):
          self.log(self.labels[i], acc, prog_bar=True, logger=True)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)

    def get_loss(self, output, truth):
        losses = sum(
            [F.cross_entropy(output[i], truth[i].type(torch.LongTensor).cuda()) for i in range(self.n_classes)])
        return losses


class MultiTagMobileNet(pl.LightningModule):

    def __init__(self, labels, lr):
        super().__init__()

        self.save_hyperparameters()
        self.labels = labels
        self.n_classes = len(labels)
        self.lr = lr
        mnet = models.mobilenet_v2()

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.base_model = mnet.features
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=mnet.last_channel, out_features=self.n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return self.fc(x)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
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
        accs = calculate_tag_metrics(output, attrs, agg=False)

        return {"val_loss": val_loss, 'accs': accs}

    def validation_epoch_end(self, outputs):
        """"""
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_accs = sum([x['accs'] for x in outputs]) / len(outputs)
        for i, acc in enumerate(avg_accs):
          self.log(self.labels[i], acc, prog_bar=True, logger=True)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)

    def get_loss(self, output, truth):
        loss = F.binary_cross_entropy(output, truth)
        return loss

