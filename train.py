import yaml
import os
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import MultiBinMobileNet, MultiTagMobileNet
from dataset import CelebDataset, valid_transform, train_transform

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

cfgs = yaml.load(open("configs/train.yaml"), Loader=yaml.FullLoader)

img_folder = os.path.join(cfgs['data_folder'], 'img_align_celeba/img_align_celeba')
attr_csv = os.path.join(cfgs['data_folder'], 'list_attr_celeba.csv')

attrs = pd.read_csv('data/list_attr_celeba.csv').replace(-1, 0)

labels = attrs.columns[1:]
n_classes = len(labels)
# id2class = {i:classes[i+1] for i in range(n_classes)}


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_df, test = train_test_split(attrs, test_size=0.1, shuffle=True, random_state=cfgs['seed'])
valid_df, test_df = train_test_split(test, test_size=0.5, random_state=cfgs['seed'])

train_data = CelebDataset(train_df, img_folder, train_transform)
valid_data = CelebDataset(valid_df, img_folder, valid_transform)
test_data = CelebDataset(test_df, img_folder, valid_transform)

pl.seed_everything(cfgs['seed'])

train_loader=DataLoader(train_data,batch_size=cfgs['batch_size'],shuffle=True,num_workers=2)
valid_loader=DataLoader(valid_data,batch_size=cfgs['batch_size'],num_workers=2)
test_loader=DataLoader(test_data,batch_size=cfgs['batch_size'],num_workers=2)

if cfgs['tagging']:
    print("Train a multi-tagging model")
    model = MultiTagMobileNet(labels, float(cfgs['lr']))
else:
    print("Train a multi-clf model")
    model = MultiBinMobileNet(n_classes, float(cfgs['lr']))

# Callbacks:
checkpoint = ModelCheckpoint(
    dirpath=cfgs['model_save_dir'],
    filename="./fx-{epoch:02d}-{val_loss:.7f}",
    monitor="val_loss"
)

earlystopping = EarlyStopping(monitor='val_loss',
                              min_delta=0.01,
                              patience=5,
                              verbose=False,
                              mode="min")

trainer = pl.Trainer(
    gpus=1,
    max_epochs=int(cfgs['n_epochs']),
    enable_checkpointing=True,
    check_val_every_n_epoch=1,
    callbacks=[checkpoint, earlystopping, LearningRateMonitor()]
)

trainer.fit(model, train_loader, valid_loader)
