
import os
import numpy as np


from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CelebDataset(Dataset):
    def __init__(self, df, image_path, transform=None, mode='train'):
        super().__init__()
        self.attr = df.drop(['image_id'], axis=1)
        self.img_folder = image_path
        self.image_id = df['image_id']
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return self.image_id.shape[0]

    def __getitem__(self, idx: int):
        image_name = self.image_id.iloc[idx]
        image = Image.open(os.path.join(self.img_folder, image_name))
        attributes = np.asarray(self.attr.iloc[idx].T, dtype=np.float32)
        if self.transform:
            image = self.transform(image)
        return image, attributes

    # function to visualize dataset


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)])

valid_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)])