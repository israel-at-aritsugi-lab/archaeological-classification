from os.path import join

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, dir, image_paths, targets, aug=None, channel_first=False, torgb=True
    ):
        self.dir = dir
        self.image_paths = image_paths
        self.targets = targets
        self.aug = aug
        self.channel_first = channel_first
        self.torgb = torgb

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        targets = self.targets[idx]
        image = Image.open(join(self.dir, self.image_paths[idx]))

        if self.torgb:
            image = image.convert("RGB")

        image = np.array(image)

        if self.aug is not None:
            augmented = self.aug(image=image)
            image = augmented["image"]

        if self.channel_first:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image),
            "targets": torch.tensor(targets),
        }
