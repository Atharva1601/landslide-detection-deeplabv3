import os
import cv2
import torch
from torch.utils.data import Dataset


class LandslideDataset(Dataset):
    def __init__(self, split_file, img_dir, mask_dir, transforms=None):
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f]

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        img_path = os.path.join(self.img_dir, id_ + ".png")
        mask_path = os.path.join(self.mask_dir, id_ + ".png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        # normalize mask to 0/1
        mask = (mask > 0).astype("float32")

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # add channel to mask
        mask = mask.unsqueeze(0)

        return image, mask
