
from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
import torch


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, training=True):
        self.image_list = []
        self.id_list = []
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = 0
        self.training = training
        if training:
            movie_data = pd.read_csv(csv_file)

            for idx, row in movie_data.iterrows():
                img_name = row['Id'] + '.jpg'
                label = []
                for lbl in row[2:]:
                    label.append(lbl)
                self.image_list.append(img_name)
                self.id_list.append(label)
            self.num_classes = len(self.id_list[0])
        else:
            for img_name in os.listdir(root_dir):
                self.image_list.append(img_name)
                self.id_list.append(0)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = torch.Tensor(self.id_list[idx])
        img_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, img_name
