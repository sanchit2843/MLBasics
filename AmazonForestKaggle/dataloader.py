from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from params import *
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import os
import torch

data = pd.read_csv(path_to_csv)
tags = data['tags']
tags = tags.str.split()
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(tags)
print(img_dir)
#custom dataset class
class Amazon_dataset(Dataset):
    def __init__(self,image_dir,y_train,transform = None):

        self.img_dir = image_dir
        self.y_train = y_train
        self.transform = transform
        self.id = os.listdir(self.img_dir)
    def __len__(self):
        return len(os.listdir(self.img_dir))
    def __getitem__(self,idx):
        img_name = os.path.join(self.img_dir, self.id[idx])
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(self.y_train[idx])
        return image,label

# Data transform
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((im_size,im_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
#Data object
amazon_data = Amazon_dataset(img_dir,y_train,transform)
data_len = len(amazon_data)
indices = list(range(data_len))
np.random.shuffle(indices)
split1 = int(np.floor(test_size * data_len))
test_idx, train_idx = indices[:split1], indices[split1:]
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)
train_loader = DataLoader(amazon_data, batch_size=batch_size , sampler=train_sampler)
test_loader = DataLoader(amazon_data, batch_size=batch_size , sampler=test_sampler)

# code to calculate mean and standard deviation of dataset
'''
mean = 0.
std = 0.
from tqdm import tqdm
nb_samples = len(amazon_data)
for data,_ in tqdm(dataloader):
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)

mean /= nb_samples
std /= nb_samples
'''
