'''
Mnist is a dataset resource provided by the torch library,
Mini-ImageNet is acquired from https://www.kaggle.com/datasets/arjunashok33/miniimagenet?resource=download,
Cifar10 is a dataset resource provided by the torch library,
NEU-Set (NEU surface defect database) is acquired from http://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm.
'''

import os
import cv2
import pyrootutils
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import torchvision
from torchvision import datasets, transforms
from Data.process_to_txt import label_pair


# Obtaining the root path ..\VersionTorch
root = pyrootutils.setup_root(search_from=__file__,
                            indicator=["pyproject.toml"],
                            pythonpath=True,
                            dotenv=True)


# Creating NEU-Set (Go ahead and generate neu_cls.txt) 
class NEUDataset(Dataset):
    def __init__(self, root_dir, transform):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transform
        self.images = []
        self.labels = {"Cr": 0, "In": 1, "Pa": 2, "PS": 3, "RS": 4, "Sc": 5}
        with open(os.path.join(self.root_dir, 'neu_cls.txt'), 'r') as f:
            for i in range(1800):
                line = f.readline()
                line = line.rstrip()
                line = line.split(' ')
                self.images.append([line[-1], line[1]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_index = self.images[index]
        img_path = os.path.join(self.root_dir, img_index[0])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1)
        label = self.labels[img_index[1]]
        if self.transforms:
            img = self.transforms(img)
        return img, label


# Creating Mini-ImageNet (Go ahead and generate mini-imagenet.txt) 
class MINIDataset(Dataset):
    def __init__(self, root_dir, transform):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transform
        self.images = []
        self.labels = label_pair
        with open(os.path.join(self.root_dir, 'mini-imagenet.txt'), 'r') as f:
            for i in range(60000):
                line = f.readline()
                line = line.rstrip()
                line = line.split(' ')
                self.images.append([line[-1], line[1]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_index = self.images[index]
        img_path = os.path.join(self.root_dir, img_index[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[img_index[1]]
        if self.transforms:
            img = self.transforms(img)
        return img, label
    pass


# Create DataLoaders (DataModules)
class DataM():
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # MNIST
        if self.params.dataset == "mnist":
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            self.train_dataset = datasets.MNIST(
                os.path.join(root, "Data"),
                train=True,
                download=True,
                transform=transform)

            # For testing ASR
            self.Random1 = self.train_dataset

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [50000, 10000])
            self.test_dataset = datasets.MNIST(
                os.path.join(root, "Data"),
                train=False,
                download=True,
                transform=transform)
            # For testing ASR
            self.Random2 = self.test_dataset
            
            print("Start from Dataset MNIST: Train size is %s " % (len(self.train_dataset)))
            print("Start from Dataset MNIST: Validation size is %s " % (len(self.val_dataset)))
            print("Start from Dataset MNIST: Testing size is %s " % (len(self.test_dataset)))
            self.all_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset, self.test_dataset])

        # Mini-ImageNet
        elif self.params.dataset == "mini-imagenet":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

            self.train_dataset = MINIDataset(root_dir=os.path.join(root, "Data"), transform=transform)

            # For testing ASR
            self.Random1 = self.Random2 = self.train_dataset

            self.train_dataset, self.other_dataset = torch.utils.data.random_split(self.train_dataset, [50000, 10000])
            self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.other_dataset, [5000, 5000])
            print("Start from Dataset MINI-IMAGRNET: Train size is %s " % (len(self.train_dataset)))
            print("Start from Dataset MINI-IMAGRNET: Validation size is %s " % (len(self.val_dataset)))
            print("Start from Dataset MINI-IMAGRNET: Testing size is %s " % (len(self.test_dataset)))
            self.all_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset, self.test_dataset])

        # NEU-Set
        elif self.params.dataset == "surface":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.train_dataset = NEUDataset(root_dir=os.path.join(root, "Data"),transform=transform)

            # For testing ASR
            self.Random1 = self.Random2 = self.train_dataset

            self.train_dataset, self.other_dataset = torch.utils.data.random_split(self.train_dataset, [1260, 540])
            self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.other_dataset, [270, 270])
            print('Start from NEU-CLS Surface: Train size-%s' % (len(self.train_dataset)))
            print('Start from NEU-CLS Surface: Validation size-%s' % (len(self.val_dataset)))
            print('Start from NEU-CLS Surface: Testing size-%s' % (len(self.test_dataset)))
            self.all_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset, self.test_dataset])

        # CIFAR10
        elif self.params.dataset == "cifar10":
            transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            self.train_dataset = datasets.CIFAR10(root=os.path.join(root, "Data"), train=True, download=True, transform=transform)

            # For testing ASR
            self.Random1 = self.train_dataset

            self.test_dataset = datasets.CIFAR10(root=os.path.join(root, "Data"), train=False, download=True, transform=transform)

            # For testing ASR
            self.Random2 = self.train_dataset

            self.test_dataset, self.val_dataset = torch.utils.data.random_split(self.test_dataset, [5000, 5000])
            print('Start from Dataset Cifar10: Train size-%s' % (len(self.train_dataset)))
            print('Start from Dataset Cifar10: Validation size-%s' % (len(self.val_dataset)))
            print('Start from Dataset Cifar10: Testing size-%s' % (len(self.test_dataset)))
            self.all_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset, self.test_dataset])

    def return_train_loader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            shuffle=True
        )

    def return_val_loader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            shuffle=False
        )

    def return_test_loader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            shuffle=False
        )

    def return_all_loader(self):
        return DataLoader(
            dataset=self.all_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            shuffle=True
        )

    def return_all_dataset(self):
        return self.Random1, self.Random2