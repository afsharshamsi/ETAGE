import torch
from torchvision.datasets import CIFAR10
from decouple import config
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



def load_data_camelyon(preprocess, train, val, test, device):
    # train = CIFAR10(root, download=True, train=True)
    # train_indices, validation_indices = train_test_split(range(len(train)), test_size=0.2, random_state=42)
    # train_set = torch.utils.data.Subset(train, train_indices)
    # validation_set = torch.utils.data.Subset(train, validation_indices)
    # train_set.dataset.transform = preprocess
    # validation_set.dataset.transform = preprocess
    # test = CIFAR10(root, download=True, train=False, transform=preprocess)

    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test.classes]).to(device)
    print(f'trainset size {len(train)}')
    print(f'validation_set size {len(val)}')
    print(f'test size {len(test)}')
    
    batch_size = int(config('batch_size'))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


def load_data_cifar(preprocess, train, test, device):

    random.seed(2295)

    # train = CIFAR10(root, download=True, train=True)
    train_indices, validation_indices = train_test_split(range(len(train)), test_size=0.2, random_state=42)
    train_set = torch.utils.data.Subset(train, train_indices)
    validation_set = torch.utils.data.Subset(train, validation_indices)
    train_set.dataset.transform = preprocess
    validation_set.dataset.transform = preprocess
   
    # print(f'trainset size {len(train_set)}')
    # print(f'validation_set size {len(validation_set)}')
    print(f'test size {len(test)}')
    
    batch_size = int(config('batch_size'))

    trainloaders = [torch.utils.data.DataLoader(train_set, batch_size=int(config('batch_size')), shuffle=True) for i in range(int(config('opt')))]
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return trainloaders, validation_loader, test_loader


def load_data_places(preprocess, train_data, val_data, test_data, device):
    # trainloaders = [train_data.pytorch(num_workers = 0, shuffle = True, transform = {'images': preprocess, 'labels': None}, batch_size = int(config('batch_size')), decode_method = {'images': 'pil'})  for i in range(int(config('opt')))]
    trainloader = train_data.pytorch(num_workers = 0, shuffle = True, transform = {'images': preprocess, 'labels': None}, batch_size = int(config('batch_size')), decode_method = {'images': 'pil'})
    validation_loader = val_data.pytorch(num_workers = 0, shuffle = False, transform = {'images': preprocess, 'labels': None}, batch_size = int(config('batch_size')), decode_method = {'images': 'pil'})
    test_loader = test_data.pytorch(num_workers = 4, shuffle = False, transform = {'images': preprocess, 'labels': None}, batch_size = int(config('batch_size')), decode_method = {'images': 'pil'})

    return trainloader, validation_loader, test_loader

