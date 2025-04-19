from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import torch

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    

def getDataLoaders(dataset_name='cifar10', batch_size=128, image_size=32, num_workers=2):
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        DatasetClass = datasets.CIFAR10
    elif dataset_name == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        DatasetClass = datasets.CIFAR100
    else:
        raise ValueError("Invalid dataset name. Choose 'cifar10' or 'cifar100'.")

    # Data augmentation + resize for ResNet/ViT
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = DatasetClass(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = DatasetClass(root='./data', train=False, download=True, transform=test_transform)

    trainLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker
    )
    testLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker
    )

    return trainLoader, testLoader
