import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

def get_mnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def load_mnist_datasets(root='./data'):
    transform = get_mnist_transforms()
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def create_subset_datasets(train_dataset, test_dataset, num_samples_train=60000, num_samples_test=10000):
    indices_train = np.random.choice(len(train_dataset), num_samples_train, replace=False)
    indices_test = np.random.choice(len(test_dataset), num_samples_test, replace=False)
    subset_train_dataset = Subset(train_dataset, indices_train)
    subset_test_dataset = Subset(test_dataset, indices_test)
    return subset_train_dataset, subset_test_dataset

def create_data_loaders(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def check_image_range(loader):
    min_val, max_val = float('inf'), float('-inf')
    for images, _ in loader:
        batch_min, batch_max = images.min().item(), images.max().item()
        min_val, max_val = min(min_val, batch_min), max(max_val, batch_max)
        break
    return min_val, max_val

def plot_sample_images(loader):
    for images, labels in loader:
        plt.figure(figsize=(5, 5))
        for i in range(4):
            ax = plt.subplot(2, 2, i + 1)
            plt.imshow(images[i].squeeze(), cmap='viridis')
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')
        plt.show()
        break

def load_mnist(num_samples_train=60000, num_samples_test=10000, batch_size=64):
    train_dataset, test_dataset = load_mnist_datasets()
    
    print(f"Total number of training images: {len(train_dataset)}")
    print(f"Total number of testing images: {len(test_dataset)}")
    
    subset_train_dataset, subset_test_dataset = create_subset_datasets(
        train_dataset, test_dataset, num_samples_train, num_samples_test
    )
    
    print(f"Total number of training images in subset: {len(subset_train_dataset)}")
    print(f"Total number of testing images in subset: {len(subset_test_dataset)}")
    
    train_loader, test_loader = create_data_loaders(subset_train_dataset, subset_test_dataset, batch_size)
    
    train_min, train_max = check_image_range(train_loader)
    test_min, test_max = check_image_range(test_loader)
    
    print(f"Training data range: min = {train_min}, max = {train_max}")
    print(f"Testing data range: min = {test_min}, max = {test_max}")
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_mnist()
    plot_sample_images(train_loader)
