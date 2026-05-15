import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def make_mnist_loader(
    root: str = "data",
    train: bool = True,
    batch_size: int = 1024,
    shuffle: bool = False,
    download: bool = True,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),
    ])

    dataset = MNIST(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )