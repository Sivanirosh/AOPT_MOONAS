from torchvision import datasets

class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, train, transform, data_root="./data"):
        super().__init__(root=data_root, train=train, download=True, transform=transform)
