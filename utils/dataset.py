import numpy as np
from torchvision import datasets, transforms
import os


data_dir = os.path.join(os.environ['HOME'], "datasets")


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iMNIST(iData):
    use_path = False  # Indicate whether to use paths or in-memory data
    train_trsf = [
        transforms.RandomCrop(28, padding=4),  # Random cropping with padding for MNIST
        transforms.RandomRotation(15),  # Random rotation for augmentation
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,), std=(0.3081,)
        ),  # Normalization for MNIST dataset
    ]

    class_order = np.arange(10).tolist()  # Classes from 0 to 9

    def download_data(self):
        train_dataset = datasets.MNIST(data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data.numpy(), np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data.numpy(), np.array(
            test_dataset.targets
        )


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(data_dir, train=True, download=False)
        test_dataset = datasets.cifar.CIFAR10(data_dir, train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(data_dir, train=True, download=False)
        test_dataset = datasets.cifar.CIFAR100(data_dir, train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


