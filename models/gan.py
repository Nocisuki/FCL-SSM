from torch import nn
from torch.nn import functional as F


class Critic(nn.Module):
    def __init__(self, image_size, image_channel_size, channel_size, num_classes):
        super(Critic, self).__init__()

        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(
            image_channel_size, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(
            channel_size, channel_size * 2,
            kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(
            channel_size * 2, channel_size * 4,
            kernel_size=4, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.fc_dis = nn.Linear((image_size // 8) ** 2 * channel_size * 4, 1)
        self.fc_aux = nn.Linear((image_size // 8) ** 2 * channel_size * 4, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, if_features=False):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)

        x = x.view(-1, (self.image_size // 8) ** 2 * self.channel_size * 4)
        features = x
        realfake = self.sigmoid(self.fc_dis(x)).squeeze()
        classes = self.softmax(self.fc_aux(x))
        logits = self.fc_aux(x)

        if not if_features:
            return realfake, classes, logits
        else:
            return realfake, classes, logits, features


class Generator(nn.Module):
    def __init__(self, z_size, image_size, image_channel_size, channel_size):
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.fc = nn.Linear(z_size, (image_size // 8) ** 2 * channel_size * 8)
        self.bn0 = nn.BatchNorm2d(channel_size * 8)
        self.deconv1 = nn.ConvTranspose2d(
            channel_size * 8, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_size * 4)
        self.deconv2 = nn.ConvTranspose2d(
            channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)
        self.deconv3 = nn.ConvTranspose2d(
            channel_size * 2, channel_size, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size)
        self.deconv4 = nn.ConvTranspose2d(
            channel_size, image_channel_size, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        g = F.relu(self.bn0(self.fc(z).view(
            z.size(0), self.channel_size * 8, self.image_size // 8, self.image_size // 8)))
        g = F.relu(self.bn1(self.deconv1(g)))
        g = F.relu(self.bn2(self.deconv2(g)))
        g = F.relu(self.bn3(self.deconv3(g)))
        g = self.deconv4(g)
        return F.tanh(g)


def get_z_size(dataset_name):
    if dataset_name == "mnist":
        return 64
    elif dataset_name == "cifar10":
        return 128
    elif dataset_name == "cifar100":
        return 256
    else:
        raise ValueError("Unsupported dataset!")


def get_generator(dataset_name):
    if dataset_name == "mnist":
        return Generator(z_size=64, image_size=28, image_channel_size=1, channel_size=64)
    elif dataset_name == "cifar10":
        return Generator(z_size=128, image_size=32, image_channel_size=3, channel_size=128)
    elif dataset_name == "cifar100":
        return Generator(z_size=256, image_size=32, image_channel_size=3, channel_size=256)
    else:
        raise ValueError("Unsupported dataset!")


def get_critic(dataset_name):
    if dataset_name == "mnist":
        return Critic(
            image_size=28,
            image_channel_size=1,
            channel_size=64,
            num_classes=10
        )
    elif dataset_name == "cifar10":
        return Critic(
            image_size=32,
            image_channel_size=3,
            channel_size=128,
            num_classes=10
        )
    elif dataset_name == "cifar100":
        return Critic(
            image_size=32,
            image_channel_size=3,
            channel_size=256,
            num_classes=80
        )
    else:
        raise ValueError("Unsupported dataset!")
