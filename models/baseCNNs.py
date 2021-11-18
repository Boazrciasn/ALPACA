import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(4 if kernel_size == 11 else kernel_size // 2)
        )


class LocalResponseNorm(nn.LocalResponseNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(size=5, alpha=1e-4, beta=0.75, k=2.0)


class BatchNorm(nn.BatchNorm2d):

    def __init__(self, channels, *args, **kwargs):
        super().__init__(num_features=channels)


class Identity(nn.Identity):

    def __init__(self, *args, **kwargs):
        super().__init__()


class AlexNet(nn.Module):

    def __init__(
            self,
            num_classes=1000,
            conv=Conv2d,
            activation=nn.ReLU,
            normalization=LocalResponseNorm,
            pooling=nn.MaxPool2d
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conv = conv
        self.pooling = pooling
        self.activation = activation
        self.normalization = normalization
        self.layers = nn.ModuleList()

        self.layers.append(self.conv(1, 32, 5, 2))
        self.layers.append(self.activation())
        self.layers.append(self.normalization(32))
        self.layers.append(self.pooling(3, 2))
        self.layers.append(self.conv(32, 64, 5, groups=2))
        self.layers.append(self.activation())
        self.layers.append(self.normalization(64))
        self.layers.append(self.pooling(3, 2))
        self.layers.append(self.conv(64, 128, 3))
        self.layers.append(self.activation())
        self.layers.append(self.conv(128, 384, 3, groups=2))
        self.layers.append(self.activation())
        self.layers.append(self.conv(384, 256, 3, groups=2))
        self.layers.append(self.activation())
        self.layers.append(self.pooling(3, 2))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Linear(256 * 7 * 7, 4096))
        self.layers.append(self.activation())
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Linear(4096, 4096))
        self.layers.append(self.activation())
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Linear(4096, self.num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class PytorchNet(nn.Module):
    def __init__(self, num_classes):
        super(PytorchNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, 11)
        self.conv2 = nn.Conv2d(32, 64, 11)
        self.conv3 = nn.Conv2d(64, 64, 5)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 10 * 10, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class smallCNN(nn.Module):
    def __init__(self, num_classes):
        super(smallCNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 16, 11)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x




def create_basecnn(opt):
    return PytorchNet(opt.DATA.NUM_CLASS)

def create_small_cnn(opt):
    return smallCNN(opt.DATA.NUM_CLASS)


