import torch
import torch.nn as nn
import models.ResidualBlocks as rb
from utils.utils import count_parameters

class BasicExtractor(nn.Module):

    def __init__(self):
        super(BasicExtractor, self).__init__()
        self.layer1 = rb.BasicPreActResBlock(1, 16, 2)
        self.layer2 = rb.BasicPreActResBlock(16, 32, 2)
        self.layer3 = rb.BasicPreActResBlock(32, 16, 2)
        self.layer4 = rb.BasicPreActResBlock(16, 16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], -1)
        return x


class Deep1Extractor(nn.Module):

    def __init__(self):
        super(Deep1Extractor, self).__init__()
        self.layer1 = rb.BasicPreActResBlock(1, 16, 1)
        self.layer2 = rb.BasicPreActResBlock(16, 16, 2)
        self.layer3 = rb.BasicPreActResBlock(16, 32, 2)
        self.layer4 = rb.BasicPreActResBlock(32, 32, 2)
        self.layer5 = rb.BasicPreActResBlock(32, 64, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], -1)
        return x



def get_feature_extractor(name):
    return {
        "basic": lambda: BasicExtractor(),
        "deep1": lambda: Deep1Extractor()
        }[name]()


if __name__ == '__main__':
    b = get_feature_extractor("deep1")
    print(count_parameters(b))
    x = torch.rand([1, 1, 32, 32])
    print(b(x).shape)



