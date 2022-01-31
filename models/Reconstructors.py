import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import count_parameters
#for teest
from utils.config import *


class DeconvDecoder(nn.Module):
    def __init__(self, opt):
        super(DeconvDecoder, self).__init__()
        self.linear1 = nn.Linear(opt.MODEL.FEAT_SIZE * opt.DATA.NUM_CLASS, 8*8*16)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(16, 16, kernel_size=(4, 4))
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(16, 16, kernel_size=(4, 4))
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), stride=2)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.deconv4 = nn.ConvTranspose2d(16, opt.DATA.CHANNELS, kernel_size=(4, 4), padding=1)


    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = x.reshape(-1, 16, 8, 8)
        x = self.leakyrelu(self.batchnorm1(self.deconv1(x)))
        x = self.leakyrelu(self.batchnorm2(self.deconv2(x)))
        x = self.leakyrelu(self.batchnorm3(self.deconv3(x)))
        x = self.leakyrelu(self.deconv4(x))
        #x = F.sigmoid(x)
        return x







if __name__ == '__main__':
    inp_t = torch.FloatTensor(32, 32*15)
    opt = get_cfg_model_v1()
    opt.DATA.CHANNELS = 3
    opt.DATA.NUM_CLASS = 15
    model = DeconvDecoder(opt)
    print(count_parameters(model))
    x = model(inp_t)
    print(x.shape)
