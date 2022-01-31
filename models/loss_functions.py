import torch.nn as nn
import torch
import re

class SpreadLoss(nn.Module):
    def __init__(self, num_class, dset):
        super(SpreadLoss, self).__init__()
        self.m = 0.2
        self.num_class = num_class
        self.num_iter = 0
        if dset == "NVPD":
            self.denom = 8500.
        elif dset == "ilab2m":
            self.denom = 20000.
        elif re.match("SmallNORB_*", dset):
            self.denom = 50000.

    def forward(self, y_hat, Y):

        Y_onehot = torch.eye(self.num_class, device=y_hat.device).index_select(dim=0, index=Y.squeeze())
        a_t = (Y_onehot * y_hat).sum(dim=1)
        margins = (self.m - (a_t.unsqueeze(1) - y_hat)) * (1 - Y_onehot)
        Loss_perCaps = (torch.max(margins, torch.zeros(margins.shape, device=y_hat.device)) ** 2)
        Loss = Loss_perCaps.sum(dim=1)
        # the m schedule stated in open review: https://openreview.net/forum?id=HJWLfGWRb
        if self.m < 0.9:
            self.m = 0.2 + 0.79 * torch.sigmoid(torch.min(torch.tensor([10, self.num_iter / self.denom - 4])))

        self.num_iter += 1
        return Loss.mean()



def createSpreadLoss(opt):

    return SpreadLoss(opt.DATA.NUM_CLASS, opt.DATA.NAME)


def createCELoss():
    return nn.CrossEntropyLoss()


def get_lossfn(opt):
    return {"cross_entropy": lambda: createCELoss(),
            "spread": lambda: createSpreadLoss(opt)
            }[opt.TRAIN.LOSS_FN.NAME]()
