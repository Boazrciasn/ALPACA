import torch.nn as nn
import torch


class SpreadLoss(nn.Module):
    def __init__(self):
        super(SpreadLoss, self).__init__()
        self.m = 0.2
        self.num_class = 10
        self.num_iter = 0
    def forward(self, y_hat, Y):

        Y_onehot = torch.eye(self.num_class, device=y_hat.device).index_select(dim=0, index=Y.squeeze())
        a_t = (Y_onehot * y_hat).sum(dim=1)
        margins = (self.m - (a_t.unsqueeze(1) - y_hat)) * (1 - Y_onehot)
        Loss_perCaps = (torch.max(margins, torch.zeros(margins.shape, device=y_hat.device)) ** 2)
        Loss = Loss_perCaps.sum(dim=1)
        # the m schedule stated in open review: https://openreview.net/forum?id=HJWLfGWRb
        if self.m < 0.9:
            self.m = 0.2 + 0.79 * torch.sigmoid(torch.min(torch.tensor([10, self.num_iter / 8500. - 4])))

        self.num_iter += 1
        return Loss.mean()


loss_functions = {"cross_entropy": nn.CrossEntropyLoss,
                  "spread": SpreadLoss}
