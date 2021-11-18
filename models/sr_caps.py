import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SelfRouting2d(nn.Module):
    def __init__(self, A, B, C, D, kernel_size=3, stride=1, padding=1, pose_out=False):
        super(SelfRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.pose_out = pose_out

        if pose_out:
            self.W1 = nn.Parameter(torch.FloatTensor(self.kkA, B*D, C))
            nn.init.kaiming_uniform_(self.W1.data)

        self.W2 = nn.Parameter(torch.FloatTensor(self.kkA, B, C))
        self.b2 = nn.Parameter(torch.FloatTensor(1, 1, self.kkA, B))

        nn.init.constant_(self.W2.data, 0)
        nn.init.constant_(self.b2.data, 0)

    def forward(self, a, pose):
        # a: [b, A, h, w]
        # pose: [b, AC, h, w]
        b, _, h, w = a.shape

        # [b, ACkk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, C, kk, l]
        pose = pose.view(b, self.A, self.C, self.kk, l)
        # [b, l, kk, A, C]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, C, 1]
        pose = pose.view(b, l, self.kkA, self.C, 1)

        if hasattr(self, 'W1'):
            # [b, l, kkA, BD]
            pose_out = torch.matmul(self.W1, pose).squeeze(-1)
            # [b, l, kkA, B, D]
            pose_out = pose_out.view(b, l, self.kkA, self.B, self.D)

        # [b, l, kkA, B]
        logit = torch.matmul(self.W2, pose).squeeze(-1) + self.b2

        # [b, l, kkA, B]
        r = torch.softmax(logit, dim=3)

        # [b, kkA, l]
        a = F.unfold(a, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a = a.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a = a.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA, 1]
        a = a.view(b, l, self.kkA, 1)

        # [b, l, kkA, B]
        ar = a * r
        # [b, l, 1, B]
        ar_sum = ar.sum(dim=2, keepdim=True)
        # [b, l, kkA, B, 1]
        coeff = (ar / (ar_sum)).unsqueeze(-1)

        # [b, l, B]
        # a_out = ar_sum.squeeze(2)
        a_out = ar_sum / a.sum(dim=2, keepdim=True)
        a_out = a_out.squeeze(2)

        # [b, B, l]
        a_out = a_out.transpose(1,2)

        if hasattr(self, 'W1'):
            # [b, l, B, D]
            pose_out = (coeff * pose_out).sum(dim=2)
            # [b, l, BD]
            pose_out = pose_out.view(b, l, -1)
            # [b, BD, l]
            pose_out = pose_out.transpose(1,2)

        oh = ow = math.floor(l**(1/2))

        a_out = a_out.view(b, -1, oh, ow)
        if hasattr(self, 'W1'):
            pose_out = pose_out.view(b, -1, oh, ow)
        else:
            pose_out = None

        return a_out, pose_out


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        channels, classes = 1, 10
        self.conv1 = nn.Conv2d(channels, 256, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.num_caps = 16

        planes = 16
        last_size = 6

        self.conv_a = nn.Conv2d(256, self.num_caps, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(256, self.num_caps*planes, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(self.num_caps)
        self.bn_pose = nn.BatchNorm2d(self.num_caps*planes)

        self.conv_caps = SelfRouting2d(self.num_caps, self.num_caps, planes, planes, kernel_size=3, stride=2, padding=1, pose_out=True)
        self.bn_pose_conv_caps = nn.BatchNorm2d(self.num_caps*planes)

        self.fc_caps = SelfRouting2d(self.num_caps, classes, planes, 1, kernel_size=last_size, padding=0, pose_out=False)

        self.apply(weights_init)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        a, pose = self.conv_a(out), self.conv_pose(out)
        a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

        a, pose = self.conv_caps(a, pose)
        pose = self.bn_pose_conv_caps(pose)

        a, _ = self.fc_caps(a, pose)

        out = a.view(a.size(0), -1)
        out = out.log()

        return out


