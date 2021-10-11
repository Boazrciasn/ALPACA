import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import models.ResidualBlocks as rb
from utils.device_setting import device
from models.Routing_Methods import EMRouting, EMRouting_old
from models.modules import QuaternionLayer


eps = 1e-10

# Pure quaternions --> 3 dimensional caps
class PrimaryQuatCaps(nn.Module):
    def __init__(self, in_channels_pose, in_channels_act, outCaps, quat_dims=3):
        super(PrimaryQuatCaps, self).__init__()
        self.activation_layer = nn.Conv2d(in_channels=in_channels_act, out_channels=outCaps, kernel_size=1)
        self.pose_layer = nn.Conv2d(in_channels=in_channels_pose,
                                    out_channels=outCaps * 3,
                                    kernel_size=1)

        torch.nn.init.xavier_uniform_(self.pose_layer.weight)
        self.batchNormPose = nn.BatchNorm2d(quat_dims * outCaps)
        self.batchNormA = nn.BatchNorm2d(outCaps)
        self.quat_dims = quat_dims
        self.stride = (1, 1)
        self.kernel_size = (1, 1)

    def forward(self, x, a):
        M = self.batchNormPose(self.pose_layer(x))
        a = torch.sigmoid(self.batchNormA(self.activation_layer(a)))
        M = M.permute(0, 2, 3, 1)
        M = M.view(M.size(0), M.size(1), M.size(2), -1, self.quat_dims)
        a = a.permute(0, 2, 3, 1)

        return M, a

# Pure quaternions --> 3 dimensional caps
class PrimaryQuatCapsNobranch(nn.Module):
    def __init__(self, in_channels, outCaps, quat_dims=3):
        super(PrimaryQuatCapsNobranch, self).__init__()
        self.activation_layer = nn.Conv2d(in_channels=in_channels, out_channels=outCaps, kernel_size=1)
        self.pose_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=outCaps * 3,
                                    kernel_size=1)

        torch.nn.init.xavier_uniform_(self.pose_layer.weight)
        self.batchNormPose = nn.BatchNorm2d(quat_dims * outCaps)
        self.batchNormA = nn.BatchNorm2d(outCaps)
        self.quat_dims = quat_dims
        self.stride = (1, 1)
        self.kernel_size = (1, 1)

    def forward(self, x):
        M = self.batchNormPose(self.pose_layer(x))
        a = torch.sigmoid(self.batchNormA(self.activation_layer(x)))
        M = M.permute(0, 2, 3, 1)
        M = M.view(M.size(0), M.size(1), M.size(2), -1, self.quat_dims)
        a = a.permute(0, 2, 3, 1)

        return M, a

class ConvQuaternionCapsLayer(QuaternionLayer):
    def __init__(self, kernel_size, stride, inCaps, outCaps, routing_iterations, routing, init_type):
        super(ConvQuaternionCapsLayer, self).__init__()

        self.W_theta = nn.Parameter(torch.zeros(1, 1, 1, *kernel_size, inCaps, outCaps, 1, 1, 1))
        if init_type == "uniform_pi":
            nn.init.uniform_(self.W_theta, -math.pi, math.pi)
        elif init_type == "normal":
            nn.init.normal_(self.W_theta)

        self.W_hat = nn.Parameter(torch.zeros(1, 1, 1, *kernel_size, inCaps, outCaps, 3, 1, 1))
        nn.init.uniform_(self.W_hat, -1, 1)
        self.Beta_a = nn.Parameter(torch.zeros(1, outCaps))
        self.Beta_u = nn.Parameter(torch.zeros(1, outCaps, 1))

        self.kernel_size = kernel_size
        self.stride = stride

        self.inCaps = inCaps
        self.outCaps = outCaps
        self.routing = routing(routing_iterations).to(device)


    #   Pose Input dims      : <B, Spatial, Caps, Quaternion>
    #   Votes dim       : <B, Spatial, Caps_in, Caps_out, Quaternion>
    def forward(self, x, a):

        x = x.unfold(1, self.kernel_size[0], self.stride[0]).unfold(2, self.kernel_size[1], self.stride[1]) \
            .permute(0,
                     1,
                     2,
                     5,
                     6,
                     3,
                     4).unsqueeze(6).unsqueeze(8).contiguous()


        a = a.unfold(1, self.kernel_size[0], self.stride[0]).unfold(2, self.kernel_size[1], self.stride[1]).permute(0,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    4,
                                                                                                                    5,
                                                                                                                    3).contiguous()

        W_unit = torch.div(self.W_hat, torch.norm(self.W_hat, dim=7, keepdim=True))
        W_theta_sin = torch.sin(self.W_theta) + eps
        W_theta_cos = torch.cos(self.W_theta) + eps
        W_rotor = torch.cat((W_theta_cos, W_theta_sin * W_unit), dim=7)
        W_ = torch.sum(self.quatEmbedder * W_rotor, dim=7)
        W_conj = self.left2right * W_.permute(0, 1, 2, 3, 4, 5, 6, 8, 7)
        V = W_conj @ W_ @ x
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1)

        return self.routing(V, a, self.Beta_u, self.Beta_a,
                            (x.size(0), x.size(1), x.size(2), self.outCaps, 4, 1), self.outCaps)


class FCQuatCaps(QuaternionLayer):

    def __init__(self, inCaps, outCaps, quat_dims, routing_iterations, routing, init_type):
        super(FCQuatCaps, self).__init__()

        self.W_theta = nn.Parameter(torch.zeros(1, 1, 1, inCaps, outCaps, 1, 1, 1))

        if init_type == "uniform_pi":
            nn.init.uniform_(self.W_theta, -math.pi, math.pi)
        elif init_type == "normal":
            nn.init.normal_(self.W_theta)

        self.W_hat = nn.Parameter(torch.zeros(1, 1, 1, inCaps, outCaps, 3, 1, 1))
        nn.init.uniform_(self.W_hat, -1, 1)
        self.Beta_a = nn.Parameter(torch.zeros(1, outCaps))
        self.Beta_u = nn.Parameter(torch.zeros(1, outCaps, 1))
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.quat_dims = quat_dims
        self.routing = routing(routing_iterations)



    def forward(self, x, a):
        W_unit = torch.div(self.W_hat, torch.norm(self.W_hat, dim=5, keepdim=True))
        W_theta_sin = torch.sin(self.W_theta) + eps
        W_theta_cos = torch.cos(self.W_theta) + eps
        W_rotor = torch.cat((W_theta_cos, W_theta_sin * W_unit), dim=5)
        W_ = torch.sum(self.quatEmbedder * W_rotor, dim=5)  # MAY TRANSFER THIS TO CTOR.
        W_conj = self.left2right * W_.permute(0, 1, 2, 3, 4, 6, 5)
        V = W_conj @ W_ @ x.unsqueeze(4)
        V = V.flatten(start_dim=1, end_dim=3)
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        a = a.flatten(start_dim=1, end_dim=a.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        return self.routing(V, a, self.Beta_u, self.Beta_a,
                            (x.size(0), self.outCaps, self.quat_dims, 1), self.outCaps)




class FCQuaternionLayer(QuaternionLayer):

    def __init__(self, inCaps, outCaps, quat_dims, routing_iterations, routing, init_type):
        super(FCQuaternionLayer, self).__init__()

        self.W_theta = nn.Parameter(torch.zeros(1,  inCaps, outCaps, 1, 1, 1))

        if init_type == "uniform_pi":
            nn.init.uniform_(self.W_theta, -math.pi, math.pi)
        elif init_type == "normal":
            nn.init.normal_(self.W_theta)

        self.W_hat = nn.Parameter(torch.zeros(1, inCaps, outCaps, 3, 1, 1))
        nn.init.uniform_(self.W_hat, -1, 1)
        self.Beta_a = nn.Parameter(torch.zeros(1, outCaps))
        self.Beta_u = nn.Parameter(torch.zeros(1, outCaps, 1))
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.quat_dims = quat_dims
        self.routing = routing(routing_iterations)
        self.left2right = self.left2right.flatten(start_dim=1, end_dim=5)


    # INPUT SHAPE:  <B, InCaps, Quaternion>, <B, InCaps, Activation>
    def forward(self, x, a):
        W_unit = torch.div(self.W_hat, torch.norm(self.W_hat, dim=3, keepdim=True))
        W_theta_sin = torch.sin(self.W_theta) + eps
        W_theta_cos = torch.cos(self.W_theta) + eps
        W_rotor = torch.cat((W_theta_cos, W_theta_sin * W_unit), dim=3)
        W_ = torch.sum(self.quatEmbedder * W_rotor, dim=4)
        W_conj = self.left2right * W_.permute(0, 1, 2, 4, 3)
        V = W_conj @ W_ @ x.unsqueeze(2).unsqueeze(-1)
        V = V.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        a = a.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        return self.routing(V.squeeze(-1), a, self.Beta_u, self.Beta_a,
                            (x.size(0), self.outCaps, self.quat_dims, 1), self.outCaps)


class ConvQuaternionCapsLayer_old(QuaternionLayer):
    def __init__(self, kernel_size, stride, inCaps, outCaps, routing_iterations, routing):
        super(ConvQuaternionCapsLayer_old, self).__init__()

        self.W_theta = nn.Parameter(torch.zeros(1, 1, 1, *kernel_size, inCaps, outCaps, 1, 1, 1))
        nn.init.uniform_(self.W_theta, -math.pi, math.pi)
        self.W_hat = nn.Parameter(torch.zeros(1, 1, 1, *kernel_size, inCaps, outCaps, 3, 1, 1))
        nn.init.uniform_(self.W_hat, -1, 1)
        self.Beta_a = nn.Parameter(torch.zeros(1, outCaps))
        self.Beta_u = nn.Parameter(torch.zeros(1, outCaps, 1))

        self.kernel_size = kernel_size
        self.stride = stride

        self.inCaps = inCaps
        self.outCaps = outCaps
        self.routing = routing(routing_iterations).to(device)


    #   Pose Input dims      : <B, Spatial, Caps, Quaternion>
    #   Votes dim       : <B, Spatial, Caps_in, Caps_out, Quaternion>
    def forward(self, x, a):

        x = x.unfold(1, self.kernel_size[0], self.stride[0]).unfold(2, self.kernel_size[1], self.stride[1]) \
            .permute(0,
                     1,
                     2,
                     5,
                     6,
                     3,
                     4).unsqueeze(6).unsqueeze(8).contiguous()


        a = a.unfold(1, self.kernel_size[0], self.stride[0]).unfold(2, self.kernel_size[1], self.stride[1]).permute(0,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    4,
                                                                                                                    5,
                                                                                                                    3).contiguous()

        W_unit = torch.div(self.W_hat, torch.norm(self.W_hat, dim=7, keepdim=True))

        W_theta_sin = torch.sin(self.W_theta) + eps
        W_theta_cos = torch.cos(self.W_theta) + eps
        W_rotor = torch.cat((W_theta_cos, W_theta_sin * W_unit), dim=7)
        W_ = torch.sum(self.quatEmbedder * W_rotor, dim=7)
        W_conj = self.left2right * W_.permute(0, 1, 2, 3, 4, 5, 6, 8, 7)
        V = W_conj @ W_ @ x
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1)

        R = (torch.ones(*a.size(), self.outCaps) / self.outCaps).to(device)
        a = a.unsqueeze(6)
        return self.routing(V, a, self.Beta_u, self.Beta_a, R,
                            (x.size(0), x.size(1), x.size(2), self.outCaps, 4, 1))


class FCQuatCaps_old(QuaternionLayer):

    def __init__(self, inCaps, outCaps, quat_dims, routing_iterations, routing):
        super(FCQuatCaps_old, self).__init__()

        self.W_theta = nn.Parameter(torch.zeros(1, 1, 1, inCaps, outCaps, 1, 1, 1))
        nn.init.uniform_(self.W_theta, -math.pi, math.pi)
        self.W_hat = nn.Parameter(torch.zeros(1, 1, 1, inCaps, outCaps, 3, 1, 1))
        nn.init.uniform_(self.W_hat, -1, 1)
        self.Beta_a = nn.Parameter(torch.zeros(1, outCaps))
        self.Beta_u = nn.Parameter(torch.zeros(1, outCaps, 1))
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.quat_dims = quat_dims
        self.routing = routing(routing_iterations)



    def forward(self, x, a):
        W_unit = torch.div(self.W_hat, torch.norm(self.W_hat, dim=5, keepdim=True))
        W_theta_sin = torch.sin(self.W_theta) + eps
        W_theta_cos = torch.cos(self.W_theta) + eps
        W_rotor = torch.cat((W_theta_cos, W_theta_sin * W_unit), dim=5)
        W_ = torch.sum(self.quatEmbedder * W_rotor, dim=5)  # MAY TRANSFER THIS TO CTOR.
        W_conj = self.left2right * W_.permute(0, 1, 2, 3, 4, 6, 5)
        V = W_conj @ W_ @ x.unsqueeze(4)
        V = V.flatten(start_dim=1, end_dim=3)
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        a = a.flatten(start_dim=1, end_dim=a.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        R = (torch.ones(*a.size(), self.outCaps) / self.outCaps).to(device)  # R: <B, Spatial, Kernel Size, Caps_in, Caps_out>
        a = a.unsqueeze(6)

        return self.routing(V, a, self.Beta_u, self.Beta_a, R,
                            (x.size(0), self.outCaps, self.quat_dims, 1))




class QCN(nn.Module):

    def forward(self, x):

        if self.opt.MODEL.BRANCHED:
            x_pose = self.resblock2_pose(self.resblock1_pose(x))
            x_activation = self.resblock1_activation(x)
            x = self.primarycaps(F.relu(x_pose), F.relu(x_activation))
        else:
            x = self.resblock2(self.resblock1(x))
            x = self.primarycaps(F.relu(x))


        z = torch.zeros(x[0].size(0), x[0].size(1), x[0].size(2), x[0].size(3), 1).to(device)
        x_quat = torch.cat((z, x[0]), 4)

        x1 = self.convquatcapmat1(x_quat, x[1])
        l2_output = self.convquatcapmat2(x1[0].squeeze(-1), x1[1])
        l3_output = self.convquatcapmat3(l2_output[0].squeeze(-1), l2_output[1])
        x = self.classquatcapmat(l3_output[0].flatten(start_dim=1, end_dim=3).squeeze(-1), l3_output[1].flatten(start_dim=1, end_dim=3))

        return x

    def __init__(self, opt):
        super(QCN, self).__init__()
        self.opt = opt
        if self.opt.MODEL.BRANCHED:
            self.resblock1_pose = rb.BasicPreActResBlock(1, 32, 1)
            self.resblock1_activation = rb.BasicPreActResBlock(1, 32, 2)
            self.resblock2_pose = rb.BasicPreActResBlock(32, 64, 2)
            self.primarycaps = PrimaryQuatCaps(in_channels_pose=64, in_channels_act=32, outCaps=32)

        else:

            self.resblock1 = rb.BasicPreActResBlock(1, 64, 1)
            self.resblock2 = rb.BasicPreActResBlock(64, 64 + 32, 2)
            self.primarycaps = PrimaryQuatCapsNobranch(in_channels=96, outCaps=32)




        self.convquatcapmat1 = ConvQuaternionCapsLayer(kernel_size=(5, 5), stride=(1, 1), inCaps=32, outCaps=16,
                                                       routing_iterations=self.opt.MODEL.ROUTING_IT,
                                                       routing=EMRouting, init_type=self.opt.MODEL.INIT_TYPE)

        self.convquatcapmat2 = ConvQuaternionCapsLayer(kernel_size=(5, 5), stride=(1, 1), inCaps=16, outCaps=16,
                                                       routing_iterations=self.opt.MODEL.ROUTING_IT,
                                                       routing=EMRouting, init_type=self.opt.MODEL.INIT_TYPE)

        self.convquatcapmat3 = ConvQuaternionCapsLayer(kernel_size=(5, 5), stride=(1, 1), inCaps=16, outCaps=16,
                                                       routing_iterations=self.opt.MODEL.ROUTING_IT,
                                                       routing=EMRouting, init_type=self.opt.MODEL.INIT_TYPE)

        self.classquatcapmat = FCQuaternionLayer(inCaps=256, outCaps=self.opt.DATA.NUM_CLASS, quat_dims=4, routing_iterations=self.opt.MODEL.ROUTING_IT,
                                                 routing=EMRouting, init_type=self.opt.MODEL.INIT_TYPE)





class MatQuatCapNet(nn.Module):

    def __init__(self):
        super(MatQuatCapNet, self).__init__()


        self.resblock1_pose = rb.BasicPreActResBlock(1, 32, 1)
        self.resblock2_pose = rb.BasicPreActResBlock(32, 64, 2)
        self.resblock1_activation = rb.BasicPreActResBlock(1, 32, 2)
        self.primarycaps = PrimaryQuatCaps(in_channels_pose=64, in_channels_act=32, outCaps=32)

        self.convquatcapmat1 = ConvQuaternionCapsLayer_old(kernel_size=(5, 5), stride=(1, 1), inCaps=32, outCaps=16,
                                                       routing_iterations=2,
                                                       routing=EMRouting_old)

        self.convquatcapmat2 = ConvQuaternionCapsLayer_old(kernel_size=(5, 5), stride=(1, 1), inCaps=16, outCaps=16,
                                                       routing_iterations=2,
                                                       routing=EMRouting_old)

        self.convquatcapmat3 = ConvQuaternionCapsLayer_old(kernel_size=(5, 5), stride=(1, 1), inCaps=16, outCaps=16,
                                                       routing_iterations=2,
                                                       routing=EMRouting_old)

        self.classquatcapmat = FCQuatCaps_old(inCaps=16, outCaps=10, quat_dims=4, routing_iterations=2, routing=EMRouting_old)

    def forward(self, x):

        x_pose = self.resblock2_pose(self.resblock1_pose(x))
        x_activation = self.resblock1_activation(x)
        x = self.primarycaps(F.relu(x_pose), F.relu(x_activation))
        z = torch.zeros(x[0].size(0), x[0].size(1), x[0].size(2), x[0].size(3), 1).to(device)
        x_quat = torch.cat((z, x[0]), 4)

        x1 = self.convquatcapmat1(x_quat, x[1])
        l2_output = self.convquatcapmat2(x1[0].squeeze(), x1[1])
        l3_output = self.convquatcapmat3(l2_output[0].squeeze(), l2_output[1])

        x = self.classquatcapmat(l3_output[0], l3_output[1])

        return x

# class st_qcn(nn.Module):
#
#     def __init__(self):
#         super(st_qcn, self).__init__()
#
#         self.resblock1_pose = rb.BasicPreActResBlock(3, 32, 1)
#         self.resblock1_activation = rb.BasicPreActResBlock(3, 32, 2)
#         self.resblock2_pose = rb.BasicPreActResBlock(32, 64, 2)
#         self.primarycaps = PrimaryQuatCaps(in_channels_pose=64, in_channels_act=32, outCaps=32)
#         # self.st_cap = STRoutedQCLayer(inCaps=32, outCaps=5, quat_dims=3)
#
#     def forward(self, x):
#
#         x_pose = self.resblock2_pose(self.resblock1_pose(x))
#         x_activation = self.resblock1_activation(x)
#         x = self.primarycaps(F.relu(x_pose), F.relu(x_activation))
#
#
#         z = torch.zeros(x[0].size(0), x[0].size(1), x[0].size(2), x[0].size(3), 1).to(device)
#         x_quat = torch.cat((z, x[0]), 4)
#
#         out = self.st_cap(x_quat.unsqueeze(5), x[1].unsqueeze(4))
#         out = F.softmax(out)
#         return None, out, None


def create_qcn(opt):
    return MatQuatCapNet()


if __name__ == '__main__':
    from utils.config import get_cfg_qcn
    opt = get_cfg_qcn()

    qcn = create_qcn(opt)
    from utils.utils import count_parameters
    print(count_parameters(qcn))
    x = torch.rand([4, 1, 32, 32])
    qcn(x)
