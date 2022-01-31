import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Routing_Methods import EMRouting


# Create receptive field ofssets NO PADDING ASSUMED
# Inputs:
def receptive_offset(imgSize, start, j_out, r_out, M_size):

    receptiveCenters_x = torch.arange(start[0], imgSize[0] - r_out[0] / 2, step=j_out[0])
    receptiveCenters_y = torch.arange(start[1], imgSize[1] - r_out[0] / 2, step=j_out[1])
    receptiveCenters_x = receptiveCenters_x.repeat(receptiveCenters_y.size(0), 1).t()
    receptiveCenters_y = receptiveCenters_y.repeat(receptiveCenters_x.size(0), 1)
    receptiveCenters = torch.cat((receptiveCenters_x.unsqueeze(2), receptiveCenters_y.unsqueeze(2)), 2)
    scale = torch.tensor(imgSize, dtype=torch.float).unsqueeze(0).unsqueeze(1)
    scaled_coords = (receptiveCenters / scale).unsqueeze(2).permute(0, 1, 3, 2)
    scaled_coords = nn.functional.pad(scaled_coords, (M_size[0] - 1, 0, 0, M_size[1] - 2), 'constant', 0)

    return scaled_coords


# Receptive field calculator:
# r     : Receptive field size
# j     : jump on original dimensions(stride but on the original spatial coordinates)
# start : Center of the receptive field of the right top feature(first one.).
# returns receptive field center given current layers stride, padding, kernel size and previous layers r_in, start_in, j_in,
# stride, kernel_size, padding vars are pytorch conv layer compatible.
def receptive_field(stride, kernel_size, padding, r_in=(1, 1), start_in=(0.5, 0.5), j_in=(1, 1)):

    j_out = torch.tensor([j_in[0] * stride[0], j_in[1] * stride[1]], dtype=torch.float)
    r_out = torch.tensor([r_in[0] + (kernel_size[0] - 1) * j_in[0],
                          r_in[1] + (kernel_size[1] - 1) * j_in[1]], dtype=torch.float)

    start_out = torch.tensor([start_in[0] + ((kernel_size[0] - 1) / 2 - padding[0]) * j_in[0],
                              start_in[1] + ((kernel_size[1] - 1) / 2 - padding[1]) * j_in[1]], dtype=torch.float)

    return r_out, start_out, j_out



class FCCapsMatrix(nn.Module):

    def __init__(self, inCaps, outCaps, M_size, routing_iterations, routing, receptive_centers):
        super(FCCapsMatrix, self).__init__()
        self.W = nn.Parameter(torch.randn(1, 1, 1, inCaps, outCaps, *M_size))
        self.Beta_a = nn.Parameter(torch.Tensor(1, outCaps))
        self.Beta_u = nn.Parameter(torch.Tensor(1, outCaps, 1))
        nn.init.uniform_(self.Beta_a.data)
        nn.init.uniform_(self.Beta_u.data)
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.M_size = M_size
        self.routing = routing(routing_iterations)
        self.register_buffer("receptive_centers", receptive_centers)
        self.receptive_centers = self.receptive_centers.unsqueeze(0).unsqueeze(3).unsqueeze(4)

    def forward(self, x, a):
        V = x.unsqueeze(4) @ self.W
        V = V + self.receptive_centers
        V = V.flatten(start_dim=1, end_dim=3)
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        a = a.flatten(start_dim=1, end_dim=a.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)


        return self.routing(V, a, self.Beta_u, self.Beta_a,
                            (x.size(0), self.outCaps, *self.M_size), self.outCaps)


class FCCapsMatrixNorecept(nn.Module):

    def __init__(self, inCaps, outCaps, M_size, routing_iterations, routing):
        super(FCCapsMatrixNorecept, self).__init__()
        self.W = nn.Parameter(torch.randn(1, 1, 1, inCaps, outCaps, *M_size))
        self.Beta_a = nn.Parameter(torch.Tensor(1, outCaps))
        self.Beta_u = nn.Parameter(torch.Tensor(1, outCaps, 1))
        nn.init.uniform_(self.Beta_a.data)
        nn.init.uniform_(self.Beta_u.data)
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.M_size = M_size
        self.routing = routing(routing_iterations)



    def forward(self, x, a):
        V = x.unsqueeze(4) @ self.W
        V = V
        V = V.flatten(start_dim=1, end_dim=3)
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        a = a.flatten(start_dim=1, end_dim=a.dim() - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        return self.routing(V, a, self.Beta_u, self.Beta_a,
                            (x.size(0), self.outCaps, *self.M_size), self.outCaps)

class PrimaryMatCaps(nn.Module):
    def __init__(self, in_channels, outCaps, M_size):
        super(PrimaryMatCaps, self).__init__()
        self.activation_layer = nn.Conv2d(in_channels=in_channels, out_channels=outCaps, kernel_size=1)
        self.pose_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=outCaps * (torch.prod(torch.tensor(M_size, dtype=torch.int)).item()),
                                    kernel_size=1)
        self.M_size = M_size
        self.stride = (1, 1)
        self.kernel_size = (1, 1)
        self.padding = (0, 0)

    def forward(self, x):
        M = self.pose_layer(x)
        a = torch.sigmoid(self.activation_layer(x))
        # reshape ops:
        M = M.permute(0, 2, 3, 1)
        M = M.view(M.size(0), M.size(1), M.size(2), -1, *self.M_size)
        a = a.permute(0, 2, 3, 1)

        return M, a


# W         : <1, 1, Kx, Ky, inCaps, outCaps, Mx,My>
class ConvCapsMatrix(nn.Module):

    def __init__(self, kernel_size, stride, inCaps, outCaps, M_size, routing_iterations, routing):
        super(ConvCapsMatrix, self).__init__()
        self.W = nn.Parameter(torch.rand(1, 1, 1, *kernel_size, inCaps, outCaps, *M_size))
        self.Beta_a = nn.Parameter(torch.Tensor(1, outCaps))
        self.Beta_u = nn.Parameter(torch.Tensor(1, outCaps, 1))
        nn.init.uniform_(self.Beta_a.data)
        nn.init.uniform_(self.Beta_u.data)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (0, 0)
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.M_size = M_size
        self.routing = routing(routing_iterations)

    #   Pose Input dims      : <B, Spatial, Caps, CapsDims<dx,dy>>
    #   Votes dim       : <B, Spatial, Caps_in, Caps_out, CapsDims<dx,dy>>
    def forward(self, x, a):
        # Messy unfold for convolutionally connected capsules. Faster computation this way.
        x = x.unfold(1, self.kernel_size[0], self.stride[0]).unfold(2, self.kernel_size[1], self.stride[1]).permute(0,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    6,
                                                                                                                    7,
                                                                                                                    3,
                                                                                                                    4,
                                                                                                                    5).unsqueeze(6).contiguous()
        a = a.unfold(1, self.kernel_size[0], self.stride[0]).unfold(2, self.kernel_size[1], self.stride[1]).permute(0,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    4,
                                                                                                                    5,
                                                                                                                    3).contiguous()
        V = (x @ self.W)
        V = V.flatten(start_dim=V.dim() - 2, end_dim=V.dim() - 1)

        return self.routing(V, a, self.Beta_u, self.Beta_a,
                            (x.size(0), x.size(1), x.size(2), self.outCaps, *self.M_size), self.outCaps)


# Exactly same network with the paper:
class MatCapNet(nn.Module):

    def __init__(self, inputSize, num_class, opt):
        super(MatCapNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=opt.DATA.CHANNELS, out_channels=32, kernel_size=5, stride=2)
        self.primarycaps = PrimaryMatCaps(in_channels=32, outCaps=32, M_size=(4, 4))
        self.convcapmat1 = ConvCapsMatrix(kernel_size=(3, 3), stride=(2, 2), inCaps=32, outCaps=32, M_size=(4, 4),
                                          routing_iterations=3, routing=EMRouting)
        self.convcapmat2 = ConvCapsMatrix(kernel_size=(3, 3), stride=(1, 1), inCaps=32, outCaps=32, M_size=(4, 4),
                                          routing_iterations=3, routing=EMRouting)
        # Find a better WAY!
        r_out, start_out, j_out = receptive_field(self.conv1.stride, self.conv1.kernel_size, self.conv1.padding)
        r_out, start_out, j_out = receptive_field(self.primarycaps.stride, self.primarycaps.kernel_size,
                                                  self.primarycaps.padding, r_out, start_out, j_out)
        r_out, start_out, j_out = receptive_field(self.convcapmat1.stride, self.convcapmat1.kernel_size,
                                                  self.convcapmat1.padding, r_out, start_out, j_out)
        r_out, start_out, j_out = receptive_field(self.convcapmat2.stride, self.convcapmat2.kernel_size,
                                                  self.convcapmat2.padding, r_out, start_out, j_out)

        scaled_receptive_centers = receptive_offset(imgSize=(inputSize, inputSize), start=start_out, j_out=j_out, M_size=(4, 4),
                                                    r_out=r_out)

        self.classcapmat = FCCapsMatrix(inCaps=32, outCaps=num_class, M_size=(4, 4),
                                        routing_iterations=3,
                                        routing=EMRouting,
                                        receptive_centers=scaled_receptive_centers)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primarycaps(F.relu(x))
        x = self.convcapmat1(x[0], x[1])
        x = self.convcapmat2(x[0], x[1])
        x = self.classcapmat(x[0], x[1])

        return x



def create_matcap():
    opt = get_cfg_baseline_defaults()
    return MatCapNet(opt.DATA.SIZE, opt.DATA.NUM_CLASS), opt
