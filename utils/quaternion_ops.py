import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Equivalent to inner product: d(q1,q2) = arccos(|q1.q2|) where . is the dot product and || is abs.
# INPUT MUST BE UNIT QUATERNIONS
# Q1 ~ V
# Distance code is referenced from https://github.com/tolgabirdal/qecnetworks/blob/master/models/qec_module.py
def geodesic_dist(Q1, Q2):
    print("batch geodesic dist,")
    dots = (Q1.unsqueeze(dim=3) @ Q2.unsqueeze(dim=-1)).squeeze(dim=-1)
    distance = torch.arccos(
        torch.clamp(torch.abs(dots), max=0.9999))  # Equivalent to 1 - | q1 . q2 |  --> Consider this.
    distance = 2 * distance / np.pi
    return distance


# outer product is equivalent to matrix multiplication if arranged correctly.
# check if works with pure quaternions,
# Distance code is referenced from https://github.com/tolgabirdal/qecnetworks/blob/master/models/qec_module.py
# MAY BE UNSTABLE DUE TO SYMEIG on similer eigenvalues. But we only use largest and do not backpropagate over other eigenvalues
def quaternion_average(Q1, weights):
    outer = weights[(...,) + (None,) * 2] * (Q1.unsqueeze(dim=-1) @ Q1.unsqueeze(dim=3))
    M = outer.sum(1, keepdim=True)
    eigs, eigvect = torch.symeig(M.view(-1, M.shape[-2], M.shape[-1]), eigenvectors=True)
    avg_quats = eigvect[:, -1, :].view(M.shape[:-1])
    normalized_eigs = torch.div(eigs, eigs.sum(dim=1).unsqueeze(1))
    normalized_max_eig = normalized_eigs[:, -1].view(M.shape[0], M.shape[-3], 1)
    return normalized_max_eig, avg_quats

class PoseAttRouter(nn.Module):
    def __init__(self):
        super(PoseAttRouter, self).__init__()


    # INPUT:
    # votes:    tuple<  activation: <B, in_caps, out_caps, 1>
    #                   quaternion: <B, in_caps, out_caps, 4> ---> last dimension is a pure quaternion
    #                   feature   : <B, in_caps, out_caps, feat_size>  >
    def forward(self, votes):
        print("heyo")
        A, Q, V = votes  #activations(A), quaternion poses (Q), feature vectors(V)
        norm_eigs, K = self.quaternion_average(Q, A)
        out_a = F.softmax(norm_eigs, dim=1)
        dists = self.geodesic_dist(Q, K)
        att_weights = F.softmax(1/dists, dim=1)
        out_f = (att_weights*V).mean(dim=1)
        out_q = K.squeeze(1)
        return out_a, out_q, out_f


    # Equivalent to inner product: d(q1,q2) = arccos(|q1.q2|) where . is the dot product and || is abs.
    # INPUT MUST BE UNIT QUATERNIONS
    # Q1 ~ V
    # Distance code is referenced from https://github.com/tolgabirdal/qecnetworks/blob/master/models/qec_module.py
    def geodesic_dist(self, Q1, Q2):
        print("batch geodesic dist,")
        dots = (Q1.unsqueeze(dim=3) @ Q2.unsqueeze(dim=-1)).squeeze(dim=-1)
        distance = torch.arccos(
            torch.clamp(torch.abs(dots), max=0.9999))  # Equivalent to 1 - | q1 . q2 |  --> Consider this.
        distance = 2 * distance / np.pi
        return distance

    # outer product is equivalent to matrix multiplication if arranged correctly.
    # check if works with pure quaternions,
    # Distance code is referenced from https://github.com/tolgabirdal/qecnetworks/blob/master/models/qec_module.py
    # MAY BE UNSTABLE DUE TO SYMEIG on similar eigenvalues. But we only use largest and do not backpropagate over other eigenvalues
    def quaternion_average(self, Q1, weights):

        outer = weights[(...,) + (None,) * 2] * (Q1.unsqueeze(dim=-1) @ Q1.unsqueeze(dim=3))
        M = outer.sum(1, keepdim=True)
        eigs, eigvect = torch.symeig(M.view(-1, M.shape[-2], M.shape[-1]), eigenvectors=True)
        normalized_eigs = torch.div(eigs, eigs.sum(dim=1).unsqueeze(1))
        normalized_max_eig = normalized_eigs[:, -1].view(M.shape[0], M.shape[-3], 1)
        avg_quats = eigvect[:, -1, :].view(M.shape[:-1])

        return normalized_max_eig, avg_quats


if __name__ == '__main__':
    print("quaternion test")
    z = torch.zeros(4, 32, 32, 1)
    quats = torch.rand([4, 32, 32, 3])
    quats = torch.div(quats, torch.norm(quats, dim=3, keepdim=True))
    quats = torch.cat((z, quats), dim=-1)
    acts = torch.rand([4, 32, 1])
    feats = torch.rand(4, 32, 32, 128)
    att_router = PoseAttRouter()
    out_a, out_q, out_f = att_router((acts, quats, feats))

