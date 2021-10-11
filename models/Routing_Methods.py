import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
import engine.engine as engine

eps = 1e-7


class EMRouting(nn.Module):

    def __init__(self, iterations=3):
        super(EMRouting, self).__init__()
        self.iterations = iterations
        self.final_lambda = 0.01
        self.register_buffer("mathpilog", torch.log(torch.FloatTensor([2 * math.pi])))
        self.register_buffer("R", torch.ones(1, 1))
        self.mathpilog.detach()
        self.R.detach()

    def forward(self, V, a, Beta_u, Beta_a, outSize, outCaps):
        self.R = (torch.ones(*a.size(), outCaps, device=a.device) / outCaps)
        a = a.unsqueeze(6)
        for i in range(self.iterations):
            Lambda = self.final_lambda * (1 - 0.95 ** (i + 1))

            # M - Step:
            self.R = (self.R * a).unsqueeze(7)
            mu_share = (self.R * V).sum(dim=[3, 4, 5], keepdim=True)
            mu_denom = self.R.sum(dim=[3, 4, 5], keepdim=True)
            mu = mu_share / (mu_denom + eps)

            V_mu_sqr = (V - mu) ** 2
            sigma_share = (self.R * V_mu_sqr).sum(dim=[3, 4, 5], keepdim=True)
            sigma_sqr = sigma_share / (mu_denom + eps)

            cost_h = (Beta_u + 0.5 * torch.log(sigma_sqr)) * mu_denom
            a_out = torch.sigmoid(F.normalize(Lambda * (Beta_a - cost_h.sum(dim=7)), dim=6))

            # E-Step:
            log_p1 = -0.5 * ((self.mathpilog + torch.log(sigma_sqr)).sum(dim=7)) + eps
            log_p2 = -(V_mu_sqr / ((2 * sigma_sqr) + eps)).sum(dim=7)
            log_p = log_p1 + log_p2
            R_ = torch.log(a_out) + log_p
            self.R = torch.softmax(R_, dim=6)

        mu = mu.squeeze()
        sigma_sqr = sigma_sqr.squeeze()
        return mu.view(outSize), \
               a_out.squeeze(), \
               sigma_sqr.view(outSize)


class EMRouting_old(nn.Module):

    def __init__(self, iterations=3 ):
        super(EMRouting_old, self).__init__()
        self.iterations = iterations
        self.final_lambda = 0.01
        self.register_buffer("mathpilog", torch.log(torch.FloatTensor([2 * math.pi])))

    def forward(self, V, a, Beta_u, Beta_a, R, outSize):


        for i in range(self.iterations):
            Lambda = self.final_lambda * (1 - 0.95 ** (i + 1))

            # M - Step:
            R = (R * a).unsqueeze(7)
            mu_share = (R * V).sum(dim=[3, 4, 5], keepdim=True)
            mu_denom = R.sum(dim=[3, 4, 5], keepdim=True)
            mu = mu_share / (mu_denom + eps)

            V_mu_sqr = (V - mu) ** 2
            sigma_share = (R * V_mu_sqr).sum(dim=[3, 4, 5], keepdim=True)
            sigma_sqr = sigma_share / (mu_denom + eps)

            cost_h = (Beta_u + 0.5*torch.log(sigma_sqr)) * mu_denom
            a_out = torch.sigmoid(F.normalize(Lambda * (Beta_a - cost_h.sum(dim=7)), dim=6))


            #E-Step:
            log_p1 = -0.5 * ((self.mathpilog + torch.log(sigma_sqr)).sum(dim=7)) + eps
            log_p2 = -(V_mu_sqr / ((2 * sigma_sqr) + eps)).sum(dim=7)
            log_p = log_p1 + log_p2
            R_ = torch.log(a_out) + log_p
            R = torch.softmax(R_, dim=6)

        mu = mu.squeeze()
        sigma_sqr = sigma_sqr.squeeze()
        return mu.view(outSize), \
               a_out.squeeze(), \
               sigma_sqr.view(outSize)

class PoseAttRouter(nn.Module):
    def __init__(self):
        super(PoseAttRouter, self).__init__()

    # INPUT:
    # votes:    tuple<  activation: <B, in_caps, 1>
    #                   quaternion: <B, in_caps, out_caps, 4> ---> last dimension is a pure quaternion
    #                   feature   : <B, in_caps, out_caps, feat_size>  >
    def forward(self, A, Q, V):
        acts, K = self.quaternion_average(Q, A)
        dists = self.geodesic_dist(Q, K)
        att_weights = F.softmax(-dists, dim=1)
        out_f = (att_weights * V).mean(dim=1)
        out_q = K.squeeze(1)
        return acts, out_q, out_f

    # Equivalent to inner product: d(q1,q2) = arccos(|q1.q2|) where . is the dot product and || is abs.
    # INPUT MUST BE UNIT QUATERNIONS
    # Q1 ~ V
    # Distance code is referenced from https://github.com/tolgabirdal/qecnetworks/blob/master/models/qec_module.py
    def geodesic_dist(self, Q1, Q2):
        dots = (Q1.unsqueeze(dim=3) @ Q2.unsqueeze(dim=-1)).squeeze(dim=-1)
        # print("\n\n-------ARCCOS X's-----------\n{}".format(dots))
        distance = torch.arccos(
            torch.clamp(torch.abs(dots), min=-1 + eps, max=1 - eps))  # Equivalent to 1 - | q1 . q2 |  --> Consider this.
        distance = 2 * distance / np.pi
        return distance

    # outer product is equivalent to matrix multiplication if arranged correctly.
    # check if works with pure quaternions,
    # Distance code is referenced from https://github.com/tolgabirdal/qecnetworks/blob/master/models/qec_module.py
    # MAY BE UNSTABLE DUE TO SYMEIG on similar eigenvalues. But we only use largest and do not backpropagate over other eigenvalues
    def quaternion_average(self, Q1, weights):
        Q1 = Q1 + eps
        # print("\n\n-------Quaternions-----------\n{}".format(Q1))
        weights = F.normalize(weights, p=1, dim=1)
        outer = weights[(...,) + (None,) * 2] * (Q1.unsqueeze(dim=-1) @ Q1.unsqueeze(dim=3))
        M = outer.sum(1, keepdim=True)
        M = M + (1e-6) * torch.randn_like(M)

        eigs, eigvect = torch.linalg.eigh(M.view(-1, M.shape[-2], M.shape[-1]).cpu())
        eigs = eigs.to(engine.device)
        eigvect = eigvect.to(engine.device)
        normalized_eigs = torch.div(eigs, eigs.sum(dim=1).unsqueeze(1))
        normalized_max_eig = normalized_eigs[:, -1].view(M.shape[0], M.shape[-3], 1)
        avg_quats = eigvect[:, -1, :].view(M.shape[:-1])

        return normalized_max_eig, avg_quats

    def plain_average(self, Q1, weights):
        averages = (Q1 * weights.unsqueeze(-1)).mean(dim=1, keepdim=True)
        activations = torch.sigmoid(-5 * self.geodesic_dist(Q1, averages).mean(1))

        return activations, averages


class NovelPoseAttRouter(PoseAttRouter):

    def __init__(self, outCaps):
        super(NovelPoseAttRouter, self).__init__()
        self.w0 = nn.Parameter(torch.ones(1, outCaps, 1))
        self.w1 = nn.Parameter(torch.ones(1, outCaps, 1))

    # INPUT:
    # votes:    tuple<  activation: <B, in_caps, 1>
    #                   quaternion: <B, in_caps, out_caps, 4> ---> last dimension is a pure quaternion
    #                   feature   : <B, in_caps, feat_size>  >
    def forward(self, A, Q, V):
        act_q, K = self.quaternion_average(Q, A)
        dists = self.geodesic_dist(Q, K)
        att_weights = F.softmax(-dists, dim=2)
        out_f = (att_weights * V).mean(dim=1)
        sqr_dists = (out_f.unsqueeze(1) - V) ** 2
        act_f = sqr_dists.mean(1).sum(-1, keepdim=True)
        out_a = torch.sigmoid(self.w0*act_q - self.w1*act_f)
        out_q = K.squeeze(1)
        return out_a, out_q, out_f


class ContrastivePoseAttentiveRouter(PoseAttRouter):
    def __init__(self):
        super(ContrastivePoseAttentiveRouter, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, A, Q, V):
        acts, K = self.quaternion_average(Q, A)
        dists = self.geodesic_dist(Q, K)
        att_weights = F.softmax(-dists, dim=1)
        most_att_weights_idx = torch.argmax(att_weights, dim=2, keepdim=True)
        att_weights_vector = torch.eye(att_weights.size(2), device=att_weights.device)[most_att_weights_idx, :].squeeze()  # y_true
        b, ic, oc = att_weights_vector.size()
        att_weights_vector = att_weights_vector.view(b * ic, oc)
        p_sims = att_weights.squeeze() / 0.07  # y_pred
        p_sims = p_sims.view(b * ic, oc)
        nce_loss = self.criterion(p_sims, att_weights_vector)
        out_f = (att_weights * V).mean(dim=1)
        out_q = K.squeeze(1)
        return acts, out_q, out_f, nce_loss


class PoseeAttNoFeatRouter(PoseAttRouter):
    def __init__(self):
        super(PoseeAttNoFeatRouter, self).__init__()

    def forward(self, A, Q):
        acts, K = self.quaternion_average(Q, A)
        return acts, K.squeeze(1)


def get_router(name, it):
    return {
        "em-routing": lambda: EMRouting(it),
        "attention-routing": lambda: PoseAttRouter()
    }[name]()


if __name__ == '__main__':
    z = torch.zeros(4, 16, 32, 1)
    quats = torch.rand([4, 16, 32, 3])
    quats = torch.div(quats, torch.norm(quats, dim=3, keepdim=True))
    quats = torch.cat((z, quats), dim=-1)
    acts = torch.rand([4, 16, 1])
    feats = torch.rand(4, 16, 32)
    att_router = PoseAttRouter()
    # from torch.autograd import gradcheck
    # acts.requires_grad = True
    # feats.requires_grad = True
    # quats.requires_grad = True
    # o = gradcheck(att_router, (acts, quats, feats))
    # print(o)
    out_a, out_q, out_f = att_router(acts, quats, feats)
    print(out_a)
