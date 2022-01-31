import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.device_setting import device

class PrimaryCapsules2d(nn.Module):
    '''Primary Capsule Layer'''
    def __init__(self, in_channels, out_caps, kernel_size, stride,
        padding=0, pose_dim=4, weight_init='xavier_uniform'):
        super().__init__()

        self.A = in_channels
        self.B = out_caps
        self.P = pose_dim
        self.K = kernel_size
        self.S = stride
        self.padding = padding

        w_kernel = torch.empty(self.B*self.P*self.P, self.A, self.K, self.K)
        a_kernel = torch.empty(self.B, self.A, self.K, self.K)

        if weight_init == 'kaiming_normal':
            nn.init.kaiming_normal_(w_kernel)
            nn.init.kaiming_normal_(a_kernel)
        elif weight_init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(w_kernel)
            nn.init.kaiming_uniform_(a_kernel)
        elif weight_init == 'xavier_normal':
            nn.init.xavier_normal_(w_kernel)
            nn.init.xavier_normal_(a_kernel)
        elif weight_init == 'xavier_uniform':
            nn.init.xavier_uniform_(w_kernel)
            nn.init.xavier_uniform_(a_kernel)
        else:
            NotImplementedError('{} not implemented.'.format(weight_init))

        # Out ← [B*(P*P+1), A, K, K]
        self.weight = nn.Parameter(torch.cat([w_kernel, a_kernel], dim=0))

        self.BN_a = nn.BatchNorm2d(self.B, affine=True)
        self.BN_p = nn.BatchNorm3d(self.B, affine=True)

    def forward(self, x): # [?, A, F, F] ← In

        # Out ← [?, B*(P*P+1), F, F]
        x = F.conv2d(x, weight=self.weight, stride=self.S, padding=self.padding)

        # Out ← ([?, B*P*P, F, F], [?, B, F, F]) ← [?, B*(P*P+1), F, F]
        poses, activations = torch.split(x, [self.B*self.P*self.P, self.B], dim=1)

        # Out ← [?, B, P*P, F, F]
        poses = self.BN_p(poses.reshape(-1, self.B, self.P*self.P, *x.shape[2:]))

        # Out ← [?, B, P, P, F, F] ← [?, B, P*P, F, F] ← In
        poses = poses.reshape(-1, self.B, self.P, self.P, *x.shape[2:])

        # Out ← [?, B, F, F])
        activations = torch.sigmoid(self.BN_a(activations))

        return (activations, poses)

class ConvCapsules2d(nn.Module):
    '''Convolutional Capsule Layer'''

    def __init__(self, in_caps, out_caps, pose_dim, kernel_size, stride, padding=0,
        weight_init='xavier_uniform', share_W_ij=False, coor_add=False):
        super().__init__()

        self.B = in_caps
        self.C = out_caps
        self.P = pose_dim
        self.PP = np.max([2, self.P*self.P])
        self.K = kernel_size
        self.S = stride
        self.padding = padding

        self.share_W_ij = share_W_ij # share the transformation matrices across (F*F)
        self.coor_add = coor_add # embed coordinates

        # Out ← [1, B, C, 1, P, P, 1, 1, K, K]
        self.W_ij = torch.empty(1, self.B, self.C, 1, self.P, self.P, 1, 1, self.K, self.K)

        if weight_init.split('_')[0] == 'xavier':
            fan_in = self.B * self.K*self.K * self.PP # in_caps types * receptive field size
            fan_out = self.C * self.K*self.K * self.PP # out_caps types * receptive field size
            std = np.sqrt(2. / (fan_in + fan_out))
            bound = np.sqrt(3.) * std

            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))

        elif weight_init.split('_')[0] == 'kaiming':
            # fan_in preserves magnitude of the variance of the weights in the forward pass.
            fan_in = self.B * self.K*self.K * self.PP # in_caps types * receptive field size
            # fan_out has same affect as fan_in for backward pass.
            # fan_out = self.C * self.K*self.K * self.PP # out_caps types * receptive field size
            std = np.sqrt(2.) / np.sqrt(fan_in)
            bound = np.sqrt(3.) * std

            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))

        elif weight_init == 'noisy_identity' and self.PP > 2:
            b = 0.01 # U(0,b)
            # Out ← [1, B, C, 1, P, P, 1, 1, K, K]
            self.W_ij = nn.Parameter(torch.clamp(.1*torch.eye(self.P,self.P).repeat( \
                1, self.B, self.C, 1, 1, 1, self.K, self.K, 1, 1) + \
                torch.empty(1, self.B, self.C, 1, 1, 1, self.K, self.K, self.P, self.P).uniform_(0,b), \
                max=1).permute(0, 1, 2, 3, -2, -1, 4, 5, 6, 7))
        else:
            raise NotImplementedError('{} not implemented.'.format(weight_init))

        if self.padding != 0:
            if isinstance(self.padding, int):
                self.padding = [self.padding]*4

    def forward(self, activations, poses): # ([?, B, F, F], [?, B, P, P, F, F]) ← In

        if self.padding != 0:
            activations = F.pad(activations, self.padding) # [1,1,1,1]
            poses = F.pad(poses, self.padding + [0]*4) # [0,0,1,1,1,1]

        if self.share_W_ij: # share the matrices over (F*F), if class caps layer
            self.K = poses.shape[-1] # out_caps (C) feature map size

        self.F = (poses.shape[-1] - self.K) // self.S + 1 # featuremap size

        # Out ← [?, B, P, P, F', F', K, K] ← [?, B, P, P, F, F]
        poses = poses.unfold(4, size=self.K, step=self.S).unfold(5, size=self.K, step=self.S)

        # Out ← [?, B, 1, P, P, 1, F', F', K, K] ← [?, B, P, P, F', F', K, K]
        poses = poses.unsqueeze(2).unsqueeze(5)

        # Out ← [?, B, F', F', K, K] ← [?, B, F, F]
        activations = activations.unfold(2, size=self.K, step=self.S).unfold(3, size=self.K, step=self.S)

        # Out ← [?, B, 1, 1, 1, F', F', K, K] ← [?, B, F', F', K, K]
        activations = activations.reshape(-1, self.B, 1, 1, 1, *activations.shape[2:4], self.K, self.K)

        # Out ← [?, B, C, P, P, F', F', K, K] ← ([?, B, 1, P, P, 1, F', F', K, K] * [1, B, C, 1, P, P, 1, 1, K, K])
        V_ji = (poses * self.W_ij).sum(dim=4) # matmul equiv.

        # Out ← [?, B, C, P*P, 1, F', F', K, K] ← [?, B, C, P, P, F', F', K, K]
        V_ji = V_ji.reshape(-1, self.B, self.C, self.P*self.P, 1, *V_ji.shape[-4:-2], self.K, self.K)

        if self.coor_add:
            if V_ji.shape[-1] == 1: # if class caps layer (featuremap size = 1)
                self.F = self.K # 1->4

            # coordinates = torch.arange(self.F, dtype=torch.float32) / self.F
            coordinates = torch.arange(self.F, dtype=torch.float32).add(1.) / (self.F*10)
            i_vals = torch.zeros(self.P*self.P,self.F,1).to(device)
            j_vals = torch.zeros(self.P*self.P,1,self.F).to(device)
            i_vals[self.P-1,:,0] = coordinates
            j_vals[2*self.P-1,0,:] = coordinates

            if V_ji.shape[-1] == 1: # if class caps layer
                # Out ← [?, B, C, P*P, 1, 1, 1, K=F, K=F] (class caps)
                V_ji = V_ji + (i_vals + j_vals).reshape(1,1,1,self.P*self.P,1,1,1,self.F,self.F)
                return activations, V_ji

            # Out ← [?, B, C, P*P, 1, F, F, K, K]
            V_ji = V_ji + (i_vals + j_vals).reshape(1,1,1,self.P*self.P,1,self.F,self.F,1,1)

        return activations, V_ji

class VariationalBayesRouting2d(nn.Module):
    '''Variational Bayes Capsule Routing Layer'''

    def __init__(self, in_caps, out_caps, pose_dim,
            kernel_size, stride,
            alpha0, # Dirichlet
            m0, kappa0, # Gaussian
            Psi0, nu0, # Wishart
            cov='diag', iter=3, class_caps=False):
        super().__init__()

        self.B = in_caps
        self.C = out_caps
        self.P = pose_dim
        self.D = np.max([2, self.P*self.P])
        self.K = kernel_size
        self.S = stride

        self.cov = cov # diag/full
        self.iter = iter # routing iters
        self.class_caps = class_caps
        self.n_classes = out_caps if class_caps else None

        # dirichlet prior parameter
        self.alpha0 = torch.tensor(alpha0).type(torch.FloatTensor)
        # self.alpha0 = nn.Parameter(torch.zeros(1,1,self.C,1,1,1,1,1,1).fill_(alpha0)) learn it by backprop

        # Out ← [?, 1, C, P*P, 1, 1, 1, 1, 1]
        self.register_buffer('m0', m0.unsqueeze(0).repeat( \
            self.C,1).reshape(1,1,self.C,self.D,1,1,1,1,1)) # gaussian prior mean parameter

        # precision scaling parameter of gaussian prior over capsule component means
        self.kappa0 = kappa0

        # scale matrix of wishart prior over capsule precisions
        if self.cov == 'diag':
            # Out ← [?, 1, C, P*P, 1, 1, 1, 1, 1]
            self.register_buffer('Psi0', torch.diag(Psi0).unsqueeze(0).repeat( \
                self.C,1).reshape(1,1,self.C,self.D,1,1,1,1,1))

        elif self.cov == 'full':
            # Out ← [?, 1, C, P*P, P*P, 1, 1, 1, 1]
            self.register_buffer('Psi0', Psi0.unsqueeze(0).repeat( \
                self.C,1,1).reshape(1,1,self.C,self.D,self.D,1,1,1,1))

        # degree of freedom parameter of wishart prior capsule precisions
        self.nu0 = nu0

        # log determinant = 0, if Psi0 is identity
        self.register_buffer('lndet_Psi0', 2*torch.diagonal(torch.linalg.cholesky(
            Psi0)).log().sum())

        # pre compute the argument of the digamma function in E[ln|lambda_j|]
        self.register_buffer('diga_arg', torch.arange(self.D).reshape(
            1,1,1,self.D,1,1,1,1,1).type(torch.FloatTensor))

        # pre define some constants
        self.register_buffer('Dlog2',
            self.D*torch.log(torch.tensor(2.)).type(torch.FloatTensor))
        self.register_buffer('Dlog2pi',
            self.D*torch.log(torch.tensor(2.*np.pi)).type(torch.FloatTensor))

        # Out ← [K*K, 1, K, K] vote collecting filter
        self.register_buffer('filter',
            torch.eye(self.K*self.K).reshape(self.K*self.K,1,self.K,self.K))

        # Out ← [1, 1, C, 1, 1, 1, 1, 1, 1] optional params
        self.beta_u = nn.Parameter(torch.zeros(1,1,self.C,1,1,1,1,1,1))
        self.beta_a = nn.Parameter(torch.zeros(1,1,self.C,1,1,1,1,1,1))

        self.BN_v = nn.BatchNorm3d(self.C, affine=False)
        self.BN_a = nn.BatchNorm2d(self.C, affine=False)

    # Out ← [?, B, 1, 1, 1, F, F, K, K], [?, B, C, P*P, 1, F, F, K, K] ← In
    def forward(self, a_i, V_ji):

        self.F_i = a_i.shape[-2:] # input capsule (B) votes feature map size (K)
        self.F_o = a_i.shape[-4:-2] # output capsule (C) feature map size (F)
        self.N = self.B*self.F_i[0]*self.F_i[1] # total num of lower level capsules

        # Out ← [1, B, C, 1, 1, 1, 1, 1, 1]
        R_ij = (1./self.C) * torch.ones(1,self.B,self.C,1,1,1,1,1,1, requires_grad=False).to(device)

        for i in range(self.iter): # routing iters

            # update capsule parameter distributions
            self.update_qparam(a_i, V_ji, R_ij)

            if i != self.iter-1: # skip last iter
                # update latent variable distributions (child to parent capsule assignments)
                R_ij = self.update_qlatent(a_i, V_ji)

        # Out ← [?, 1, C, 1, 1, F, F, 1, 1]
        self.Elnlambda_j = self.reduce_poses(
            torch.digamma(.5*(self.nu_j - self.diga_arg))) \
                + self.Dlog2 + self.lndet_Psi_j

        # Out ← [?, 1, C, 1, 1, F, F, 1, 1]
        self.Elnpi_j = torch.digamma(self.alpha_j) \
            - torch.digamma(self.alpha_j.sum(dim=2, keepdim=True))

        # subtract "- .5*ln|lmbda|" due to precision matrix, instead of adding "+ .5*ln|sigma|" for covariance matrix
        H_q_j = .5*self.D * torch.log(torch.tensor(2*np.pi*np.e)) - .5*self.Elnlambda_j # posterior entropy H[q*(mu_j, sigma_j)]

        # Out ← [?, 1, C, 1, 1, F, F, 1, 1] weighted negative entropy with optional beta params and R_j weight
        a_j = self.beta_a - (torch.exp(self.Elnpi_j) * H_q_j + self.beta_u) #* self.R_j

        # Out ← [?, C, F, F]
        a_j = a_j.squeeze()

        # Out ← [?, C, P*P, F, F] ← [?, 1, C, P*P, 1, F, F, 1, 1]
        self.m_j = self.m_j.squeeze()

        # so BN works in the classcaps layer
        if self.class_caps:
            # Out ← [?, C, 1, 1] ← [?, C]
            a_j = a_j[...,None,None]

            # Out ← [?, C, P*P, 1, 1] ← [?, C, P*P]
            self.m_j = self.m_j[...,None,None]
        # else:
        #     self.m_j = self.BN_v(self.m_j)

        # Out ← [?, C, P*P, F, F]
        self.m_j = self.BN_v(self.m_j) # use 'else' above to deactivate BN_v for class_caps

        # Out ← [?, C, P, P, F, F] ← [?, C, P*P, F, F]
        self.m_j = self.m_j.reshape(-1, self.C, self.P, self.P, *self.F_o)

        # Out ← [?, C, F, F]
        a_j = torch.sigmoid(self.BN_a(a_j))

        return a_j.squeeze(), self.m_j.squeeze() # propagate posterior means to next layer

    def update_qparam(self, a_i, V_ji, R_ij):

        # Out ← [?, B, C, 1, 1, F, F, K, K]
        R_ij = R_ij * a_i # broadcast a_i 1->C, and R_ij (1,1,1,1)->(F,F,K,K), 1->batch

        # Out ← [?, 1, C, 1, 1, F, F, 1, 1]
        self.R_j = self.reduce_icaps(R_ij)

        # Out ← [?, 1, C, 1, 1, F, F, 1, 1]
        self.alpha_j = self.alpha0 + self.R_j
        # self.alpha_j = torch.exp(self.alpha0) + self.R_j # when alpha's a param
        self.kappa_j = self.kappa0 + self.R_j
        self.nu_j = self.nu0 + self.R_j

        # Out ← [?, 1, C, P*P, 1, F, F, 1, 1]
        mu_j = (1./self.R_j) * self.reduce_icaps(R_ij * V_ji)

        # Out ← [?, 1, C, P*P, 1, F, F, 1, 1]
        # self.m_j = (1./self.kappa_j) * (self.R_j * mu_j + self.kappa0 * self.m0) # use this if self.m0 != 0
        self.m_j = (1./self.kappa_j) * (self.R_j * mu_j) # priors removed for faster computation

        if self.cov == 'diag':
            # Out ← [?, 1, C, P*P, 1, F, F, 1, 1] (1./R_j) not needed because Psi_j calc
            sigma_j = self.reduce_icaps(R_ij * (V_ji - mu_j).pow(2))

            # Out ← [?, 1, C, P*P, 1, F, F, 1, 1]
            # self.invPsi_j = self.Psi0 + sigma_j + (self.kappa0*self.R_j / self.kappa_j) \
            #     * (mu_j - self.m0).pow(2) # use this if m0 != 0 or kappa0 != 1
            self.invPsi_j = self.Psi0 + sigma_j + (self.R_j / self.kappa_j) * (mu_j).pow(2) # priors removed for faster computation

            # Out ← [?, 1, C, 1, 1, F, F, 1, 1] (-) sign as inv. Psi_j
            self.lndet_Psi_j = -self.reduce_poses(torch.log(self.invPsi_j)) # log det of diag precision matrix

        elif self.cov == 'full':
            #[?, B, C, P*P, P*P, F, F, K, K]
            sigma_j = self.reduce_icaps(
                R_ij * (V_ji - mu_j) * (V_ji - mu_j).transpose(3,4))

            # Out ← [?, 1, C, P*P, P*P, F, F, 1, 1] full cov, torch.inverse(self.Psi0)
            self.invPsi_j = self.Psi0 + sigma_j + (self.kappa0*self.R_j / self.kappa_j) \
                * (mu_j - self.m0) * (mu_j - self.m0).transpose(3,4)

            # Out ← [?, 1, C, F, F, 1, 1 , P*P, P*P]
            # needed for pytorch (*,n,n) dim requirements in .cholesky and .inverse
            self.invPsi_j = self.invPsi_j.permute(0,1,2,5,6,7,8,3,4)

            # Out ← [?, 1, 1, 1, C, F, F, 1, 1] (-) sign as inv. Psi_j
            self.lndet_Psi_j = -2*torch.diagonal(torch.cholesky(
                self.invPsi_j), dim1=-2, dim2=-1).log().sum(-1, keepdim=True)[...,None]

    def update_qlatent(self, a_i, V_ji):

        # Out ← [?, 1, C, 1, 1, F, F, 1, 1]
        self.Elnpi_j = torch.digamma(self.alpha_j) \
            - torch.digamma(self.alpha_j.sum(dim=2, keepdim=True))

        # Out ← [?, 1, C, 1, 1, F, F, 1, 1] broadcasting diga_arg
        self.Elnlambda_j = self.reduce_poses(
            torch.digamma(.5*(self.nu_j - self.diga_arg))) \
                + self.Dlog2 + self.lndet_Psi_j

        if self.cov == 'diag':
            # Out ← [?, B, C, 1, 1, F, F, K, K]
            ElnQ = (self.D/self.kappa_j) + self.nu_j \
                * self.reduce_poses((1./self.invPsi_j) * (V_ji - self.m_j).pow(2))

        elif self.cov == 'full':
            # Out ← [?, B, C, 1, 1, F, F, K, K]
            Vm_j = V_ji - self.m_j
            ElnQ = (self.D/self.kappa_j) + self.nu_j * self.reduce_poses(
                Vm_j.transpose(3,4) * torch.inverse(
                    self.invPsi_j).permute(0,1,2,7,8,3,4,5,6) * Vm_j)

        # Out ← [?, B, C, 1, 1, F, F, K, K]
        lnp_j = .5*self.Elnlambda_j -.5*self.Dlog2pi -.5*ElnQ

        # Out ← [?, B, C, 1, 1, F, F, K, K]
        p_j = torch.exp(self.Elnpi_j + lnp_j)

        # Out ← [?*B, 1, F', F'] ← [?*B, K*K, F, F] ← [?, B, 1, 1, 1, F, F, K, K]
        sum_p_j = F.conv_transpose2d(
            input=p_j.sum(dim=2, keepdim=True).reshape(
                -1, *self.F_o, self.K*self.K).permute(0, -1, 1, 2).contiguous(),
            weight=self.filter,
            stride=[self.S, self.S])

        # Out ← [?*B, 1, F, F, K, K]
        sum_p_j = sum_p_j.unfold(2, size=self.K, step=self.S).unfold(3, size=self.K, step=self.S)

        # Out ← [?, B, 1, 1, 1, F, F, K, K]
        sum_p_j = sum_p_j.reshape([-1, self.B, 1, 1, 1, *self.F_o, self.K, self.K])

        # Out ← [?, B, C, 1, 1, F, F, K, K] # normalise over out_caps j
        return 1. / torch.clamp(sum_p_j, min=1e-11) * p_j

    def reduce_icaps(self, x):
        return x.sum(dim=(1,-2,-1), keepdim=True)

    def reduce_poses(self, x):
        return x.sum(dim=(3,4), keepdim=True)

class VBCapsuleNet(nn.Module):
    ''' Example: Simple 3 layer CapsNet '''
    def __init__(self, opt):
        super(VBCapsuleNet, self).__init__()

        arch = [64,16,16,16,10]
        self.P = 4
        self.PP = int(np.max([2, self.P*self.P]))
        self.A, self.B, self.C, self.D = arch[:-1]
        self.n_classes = opt.DATA.NUM_CLASS
        self.in_channels = opt.DATA.CHANNELS

        self.Conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.A,
            kernel_size=5, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.Conv_1.weight)
        self.BN_1 = nn.BatchNorm2d(self.A)

        self.PrimaryCaps = PrimaryCapsules2d(in_channels=self.A, out_caps=self.B,
            kernel_size=1, stride=1, pose_dim=self.P)

        self.ConvCaps_1 = ConvCapsules2d(in_caps=self.B, out_caps=self.C,
            kernel_size=3, stride=2, pose_dim=self.P)

        self.ConvRouting_1 = VariationalBayesRouting2d(in_caps=self.B, out_caps=self.C,
            kernel_size=3, stride=2, pose_dim=self.P,
            cov='diag', iter=3,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1)

        self.ConvCaps_2 = ConvCapsules2d(in_caps=self.C, out_caps=self.D,
            kernel_size=3, stride=1, pose_dim=self.P)

        self.ConvRouting_2 = VariationalBayesRouting2d(in_caps=self.C, out_caps=self.D,
            kernel_size=3, stride=1, pose_dim=self.P,
            cov='diag', iter=3,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1)

        self.ClassCaps = ConvCapsules2d(in_caps=self.D, out_caps=self.n_classes,
            kernel_size=1, stride=1, pose_dim=self.P, share_W_ij=True, coor_add=True)

        self.ClassRouting = VariationalBayesRouting2d(in_caps=self.D, out_caps=self.n_classes,
            kernel_size=4, stride=1, pose_dim=self.P, # adjust final kernel_size K depending on input H/W, for H=W=32, K=4.
            cov='diag', iter=3,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1, class_caps=True)

    def forward(self, x):

        # Out ← [?, A, F, F]
        x = F.relu(self.BN_1(self.Conv_1(x)))

        # Out ← a [?, B, F, F], v [?, B, P, P, F, F]
        a,v = self.PrimaryCaps(x)

        # Out ← a [?, B, 1, 1, 1, F, F, K, K], v [?, B, C, P*P, 1, F, F, K, K]
        a,v = self.ConvCaps_1(a,v)

        # Out ← a [?, C, F, F], v [?, C, P, P, F, F]
        a,v = self.ConvRouting_1(a,v)

        # Out ← a [?, C, 1, 1, 1, F, F, K, K], v [?, C, D, P*P, 1, F, F, K, K]
        a,v = self.ConvCaps_2(a,v)

        # Out ← a [?, D, F, F], v [?, D, P, P, F, F]
        a,v = self.ConvRouting_2(a,v)

        # Out ← a [?, D, 1, 1, 1, F, F, K, K], v [?, D, n_classes, P*P, 1, F, F, K, K]
        a,v = self.ClassCaps(a,v)

        # Out ← yhat [?, n_classes], v [?, n_classes, P, P]
        yhat, v = self.ClassRouting(a,v)

        return yhat
