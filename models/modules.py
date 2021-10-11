from models.Routing_Methods import *
import math

eps = 1e-10

# Layer template using quaternion ops, has quatembedder and left2right for quaternion operations.
# quadembedder: Matrix embedding for quaternion q1 to apply quaternion product from left (q1 . q').
# left2right: matrix embedding for converting quaternion product from left matrix to from right matrix (q' . q1).
class QuaternionLayer(nn.Module):
    def __init__(self):
        super(QuaternionLayer, self).__init__()
        self.register_buffer("quatEmbedder", torch.stack([torch.eye(4),
                                                          torch.tensor([[0, -1, 0, 0],
                                                                        [1, 0, 0, 0],
                                                                        [0, 0, 0, -1],
                                                                        [0, 0, 1, 0]], dtype=torch.float),

                                                          torch.tensor([[0, 0, -1, 0],
                                                                        [0, 0, 0, 1],
                                                                        [1, 0, 0, 0],
                                                                        [0, -1, 0, 0]], dtype=torch.float),

                                                          torch.tensor([[0, 0, 0, -1],
                                                                        [0, 0, -1, 0],
                                                                        [0, 1, 0, 0],
                                                                        [1, 0, 0, 0]], dtype=torch.float)]).unsqueeze(0).unsqueeze(1))
        # convert quaternion multiplication from left to multiplication from right.
        self.register_buffer("left2right", torch.tensor([[1, 1, 1, 1],
                                                         [1, 1, -1, -1],
                                                         [1, -1, 1, -1],
                                                         [1, -1, -1, 1]], dtype=torch.float)[(None,) * 5])


class NovelCapsuleLayer(QuaternionLayer):

    def __init__(self, inCaps, outCaps, quat_dims, feat_size, init_type):
        super(NovelCapsuleLayer, self).__init__()
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.quat_dims = quat_dims
        self.feat_size = feat_size
        self.W_theta = nn.Parameter(torch.zeros(1,  inCaps, outCaps, 1, 1, 1))
        if init_type == "uniform_pi":
            nn.init.uniform_(self.W_theta, -math.pi, math.pi)
        elif init_type == "normal":
            nn.init.normal_(self.W_theta)
        self.W_hat = nn.Parameter(torch.zeros(1, inCaps, outCaps, 3, 1, 1))
        nn.init.uniform_(self.W_hat, 0, 1)
        self.phi = nn.Parameter(torch.zeros(1, inCaps, outCaps, 1, 1, 1))
        self.sigma = 1/math.sqrt(2*(inCaps + outCaps))
        nn.init.uniform_(self.phi, -self.sigma, -self.sigma)
        self.router = PoseAttRouter()
        self.left2right = self.left2right.flatten(start_dim=1, end_dim=5)
        self.feat_layer = nn.Linear(feat_size, outCaps*feat_size)

    # INPUT SHAPE:  <B, InCaps, Quaternion>, <B, InCaps, Activation>
    def forward(self, a, q, f):
        votes_q = self.calculate_pose_votes(q)
        votes_f = self.calculate_feat_votes(f)
        return self.router(a, votes_q, votes_f)

    def calculate_pose_votes(self, q):
        W_theta_sin = self.phi*torch.sin(self.W_theta) + eps
        W_theta_cos = self.phi*torch.cos(self.W_theta) + eps
        W_unit = torch.div(self.W_hat, torch.norm(self.W_hat, dim=3, keepdim=True))
        W_rotor = torch.cat((W_theta_cos, W_theta_sin * W_unit), dim=3)
        W_ = torch.sum(self.quatEmbedder * W_rotor, dim=3)
        W_conj = self.left2right * W_.permute(0, 1, 2, 4, 3)
        votes_q = W_conj @ W_ @ q.unsqueeze(2).unsqueeze(-1)
        return votes_q.squeeze(-1)

    def calculate_feat_votes(self, f):
        votes_f = list()
        for f_single in f.split(1, dim=1):
            votes_f.append(torch.relu(self.feat_layer(f_single)).view(-1, 1, self.outCaps, self.feat_size))
        return torch.cat(votes_f, dim=1)


class NovelCapsuleLayer_NoFeatVote(NovelCapsuleLayer):
    # INPUT SHAPE:  <B, InCaps, Quaternion>, <B, InCaps, Activation>
    def forward(self, a, q, f):
        votes_q = self.calculate_pose_votes(q)
        return self.router(a, votes_q, f)


class NovelCapsuleLayer_C(NovelCapsuleLayer):
    def __init__(self, inCaps, outCaps, quat_dims, feat_size, init_type):
        super(NovelCapsuleLayer_C, self).__init__(inCaps, outCaps, quat_dims, feat_size, init_type)
        self.router = ContrastivePoseAttentiveRouter()


class NovelCapsuleLayer_NoFeat(QuaternionLayer):

    def __init__(self, inCaps, outCaps, quat_dims, init_type):
        super(NovelCapsuleLayer_NoFeat, self).__init__()
        self.inCaps = inCaps
        self.outCaps = outCaps
        self.quat_dims = quat_dims
        self.W_theta = nn.Parameter(torch.zeros(1,  inCaps, outCaps, 1, 1, 1))
        if init_type == "uniform_pi":
            nn.init.uniform_(self.W_theta, -math.pi, math.pi)
        elif init_type == "normal":
            nn.init.normal_(self.W_theta)
        self.W_hat = nn.Parameter(torch.zeros(1, inCaps, outCaps, 3, 1, 1))
        nn.init.uniform_(self.W_hat, -1, 1)


        self.router = PoseeAttNoFeatRouter()
        self.left2right = self.left2right.flatten(start_dim=1, end_dim=5)


    # INPUT SHAPE:  <B, InCaps, Quaternion>, <B, InCaps, Activation>
    def forward(self, a, q):
        votes_q = self.calculate_pose_votes(q)
        return self.router(a, votes_q)

    def calculate_pose_votes(self, q):
        W_theta_sin = torch.sin(self.W_theta) + eps
        W_theta_cos = torch.cos(self.W_theta) + eps
        W_rotor = torch.cat((W_theta_cos, W_theta_sin * self.W_hat), dim=3)
        W_rotor = torch.div(W_rotor, torch.norm(W_rotor, dim=3, keepdim=True))
        W_ = torch.sum(self.quatEmbedder * W_rotor, dim=3)
        W_conj = self.left2right * W_.permute(0, 1, 2, 4, 3)
        votes_q = W_conj @ W_ @ q.unsqueeze(2).unsqueeze(-1)
        return votes_q.squeeze(-1)


class NovelCapsuleLayer_alt(NovelCapsuleLayer):
    def __init__(self, inCaps, outCaps, quat_dims, feat_size, init_type):
        super(NovelCapsuleLayer_alt, self).__init__(inCaps, outCaps, quat_dims, feat_size, init_type)
        self.router = NovelPoseAttRouter(outCaps)


class DecoderModel(nn.Module):
    def __init__(self, feat_size, n_classes, image_size):
        super(DecoderModel, self).__init__()
        self.fc1 = nn.Linear(feat_size * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, image_size**2)
        self.n_classes= n_classes

    def forward(self, x, target):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

if __name__ == '__main__':
    print("module tests")
    a = torch.rand(4, 32, 1)
    q = torch.rand(4, 32, 4)
    f = torch.rand(4, 32, 32)
    layer = NovelCapsuleLayer_alt(32, 10, 4, 32, "uniform_pi")
    from utils.utils import count_parameters
    from utils.device_setting import device
    print(count_parameters(layer))
    a, q, f = layer(a, q, f)
    dec = DecoderModel(32, 10, 32).to(device)
    o = dec(f, torch.eye(4, 10, dtype=torch.int64))

