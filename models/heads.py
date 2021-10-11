import torch
import torch.nn as nn
import torch.nn.functional as F

# POSE HEADS
class BasicPoseHead(nn.Module):
    def __init__(self, input_dim, num_pose, pose_dim):
        super(BasicPoseHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_pose*pose_dim)
        self.num_pose = num_pose
        self.pose_dim = pose_dim

    def forward(self, x):
        quats = self.fc(x).view(-1, self.num_pose, self.pose_dim)
        #normalize quaternions:
        quats = quats / torch.norm(quats, 2, -1, keepdim=True)
        return quats


class MLPPoseHead(nn.Module):
    def __init__(self, input_dim, num_pose, pose_dim):
        super(MLPPoseHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_pose * pose_dim)
        self.num_pose = num_pose
        self.pose_dim = pose_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        quats = self.fc2(x).view(-1, self.num_pose, self.pose_dim)
        #normalize quaternions:
        quats = quats / torch.norm(quats, 2, -1, keepdim=True)
        return quats


def get_pose_head(name, input_dim, num_caps, pose_dim):
    return {
        "basic": lambda: BasicPoseHead(input_dim, num_caps, pose_dim),
        "MLP": lambda: MLPPoseHead(input_dim, num_caps, pose_dim)
        }[name]()


# ACTIVATION HEADS
class BasicActivationHead(nn.Module):
    def __init__(self, input_dim, num_caps):
        super(BasicActivationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_caps)
        self.num_pose = num_caps

    def forward(self, x):
        return F.sigmoid(self.fc(x)).unsqueeze(-1)


class MLPActivationHead(nn.Module):
    def __init__(self, input_dim, num_caps):
        super(MLPActivationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_caps)
        self.num_pose = num_caps

    def forward(self, x):
        return F.sigmoid(self.fc2(F.sigmoid(self.fc1(x)))).unsqueeze(-1)


def get_activation_head(name, input_dim, num_caps):
    return {
        "basic": lambda: BasicActivationHead(input_dim, num_caps),
        "MLP": lambda: MLPActivationHead(input_dim, num_caps)
        }[name]()


# FEATURE HEADS
class BasicFeatureHead(nn.Module):
    def __init__(self, input_dim, num_pose, feature_size):
        super(BasicFeatureHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_pose*feature_size)
        self.num_pose = num_pose
        self.feature_size = feature_size


    def forward(self, x):
        features = F.relu(self.fc(x)).view(-1, self.num_pose, self.feature_size)
        return features


class MLPFeatureHead(nn.Module):
    def __init__(self, input_dim, num_pose, feature_size):
        super(MLPFeatureHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_pose * feature_size)
        self.num_pose = num_pose
        self.feature_size = feature_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x)).view(-1, self.num_pose, self.feature_size)
        return features


def get_feat_head(name, input_dim, num_caps, feat_size):
    return {
        "basic": lambda: BasicFeatureHead(input_dim, num_caps, feat_size),
        "MLP": lambda: MLPFeatureHead(input_dim, num_caps, feat_size)
        }[name]()


if __name__ == '__main__':
    inp = torch.rand(1, 128)
    b = BasicPoseHead(128, 16, 4)
    x = torch.rand([1, 1, 32, 32])
    print(b(inp))
