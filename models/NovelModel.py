from models.feature_extractors import get_feature_extractor
from models.heads import get_pose_head, get_activation_head, get_feat_head
from models.QuaternionCapsules import *
from models.modules import *
from utils.config import get_cfg_model_v1



class Model_V1(nn.Module):
    """This module is the general model.
    Has a feature extractor from input images(Grayscale, Depth or both).
    From those features, encodes capsules with given heads. (feature vector, orientation, translation, activation etc.)
    """

    def __init__(self, opt, feature_extractor, pose_head, activation_head, feature_head):
        super(Model_V1, self).__init__()
        self.opt = opt
        self.feature_extractor = feature_extractor
        self.pose_head = pose_head
        self.activation_head = activation_head
        self.fccaps = FCQuaternionLayer(inCaps=32, outCaps=32, quat_dims=4,
                                                 routing_iterations=2,
                                                 routing=EMRouting, init_type=self.opt.MODEL.INIT_TYPE)
        self.classcaps = FCQuaternionLayer(inCaps=32, outCaps=10, quat_dims=4,
                                                 routing_iterations=2,
                                                 routing=EMRouting, init_type=self.opt.MODEL.INIT_TYPE)

    def forward(self, x):

        x = self.feature_extractor(x)
        q = self.pose_head(x)
        a = self.activation_head(x)
        q = torch.cat((torch.zeros(q.size(0), q.size(1), 1, device=q.device), q), 2)
        x = self.fccaps(q, a)
        x = self.classcaps(x[0].squeeze(-1), x[1])
        return x[0], x[1], x[2]




class Model_V2(nn.Module):
    """This module is the general model.
    Has a feature extractor from input images(Grayscale, Depth or both).
    From those features, encodes capsules with given heads. (feature vector, orientation, translation, activation etc.)
    """

    def __init__(self, opt, feature_extractor, pose_head, activation_head, feature_head):
        super(Model_V2, self).__init__()
        self.opt = opt
        self.feature_extractor = feature_extractor
        self.pose_head = pose_head
        self.feature_head = feature_head
        self.activation_head = activation_head
        self.fccaps = NovelCapsuleLayer(inCaps=32, outCaps=16, quat_dims=4,
                                        feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)
        self.fccaps2 = NovelCapsuleLayer(inCaps=48, outCaps=16, quat_dims=4,
                                         feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)
        self.classcaps = NovelCapsuleLayer(inCaps=64, outCaps=10, quat_dims=4,
                                           feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)

    def forward(self, x):
        x = self.feature_extractor(x)
        q = self.pose_head(x)
        a0 = self.activation_head(x)
        f0 = self.feature_head(x)
        q0 = torch.cat((torch.zeros(q.size(0), q.size(1), 1, device=q.device), q), 2)
        a1, q1, f1 = self.fccaps(a0, q0, f0)
        # print("\n\n------------------- Quaternion 1: {}".format(q1))
        a1_ = torch.cat([a0, a1], dim=1)
        q1_ = torch.cat([q0, q1], dim=1)
        f1_ = torch.cat([f0, f1], dim=1)
        a2, q2, f2 = self.fccaps2(a1_, q1_, f1_)
        # print("\n\n------------------- Quaternion 2: {}".format(q2))
        a2_ = torch.cat([a1_, a2], dim=1)
        q2_ = torch.cat([q1_, q2], dim=1)
        f2_ = torch.cat([f1_, f2], dim=1)
        a3, q3, f3 = self.classcaps(a2_, q2_, f2_)
        # print("\n\n------------------- Quaternion 3: {}".format(q3))

        return a3, q3, f3


class Model_V2_C(Model_V2):
    def __init__(self, opt, feature_extractor, pose_head, activation_head, feature_head):
        super(Model_V2_C, self).__init__(opt, feature_extractor, pose_head, activation_head, feature_head)
        self.fccaps = NovelCapsuleLayer_C(inCaps=32, outCaps=16, quat_dims=4,
                                        feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)
        self.fccaps2 = NovelCapsuleLayer_C(inCaps=48, outCaps=16, quat_dims=4,
                                         feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)
        self.classcaps = NovelCapsuleLayer_C(inCaps=64, outCaps=opt.DATA.NUM_CLASS, quat_dims=4,
                                           feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)

    def forward(self, x):
        x = self.feature_extractor(x)
        q = self.pose_head(x)
        a0 = self.activation_head(x)
        f0 = self.feature_head(x)
        q0 = torch.cat((torch.zeros(q.size(0), q.size(1), 1, device=q.device), q), 2)
        a1, q1, f1, nce1 = self.fccaps(a0, q0, f0)
        # print("\n\n------------------- Quaternion 1: {}".format(q1))
        a1_ = torch.cat([a0, a1], dim=1)
        q1_ = torch.cat([q0, q1], dim=1)
        f1_ = torch.cat([f0, f1], dim=1)
        a2, q2, f2, nce2 = self.fccaps2(a1_, q1_, f1_)
        # print("\n\n------------------- Quaternion 2: {}".format(q2))
        a2_ = torch.cat([a1_, a2], dim=1)
        q2_ = torch.cat([q1_, q2], dim=1)
        f2_ = torch.cat([f1_, f2], dim=1)
        a3, q3, f3, nce3 = self.classcaps(a2_, q2_, f2_)
        # print("\n\n------------------- Quaternion 3: {}".format(q3))
        nce = (nce1 + nce2 + nce3) / 3
        return a3, q3, f3, nce


class Model_V2_64(nn.Module):
    """This module is the general model.
    Has a feature extractor from input images(Grayscale, Depth or both).
    From those features, encodes capsules with given heads. (feature vector, orientation, translation, activation etc.)
    """

    def __init__(self, opt, feature_extractor, pose_head, activation_head, feature_head):
        super(Model_V2_64, self).__init__()
        self.opt = opt
        self.feature_extractor = feature_extractor
        self.pose_head = pose_head
        self.feature_head = feature_head
        self.activation_head = activation_head
        self.fccaps = NovelCapsuleLayer(inCaps=32, outCaps=16, quat_dims=4,
                                        feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)
        self.fccaps2 = NovelCapsuleLayer(inCaps=48, outCaps=16, quat_dims=4,
                                         feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)
        self.classcaps = NovelCapsuleLayer(inCaps=64, outCaps=10, quat_dims=4,
                                           feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)

    def forward(self, x):
        x = self.feature_extractor(x)
        q = self.pose_head(x)
        a0 = self.activation_head(x)
        f0 = self.feature_head(x)
        q0 = torch.cat((torch.zeros(q.size(0), q.size(1), 1, device=q.device), q), 2)
        a1, q1, f1 = self.fccaps(a0, q0, f0)
        # print("\n\n------------------- Quaternion 1: {}".format(q1))
        a1_ = torch.cat([a0, a1], dim=1)
        q1_ = torch.cat([q0, q1], dim=1)
        f1_ = torch.cat([f0, f1], dim=1)
        a2, q2, f2 = self.fccaps2(a1_, q1_, f1_)
        # print("\n\n------------------- Quaternion 2: {}".format(q2))
        a2_ = torch.cat([a1_, a2], dim=1)
        q2_ = torch.cat([q1_, q2], dim=1)
        f2_ = torch.cat([f1_, f2], dim=1)
        a3, q3, f3 = self.classcaps(a2_, q2_, f2_)
        # print("\n\n------------------- Quaternion 3: {}".format(q3))

        return a3, q3, f3


class Model_V2_alt(nn.Module):
    """This module is the general model.
    Has a feature extractor from input images(Grayscale, Depth or both).
    From those features, encodes capsules with given heads. (feature vector, orientation, translation, activation etc.)
    """

    def __init__(self, opt, feature_extractor, pose_head, activation_head, feature_head):
        super(Model_V2_alt, self).__init__()
        self.opt = opt
        self.feature_extractor = feature_extractor
        self.pose_head = pose_head
        self.feature_head = feature_head
        self.activation_head = activation_head
        self.fccaps = NovelCapsuleLayer_alt(inCaps=32, outCaps=16, quat_dims=4,
                                        feat_size=self.opt.MODEL.FEAT_SIZE, init_type=self.opt.MODEL.INIT_TYPE)
        self.fccaps2 = NovelCapsuleLayer_alt(inCaps=48, outCaps=16, quat_dims=4,
                                         feat_size=self.opt.MODEL.FEAT_SIZE, init_type=self.opt.MODEL.INIT_TYPE)
        self.classcaps = NovelCapsuleLayer_alt(inCaps=64, outCaps=opt.DATA.NUM_CLASS, quat_dims=4,
                                           feat_size=self.opt.MODEL.FEAT_SIZE, init_type=self.opt.MODEL.INIT_TYPE)

    def forward(self, x):
        x = self.feature_extractor(x)
        q = self.pose_head(x)
        a0 = self.activation_head(x)
        f0 = self.feature_head(x)
        q0 = torch.cat((torch.zeros(q.size(0), q.size(1), 1, device=q.device), q), 2)
        a1, q1, f1 = self.fccaps(a0, q0, f0)
        # print("\n\n------------------- Quaternion 1: {}".format(q1))
        a1_ = torch.cat([a0, a1], dim=1)
        q1_ = torch.cat([q0, q1], dim=1)
        f1_ = torch.cat([f0, f1], dim=1)
        a2, q2, f2 = self.fccaps2(a1_, q1_, f1_)
        # print("\n\n------------------- Quaternion 2: {}".format(q2))
        a2_ = torch.cat([a1_, a2], dim=1)
        q2_ = torch.cat([q1_, q2], dim=1)
        f2_ = torch.cat([f1_, f2], dim=1)
        a3, q3, f3 = self.classcaps(a2_, q2_, f2_)
        # print("\n\n------------------- Quaternion 3: {}".format(q3))
        return a3, q3, f3



class Model_V2_Deep(nn.Module):
    """This module is the general model.
    Has a feature extractor from input images(Grayscale, Depth or both).
    From those features, encodes capsules with given heads. (feature vector, orientation, translation, activation etc.)
    """

    def __init__(self, opt, feature_extractor, pose_head, activation_head, feature_head):
        super(Model_V2_Deep, self).__init__()
        self.opt = opt
        self.feature_extractor = feature_extractor
        self.pose_head = pose_head
        self.feature_head = feature_head
        self.activation_head = activation_head
        self.fccaps = NovelCapsuleLayer(inCaps=32, outCaps=32, quat_dims=4,
                                               feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)
        self.fccaps2 = NovelCapsuleLayer(inCaps=64, outCaps=16, quat_dims=4,
                                                feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)
        self.fccaps3 = NovelCapsuleLayer(inCaps=48, outCaps=16, quat_dims=4,
                                                feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)
        self.fccaps4 = NovelCapsuleLayer(inCaps=32, outCaps=16, quat_dims=4,
                                                feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)
        self.classcaps = NovelCapsuleLayer(inCaps=32, outCaps=10, quat_dims=4,
                                                  feat_size=32, init_type=self.opt.MODEL.INIT_TYPE)


    def forward(self, x):

        x = self.feature_extractor(x)
        q = self.pose_head(x)
        a0 = self.activation_head(x)
        f0 = self.feature_head(x)
        q0 = torch.cat((torch.zeros(q.size(0), q.size(1), 1, device=q.device), q), 2)
        a1_out, q1_out, f1_out = self.fccaps(a0, q0, f0)
        a2_in = torch.cat([a0, a1_out], dim=1)
        q2_in = torch.cat([q0, q1_out], dim=1)
        f2_in = torch.cat([f0, f1_out], dim=1)
        a2_out, q2_out, f2_out = self.fccaps2(a2_in, q2_in, f2_in)

        a3_in = torch.cat([a2_out, a1_out], dim=1)
        q3_in = torch.cat([q2_out, q1_out], dim=1)
        f3_in = torch.cat([f2_out, f1_out], dim=1)
        a3_out, q3_out, f3_out = self.fccaps3(a3_in, q3_in, f3_in)

        a4_in = torch.cat([a3_out, a2_out], dim=1)
        q4_in = torch.cat([q3_out, q2_out], dim=1)
        f4_in = torch.cat([f3_out, f2_out], dim=1)
        a4_out, q4_out, f4_out = self.fccaps4(a4_in, q4_in, f4_in)
        a5_in = torch.cat([a4_out, a3_out], dim=1)
        q5_in = torch.cat([q4_out, q3_out], dim=1)
        f5_in = torch.cat([f4_out, f3_out], dim=1)
        a, q, f = self.classcaps(a5_in, q5_in, f5_in)


        return a, q, f


class Model_V2_NoFeat(nn.Module):
    """This module is the general model.
    Has a feature extractor from input images(Grayscale, Depth or both).
    From those features, encodes capsules with given heads. (feature vector, orientation, translation, activation etc.)
    """

    def __init__(self, opt, feature_extractor, pose_head, activation_head):
        super(Model_V2_NoFeat, self).__init__()
        self.opt = opt
        self.feature_extractor = feature_extractor
        self.pose_head = pose_head
        self.activation_head = activation_head
        self.fccaps = NovelCapsuleLayer_NoFeat(inCaps=32, outCaps=16, quat_dims=4,
                                                init_type=self.opt.MODEL.INIT_TYPE)
        self.fccaps2 = NovelCapsuleLayer_NoFeat(inCaps=48, outCaps=16, quat_dims=4,
                                                 init_type=self.opt.MODEL.INIT_TYPE)
        self.classcaps = NovelCapsuleLayer_NoFeat(inCaps=64, outCaps=10, quat_dims=4,
                                                   init_type=self.opt.MODEL.INIT_TYPE)


    def forward(self, x):

        x = self.feature_extractor(x)
        q = self.pose_head(x)
        a0 = self.activation_head(x)
        q0 = torch.cat((torch.zeros(q.size(0), q.size(1), 1, device=q.device), q), 2)
        a1, q1 = self.fccaps(a0, q0)
        #print("\n\n------------------- Quaternion 1: {}".format(q1))
        a1_ = torch.cat([a0, a1], dim=1)
        q1_ = torch.cat([q0, q1], dim=1)
        a2, q2 = self.fccaps2(a1_, q1_)
        #print("\n\n------------------- Quaternion 2: {}".format(q2))
        a2_ = torch.cat([a1_, a2], dim=1)
        q2_ = torch.cat([q1_, q2], dim=1)
        a3, q3 = self.classcaps(a2_, q2_)
        #print("\n\n------------------- Quaternion 3: {}".format(q3))

        return a3, q3


def create_novel_model_v1():
    opt = get_cfg_model_v1()
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 256, 32)
    p_head = get_pose_head("basic", 256, 32, 3)
    f_head = get_pose_head("basic", 256, 32, 128)
    model = Model_V1(opt, f_ext, p_head, a_head, f_head)
    return model, opt

def create_novel_model_v2_64():
    opt = get_cfg_model_v1()
    opt.MODEL.FEAT_SIZE = 32
    opt.OUTPUT_DIR = "./output_v2_64"
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 1024, 32)
    p_head = get_pose_head("basic", 1024, 32, 3)
    f_head = get_pose_head("basic", 1024, 32, 32)
    model = Model_V2_64(opt, f_ext, p_head, a_head, f_head)
    return model, opt

def create_novel_model_v2_alt():
    opt = get_cfg_model_v1()
    opt.MODEL.FEAT_SIZE = 32
    opt.OUTPUT_DIR = "./output_v2_alt"
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 1024, 32)
    p_head = get_pose_head("basic", 1024, 32, 3)
    f_head = get_pose_head("basic", 1024, 32, 32)
    model = Model_V2_64(opt, f_ext, p_head, a_head, f_head)
    return model, opt


def create_novel_model_v2():
    opt = get_cfg_model_v1()
    opt.MODEL.FEAT_SIZE = 32
    opt.OUTPUT_DIR = "./output_v2"
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 256, 32)
    p_head = get_pose_head("basic", 256, 32, 3)
    f_head = get_pose_head("basic", 256, 32, 32)
    model = Model_V2(opt, f_ext, p_head, a_head, f_head)
    return model, opt

def create_novel_model_v2_deep():
    opt = get_cfg_model_v1()
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 256, 32)
    p_head = get_pose_head("basic", 256, 32, 3)
    f_head = get_pose_head("basic", 256, 32, 32)
    model = Model_V2_Deep(opt, f_ext, p_head, a_head, f_head)
    return model, opt

def create_novel_model_v2_nofeat():
    opt = get_cfg_model_v1()
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 256, 32)
    p_head = get_pose_head("basic", 256, 32, 3)
    model = Model_V2_NoFeat(opt, f_ext, p_head, a_head)

    return model, opt



if __name__ == '__main__':
    print("testing")
    x = torch.rand([4, 1, 32, 32])
    opt = get_cfg_model_v1()
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 256, 16)
    p_head = get_pose_head("basic", 256, 16, 3)
    f_head = get_feat_head("basic", 256, 16, 32)
    model = Model_V2_alt(opt, f_ext, p_head, a_head, f_head)
    a, q, f = model(x)
