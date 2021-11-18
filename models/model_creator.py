from models.NovelModel import *
from models.MatrixCapsulesEMRouting import MatCapNet
from models.QuaternionCapsules import QCN
from models.sr_caps import SmallNet
from models.inv_dot_att_rout_caps import CapsModel
from models.VBRouting import VBCapsuleNet
from models.baseCNNs import PytorchNet, smallCNN
from engine.engine import *
from models.feature_extractors import get_feature_extractor
from models.heads import get_feat_head, get_pose_head, get_activation_head
from utils.utils import count_parameters
from torchvision import models

def test():
    print("hello test")
    return 3


def update_opt(opt, args):
    opt.DATA.SAMPLING = args.split
    opt.OUTPUT_DIR = args.outdir
    opt.TRAIN.RESUME = args.resume
    opt.TRAIN.BATCH_SIZE = args.batchsize
    opt.TRAIN.OPTIMIZER.LR = args.lr_rate
    opt.DATA.ROOT = args.dataset
    opt.DATA.NUM_INSTANCES = args.num_instance
    opt.CPU = args.cpu
    opt.DATA.SIZE = args.dsize
    return opt


def create_matcap(args):
    opt = get_cfg_matcap()
    opt.OUTPUT_DIR = "./output_matcap"
    opt.MODEL.NAME = "MatrixCaps"
    opt = update_opt(opt, args)
    model = MatCapNet(opt.DATA.SIZE, opt.DATA.NUM_CLASS)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineCap(model, opt)
    return engine


def create_sr_caps(args):
    opt = get_cfg_matcap()
    opt.OUTPUT_DUR = "./output_srcap"
    opt.MODEL.NAME = "SR-CAPS"
    opt = update_opt(opt, args)
    model = SmallNet()
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineSRCap(model, opt)
    return engine


def create_inv_dot_att_rout_caps(args):
    opt = get_cfg_matcap()
    opt.OUTPUT_DUR = "./output_inv_dot_att_rout_cap"
    opt.MODEL.NAME = "ID-AR-CAPS"
    opt = update_opt(opt, args)
    model = CapsModel(opt.DATA.SIZE, {
        "backbone": {
            "kernel_size": 3,
            "output_dim": 128,
            "input_dim": 1,
            "stride": 2,
            "padding": 1,
            "out_img_size": 16
        },
        "primary_capsules": {
            "kernel_size": 1,
            "stride": 1,
            "input_dim": 128,
            "caps_dim": 16,
            "num_caps": 32,
            "padding": 0,
            "out_img_size": 16
        },
        "capsules": [
            {
                "type": "CONV",
                "num_caps": 32,
                "caps_dim": 16,
                "kernel_size": 3,
                "stride": 2,
                "matrix_pose": True,
                "out_img_size": 7
            },
            {
                "type": "CONV",
                "num_caps": 32,
                "caps_dim": 16,
                "kernel_size": 3,
                "stride": 1,
                "matrix_pose": True,
                "out_img_size": 5
            }
        ],
        "class_capsules": {
            "num_caps": opt.DATA.NUM_CLASS,
            "caps_dim": 16,
            "matrix_pose": True
        }
    }, "resnet", 0.25, 1)  # TODO
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineIDARCap(model, opt)
    return engine


def create_vb_caps(args):
    opt = get_cfg_matcap()
    opt.OUTPUT_DUR = "./output_vbcap"
    opt.MODEL.NAME = "VB-CAPS"
    opt = update_opt(opt, args)
    model = VBCapsuleNet()
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineSRCap(model, opt)
    return engine


def create_basecnn(args):
    opt = get_cfg_baseline_defaults()
    opt.OUTPUT_DIR = "./output_base_cnn"
    opt = update_opt(opt, args)
    model = PytorchNet(opt.DATA.NUM_CLASS)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    opt.DATA.SIZE = args.dsize
    engine = EngineBasic(model, opt)
    return engine


def create_small_cnn(args):
    opt = get_cfg_baseline_defaults()
    opt.OUTPUT_DIR = "./small_cnn"
    opt = update_opt(opt, args)
    model = smallCNN(opt.DATA.NUM_CLASS)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineBasic(model, opt)
    return engine


def create_qcn(args):
    opt = get_cfg_qcn()
    opt.OUTPUT_DIR = "./output_QCN"
    opt.MODEL.NAME = "BaseQCN_" + opt.TRAIN.LOSS_FN.NAME
    opt = update_opt(opt, args)
    model = QCN(opt)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineCap(model, opt)
    return engine


def create_icprqcn(args):
    opt = get_cfg_qcn()
    opt.OUTPUT_DIR = "./output_QCN"
    opt.MODEL.NAME = "ICPRQCN_" + opt.TRAIN.LOSS_FN.NAME
    opt = update_opt(opt, args)
    model = MatQuatCapNet()
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineCap(model, opt)
    return engine


def create_novel_model_v2(args):
    opt = get_cfg_model_v1()
    opt.MODEL.FEAT_SIZE = 32
    opt.DATA.SIZE = 32
    opt.OUTPUT_DIR = "./output_v2"
    opt.MODEL.NAME = "NovelModelV2"
    opt = update_opt(opt, args)
    f_ext = get_feature_extractor("deep1")
    a_head = get_activation_head("basic", 256, 32)
    p_head = get_pose_head("basic", 256, 32, 3)
    f_head = get_feat_head("basic", 256, 32, 32)
    model = Model_V2(opt, f_ext, p_head, a_head, f_head)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineNovel(model, opt)
    return engine


def create_novel_model_v2_MLP(args):
    opt = get_cfg_model_v1()
    opt.MODEL.FEAT_SIZE = 32
    opt.DATA.SIZE = 32
    opt.OUTPUT_DIR = "./output_v2"
    opt.MODEL.NAME = "NovelModelV2"
    opt = update_opt(opt, args)
    f_ext = get_feature_extractor("MLP")
    a_head = get_activation_head("MLP", 256, 32)
    p_head = get_pose_head("MLP", 256, 32, 3)
    f_head = get_feat_head("MLP", 256, 32, 32)
    model = Model_V2(opt, f_ext, p_head, a_head, f_head)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineNovel(model, opt)
    return engine


def create_novel_model_v2_C(args):
    opt = get_cfg_model_v1()
    opt.MODEL.FEAT_SIZE = 32
    opt.DATA.SIZE = 32
    opt.OUTPUT_DIR = "./output_v2_C"
    opt.MODEL.NAME = "NovelModelV2_C"
    opt = update_opt(opt, args)
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 256, 32)
    p_head = get_pose_head("basic", 256, 32, 3)
    f_head = get_feat_head("basic", 256, 32, 32)
    model = Model_V2_C(opt, f_ext, p_head, a_head, f_head)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineNovelContrastive(model, opt)
    return engine


def create_novel_model_v2_deep(args):
    opt = get_cfg_model_v1()
    opt.OUTPUT_DIR = "./output_v2_deep"
    opt.MODEL.NAME = "NovelModelV2_Deep"
    opt = update_opt(opt, args)
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 256, 32)
    p_head = get_pose_head("basic", 256, 32, 3)
    f_head = get_feat_head("basic", 256, 32, 32)
    model = Model_V2_Deep(opt, f_ext, p_head, a_head, f_head)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineNovel(model, opt)
    return engine


def create_novel_model_v2_nofeat(args):
    opt = get_cfg_model_v1()
    opt.OUTPUT_DIR = "./output_v2_nofeat"
    opt.MODEL.NAME = "NovelModelV2_NoFeat"
    opt = update_opt(opt, args)
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 256, 32)
    p_head = get_pose_head("basic", 256, 32, 3)
    model = Model_V2_NoFeat(opt, f_ext, p_head, a_head)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineNovelNoFeat(model, opt)
    return engine


def create_novel_model_v2_alt(args):
    opt = get_cfg_model_v1()
    opt.MODEL.FEAT_SIZE = 32
    opt.OUTPUT_DIR = "./output_v2_alt"
    opt.MODEL.NAME = "NovelModelV2_Alt_" + opt.TRAIN.LOSS_FN.NAME
    opt = update_opt(opt, args)
    f_ext = get_feature_extractor("deep1")
    a_head = get_activation_head("basic", 256, 32)
    p_head = get_pose_head("basic", 256, 32, 3)
    f_head = get_feat_head("basic", 256, 32, opt.MODEL.FEAT_SIZE)
    model = Model_V2_alt(opt, f_ext, p_head, a_head, f_head)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineNovel(model, opt)
    return engine


def create_novel_model_v2_64(args):
    opt = get_cfg_model_v1()
    opt.MODEL.FEAT_SIZE = 32
    opt.DATA.SIZE = 32
    opt.OUTPUT_DIR = "./output_v2_64"
    opt.MODEL.NAME = "NovelModelV2_64"
    opt = update_opt(opt, args)
    f_ext = get_feature_extractor("basic")
    a_head = get_activation_head("basic", 1024, 32)
    p_head = get_pose_head("basic", 1024, 32, 3)
    f_head = get_feat_head("basic", 1024, 32, 32)
    model = Model_V2_alt(opt, f_ext, p_head, a_head, f_head)
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    engine = EngineNovel(model, opt)
    return engine


class MyVGG(nn.Module):
    def __init__(self):
        super(MyVGG, self).__init__()
        self.vgg = models.vgg11(num_classes=10)
        self.conv = nn.Conv2d(kernel_size=(3, 3), padding=1, out_channels=3, in_channels=1)

    def forward(self, input):
        x = self.conv(input)
        return self.vgg(x)


class MySqueeze(nn.Module):
    def __init__(self):
        super(MySqueeze, self).__init__()
        self.vgg = models.squeezenet1_0(num_classes=10)
        self.conv = nn.Conv2d(kernel_size=(3, 3), padding=1, out_channels=3, in_channels=1)

    def forward(self, input):
        x = self.conv(input)
        return self.vgg(x)


class MyEfficient(nn.Module):
    def __init__(self):
        super(MyEfficient, self).__init__()
        self.vgg = models.efficientnet_b0(num_classes=10)
        self.conv = nn.Conv2d(kernel_size=(3, 3), padding=1, out_channels=3, in_channels=1)

    def forward(self, input):
        x = self.conv(input)
        return self.vgg(x)


class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.vgg = models.densenet121(num_classes=10)
        self.conv = nn.Conv2d(kernel_size=(3, 3), padding=1, out_channels=3, in_channels=1)

    def forward(self, input):
        x = self.conv(input)
        return self.vgg(x)


class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.vgg = models.resnet50(num_classes=10)
        self.conv = nn.Conv2d(kernel_size=(3, 3), padding=1, out_channels=3, in_channels=1)

    def forward(self, input):
        x = self.conv(input)
        return self.vgg(x)


class MyMobile(nn.Module):
    def __init__(self):
        super(MyMobile, self).__init__()
        self.vgg = models.mobilenet_v2(num_classes=10)
        self.conv = nn.Conv2d(kernel_size=(3, 3), padding=1, out_channels=3, in_channels=1)

    def forward(self, input):
        x = self.conv(input)
        return self.vgg(x)


class MyEff(nn.Module):
    def __init__(self):
        super(MyEff, self).__init__()
        self.vgg = models.efficientnet_b0(num_classes=10)
        self.conv = nn.Conv2d(kernel_size=(3, 3), padding=1, out_channels=3, in_channels=1)

    def forward(self, input):
        x = self.conv(input)
        return self.vgg(x)


def create_vgg_model(args):
    opt = get_cfg_baseline_defaults()
    opt.OUTPUT_DIR = "./output_base_cnn"
    opt.MODEL.NAME = "VGGNet"
    opt = update_opt(opt, args)
    model = MyVGG()
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    opt.DATA.SIZE = args.dsize
    engine = EngineBasic(model, opt)
    return engine


def create_squeezenet_model(args):
    opt = get_cfg_baseline_defaults()
    opt.OUTPUT_DIR = "./output_base_cnn"
    opt.MODEL.NAME = "SqueezeNet"
    opt = update_opt(opt, args)
    model = MySqueeze()
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    opt.DATA.SIZE = args.dsize
    engine = EngineBasic(model, opt)
    return engine


def create_densenet_model(args):
    opt = get_cfg_baseline_defaults()
    opt.OUTPUT_DIR = "./output_base_cnn"
    opt.MODEL.NAME = "DenseNet"
    opt = update_opt(opt, args)
    model = MyDense()
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    opt.DATA.SIZE = args.dsize
    engine = EngineBasic(model, opt)
    return engine


def create_resnet_model(args):
    opt = get_cfg_baseline_defaults()
    opt.OUTPUT_DIR = "./output_base_cnn"
    opt.MODEL.NAME = "Resnet"
    opt = update_opt(opt, args)
    model = MyResnet()
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    opt.DATA.SIZE = args.dsize
    engine = EngineBasic(model, opt)
    return engine


def create_mobilenet_model(args):
    opt = get_cfg_baseline_defaults()
    opt.OUTPUT_DIR = "./output_base_cnn"
    opt.MODEL.NAME = "MobileNet"
    opt = update_opt(opt, args)
    model = MyMobile()
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    opt.DATA.SIZE = args.dsize
    engine = EngineBasic(model, opt)
    return engine


def create_efficientnet_model(args):
    opt = get_cfg_baseline_defaults()
    opt.OUTPUT_DIR = "./output_base_cnn"
    opt.MODEL.NAME = "EffNet"
    opt = update_opt(opt, args)
    model = MyEff()
    opt.MODEL.NUM_PARAMS = count_parameters(model)
    opt.DATA.SIZE = args.dsize
    engine = EngineBasic(model, opt)
    return engine




# dict returns model, opt and engine for a given model name.
def get_model(args):
    return {
        "BaseCNN": lambda: create_basecnn(args),
        "SmallCNN": lambda: create_small_cnn(args),
        "BaseQCN": lambda: create_qcn(args),
        "ICPRQCN": lambda: create_icprqcn(args),
        "NovelModelV2": lambda: create_novel_model_v2(args),
        "NovelModelV2_Deep": lambda: create_novel_model_v2_deep(args),
        "NovelModelV2_deep_NotDense": lambda: (args),
        "NovelModelV2_deep_SkipCon": lambda: (args),
        "NovelModelV2_NoFeat": lambda: create_novel_model_v2_nofeat(args),
        "MatrixCaps": lambda: create_matcap(args),
        "NovelModelV2_Alt": lambda: create_novel_model_v2_alt(args),
        "NovelModelV2_64": lambda: create_novel_model_v2_64(args),
        "NovelModelV2_C": lambda: create_novel_model_v2_C(args),
        "VGGNet": lambda: create_vgg_model(args),
        "SqueezeNet": lambda: create_squeezenet_model(args),
        "DenseNet": lambda: create_densenet_model(args),
        "Resnet": lambda: create_resnet_model(args),
        "Mobilenet": lambda: create_mobilenet_model(args),
        "EfficientNet": lambda: create_efficientnet_model(args),
        "SRCAPS": lambda: create_sr_caps(args),
        "IDARCAPS": lambda: create_inv_dot_att_rout_caps(args),
        "VBCAPS": lambda: create_vb_caps(args)
    }[args.model]()


if __name__ == '__main__':
    print("hello")
