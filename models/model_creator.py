from models.NovelModel import *
from models.MatrixCapsulesEMRouting import MatCapNet
from models.QuaternionCapsules import QCN
from models.baseCNNs import PytorchNet, smallCNN
from engine.engine import *
from models.feature_extractors import get_feature_extractor
from models.heads import get_feat_head, get_pose_head, get_activation_head
from utils.utils import count_parameters


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
    }[args.model]()


if __name__ == '__main__':
    print("hello")
