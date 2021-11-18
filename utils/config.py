from yacs.config import CfgNode as CN
import os.path as osp


def save_cfg(cfg):
    f = open(osp.join(cfg.OUTPUT_DIR, "setup.yaml"), "a")
    f.write(cfg.dump())
    f.close()


def get_cfg_baseline_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    _C = CN()

    _C.SYSTEM = CN()
    _C.SYSTEM.NUM_GPU = 1
    _C.SYSTEM.NUM_WORKERS = 4
    _C.OUTPUT_DIR = "./output"

    _C.WANDB = CN()
    _C.WANDB.PROJECT_NAME = "novel-viewpoint-analysis"
    _C.WANDB.ENTITY = "vvgl-ozu"
    _C.WANDB.LOG_DIR = ""
    _C.WANDB.NUM_ROW = 0

    _C.DATA = CN()
    _C.DATA.ROOT = ""
    _C.DATA.SAMPLING = "azimuth_dark_side"  # or "skip_sample"
    _C.DATA.INPUT = "gray"
    _C.DATA.SIZE = 128
    _C.DATA.NUM_WORKERS = 1

    _C.TRAIN = CN()
    _C.TRAIN.START_STEP = 0
    _C.TRAIN.SHUFFLE = True
    _C.TRAIN.MAX_IT = 5000000
    _C.TRAIN.BATCH_SIZE = 4
    _C.TRAIN.OPTIMIZER = "adam"
    _C.TRAIN.RESUME = False
    _C.TRAIN.LOG_INTERVAL = 200
    _C.TRAIN.SAVE_INTERVAL = 300000
    _C.TRAIN.TEST_TEST_INTERVAL = 10000
    _C.TRAIN.TRAIN_TEST_INTERVAL = 200000

    _C.TRAIN.LOSS_FN = CN()
    _C.TRAIN.LOSS_FN.NAME = "cross_entropy"

    _C.TRAIN.SCHEDULER = CN()
    _C.TRAIN.SCHEDULER.NAME = "step_lr"
    _C.TRAIN.SCHEDULER.STEP_SIZE = 50000
    _C.TRAIN.SCHEDULER.GAMMA = .8
    _C.TRAIN.SCHEDULER.LAST_EPOCH = -1

    _C.TRAIN.OPTIMIZER = CN()
    _C.TRAIN.OPTIMIZER.NAME = "adam"
    _C.TRAIN.OPTIMIZER.LR = 3e-3

    _C.TEST = CN()
    _C.TEST.BATCH_SIZE = 8
    _C.TEST.SHUFFLE = False

    _C.MODEL = CN()
    _C.MODEL.NAME = ""
    _C.MODEL.TRAIN = True

    return _C.clone()


def get_cfg_matcap():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    _C = CN()

    _C.SYSTEM = CN()
    _C.SYSTEM.NUM_GPU = 1
    _C.SYSTEM.NUM_WORKERS = 4
    _C.OUTPUT_DIR = "./output_matcap"

    _C.WANDB = CN()
    _C.WANDB.PROJECT_NAME = "novel-viewpoint-analysis"
    _C.WANDB.ENTITY = "vvgl-ozu"
    _C.WANDB.LOG_DIR = ""
    _C.WANDB.NUM_ROW = 0

    _C.DATA = CN()
    _C.DATA.ROOT = ""
    _C.DATA.SAMPLING = "azimuth_dark_side"  # or "skip_sample"
    _C.DATA.INPUT = "gray"
    _C.DATA.SIZE = 32
    _C.DATA.NUM_WORKERS = 1
    _C.DATA.NUM_CLASS = 10

    _C.TRAIN = CN()
    _C.TRAIN.START_STEP = 0
    _C.TRAIN.SHUFFLE = True
    _C.TRAIN.MAX_IT = 200000
    _C.TRAIN.BATCH_SIZE = 4
    _C.TRAIN.OPTIMIZER = "adam"
    _C.TRAIN.RESUME = False
    _C.TRAIN.LOG_INTERVAL = 200
    _C.TRAIN.SAVE_INTERVAL = 50000
    _C.TRAIN.TEST_TEST_INTERVAL = 10000
    _C.TRAIN.TRAIN_TEST_INTERVAL = 2000000

    _C.TRAIN.LOSS_FN = CN()
    _C.TRAIN.LOSS_FN.NAME = "spread"

    _C.TRAIN.SCHEDULER = CN()
    _C.TRAIN.SCHEDULER.NAME = "step_lr"
    _C.TRAIN.SCHEDULER.STEP_SIZE = 10000
    _C.TRAIN.SCHEDULER.GAMMA = .4
    _C.TRAIN.SCHEDULER.LAST_EPOCH = -1

    _C.TRAIN.OPTIMIZER = CN()
    _C.TRAIN.OPTIMIZER.NAME = "adam"
    _C.TRAIN.OPTIMIZER.LR = 3e-3

    _C.TEST = CN()
    _C.TEST.BATCH_SIZE = 8
    _C.TEST.SHUFFLE = False

    _C.MODEL = CN()
    _C.MODEL.NAME = ""
    _C.MODEL.TRAIN = True
    _C.MODEL.BRANCHED = True
    _C.MODEL.ROUTING_METHOD = "EM-Routing"  # add alternatives here
    _C.MODEL.ROUTING_IT = 2

    return _C.clone()


def get_cfg_qcn():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    _C = CN()

    _C.SYSTEM = CN()
    _C.SYSTEM.NUM_GPU = 1
    _C.SYSTEM.NUM_WORKERS = 4
    _C.OUTPUT_DIR = "./output_qcn"

    _C.WANDB = CN()
    _C.WANDB.PROJECT_NAME = "novel-viewpoint-analysis"
    _C.WANDB.ENTITY = "vvgl-ozu"
    _C.WANDB.LOG_DIR = ""
    _C.WANDB.NUM_ROW = 0

    _C.DATA = CN()
    _C.DATA.ROOT = ""
    _C.DATA.SAMPLING = "azimuth_dark_side"  # or "skip_sample"
    _C.DATA.INPUT = "gray"
    _C.DATA.SIZE = 32
    _C.DATA.NUM_WORKERS = 1
    _C.DATA.NUM_CLASS = 10

    _C.TRAIN = CN()
    _C.TRAIN.START_STEP = 0
    _C.TRAIN.SHUFFLE = True
    _C.TRAIN.MAX_IT = 10000000
    _C.TRAIN.BATCH_SIZE = 4
    _C.TRAIN.OPTIMIZER = "adam"
    _C.TRAIN.RESUME = False
    _C.TRAIN.LOG_INTERVAL = 200
    _C.TRAIN.SAVE_INTERVAL = 50000
    _C.TRAIN.TEST_TEST_INTERVAL = 200000
    _C.TRAIN.TRAIN_TEST_INTERVAL = 10000

    _C.TRAIN.LOSS_FN = CN()
    _C.TRAIN.LOSS_FN.NAME = "spread"

    _C.TRAIN.SCHEDULER = CN()
    _C.TRAIN.SCHEDULER.NAME = "step_lr"
    _C.TRAIN.SCHEDULER.STEP_SIZE = 50000
    _C.TRAIN.SCHEDULER.GAMMA = .8
    _C.TRAIN.SCHEDULER.LAST_EPOCH = -1

    _C.TRAIN.OPTIMIZER = CN()
    _C.TRAIN.OPTIMIZER.NAME = "adam"
    _C.TRAIN.OPTIMIZER.LR = 3e-3

    _C.TEST = CN()
    _C.TEST.BATCH_SIZE = 8
    _C.TEST.SHUFFLE = False

    _C.MODEL = CN()
    _C.MODEL.NAME = ""
    _C.MODEL.TRAIN = True
    _C.MODEL.INIT_TYPE = "uniform_pi"  # or normal
    _C.MODEL.BRANCHED = True
    _C.MODEL.ROUTING_METHOD = "EM-Routing"  # add alternatives here
    _C.MODEL.ROUTING_IT = 2

    return _C.clone()


def get_cfg_model_v1():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    _C = CN()

    _C.SYSTEM = CN()
    _C.SYSTEM.NUM_GPU = 1
    _C.SYSTEM.NUM_WORKERS = 1
    _C.OUTPUT_DIR = "./output_qcn"

    _C.WANDB = CN()
    _C.WANDB.PROJECT_NAME = "novel-viewpoint-analysis"
    _C.WANDB.ENTITY = "vvgl-ozu"
    _C.WANDB.LOG_DIR = ""
    _C.WANDB.NUM_ROW = 0

    _C.DATA = CN()
    _C.DATA.ROOT = ""
    _C.DATA.SAMPLING = "azimuth_dark_side"  # or "skip_sample"
    _C.DATA.INPUT = "gray"
    _C.DATA.SIZE = 32
    _C.DATA.NUM_WORKERS = 1
    _C.DATA.NUM_CLASS = 10

    _C.TRAIN = CN()
    _C.TRAIN.START_STEP = 0
    _C.TRAIN.SHUFFLE = True
    _C.TRAIN.MAX_IT = 10000000
    _C.TRAIN.BATCH_SIZE = 4
    _C.TRAIN.OPTIMIZER = "adam"
    _C.TRAIN.RESUME = False
    _C.TRAIN.LOG_INTERVAL = 200
    _C.TRAIN.SAVE_INTERVAL = 10000
    _C.TRAIN.TEST_TEST_INTERVAL = 10000
    _C.TRAIN.TRAIN_TEST_INTERVAL = 10000

    _C.TRAIN.LOSS_FN = CN()
    _C.TRAIN.LOSS_FN.NAME = "spread"  # spread

    _C.TRAIN.SCHEDULER = CN()
    _C.TRAIN.SCHEDULER.NAME = "step_lr"  # step_lr, reduce_on_plat
    _C.TRAIN.SCHEDULER.STEP_SIZE = 40000
    _C.TRAIN.SCHEDULER.GAMMA = .9
    _C.TRAIN.SCHEDULER.LAST_EPOCH = -1

    _C.TRAIN.OPTIMIZER = CN()
    _C.TRAIN.OPTIMIZER.NAME = "adam"
    _C.TRAIN.OPTIMIZER.LR = 1e-3

    _C.TEST = CN()
    _C.TEST.BATCH_SIZE = 8
    _C.TEST.SHUFFLE = False

    _C.MODEL = CN()
    _C.MODEL.NAME = ""
    _C.MODEL.TRAIN = True
    _C.MODEL.INIT_TYPE = "uniform_pi"  # or normal
    _C.MODEL.ROUTING_METHOD = "attention_router"  # add alternatives here
    _C.MODEL.ROUTING_IT = 2
    _C.MODEL.ACTIVATION_HEAD_NAME = "basic"
    _C.MODEL.POSE_HEAD_NAME = "basic"
    _C.MODEL.FEAT_SIZE = 32

    return _C.clone()


if __name__ == '__main__':
    opt = get_cfg_baseline_defaults()
    opt.dump()
