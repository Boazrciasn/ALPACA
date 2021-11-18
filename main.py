import argparse
from engine.engine import *
import engine.engine as ENGINE
from models.model_creator import get_model




parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="dataset root", type=str, default="")
parser.add_argument("-b", "--batchsize", help="batchsize for (train, test)", type=int, default=32)
parser.add_argument("-w", "--numworkers", help="number of workers for dataloader", type=int, default=8)
parser.add_argument("-e", "--epoch", help="training epochs", type=int, default=250)
parser.add_argument("-r", "--resume", help="resume or from scratch", action="store_true")
parser.add_argument("-o", "--outdir", help="training output directory", type=str, default="./output")
parser.add_argument("-lr", "--lr_rate", help="learning rate", type=float, default=1e-3)
parser.add_argument("--num_instance", help="number of object instances, maximum 100", type=int, default=50)
parser.add_argument("--split", help="train-test split, current options are: \nazimuth_dark_side,\n"
                                    "azimuth_stepover,\n"
                                    "azimuth_steplarger,\n"
                                    "elevation_stepdarkside,\n"
                                    "elevation_stepover,\n"
                                    "distance_stepover,\n", type=str, default="distance_stepover")
parser.add_argument("--cpu", help="run on cpu? Note that it is faster at the moment", type=bool, default=False)
parser.add_argument("-m", "--model", help="Which model to run?", type=str, default="VBCAPS")
parser.add_argument( "--dsize", help="input size", type=int, default=32)
"""
model list: 
            BaseCNN
            SmallCNN
            BaseQCN
            NovelModelV2
            NovelModelV2_Deep
            NovelModelV2_deep_NotDense
            NovelModelV2_deep_SkipCon
            NovelModelV2_NoFeat
            MatrixCaps
            NovelModelV2_Alt
            NovelModelV2_C
            VGGNet
            SqueezeNet
            DenseNet
            ResNet
            MobileNet
            EfficientNet
            SRCAPS
            IDARCAPS
            VBCAPS
"""


if __name__ == '__main__':
    args = parser.parse_args()
    ENGINE.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(torch.cuda.is_available())
    print(args.cpu)
    print(torch.cuda.is_available() and not args.cpu)
    print("DEVICE:{}".format(ENGINE.device))
    engine = get_model(args)
    engine.train()

