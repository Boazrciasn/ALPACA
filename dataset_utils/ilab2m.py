import glob
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import os.path as osp
from torch.utils import data

cname_map = {"boat": 0,
              "bus": 1,
              "car": 2,
              "equip": 3,
              "f1car": 4,
              "heli": 5,
              "mil": 6,
              "monster": 7,
              "pickup": 8,
              "plane": 9,
              "semi": 10,
              "tank": 11,
              "train": 12,
              "ufo": 13,
              "van": 14}



class ILAB2M(data.Dataset):

    def __init__(self, root_dir, size):
        self.cname_map = {"boat": 0, "bus": 1, "car": 2, "equip": 3, "f1car": 4, "heli": 5, "mil": 6, "monster": 7, "pickup": 8, "plane": 9,
                          "semi": 10, "tank": 11, "train": 12, "ufo": 13, "van": 14}

        self.dataset_info = "--Cameras:\t\t{c00, c02, c04, c07, c09}\n" \
                            "--Rotations:\t{r01, r02, r03, r04, r06, r07}\n" \
                            "--Lighting Conditions:\t{l0}\n" \
                            "--Focus Values:\t{f2}\n" \
                            "All categories and all instances."
        self.num_class = len(self.cname_map.keys())
        self.root_dir = root_dir
        self.samples = [osp.basename(x) for x in glob.glob(root_dir + "/*.jpg")]
        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        info = self.nameparse(sample)
        gt_class = info["class"]
        img_file = osp.join(self.root_dir, sample)
        image = Image.open(img_file)
        image = TF.resize(image, self.size)
        image = transforms.ToTensor()(image)
        n_image = transforms.Normalize(mean=image.mean([1, 2]), std=image.std([1, 2])*2)(image)
        return n_image, gt_class


    def nameparse(self, name):

        cat = self.cname_map[osp.splitext(name)[0].split("-")[0]]
        inst = int(osp.splitext(name)[0].split("-")[1][1:])
        bg = int(osp.splitext(name)[0].split("-")[2][1:])
        cam = int(osp.splitext(name)[0].split("-")[3][1:])
        rot = int(osp.splitext(name)[0].split("-")[4][1:])
        light = int(osp.splitext(name)[0].split("-")[5][1:])
        focus = int(osp.splitext(name)[0].split("-")[6][1:])

        info_dict = {"class": cat,
                     "camera": cam,
                     "rotation": rot,
                     "light": light,
                     "focus": focus,
                     "instance": inst,
                     "background": bg}

        return info_dict


if __name__ == '__main__':
    print("ILAB2M test")
    root = " "
    dset = ILAB2M(root, 32)
    sample = dset[0]
    print("the end.")
    for xx in dset:
        print(xx)

