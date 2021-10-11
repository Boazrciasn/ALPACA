import json
import random
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import os.path as osp
from torch.utils import data



################### SPLIT CRITERIONS: ##################################

def dark_side(sample):
    return 0 <= sample['azimuth'] < 180


def skip_sample(sample):
    return sample['azimuth'] % 40



def even_random(_):
    out = even_random.list[even_random.s]
    even_random.s +=1
    return out

even_random.s = 0
even_random.r = 2

def onethird_random(_):
    out = onethird_random.list[onethird_random.s]
    onethird_random.s +=1
    return out

onethird_random.s = 0
onethird_random.r = 3

split_criterion_dict = {"azimuth_dark_side": lambda a: (a['azimuth'] < 180),
                        "azimuth_stepover": lambda a: (a['azimuth'] % 40 == 0),
                        "azimuth_steplarger": lambda a: (a['azimuth'] % 60 == 0),
                        "elevation_stepdarkside": lambda a: (a['elevation'] < -0),
                        "elevation_stepover": lambda a: (a['elevation'] == -30 or a['elevation'] == -6 or a['elevation'] == 18),
                        "distance_stepover": lambda a: (a['dist'] == 1.75),
                        "random_even": even_random,
                        "random_onethird": onethird_random}

#########################################################################

class NovelViewDataset(data.Dataset):

    def __init__(self, sample_list, cat_map, root_dir, size):
        self.samples = sample_list
        self.cat_map = cat_map
        self.id_map = dict()
        for k in self.cat_map.keys():
            self.id_map[self.cat_map[k]['id']] = self.cat_map[k]['name']

        self.num_class = len(self.cat_map.keys())
        self.root_dir = root_dir
        self.size = size

    def __len__(self):
        return len(self.samples)


########################  GET ITEM SPECIALIZATIONS ######################################

# GRAYSCALE OUTPUT ONLY:
class GDataset(NovelViewDataset):
    def __init__(self, sample_list, cat_map, root_dir, size=128):
        super(GDataset, self).__init__(sample_list, cat_map, root_dir, size)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_dir = osp.join(self.root_dir, "imgs")
        img_file = osp.join(img_dir, sample['image_file'])
        # Read image and apply transforms:
        image = Image.open(img_file).convert("L")
        image = TF.resize(image, self.size)
        image = transforms.ToTensor()(image)
        #image = transforms.Normalize(mean=[image.mean()], std=[image.std()])(image)

        gt_class = self.cat_map[sample['class_id']]['id']
        gt_pose = sample['pose']

        return image, gt_class, gt_pose


class DepthDataset(NovelViewDataset):
    def __init__(self, sample_list, cat_map, root_dir, size=128):
        super(DepthDataset, self).__init__(sample_list, cat_map, root_dir, size)

    # Depth image is scaled with distance since all depth images are normalized in itself.
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_dir = osp.join(self.root_dir, "imgs")
        img_file = osp.join(img_dir, sample['image_file'])
        img_file = img_file.split('.')[0] + '_d' + img_file.split('.')[1]
        # Read image and apply transforms:
        depth_image = Image.open(img_file).convert("L") * sample['dist']
        depth_image = transforms.ToTensor()(depth_image)
        depth_image = TF.resize(depth_image, self.size)
        gt_class = self.cat_map[sample['class_id']]['id']
        gt_pose = sample['pose']

        return depth_image, gt_class, gt_pose


class GDepthDataset(NovelViewDataset):
    def __init__(self, sample_list, cat_map, root_dir, size=128):
        super(GDepthDataset, self).__init__(sample_list, cat_map, root_dir, size)

    # Depth image is scaled with distance since all depth images are normalized in itself.
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_dir = osp.join(self.root_dir, "imgs")
        d_img_file = osp.join(img_dir, sample['image_file'])
        d_img_file = d_img_file.split('.')[0] + '_d' + d_img_file.split('.')[1]
        img_file = osp.join(img_dir, sample['image_file'])

        depth_image = Image.open(d_img_file).convert("L") * sample['dist']
        depth_image = transforms.ToTensor()(depth_image)
        image = Image.open(img_file).convert("L")
        image = transforms.ToTensor()(image)
        image = TF.resize(image, self.size)
        depth_image = TF.resize(depth_image, self.size)
        gt_class = self.cat_map[sample['class_id']]['id']
        gt_pose = sample['pose']

        return image, depth_image, gt_class, gt_pose


get_item_types = {'gray': GDataset,
                  'depth': DepthDataset,
                  'gray_depth': GDepthDataset}


def create_datasets(root_dir, split_mode="azimuth_dark_side", data_mode="gray", size=128, num_instances=50):
    with open(osp.join(root_dir, "dataset.json"), 'r') as f:
        dataset_file = json.load(f)

    cat_map = dict()
    for idx, c in enumerate(dataset_file['dataset_info']['classes']):
        cat_map[dataset_file['dataset_info']['classes'][c]] = {"name": c, "id": idx, "objects": dict()}

    criterion = split_criterion_dict[split_mode]
    train_samples, test_samples = list(), list()
    samples = reduce_instances(cat_map, dataset_file, num_instances)
    num_samples = len(samples)
    if hasattr(criterion, 's'):
        criterion.list = [True]*(num_samples//criterion.r) + [False]*(num_samples - num_samples//criterion.r)
        random.shuffle(criterion.list)

    for s in samples:
        target = train_samples if criterion(s) else test_samples
        target.append(s)

    train_set = get_item_types[data_mode](train_samples, cat_map, root_dir, size)
    test_set = get_item_types[data_mode](test_samples, cat_map, root_dir, size)

    del dataset_file

    return train_set, test_set


def reduce_instances(cat_map, dataset_file, num_instances):
    for s in dataset_file['data']:
        if s["obj_id"] in cat_map[s["class_id"]]['objects']:
            if cat_map[s["class_id"]]['objects'][s["obj_id"]]:
                cat_map[s["class_id"]]['objects'][s["obj_id"]].append(s)
        else:
            cat_map[s["class_id"]]['objects'][s["obj_id"]] = [s]
    samples = []
    for c in cat_map:
        obj_samples = []
        count = 0
        for obj in cat_map[c]['objects']:
            obj_samples += cat_map[c]['objects'][obj]
            count += 1
            if count == num_instances:
                print("reached instance limit.")
                print("{}: {}".format(cat_map[c]['name'], len(obj_samples)))
                break

        samples += obj_samples
    return samples


if __name__ == '__main__':
    file = ""
    train, test = create_datasets(file, "random_onethird")
    train_set = data.DataLoader(train, batch_size=4, shuffle=True)

    for t in train_set:
        print(t[0].size(), t[1], t[2])
