import glob
import json
from .blender_render import render
import random
import os.path as osp
import os
from tqdm import tqdm


def find_cat_name(synsey_id, data):

    for d in data:
        if d['synsetId'] == synsey_id:
            return d

with open("/Volumes/Storage/ShapeNetCore.v2/taxonomy.json") as jfile:
    data = json.load(jfile)

path = "/Volumes/Storage/ShapeNetCore.v2/"

category_ids = {
    'Airplane': '02691156',
    'Car': '02958343',
    'Chair': '03001627',
    'Guitar': '03467517',
    'Train': '04468005',
    'Motorbike': '03790512',
    'Sofa': '04256520',
    'Mug': '03797390',
    'Table': '04379243',
    'Lamp': '03636649'
}


if __name__ == '__main__':

    dataset_dict = {"dataset_info": {"classes": category_ids,
                                     "image_size": [128, 128],
                                     "num_samples:": 0,
                                     "view_dist": {"azimuth": {"range": (0, 360),
                                                               "views": 18},
                                                   "elevation": {"range": (-30, 30),
                                                                 "views": 6},
                                                   "dist": {"range": (1, 2.5),
                                                            "views": 3}}
                                     },
                    "data": []}

    cat_samples = [glob.glob(d + "/*") for d in glob.glob(path+"*[!.json]")]
    cats = glob.glob(path+"*[!.json]")
    output_folder = "/Volumes/Storage/myDataset/"
    cats = []
    for c in category_ids:
        print(category_ids[c])
        cats.append(glob.glob(path + category_ids[c] + "/*"))
    num_samples = 100 #min([len(c) for c in cats])
    shapenetNovel = dict()

    for c, s in zip(category_ids, cats):
        shapenetNovel[c] = random.sample(s, num_samples)

    for c_name in tqdm(shapenetNovel):

        dataset_dict = {"dataset_info": {"classes": category_ids,
                                         "image_size": [128, 128],
                                         "num_samples:": 0,
                                         "view_dist": {"azimuth": {"range": (0, 360),
                                                                   "views": 18},
                                                       "elevation": {"range": (-60, 60),
                                                                     "views": 6},
                                                       "dist": {"range": (1, 2.5),
                                                                "views": 3}}
                                         },
                        "data": []}

        obj_img_folder = osp.join(output_folder, c_name)
        os.makedirs(obj_img_folder, exist_ok=True)

        for obj in tqdm(shapenetNovel[c_name]):
            obj_file = osp.join(obj, "models", "model_normalized") + ".obj"
            dataset_dict["data"] += render(obj_file, 1, True, False, obj_img_folder, dataset_dict["dataset_info"]["view_dist"], dataset_dict["dataset_info"]["image_size"])

        with open(osp.join(output_folder, '{}.json'.format(c_name)), 'w') as jfile:
            json.dump(dataset_dict, jfile)

