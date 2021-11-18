import json
import os
import imageio
import os.path as pth


def namegen(c_name, sample):
    return "{}_{}_{}.png".format(c_name, sample['azimuth'], sample['elevation'])


dataset_dir = ""

with open(os.path.join(dataset_dir, "dataset.json")) as dfile:
    dataset = json.load(dfile)

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

id_cat_map = dict()

for k in category_ids:
    id_cat_map[category_ids[k]] = k


categories = dict()
for k in category_ids:
    categories[k] = list()

object_ids = []
for sample in dataset['data']:
    if sample['obj_id'] not in categories[id_cat_map[sample['class_id']]]:
        categories[id_cat_map[sample['class_id']]].append(sample['obj_id'])





########### CHANGED HERE #####################
samples_for_gif = dict()
for c in categories:
    samples_for_gif[c] = list()
    for sample in dataset['data']:
        if sample['obj_id'] == categories[c][0]:
            print(sample['elevation'])
            if (sample['azimuth'] % 80 == 0) and (sample['dist'] == 1):
                print(sample['azimuth'], sample['dist'], sample['elevation'])
                samples_for_gif[c].append(sample)




out_path = ""
for cat in categories:
    for sample in samples_for_gif[cat]:
        image = imageio.imread(os.path.join(dataset_dir, 'imgs', sample['image_file']))
        d_path = pth.join(out_path, namegen(cat, sample))
        imageio.imwrite(d_path, image)


