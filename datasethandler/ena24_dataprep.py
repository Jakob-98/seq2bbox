# This file prepares the dicts for the ENA dataset.
#%%
import json
import random
import numpy as np
from operator import itemgetter
from itertools import groupby
from sklearn.model_selection import train_test_split
import pickle

# seeds for reproducibility
random.seed(42)
np.random.seed(42)

class config:
    metadata_path = 'C:\Projects\wild\data\ENA24\ena24_public.json'
    ena_local_base = 'c:/temp/ena/images/'
    pickle_dir = './data/pickle/'

with open(config.metadata_path) as f:
    d = json.load(f)


# changing the labels to remove human labels but keep [1,..., n] order
d['categories'][-1]['id'] = 8
for anno in d['annotations']:
    if anno.get('category_id') == 22:
        anno['category_id'] = 8


# merging the annotations and image metadata
for i in range(len(d['images'])):
    d['images'][i]['image_id'] = d['images'][i].pop('id')

my_id = itemgetter('image_id')
meta_anno = []

for k, v in groupby(sorted((d['annotations'] + d['images']), key=my_id), key=my_id):
    meta_anno.append({key:val for d in v for key, val in d.items()})

# removing missing images
missing_ids = set()
for idx, image in enumerate(meta_anno):
    if not image.get('category_id'):
        missing_ids.add(image['image_id'])
print('number of missing ids: {}'.format(len(missing_ids)))
meta_anno = [img for img in meta_anno if img.get('image_id') not in missing_ids]

# Generating subsets... 
full_length = len(meta_anno)
labels = [i.get('category_id') for i in meta_anno]
train, test = train_test_split(meta_anno, test_size=(full_length//5), random_state=42, stratify=labels)
labels = [i.get('category_id') for i in train]
train, validate = train_test_split(train, test_size=(full_length//10), random_state=42, stratify=labels)
labels = [i.get('category_id') for i in train]
full_train = len(train)
trainsubsets = [train]
for i in (0.5, 0.8, 0.9, 0.95): 
    subset, throwaway = train_test_split(train, test_size=i, random_state=42, stratify=labels)
    trainsubsets.append(subset)

#saving the subsets to pickle
for name, dset in zip(['test', 'val', 'train100', 'test20', 'val20', 'train50', 'train20', 'train10', 'train5'], [test, val, train100, test20, val20, train50, train20, train10, train5]):
    with open(config.pickle_path + name + '.pk', 'wb') as f:
        pickle.dump(dset, f, protocol=pickle.HIGHEST_PROTOCOL)


# EDA....
# https://www.statology.org/seaborn-barplot-show-values/
# plotting...
