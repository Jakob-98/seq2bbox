#%%
import glob
from pathlib import Path
import pickle
import os
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
import cv2
from functools import partial

# ... local utils import
import utils

from importlib import reload
reload(utils)
# %%

def main():
    filepath = './val20.pk'
    with open(filepath, 'rb') as f:
        meta_anno = pickle.load(f)
    with open('./result.pkl', 'rb') as f:
        res = pickle.load(f)

    #%%
    get_label = {i.get('image_id'): i.get('category_id') for i in meta_anno}
    cat_lookup = {i.get('id') : i.get('name') for i in 
    [{'id': 0, 'name': 'empty'},
    {'id': 1, 'name': 'human'},
    {'id': 2, 'name': 'fox'},
    {'id': 3, 'name': 'skunk'},
    {'id': 4, 'name': 'rodent'},
    {'id': 5, 'name': 'bird'},
    {'id': 6, 'name': 'other'}]
    }

    tp, fp, tn, fn = 0, 0, 0, 0
    # %%
    i = 0
    counter = {}
    for seq in res:
        for im in seq:
            i+=1 
            imid, cat = im
            true_label = get_label.get(imid)
            counter[(cat, cat_lookup.get(true_label))] = counter.get((cat, cat_lookup.get(true_label)), 0) + 1
            if true_label == 0 and cat == 0:
                tp += 1
            if true_label != 0 and cat == 0:
                fp += 1
            if true_label != 0 and cat == 1:
                tn += 1
            if true_label == 0 and cat == 1:
                fn += 1

    # %%
    print(tp, fp, tn, fn)
    print('precision:', tp/(tp + fp))
    print('recall:', tp/(tp + fn))
    print(counter)
    # %%
    # %%
