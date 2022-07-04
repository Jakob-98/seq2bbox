#%%
import glob
import imp
from pathlib import Path
import pickle
import os
import cv2
import immods.sequence

# ... local utils import
import utils

from importlib import reload
reload(utils)

basepath = './examples'

# %%
filepath = './val20.pk'
with open(filepath, 'rb') as f:
    meta_anno = pickle.load(f)

# %%
imgs = []
for folder in os.listdir(basepath):
    fpath = Path(basepath) / folder
    filepaths = []
    for file in os.listdir(fpath):
        if file.endswith('.jpg'):
            filepaths.append(fpath / file)
    imgs = utils.generate_boxed_by_sequence(filepaths, 128)
    raise

# %%
