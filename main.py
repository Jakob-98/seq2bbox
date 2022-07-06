#%%
import glob
import imp
from pathlib import Path
import pickle
import os
import cv2
import immods.sequence
import numpy as np
import immods.jplots
# ... local utils import
import utils

from importlib import reload
reload(immods.sequence)
reload(immods.jplots)

basepath = './examples'

# %%
filepath = './val20.pk'
with open(filepath, 'rb') as f:
    meta_anno = pickle.load(f)

# %%
imgs = []
for folder in os.listdir(basepath):
    fpath = Path(basepath) / folder
    filepaths, filenames = [], []
    for file in os.listdir(fpath):
        if file.endswith('.jpg'):
            filenames.append(file)
            filepaths.append(fpath / file)
    imgs, bgs, _ = immods.sequence.generate_boxed_by_sequence(filepaths, 256)
    immods.jplots.plotMultiImg(imgs)
    for i, (im, fn) in enumerate(zip(imgs, filenames)): 
        cv2.imwrite('./tmp/' + fn, np.array(im))
        cv2.imwrite('./tmp/' + 'bg' + fn, np.array(bgs[i]))

# %%
