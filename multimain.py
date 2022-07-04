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

basepath = Path('C:\Projects\wild\data\islands\images\images\\')

# %%
filepath = './val20.pk'
with open(filepath, 'rb') as f:
    meta_anno = pickle.load(f)
# %%
def _getfeats(res, sequence):
    filenames = [i.get('file_name') for i in imgs_seq_lookup.get(sequence)]
    ids = [i.get('image_id') for i in imgs_seq_lookup.get(sequence)]
    p = lambda x: str(Path(basepath) / x)
    paths = [p(x) for x in filenames]
    _, preds = utils.generate_boxed_by_sequence(paths, 64)
    res.append([i for i in zip(ids, preds)])


# %%
sequences = list(set([i.get('seq_id') for i in meta_anno]))
imgs_seq_lookup = {}
for ma in meta_anno:
    imgs_seq_lookup.setdefault(ma.get('seq_id','empty'),[]).append(ma)

# %%
if __name__ == '__main__':
    with mp.Manager() as manager:
        res = manager.list()
        process_map(partial(_getfeats, res), sequences, max_workers=12, chunksize=2)
        with open('./result.pkl', 'wb') as f: 
            pickle.dump(list(res), f)
# %%
