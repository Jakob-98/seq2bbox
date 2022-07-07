#%%
import glob
from pathlib import Path
import pickle
import os
from cv2 import erode
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
import cv2
from functools import partial
import immods.sequencev2
import warnings
import testresults
import numpy as np

warnings.filterwarnings("ignore")
# ... local utils import

from importlib import reload
reload(immods.sequencev2)

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
    imgs, _, preds = immods.sequencev2.generate_boxed_by_sequence(paths, 256)

    for im, i in zip(imgs, ids): 
        cv2.imwrite('./tmp2/' + i + '.jpg', np.array(im))
    res.append([i for i in zip(ids, preds)])


# %%
sequences = sorted(list(set([i.get('seq_id') for i in meta_anno])))
imgs_seq_lookup = {}
for ma in meta_anno:
    imgs_seq_lookup.setdefault(ma.get('seq_id','empty'),[]).append(ma)

# if __name__ == '__main__':
#     for thresh in (25, 50, 75, 100, 125):
#         for erodecount in (1, 2, 3, 4):   
#             with open('C:\Projects\seq2bbox\config.txt', 'w') as f:
#                 f.write('{} {}'.format(thresh, erodecount))
#             print('threshold, erodecount:', thresh, erodecount) 
#             with mp.Manager() as manager:
#                 res = manager.list()
#                 process_map(partial(_getfeats, res), sequences[:50], max_workers=4, chunksize=2)
#                 with open('./result.pkl', 'wb') as f: 
#                     pickle.dump(list(res), f)
#                 testresults.main()
# #
if __name__ == '__main__':
    print(immods.sequencev2.config.threshold, immods.sequencev2.config.erodecount) 
    with mp.Manager() as manager:
        res = manager.list()
        process_map(partial(_getfeats, res), sequences[:50], max_workers=4, chunksize=2)
        with open('./result.pkl', 'wb') as f: 
            pickle.dump(list(res), f)
        testresults.main()
# %%


