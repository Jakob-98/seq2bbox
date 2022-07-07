import cv2
import numpy as np
import json
import shutil
import os
import glob
from matplotlib import pyplot as plt
from scipy import ndimage
import cv2 as cv
import PIL
import pickle
from immods import colorhistslbp, labelgen, sequencev1
from pathlib import Path
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
from functools import partial


class config:
    pickle_path = './train5.pk'
    dataset_path = 'C:\Projects\wild\data\islands\images\images\\'
    image_path = ""
    label_path = ""
    histlbp_path = ""
    generate_histlp = True
    generate_labels = True
    convert_grayscale = False
    wavelet_compress = False
    naive_compress = False
    resize = True
    image_size = 224

    # multiprocessing
    chunksize = 2
    max_workers = 4


filepath = config.pickle_path
with open(filepath, 'rb') as f:
    meta_anno = pickle.load(f)

sequences = sorted(list(set([i.get('seq_id') for i in meta_anno])))
imgs_seq_lookup = {}
for ma in meta_anno:
    imgs_seq_lookup.setdefault(ma.get('seq_id','empty'),[]).append(ma)


def _createFiles(imagelist):
    if len(imagelist) == 1:
        print('WARNING: single image found - sequential pipeline')
        raise NotImplementedError
    else: 
        p = lambda x:(str(Path(config.dataset_path) / x))
        originalFileNames = [i.get('file_name') for i in imgs_seq_lookup.get(imagelist)]
        baseFilePaths = [p(i) for i in originalFileNames]

        cats = [i.get('category_id') for i in imgs_seq_lookup.get(imagelist)]
        ids = [i.get('image_id') for i in imgs_seq_lookup.get(imagelist)]

        loadedImages = []

        for baseFilePath, categoryId, imageId in zip(baseFilePaths, cats, ids):
            rawImage = np.array(PIL.Image.open(baseFilePath))
            loadedImages.append(rawImage)

            if config.generate_histlp: 
                lbp = colorhistslbp.getLpb(rawImage)

                result = "1234542 511234 1231234312"
                with open(Path(config.histlbp_path) / imageId / '.txt') as f:
                    f.writeline(result)
    
            if config.generate_labels: 
                label = labelgen.generateLabel(categoryId)
                with open(Path(config.label_path) / imageId / '.txt') as f:
                    f.writeline(label)

        sequencedImages, _, _ = sequencev1.generate_boxed_by_sequence(loadedImages, config.image_size)

        for sequencedImage, imageId in zip(sequencedImages, ids):
            if config.convert_grayscale: 
                raise NotImplementedError
            if config.wavelet_compress:
                raise NotImplementedError
            if config.naive_compress:
                raise NotImplementedError
            
            cv2.imwrite(Path(config.image_path) / imageId / '.jpg', sequencedImage)

            

if __name__ == '__main__':
    with mp.Manager() as manager: 
        process_map(partial(_createFiles), sequences, max_workers = config.max_workers, chunksize = config.chunksize)
