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
from immods import colorhistslbp, labelgen, sequencev1, jtimer
from pathlib import Path
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
from functools import partial

import warnings
warnings.filterwarnings("ignore")

## Configuration

class config:
    pickle_path = './val20.pk'
    dataset_path = 'C:/Projects/wild/data/islands/images/images/'
    image_path = "C:/temp/ispipeline/images/224xCropRGBval20/"
    label_path = "C:/temp/ispipeline/labels/224xCropRGBval20/"
    histlbp_path = "C:/temp/ispipeline/histlbp/224xCropRGBval20/"  
    ## pickle_path = './train5.pk'
    # dataset_path = 'C:/Projects/wild/data/islands/images/images/'
    # image_path = "C:/temp/ispipeline/images/224xSeqRGBTrain5/"
    # label_path = "C:/temp/ispipeline/labels/224xSeqRGBTrain5/"
    # histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBTrain5/"
    generate_histlp = False
    sequence_bboxer = False
    generate_labels = True
    convert_grayscale = False
    wavelet_compress = False
    naive_compress = False
    resize = True
    image_size = 224
    labelfix = True

    # multiprocessing
    chunksize = 8
    max_workers = 8

# init timer and 'globals'

timer = jtimer.Timer(printupdates=False)
filepath = config.pickle_path
with open(filepath, 'rb') as f:
    meta_anno = pickle.load(f)

sequences = sorted(list(set([i.get('seq_id') for i in meta_anno])))
imgs_seq_lookup = {}
for ma in meta_anno:
    imgs_seq_lookup.setdefault(ma.get('seq_id','empty'),[]).append(ma)

def hacky_labelfix(cats):
        # replacing cat id 6 with 1. 
    cats = [i if i!=0 else 6 for i in cats]
    return cats

# main function in multiprocessor

def _createFiles(imagelist):
    timer.updatetime('Init new sequence: ')

    # logic for single images
    if len(imagelist) == 1:
        print('WARNING: single image found - sequential pipeline')
        raise NotImplementedError

    # logic for sequential images
    else: 

        # preparing useful vars
        p = lambda x:(str(Path(config.dataset_path) / x))
        originalFileNames = [i.get('file_name') for i in imgs_seq_lookup.get(imagelist)]
        baseFilePaths = [p(i) for i in originalFileNames]

        cats = [i.get('category_id') for i in imgs_seq_lookup.get(imagelist)] 

        cats = hacky_labelfix(cats)

        ids = [i.get('image_id') for i in imgs_seq_lookup.get(imagelist)]

        loadedImages = []
        # generating color histograms/LBP, labels. 
        for baseFilePath, categoryId, imageId in zip(baseFilePaths, cats, ids):
            rawImage = np.array(PIL.Image.open(baseFilePath))
            loadedImages.append(rawImage)

            if config.generate_histlp: 
                result = colorhistslbp.getLpb(rawImage)

                np.save(Path(config.histlbp_path) / (imageId), result, allow_pickle=False)
                # with open(Path(config.histlbp_path) / (imageId + '.txt'), 'w') as f:
                #     f.write(result)
    
            if config.generate_labels: 
                label = labelgen.generateLabel(categoryId)
                with open(Path(config.label_path) / (imageId + '.txt'), 'w') as f:
                    f.write(label)

        timer.updatetime('generating lpb and label: ')
        
        # generating sequenced images with bounding box if found if config.sequence_bboxer. 
        if config.sequence_bboxer: 
            sequencedImages, _, _ = sequencev1.generateSequence(loadedImages, cats, ids, config.image_size)
            timer.updatetime('generating sequence: ')
        elif config.resize: 
            sequencedImages = [sequencev1.letterbox(im, config.image_size, auto=False)[0] for im in loadedImages]
            timer.updatetime('resizing images: ')
        
        # saving sequenced images
        for sequencedImage, imageId in zip(sequencedImages, ids):
            if config.convert_grayscale: 
                raise NotImplementedError
            if config.wavelet_compress:
                raise NotImplementedError
            if config.naive_compress:
                raise NotImplementedError
            
            cv2.imwrite(str(Path(config.image_path) / (imageId + '.jpg')), sequencedImage)


# multiprocessing

if __name__ == '__main__':
    # _createFiles(sequences) #listunhashableerror
    with mp.Manager() as manager: 
        process_map(partial(_createFiles), sequences, max_workers = config.max_workers, chunksize = config.chunksize)
