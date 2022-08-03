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
from immods import colorhistslbp, labelgen, sequencev1, jtimer, compression
from pathlib import Path
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
from functools import partial

import warnings
warnings.filterwarnings("ignore")

## Configuration

class configTMP:
    remove_existing = False
    sequential_data = True 
    pickle_path = "C:\Projects\seq2bbox\data\pickle\islands\\test20.pk"
    dataset_path = "C:\Projects\wild\data\islands\images\\images\\"
    image_path = "C:\\temp\data_final\\islands\\images\\ISL64xSeqRGBTest20\\"
    histlbp_path = "C:\\temp\data_final\\islands\\histlbp\\ISL64xSeqRGBTest20\\"
    label_path = "C:\\temp\data_final\\islands\\labels\\ISL64xSeqRGBTest20\\"

    # dataset_path = "C:\\temp\ENA_full\\"
    # image_path = "C:\\temp\data_final\\ENA\\images\\ENA224xCropRGBTrain5\\"
    # label_path = "C:\\temp\data_final\\ENA\\labels\\ENA224xCropRGBTrain5\\"
    # histlbp_path = "C:\\temp\data_final\\ENA\\histlbp\\ENA224xCropRGBTrain5\\"
    # pickle_path = './val20.pk'
    # dataset_path = 'C:/Projects/wild/data/islands/images/images/'
    # image_path = "C:/temp/ispipeline/images/224xCropRGBval20/"
    # label_path = "C:/temp/ispipeline/labels/224xCropRGBval20/"
    # histlbp_path = "C:/temp/ispipeline/histlbp/224xCropRGBval20/"  
    ## pickle_path = './train5.pk'
    # dataset_path = 'C:/Projects/wild/data/islands/images/images/'
    # image_path = "C:/temp/ispipeline/images/224xSeqRGBTrain5/"
    # label_path = "C:/temp/ispipeline/labels/224xSeqRGBTrain5/"
    # histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBTrain5/"
    generate_histlp = False
    generate_labels = True
    convert_grayscale = False
    wavelet_compress = False
    naive_compress = False
    resize = True
    image_size = 64
    # multiprocessing
    chunksize = 8
    max_workers = 8

class config:
    remove_existing= False
    sequential_data = False
    pickle_path = "C:\Projects\seq2bbox\data\pickle\ENA\\train100.pk"
    dataset_path = "C:\\temp\ENA_full\\"
    image_path = "C:\\temp\data_final\\ENA\\images\\ENAORIGxCropRGBTrain100\\"
    histlbp_path = "C:\\temp\data_final\\ENA\\histlbp\\ENAORIGxCropRGBTrain100\\"
    label_path = "C:\\temp\data_final\\ENA\\labels\\ENAORIGxCropRGBTrain100\\"

    # pickle_path = "C:\Projects\seq2bbox\data\pickle\islands\\train5.pk"
    # dataset_path = "C:\Projects\wild\data\islands\images\\images\\"
    # image_path = "C:\\temp\data_final\\islands\\images\\ISL224xCropRGBTrain5\\"
    # histlbp_path = "C:\\temp\data_final\\islands\\histlbp\\ISL224xCropRGBTrain5\\"
    # label_path = "C:\\temp\data_final\\islands\\labels\\ISL224xCropRGBTrain5\\"


    # image_path = "C:\\temp\data_final\\ENA\\images\\ENA224xCropRGBTrain5\\"
    # label_path = "C:\\temp\data_final\\ENA\\labels\\ENA224xCropRGBTrain5\\"
    # histlbp_path = "C:\\temp\data_final\\ENA\\histlbp\\ENA224xCropRGBTrain5\\"
    # pickle_path = './val20.pk'
    # dataset_path = 'C:/Projects/wild/data/islands/images/images/'
    # image_path = "C:/temp/ispipeline/images/224xCropRGBval20/"
    # label_path = "C:/temp/ispipeline/labels/224xCropRGBval20/"
    # histlbp_path = "C:/temp/ispipeline/histlbp/224xCropRGBval20/"  
    ## pickle_path = './train5.pk'
    # dataset_path = 'C:/Projects/wild/data/islands/images/images/'
    # image_path = "C:/temp/ispipeline/images/224xSeqRGBTrain5/"
    # label_path = "C:/temp/ispipeline/labels/224xSeqRGBTrain5/"
    # histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBTrain5/"
    generate_histlp = False 
    generate_labels = True
    convert_grayscale = False
    wavelet_compress = False
    naive_compress = False 
    resize = False 
    image_size = 64 
    # multiprocessing
    chunksize = 8
    max_workers = 8

# create image, label and histlbp directories if they don't exist:
for path in [config.image_path, config.label_path, config.histlbp_path]:
    if not os.path.exists(path):
        os.makedirs(path)
    # delete contents of directories
    if config.remove_existing:
        for file in glob.glob(path + '*'):
            os.remove(file)

# init timer and 'globals'
timer = jtimer.Timer(printupdates=False)
filepath = config.pickle_path
with open(filepath, 'rb') as f:
    meta_anno = pickle.load(f)

if config.sequential_data:
    sequences = sorted(list(set([i.get('seq_id') for i in meta_anno])))
    imgs_seq_lookup = {}
    for ma in meta_anno:
        imgs_seq_lookup.setdefault(ma.get('seq_id','empty'),[]).append(ma)

# main function in multiprocessor
def _createFilesSingular(meta_anno_item):
    # preparing useful vars
    # originalFileNames = [i.get('file_name') for i in meta_anno_subset]
    # baseFilePaths = [p(i) for i in originalFileNames]
    # cats = [i.get('category_id') for i in meta_anno_subset]
    # ids = [i.get('image_id') for i in meta_anno_subset]
    # dims = [(i.get('width'), i.get('height')) for i in meta_anno_subset]
    # bboxs = [i.get('bbox') for i in meta_anno_subset]

    p = lambda x:(str(Path(config.dataset_path) / x))
    baseFilePath = p(meta_anno_item.get('file_name'))
    categoryId = meta_anno_item.get('category_id')
    imageId = meta_anno_item.get('image_id')
    dim = (meta_anno_item.get('width'), meta_anno_item.get('height'))
    bb = meta_anno_item.get('bbox')

    try:
        rawImage = np.array(PIL.Image.open(baseFilePath))
    except OSError:
        print("Error opening file: " + baseFilePath)
        return

    if config.generate_histlp:
        result = colorhistslbp.getLpb(rawImage)
        np.save(Path(config.histlbp_path) / f'{imageId}.npy', result, allow_pickle=False)



    if config.resize:
        resizedImage, ratio, pad = sequencev1.letterbox(rawImage, config.image_size, auto=False)

        fpath = str(Path(config.image_path) / f'{imageId}.jpg')
        if config.convert_grayscale:
            resizedImage = compression.converGrayscale(resizedImage)
            if config.wavelet_compress:
                resizedImage = compression.waveletCompress(resizedImage)
            if config.naive_compress:
                resizedImage = compression.saveCompressed(fpath, resizedImage, 75)
            else: 
                cv.imwrite(fpath, resizedImage)
        else: 
            cv.imwrite(fpath, resizedImage)

        # gen labels but fix the size of the bbox
        if config.generate_labels:
            try:
                bbox = np.array(bb) * ratio[0] # multiply by ratio of resized image 
                bbox[1] = bbox[1] + pad[1]
            except TypeError:
                print('no bbox, skipping...')
                bbox = np.array([0.5,0.5,1,1])
            label = labelgen.generateSingleLabel(cat=categoryId, dimensions=(config.image_size, config.image_size), bbox=bbox)
            with open(Path(config.label_path) / f'{imageId}.txt', 'w') as f:
                f.write(label)

    else: 
        if config.generate_labels:
            # check if bbox exists: 
            if bb is not None:
                bbox = np.array(bb)
                label = labelgen.generateSingleLabel(cat=categoryId, dimensions=dim, bbox=bbox)
                with open(Path(config.label_path) / f'{imageId}.txt', 'w') as f:
                    f.write(label)
            else: 
                print('no bbox, skipping...')
                bbox = np.array([0.5,0.5,1,1])
                label = labelgen.generateSingleLabel(cat=categoryId, dimensions=dim, bbox=bbox)
                with open(Path(config.label_path) / f'{imageId}.txt', 'w') as f:
                    f.write(label)

        # save image
        fpath = str(Path(config.image_path) / f'{imageId}.jpg')
        # save raw image:
        cv.imwrite(fpath, rawImage)




def _createFilesSequential(imagelist):
    timer.updatetime('Init new sequence: ')

    # logic for single images
    if len(imagelist) == 1:
            raise Exception('sequential_data flag is enabled - not supported for single images')

    # logic for sequential images
    else: 

        # preparing useful vars
        p = lambda x:(str(Path(config.dataset_path) / x))
        originalFileNames = [i.get('file_name') for i in imgs_seq_lookup.get(imagelist)]
        baseFilePaths = [p(i) for i in originalFileNames]

        cats = [i.get('category_id') for i in imgs_seq_lookup.get(imagelist)] 

        ids = [i.get('image_id') for i in imgs_seq_lookup.get(imagelist)]

        loadedImages = []
        # generating color histograms/LBP, labels. 
        for baseFilePath, categoryId, imageId in zip(baseFilePaths, cats, ids):
            try:
                rawImage = np.array(PIL.Image.open(baseFilePath))
            except OSError:
                print("Error opening file: " + baseFilePath)
                return
            loadedImages.append(rawImage)

            if config.generate_histlp: 
                result = colorhistslbp.getLpb(rawImage)

                np.save(Path(config.histlbp_path) / (imageId), result, allow_pickle=False)
                # with open(Path(config.histlbp_path) / (imageId + '.txt'), 'w') as f:
                #     f.write(result)
    
            if config.generate_labels: 
                label = labelgen.generateSeqLabel(categoryId)
                with open(Path(config.label_path) / (imageId + '.txt'), 'w') as f:
                    f.write(label)

        timer.updatetime('generating lpb and label: ')
        
        # generating sequenced images with bounding box
        sequencedImages, _, _ = sequencev1.generate_boxed_by_sequence(loadedImages, config.image_size)
        timer.updatetime('generating sequence: ')

        
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
        if config.sequential_data:
            process_map(partial(_createFilesSequential), sequences, max_workers = config.max_workers, chunksize = config.chunksize)
        else:
            process_map(partial(_createFilesSingular), meta_anno, max_workers = config.max_workers, chunksize = config.chunksize)