import json
from itertools import groupby
from operator import itemgetter
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

random.seed(42)
np.random.seed(42)


class config: 
    metadata_path = "C:\Projects\wild\data\islands\metadata.json"
    pickle_path = 'C:\Projects\seq2bbox\data\pickle\islands\\'

with open(config.metadata_path) as f:
    d = json.load(f)


# some basic data prep..


human_imageid = [i.get('image_id') for i in d['annotations'] if i.get('category_id') == 1] #5981 human labeled. 

# filter out all images that are in a sequence shorter than 3
noseq_imageid = [i.get('id') for i in d['images'] if i.get('seq_num_frames') <= 2]
id_filter = set(noseq_imageid + human_imageid)
del human_imageid
del noseq_imageid
d['annotations'] = [i for i in d['annotations'] if i.get('image_id') not in id_filter]
d['images'] = [i for i in d['images'] if i.get('id') not in id_filter]

# change the category id of 'other' to that of human
print('Changing category id of "other" to that of human...')

for anno in d['annotations']:
    if anno.get('category_id') == 6:
        anno['category_id'] = 1


# merging the annotations and image metadata
for i in range(len(d['images'])):
    d['images'][i]['image_id'] = d['images'][i].pop('id')

my_id = itemgetter('image_id')
meta_anno = []

for k, v in groupby(sorted((d['annotations'] + d['images']), key=my_id), key=my_id):
    meta_anno.append({key:val for d in v for key, val in d.items()})



# this logic helps with stratification by all categories in the sequence...
def get_ordered_sequence_categories(dataset, all_sequences):
    imgs_seq_lookup = {}
    for ma in dataset:
        imgs_seq_lookup.setdefault(ma.get('seq_id','empty'),[]).append(ma)
    orderedseqcats = []
    for seq in all_sequences: 
        cats = [i.get('category_id') for i in imgs_seq_lookup.get(seq)]

        # the following logic finds the most frequent category in the sequence, not including 'empty'. 
        catsnozero = [c for c in cats if c != 0]
        if not catsnozero:  
            maxcats = 0
        else: 
             maxcats = max(set(catsnozero), key=catsnozero.count)

        orderedseqcats.append(maxcats)
    return orderedseqcats

def sequences_to_image_annotations(sequences, annotations):
    sequences = set(sequences)
    images_anns = []
    for ann in annotations:
        if ann.get('seq_id') in sequences:
            images_anns.append(ann)
    return images_anns

def stratified_sequence_split(annotations, partition):
    all_sequences = sorted(list(set([i.get('seq_id') for i in annotations])))
    full_length = len(all_sequences)
    orderedseqcats = get_ordered_sequence_categories(meta_anno,all_sequences)
    labels = [i for i in orderedseqcats]
    large_seq, small_seq = train_test_split(all_sequences, test_size=(full_length//partition), random_state=42, stratify=labels)
    large_anno, small_anno = sequences_to_image_annotations(large_seq, annotations), sequences_to_image_annotations(small_seq, annotations) 
    return large_anno, small_anno

train100, test = stratified_sequence_split(meta_anno, 5)
train100, val =  stratified_sequence_split(train100, 5)
_, test20  = stratified_sequence_split(test, 5)
_, val20 =  stratified_sequence_split(val, 5)
_, train50 = stratified_sequence_split(train100, 2)
_, train20 = stratified_sequence_split(train100, 5)
_, train10 = stratified_sequence_split(train100, 10)
_, train5 = stratified_sequence_split(train100, 20)

for name, dset in zip(['test', 'val', 'train100', 'test20', 'val20', 'train50', 'train20', 'train10', 'train5'], [test, val, train100, test20, val20, train50, train20, train10, train5]):
    with open(config.pickle_path + name + '.pk', 'wb') as f:
        pickle.dump(dset, f, protocol=pickle.HIGHEST_PROTOCOL)