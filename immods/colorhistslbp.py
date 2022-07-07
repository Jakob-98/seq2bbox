from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import mahotas
from tqdm import tqdm
from six.moves import cPickle as pickle #for performance
from functools import partial
import cv2

def getLpb(imgpath, singlechannel=False):
    return
    if singlechannel: 
        if np.shape(img)[2] == 3: 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    else: 
        b, g, r =  (mahotas.features.lbp(img[:, :, i], 2.5, 12) for i in (0, 1, 2))

    