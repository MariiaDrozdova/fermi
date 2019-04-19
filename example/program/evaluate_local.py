#!/usr/bin/env python
import sys, os, os.path
import warnings
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
warnings.filterwarnings('ignore')
from scipy.spatial import distance as sc_distance

from sys import argv

import libscores
#import yaml
from libscores import ls, filesep, mkdir, read_array, compute_all_scores, write_scores
set_ = 'data'#!!!!!!!!!!CHANGE HERE
ch = '4'#!!!!!!!!!!CHANGE HERE
# Default I/O directories:
root_dir = "../"
#!!!!!!!!!!CHANGE HERE
default_input_dir = root_dir + "sample_result_submission/"
default_output_dir =  '../cnn_4layers/'+set_+'/'
default_output_dir =  '../yolo_res/'+set_+'/'
default_output_dir =  '../sample_trivial1/'+set_+'/'
#default_output_dir =  '/media/mariia/Maxtor/new_ubuntu/gsl/d3po-master/res/'
#
#default_output_dir =  '/media/mariia/Maxtor/fermicomp/sample_cnnGREATFILTERS/'+set_+'/'
#default_output_dir =  '/media/mariia/Maxtor/fermicomp/sample_cnn2/'+set_+'/'
default_hidden_dir = root_dir + "reference_data_1/"

# Debug flag 0: no debug, 1: show all scores, 2: also show version amd listing of dir
debug_mode = 1

# Constant used for a missing score
missing_score = -0.999999

# Version number
scoring_version = 1.0
def openAsImage(fname):        
    fits_image = fits.open(fname)[0].data
    return fits_image

import binascii

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int(binascii.hexlify(text.encode(encoding, errors)), 16))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return int2bytes(n).decode(encoding, errors)

def int2bytes(i):
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))

def distance(true_coords_vector, coords_vector):
    if len(coords_vector) == 0 and len(true_coords_vector) != 0:
        res = 0
        return np.sum(sc_distance.cdist(np.array([[100,100]]), true_coords_vector, 'euclidean')) + 16*len(true_coords_vector)
    if len(coords_vector) != 0 and len(true_coords_vector) == 0:
        return np.sum(sc_distance.cdist(np.array([[100,100]]), coords_vector, 'euclidean')) + 16*len(coords_vector)
    if len(coords_vector) == 0 and len(true_coords_vector) == 0:
        return 0
    true_coords = np.vstack({tuple(row) for row in true_coords_vector})#np.unique(true_coords_vector, axis=0)
    coords = np.vstack({tuple(row) for row in coords_vector})#np.unique(coords_vector, axis=0)
    tree_true_coords = BallTree(true_coords)
    tree_coords = BallTree(coords)
    distance_from_true_array, _ = tree_true_coords.query(coords)
    distance_from_found_array, _ = tree_coords.query(true_coords)
    distance_from_found = np.sum(distance_from_found_array)
    distance_from_true = np.sum(distance_from_true_array)
    return np.sum(distance_from_found_array) + np.sum(distance_from_true_array) 

def distance_extend(true_coords_vector, coords_vector):
    #print("extend")
    if len(coords_vector) == 0 and len(true_coords_vector) != 0:
        res = 0
        return np.sum(sc_distance.cdist(np.array([[100,100]]), true_coords_vector, 'euclidean')) + 25*len(true_coords_vector), [[0, len(true_coords_vector)], [0, 200*200-len(true_coords_vector)]]
    if len(coords_vector) != 0 and len(true_coords_vector) == 0:
        return np.sum(sc_distance.cdist(np.array([[100,100]]), coords_vector, 'euclidean')) + 25*len(coords_vector), [[0, 0], [len(coords_vector), 200*200-len(coords_vector)]]
    if len(coords_vector) == 0 and len(true_coords_vector) == 0:
        return 0, [[0, 0], [0, 200*200]]
    true_coords = np.vstack({tuple(row) for row in true_coords_vector})#np.unique(true_coords_vector, axis=0)
    coords = np.vstack({tuple(row) for row in coords_vector})#np.unique(coords_vector, axis=0)
    tree_true_coords = BallTree(true_coords)
    tree_coords = BallTree(coords)
    distance_from_true_array, _ = tree_true_coords.query(coords)
    distance_from_found_array, _ = tree_coords.query(true_coords)
    #print(distance_from_true_array)
    #print(distance_from_found_array)
    threshold = 25
    if False:
        TP = np.sum((distance_from_true_array<threshold)*1)
        FN = np.sum((distance_from_true_array>threshold)*1)
        FP = np.sum((distance_from_found_array>threshold)*1)
    else:
        TP = np.sum((distance_from_found_array<threshold)*1)
        FN = np.sum((distance_from_found_array>threshold)*1)
        FP = np.sum((distance_from_true_array>threshold)*1)
    m = [[TP, FN],[FP, 200*200-FN-len(true_coords_vector)]]
    distance_from_found = np.sum(distance_from_found_array)
    distance_from_true = np.sum(distance_from_true_array)
    #print(np.sum(distance_from_found_array) + np.sum(distance_from_true_array))
    return np.sum(distance_from_found_array) + np.sum(distance_from_true_array), np.array(m)


def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if True:
    #res1_names = os.listdir(res1_dir)
    #res2_names = os.listdir(res2_dir)
    #ref1_names = os.listdir(ref1_dir)
    #ref2_names = os.listdir(ref2_dir)
    res_dir  =default_output_dir
    ref_dir  = "/media/mariia/Maxtor/final_datasets/dataset2/reference_data_"+ch+"/" +set_+'/' #!!!!!!!!!!CHANGE HERE
    ref_names = os.listdir(ref_dir)
    res_names = os.listdir(default_output_dir)
    distances1 = []
    distances2 = []
    distances_ = []
    """
    for i in range(len(res1_names)):                 
        y_train = np.load(res1_dir + res1_names[i])
        Y_train = np.load(ref1_dir + res1_names[i])

        dist = distance(np.array(Y_train), np.array(y_train))
        if dist != np.inf:
            distances1.append(dist)

    for i in range(len(res2_names)):
        y_train = np.load(res2_dir + res2_names[i])
        Y_train = np.load(ref2_dir + res2_names[i])


        dist = distance(np.array(Y_train), np.array(y_train))
        if dist != np.inf:
            distances2.append(dist)

    print("Your training score is %f" % (np.mean(distances1)))
    print("Your testing score is %f" % (np.mean(distances2)))
    """
    matrix = np.zeros((2,2))
    roc = []
    stars = 0
    for i in range(len(res_names)):  
        if res_names[i][0] != 's':
           continue
        print(res_names[i])               
        y_train = np.load(res_dir + res_names[i])
        print(ref_dir + res_names[i])
        Y_train = np.load(ref_dir + res_names[i])

        #new_y = []
        #for i in y_train:
        #    #print(i)
        #    if i[0] < 1e-15:
        #        #print(i)
        #        new_y.append(np.array([i[1],i[2]]))
        #y_train = np.array(new_y)
        stars += len(Y_train)
        dist, m = distance_extend(np.array(Y_train), np.array(y_train))
        if dist != np.inf:
            distances_.append(dist) 
        print(dist)  
        matrix = matrix + m
        try:
            SEN = m[0][0]/(m[0][0]+m[0][1])
            SPE = m[1][1]/(m[1][1]+m[1][0])
            roc.append([1-SPE, SEN])
        except:
            continue
    print(distances_) 
    print(str(np.mean(distances_)))
    print(str(np.std(distances_)))
    print(matrix)
    print(stars)
    plt.hist(distances_, bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Histogram of distance distribution on '+set_+' set')
    #plt.text(1040, 17, r'CNN1')
    plt.grid(True)
    plt.show()
    
    roc.sort()
    x = [roc[i][0] for i in range(len(roc))]
    y = [roc[i][1] for i in range(len(roc))]
    plt.plot(x, y)
    plt.show()
    #print(roc)


