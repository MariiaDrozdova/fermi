#!/usr/bin/env python
import sys, os, os.path
import warnings
import numpy as np
import pickle
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.neighbors import BallTree
warnings.filterwarnings('ignore')

from sys import argv

import libscores
#import yaml
from libscores import ls, filesep, mkdir, read_array, compute_all_scores, write_scores

# Default I/O directories:
root_dir = "../"
default_input_dir = root_dir + "sample_result_submission/"
default_output_dir = root_dir + "scoring_output/"
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
        return np.inf
    if len(coords_vector) != 0 and len(true_coords_vector) == 0:
        return np.inf
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

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if False:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')          
    

    #res1_names = os.listdir(res1_dir)
    #res2_names = os.listdir(res2_dir)
    #ref1_names = os.listdir(ref1_dir)
    #ref2_names = os.listdir(ref2_dir)
    ref_names = os.listdir(ref_dir)
    res_names = os.listdir(res_dir)
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
    for i in range(len(res_names)):                 
        y_train = np.load(res_dir + res_names[i])
        Y_train = np.load(ref_dir + res_names[i])

        dist = distance(np.array(Y_train), np.array(y_train))
        if dist != np.inf:
            distances_.append(dist)    
    f = open(output_filename, "w")
    f.write(str(np.mean(distances_)))
    f.close()



if __name__ == "__main__":

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        hidden_dir = default_hidden_dir
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        hidden_dir = argv[3]
        # Create the output directory, if it does not already exist and open output files 
    try:
        print(output_dir)
        os.listdir(output_dir)
    except:
        mkdir(output_dir)
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'w')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'w')

    # Get the metric
    metric_name, scoring_function = 'distance', distance

    # Get all the solution files from the solution directory    
    #print(hidden_dir)
    #reference_path = hidden_dir    
    #solution_folder_names = sorted(os.listdir(reference_path))


    reference_path = ls(os.path.join(input_dir, 'ref'))
    res_path = ls(os.path.join(input_dir, 'res'))[0]
 

    solution_folder_name = sorted(ls(os.path.join(input_dir, 'ref', '*')))
    res_name = res_path[0]

    res_folder_names = []
    solution_folder_names = []
    for i in solution_folder_name:
        if i.find('zip') == -1:
            for j in os.listdir(i):
                solution_folder_names.append(j)


    solution_folder_name = solution_folder_name[0]
    reference_path = reference_path[0]
    # Loop over files in solution directory and search for predictions with extension .predict having the same basename

    for i, solution_folder in enumerate(solution_folder_names):
        distances_ = []
        set_num = i + 1  # 1-indexed
        score_name = 'set%s_score' % set_num

        # Extract the dataset name from the file name
        
        res_dir = os.path.join(solution_folder_name, solution_folder)
        ref_dir = os.path.join(input_dir, 'res', solution_folder)
        solution_names = sorted(os.listdir(res_dir))
        for j, solution_file in enumerate(solution_names):
            y_train = np.load(os.path.join(res_dir, solution_file))
            Y_train = np.load(os.path.join(ref_dir, solution_file))

            dist = distance(np.array(Y_train), np.array(y_train))
            if dist != np.inf:
                distances_.append(dist)    


        # Write score corresponding to selected task and metric to the output file
        score_file.write(score_name + ': %0.12f\n' % (np.mean(distances_)))

    # End loop for solution_file in solution_names

    # Read the execution time and add it to the scores:
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'res', 'metadata'), 'r'))
        score_file.write(b"Duration: %0.6f\n" % metadata['elapsedTime'])
    except:
        score_file.write('Duration: 0\n')

        html_file.close()
    score_file.close()

    # Lots of debug stuff
    if debug_mode > 1:
        swrite('\n*** SCORING PROGRAM: PLATFORM SPECIFICATIONS ***\n\n')
        show_platform()
        show_io(input_dir, output_dir)
        show_version(scoring_version)

        # exit(0)
