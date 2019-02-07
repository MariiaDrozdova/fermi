import sys
import os
#import cv2
import random
import pickle
import astropy.wcs as wcs
import numpy as np

#import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import poisson
import scipy.optimize as optimize


#from ds9_cmap import *
#from make_fermi_maps import *


import scipy.optimize as optimize

_fname_prefold = ''
_fname_data = 'trash/'
_fname_database = _fname_prefold + _fname_data


def save_obj(obj, name ):
    with open('../res/'+ name, 'wb') as f:
        pickle.dump(obj, f, 2)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def f(params, params2):#, observ, models):
    observ, models = params2

    params_array = [0]*len(models)
    #print(params)  # <-- you'll see that params is a NumPy array
    params_array = [i for i in params] # <-- for readability you may wish to assign names to the component variables
    im = np.sum([models[i]*params[i] for i in range(len(models))], axis=0)
    return likelihood(im, observ)

def openAsImage(fname):        
    fits_image = fits.open(fname)[0].data
    return fits_image

def findBrightest(picture):
    m = len(picture[0])
    index = np.argmax(picture)
    i = index // m
    j = index % m
    return([i, j])    


def likelihood(model, observations):
    ll = 1.00
    #print(observations)
    #print(model)
    x = poisson.pmf(observations,model) 
    #print(np.sum(x))
    x = (x <= 0) + x
    #print(np.sum(-np.log(x)))
    return np.sum(-np.log(x))
            

def get_center_from_wcs_string(string):
    crval_s = string.find('CRVAL : ') + len('CRVAL : ') 
    space_s = crval_s + string[crval_s:].find(" ")
    cprix_s = string.find('CRPIX : ')
    ra = float(string[crval_s:space_s])
    dec = float(string[space_s+1:cprix_s])
    return ra, dec

def test_prog(observations, output_dir):
    fname_xml = "test_src_model_const.xml"
    d = {}
    d["chosen_fits"] = []
    d["koefs"] = []
    d["fin_fits"] = None
    diff_back = np.zeros((200,200))#generate_diffuse_background(fname_xml, get_center_from_wcs_string(str(w)), output_dir)
    
    model = diff_back


    max_likelihood = np.inf
    koefs = []
    chosen_fits = []
    koef = 1
    for k in range(0,50):
            model = diff_back * k*0.1
            new_likelihood = likelihood(model, observations)

            if max_likelihood > new_likelihood:
                max_likelihood = new_likelihood
                koef = k            

    d["chosen_fits"].append(diff_back)
    d["koefs"].append(koef*0.1)
    d["fin_fits"] = koef *diff_back
    d["indexes"] = []
    int_cr =4
    new_likelihood = 0
    likelihood0 = np.inf + 5
    star = 0     
    bnds = [(koef, 10*(koef+1))]
    params_k = [1.0] 
    values = []
    while (-new_likelihood + likelihood0 > int_cr  or likelihood0 == 0):
        star += 1
        index = findBrightest(observations-d["fin_fits"])
        values.append(observations[index[0], index[1]] - d["fin_fits"][index[0], index[1]])

        likelihood0 = new_likelihood
        max_likelihood = -np.inf

        test1 = np.loadtxt("test1.fits")
        test=np.zeros((600,600))
        x = index[0]
        y = index[1]
        test[x:x+400, y:y+400] = test1
        point_model = test[200:400, 200:400]

        d['indexes'].append(index)
        if ((observations[index[0], index[1]] - d["fin_fits"][index[0], index[1]]) < 0):
            break
        bnds.append((1e-20, 1+(observations[index[0], index[1]] - d["fin_fits"][index[0], index[1]])/point_model[index[0], index[1]]*1000))
        params_k = list(d["koefs"])
        d["chosen_fits"].append(point_model)
        params_k.append(1.2*np.abs(observations[index[0], index[1]] - d["fin_fits"][index[0], index[1]])/point_model[index[0], index[1]])
        results_o = optimize.minimize(f, params_k, [observations,d["chosen_fits"]], method='SLSQP',bounds=bnds, options={'eps':1e0, 'ftol':10})
        d["koefs"] = results_o.x
        new_likelihood = results_o.fun
        d["fin_fits"] = np.sum([d["chosen_fits"][i]*d["koefs"][i] for i in range(len(d["chosen_fits"]))], axis=0)
        #print("-------------------")     
    return d['indexes']

if __name__=="__main__":
    if len(sys.argv)==1:
        input_dir = '../dataset_1/'
        output_dir = '../output_1/'
    else:
        input_dir = os.path.abspath(sys.argv[1])
        output_dir = os.path.abspath(sys.argv[2])
        
    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    y = {}
    for file_pkl in os.listdir(input_dir):

        #print(file_pkl)
        #sys.stdout = open(os.devnull, 'w')
        observations = np.load(input_dir+file_pkl[:-4] + '.npy')
        result = test_prog(observations, output_dir)
        np.save('../res/' + file_pkl[:-4] + '.npy',result)
        #sys.stdout = sys.__stdout__

        #clean_trash(fname_prefold = output_dir)

    
#"""
