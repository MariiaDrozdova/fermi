import pickle
import os, sys

import random
import pickle
import astropy.wcs as wcs
import numpy as np

#import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import poisson
import scipy.optimize as optimize


import scipy.optimize as optimize


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


class model:
    def __init__(self, shared=''):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_labels=1
        self.is_trained=False
        self.shared=shared

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        
        #self.num_train_samples = X.shape[0]
        self.is_trained=True
        print("Training is done!")

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
	In out case X is a dictionary of two fields: Fermi Image and its WCS object.
90
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        y = []
        for i in range(len(X)):
            print(str(i) + '/' + str(len(X)))
            observations = X[i]
  
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
            int_cr =10
            new_likelihood = 0
            likelihood0 = np.inf + 5
            star = 0     
            bnds = [(koef, 10*(koef+1))]
            params_k = [1.0] 
            values = []
            iterations = 100
            test1 = np.load(self.shared + "/test1.npy")
            while ((-new_likelihood + likelihood0 > int_cr  or likelihood0 == 0) and star<iterations): 
                if star > 10:
                    break
                
                
                star += 1
                index = findBrightest(observations-d["fin_fits"])
                values.append(observations[index[0], index[1]] - d["fin_fits"][index[0], index[1]])

                likelihood0 = new_likelihood
                max_likelihood = -np.inf

                
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
            print(star)     
            #plt.imshow(d["fin_fits"]) 
            #plt.show()
        return [d['indexes'] for _ in range(len(X))]

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"), 2) 

    def load(self, path="./"):
        print(path)
        modelfile = path + '_model.pickle'
        print(modelfile)
        if os.path.isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self




