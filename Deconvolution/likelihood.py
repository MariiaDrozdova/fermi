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


from ds9_cmap import *
from make_fermi_maps import *


import scipy.optimize as optimize

_fname_prefold = '/media/mariia/Maxtor/'
_fname_data = 'database2/'
_fname_database = _fname_prefold + _fname_data

def f(params, params2):#, observ, models):
    observ, models = params2

    params_array = [0]*len(models)
    #print(params)  # <-- you'll see that params is a NumPy array
    params_array = [i for i in params] # <-- for readability you may wish to assign names to the component variables
    im = np.sum([models[i]*params[i] for i in range(len(models))], axis=0)
    return likelihood(im, observ)**2

def makeNoisy(data):

       return np.random.poisson(data)

def openAsImage(fname):        
    fits_image = fits.open(fname)[0].data
    #img = np.zeros((200,200,3))
    #img2 = np.zeros_like(img)
    #img2[:,:,0] = fits_image
    #img2[:,:,1] = fits_image
    #img2[:,:,2] = fits_image
    #print(fits_image.info())
    return fits_image


class XML():
    def __init__(self, name, center):

        self.name = name
        self.center = center
        self.start = """<?xml version="1.0" ?>
     <source_library title="source library">

     <!-- Point Sources -->"""
        self.end = """
        
     </source_library>
        """
        self.point_src = []
        self.diffuse = ""
        sky_f = self.name.find("sky")+3
        new_str = self.name[sky_f:]
        self.sim = new_str[:new_str.find("_")]
        
    def generate_diffuse(self):
        self.diffuse = """
        <!-- Diffuse Sources -->
        <source name="gll_iem_v06" type="DiffuseSource">
            <spectrum apply_edisp="false" type="PowerLaw">
                <parameter free="1" max="100" min="0" name="Prefactor" scale="1" value="1"/>
                <parameter free="0" max="1" min="-1" name="Index" scale="1.0" value="0"/>
                <parameter free="0" max="2e2" min="5e1" name="Scale" scale="1.0" value="1e2"/>
            </spectrum>
            <spatialModel file="glprp.fits" type="MapCubeFunction">
                <parameter free="0" max="1e3" min="1e-3" name="Normalization" scale="1.0" value="1.0"/>
            </spatialModel>
        </source>
        <source name="iso_P8R2_CLEAN_V6_v06" type="DiffuseSource">
            <spectrum apply_edisp="false" file="iso_P8R2_CLEAN_V6_v06.txt" type="FileFunction">
                <parameter free="1" max="100" min="0" name="Normalization" scale="1" value="1"/>
            </spectrum>
            <spatialModel type="ConstantValue">
                <parameter free="0" max="10.0" min="0.0" name="Value" scale="1.0" value="1.0"/>
            </spatialModel>
        </source>
        """

    def generate_diffuse_k(self, k):
        self.diffuse = '''
        <!-- Diffuse Sources -->
        <source name="gll_iem_v06" type="DiffuseSource">
            <spectrum apply_edisp="false" type="PowerLaw">
                <parameter free="1" max="100" min="0" name="Prefactor" scale="1" value="''' + str(k)+''' "/>
                <parameter free="0" max="1" min="-1" name="Index" scale="1.0" value="0"/>
                <parameter free="0" max="2e2" min="5e1" name="Scale" scale="1.0" value="1e2"/>
            </spectrum>
            <spatialModel file="glprp.fits" type="MapCubeFunction">
                <parameter free="0" max="1e3" min="1e-3" name="Normalization" scale="1.0" value="1.0"/>
            </spatialModel>
        </source>
        <source name="iso_P8R2_CLEAN_V6_v06" type="DiffuseSource">
            <spectrum apply_edisp="false" file="iso_P8R2_CLEAN_V6_v06.txt" type="FileFunction">
                <parameter free="1" max="100" min="0" name="Normalization" scale="1" value="1"/>
            </spectrum>
            <spatialModel type="ConstantValue">
                <parameter free="0" max="10.0" min="0.0" name="Value" scale="1.0" value="1.0"/>
            </spatialModel>
        </source>
        '''      
        
    def add_point_source(self, val1, val2):
        print(val1)
        print(val2)
 
        point_str = '''
            <source ROI_Center_Distance="8.583" name="obj_0" type="PointSource">
                <spectrum apply_edisp="false" type="PowerLaw">
                <!-- Source is 8.582976793871993 degrees away from ROI center -->
                <!-- Source is outside ROI, all parameters should remain fixed -->
                    <parameter free="0" max="1e4" min="1e-4" name="Prefactor" scale="1e-10" value="10.0"/>
                    <parameter free="0" max="10.0" min="0.0" name="Index" scale="-1.0" value="1.0"/>
                    <parameter free="0" max="5e5" min="30" name="Scale" scale="1.0" value="590.3473105381578"/>
                </spectrum>
                <spatialModel type="SkyDirFunction">
                    <parameter free="0" max="360.0" min="-360.0" name="RA" scale="1.0" value="'''+ str(val1) + '''"/>
                    <parameter free="0" max="90" min="-90" name="DEC" scale="1.0" value="''' + str(val2) + '''"/>
                </spatialModel>
            </source>        
        '''
        self.point_src.append(point_str)
        pass
    
    def modify_last_point_source(self, factor):
        point_str = self.point_src[-1]
        start = point_str.find('name="Prefactor" scale="1e-15" value="') + len('name="Prefactor" scale="1e-15" value="')
        new_value = float(point_str[start:start+2])*factor
        new_point_str = point_str[:start] + str(new_value) + point_str[start+3:]
        self.point_src[-1] = new_point_str
    
    def full_string(self):
        res = ""
        res = self.start
        for i in self.point_src:
            res += str(i)
        res += self.diffuse
        res += self.end        
        return res
        
    def get_xml(self):
        res = self.full_string()
        print(res)
        f = open(self.name, 'w')
        f.write(res)
        f.close()
        
    def run_back(self):
        self.generate_diffuse()
        res = self.start
        res += self.diffuse
        res += self.end     
        print(self.name[:self.name.rfind("/")] + 'sky' + str(self.sim) + '_coord_ra' + str(self.center[0])+'_dec' + str(self.center[1]) + '_test' + '_src_model_const.xml')    
        #exit()     
        f = open(self.name[:self.name.rfind("/")+1] + 'sky' + str(self.sim) + '_coord_ra' + str(self.center[0])+'_dec' + str(self.center[1]) + '_test' + '_src_model_const.xml', 'w')
        f.write(res)
        f.close()   
        name = self.name
        simulateMaps(ra=self.center[0], dec=self.center[1], database=_fname_data, prefold=_fname_prefold, sim=self.sim, flag='_test')
        model_name = name[:name.find("_src_model_const.xml")]+"_model_map.fits"
        new_model = model_name
        print(new_model)
        res = openAsImage(model_name)

        return res
    
    def run_point(self, i):
        res = self.start
        res += self.point_src[-1]
        res += self.end              
        print(self.name[:self.name.rfind("/")] + 'sky' + str(self.sim) + '_coord_ra' + str(self.center[0])+'_dec' + str(self.center[1]) + '_test' + str(i) + '_src_model_const.xml')    
        #exit()     
        f = open(self.name[:self.name.rfind("/")+1] + 'sky' + str(self.sim) + '_coord_ra' + str(self.center[0])+'_dec' + str(self.center[1]) + '_test' + str(i) + '_src_model_const.xml', 'w')
        f.write(res)
        f.close()   
        name = self.name
        simulateMaps(ra=self.center[0], dec=self.center[1], database=_fname_data, prefold=_fname_prefold, sim=self.sim, xml=0, flag='_test' + str(i))
        model_name = name[:name.find("_src_model_const.xml")]+ str(i)+"_model_map.fits"
        new_model = ""+model_name
        res = openAsImage(model_name)
        #f = open(new_model, 'w')
        #f.write(res)
        #f.close()     
        return res
        
    def print_string(self):
        print(self.start)

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
            
def sources(model, observations, mask):
    int_cr = 0.2#34.024
    new_likelihood = int_cr+1
    likelihood0 = 0
    new_mask = mask
    while (new_likelihood - likelihood0 > int_cr):
        observations = np.multiply(observations, new_mask)
        index = findBrightest(observations)
        #print(index)
        likelihood0 = new_likelihood
        mask = new_mask
        new_mask = mask.copy()
        new_mask[index[0], index[1]] = 0
        max_likelihood = - np.inf
        koef = 0
        for k in range(10):
            model[index[0], index[1]] += observations[index[0], index[1]]
            new_likelihood = likelihood(model, observations)
            if max_likelihood < new_likelihood:
                max_likelihood = new_likelihood
                koef = k
        new_likelihood = max_likelihood
        model[index[0], index[1]]=observations[index[0], index[1]] * (k+1)
        #print(new_likelihood)
    return model



def get_center(string):
    ra = string.find("ra")
    dec = string.find("dec")
    end = string[dec:].find("_")+dec
    center = [float(string[ra+2:dec-1]), float(string[dec+3:end])]
    return center

def save_obj(obj, name ):
    with open('/media/mariia/Maxtor/database5/results_test/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/media/mariia/Maxtor/database5/results_test/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def extract_names(string):
    option_position = string.find("back")
    string[option_position:]
    d = {}
    d.update({"fname_points" : string[:option_position] + "points_model_map.fits"})
    d.update({"fname_back" : string[:option_position] + "back_model_map.fits"})
    d.update({"fname_xml" : string[:option_position] + "test_src_model_const.xml"})    
    d.update({"final_fits_folder" : string[:string.rfind("/")+1]})
    d.update({"sky_number" : string[string.rfind("/")+1:][:string[string.rfind("/")+1:].find("_")]})
    d.update({"final_fits" : d["final_fits_folder"] + d["sky_number"] + ".fits"})
    return d

def run_prog(d_name, test=False):

    final_fits_folder = d_name["final_fits_folder"]
    final_fits = d_name["final_fits"]
    sky_number = d_name["sky_number"]
    fname_points = d_name["fname_points"]
    fname_back = d_name["fname_back"]
    fname_xml = d_name["fname_xml"]
    d = d_name
    if test:
        x = XML(fname_xml, get_center(d_name["fname_points"]))
        x.generate_diffuse()
        diff_back = x.run_back()
    else:
        diff_back = openAsImage(fname_back) 
    model = diff_back
    observations = d["observations"]

    w = d["w"]

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
                print(koef)
                print(new_likelihood)
    d["chosen_fits"].append(diff_back)
    d["koefs"].append(koef*0.1)
    d["fin_fits"] = koef *diff_back
    #x.generate_diffuse_k(koef)
    int_cr =d["int_cr"]
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
        print(np.min(observations[index[0], index[1]] - d["fin_fits"][index[0], index[1]]))
        print(observations[index[0], index[1]])
        print(d["fin_fits"][index[0], index[1]])
        #while index in indexes:
        #    index[1] = index[1] + 3
        likelihood0 = new_likelihood
        max_likelihood = -np.inf
        ra, dec = w.all_pix2world([index[1]+1], [index[0]+1], 1)
        print(ra)
        print(dec)
        test1 = np.loadtxt("test1.fits")
        test=np.zeros((600,600))
        x = index[0]
        y = index[1]
        test[x:x+400, y:y+400] = test1
        point_model = test[200:400, 200:400]
        #x.add_point_source(ra[0], dec[0])
        
        
        d['indexes'].append(index)
        #point_model = x.run_point(star)
        #plt.imshow(point_model)
        #plt.show()
        if ((observations[index[0], index[1]] - d["fin_fits"][index[0], index[1]]) < 0):
            break
        bnds.append((1e-20, 1+(observations[index[0], index[1]] - d["fin_fits"][index[0], index[1]])/point_model[index[0], index[1]]*1000))
        params_k = list(d["koefs"])
        d["chosen_fits"].append(point_model)
        params_k.append(1.2*np.abs(observations[index[0], index[1]] - d["fin_fits"][index[0], index[1]])/point_model[index[0], index[1]])#options={'maxinter':10000,'xtol':1e-8, 'disp':True, 'stepmx':1.0}, )#'SLSQP',options={'disp':True, 'maxiter':10000000, 'eps':1e-8, 'ftol':1e-8})
        print(bnds)
        print(params_k)
        results_o = optimize.minimize(f, params_k, [observations,d["chosen_fits"]], method='SLSQP',bounds=bnds, options={'eps':1e0, 'ftol':10})
        d["koefs"] = results_o.x
        print(results_o)

        new_likelihood = results_o.fun
        print(d["koefs"])
        print(d["chosen_fits"])
        d["fin_fits"] = np.sum([d["chosen_fits"][i]*d["koefs"][i] for i in range(len(d["chosen_fits"]))], axis=0)

        #print(np.sum(fin_fits))        
        print("new_likelihhodd=" + str(new_likelihood))
        print("likelihhodd=" + str(likelihood0))
        #print(koefs)
        print(bnds)
        #print(indexes)
        #print(values)
        print(observations[index[0], index[1]] - d["fin_fits"][index[0], index[1]])
        x0, y0 = w.all_world2pix(ra, dec, 1)
        print(x0)  
        #print(y0)
        #plt.imshow(observations-d["fin_fits"])
        #plt.show()
        print("-------------------")
    save_obj(d, d["file_to_write"] )        
    #plt.imshow(observations-d["fin_fits"])
    #plt.show()
    return model

def generate_linear_koefs():
    r = random.random()
    points_k = 1 + random.random() * 9
    a = (r < 0.1)*points_k*0.1 + (r>=0.1)*points_k
    b = 0.3 + (3-0.3)*random.random()
    return a, b

def get_center(string):
    ra = string.find("ra")
    dec = string.find("dec")
    end = string[dec:].find("_")+dec
    center = [float(string[ra+2:dec-1]), float(string[dec+3:end])]
    return center

def get_original_points(xml_file, w):
    f_points = open(xml_file)

    lines = f_points.readlines()
    ra = []
    dec = []
    koefs = []
    k = 0
    for i in lines:
        if (i.find("degrees away from ROI center -->") >= 0.0):
            k = k + 1
        if (i.find('max="1e4" min="1e-4" name="Prefactor"') >= 0.0):
            value_str = i.split(" ")[-1]
            first = value_str.find('"')+1
            second = first + 10#value_str.rfind('"')+1
            value = float(value_str[first:second])                
            value_scale = i.split(" ")[-2]
            value_s = float(value_scale[value_scale.find('"')+1 : value_scale.rfind('"')])
            koef = value_s/(1e-10)
            koefs.append(koef)
        if (i.find('<parameter free="0" max="360.0" min="-360.0" name="RA"') >= 0.0):
            ra_coord = i[i.find("value")+7:]
            ra0 = float(ra_coord[:ra_coord.find('.')+3])
            ra.append(ra0)
        if (i.find('<parameter free="0" max="90" min="-90" name="DEC"') >= 0.0):
            dec_coord = i[i.find("value")+7:]
            try:
                dec0 = float(dec_coord[:dec_coord.find('.')+3]) 
            except:
                dec0 = float(dec_coord[:dec_coord.find('.')+2]) 
            dec.append(dec0)
    #ra, dec = w.all_pix2world([50, 100], [0, 50], 1)
    try:  
        xy = w.all_world2pix(ra, dec, 1, maxiter=20,
                       tolerance=1.0e-4, adaptive=False,
                       detect_divergence=False,
                       quiet=False)
    except wcs.wcs.NoConvergence as e:
        print("Indices of diverging points: {0}"
            .format(e.divergent))
        print("Indices of poorly converging points: {0}"
            .format(e.slow_conv))
        print("Best solution:\n{0}".format(e.best_solution))
        print("Achieved accuracy:\n{0}".format(e.accuracy))
        #xy = w.all_world2pix(ra[1], dec[1], 1)
    x, y = xy
    return x, y, koefs

def extract_names(string_points, string_back, a, b):
    string=string_back
    option_position = string.find("back")
    points_option_position = string_points.find("points")
    string[option_position:]
    d = {}
    d.update({"fname_points" : string_points[:points_option_position] + "points_model_map.fits"})
    d.update({"fname_back" : string[:option_position] + "back_model_map.fits"})
    d.update({"fname_xml" : string_points[:points_option_position] + "test_src_model_const.xml"})    
    d.update({"final_fits_folder" : string[:string.rfind("/")+1]})
    d.update({"sky_number" : string[string.rfind("/")+1:][:string[string.rfind("/")+1:].find("_")]})
    d.update({"sky_number_points" : string_points[string_points.rfind("/")+1:][:string_points[string_points.rfind("/")+1:].find("_")]})
    d.update({"final_fits" : d["final_fits_folder"] + d["sky_number"] + ".fits"})
    d.update({"center" : get_center(string)})
    d.update({"a" : a})
    d.update({"b" : b})
    d.update({"original": a*openAsImage(d["fname_points"]) + b*openAsImage(d["fname_back"])})
    d.update({"observations": makeNoisy(d["original"])})
    d['sources'] = []
    d['koefs'] = []
    d['indexes'] = []
    d["chosen_fits"] = []
    hdulist = fits.open(d["fname_points"])
    d["w"] = wcs.WCS(hdulist[0].header, hdulist)
    d["fin_fits"] = np.zeros(d["original"].shape)
    d["src_points"] =  string_points[:points_option_position] + "points_src_model_const.xml"
    x, y, koefs = get_original_points(d["src_points"], d["w"])
    d["original_coords"] = [[y[i], x[i]] for i in range(len(x))]
    d["original_brightness"] = koefs
    d["file_to_write"] =  "combination_" + str(a)+"_"+ d["sky_number_points"] +"_" +str(b)+"_"+ d["sky_number"]
    d["f"] = "likelihood**2"
    d["int_cr"] = 4
    #x, y, koefs = 
    #observations = (makeNoisy(obs))
    return d

def extract_names_test(string, original=True):
    option_position = string.find("_model")
    string_points = string
    points_option_position = option_position
    d = {}
    d['test'] = True
    d.update({"fname_points" : string_points[:points_option_position] + "_model_map.fits"})
    d.update({"fname_back" : string[:option_position] + "_model_map.fits"})
    d.update({"fname_xml" : string_points[:option_position] + "_test_src_model_const.xml"})    
    d.update({"final_fits_folder" : string[:string.rfind("/")+1]})
    d.update({"sky_number" : string[string.rfind("/")+1:][:string[string.rfind("/")+1:].find("_")]})
    d.update({"sky_number_points" : string_points[string_points.rfind("/")+1:][:string_points[string_points.rfind("/")+1:].find("_")]})
    d.update({"final_fits" : d["final_fits_folder"] + d["sky_number"] + ".fits"})
    d.update({"center" : get_center(string)})
    if original:
        d["original"] = openAsImage(string[:-6])
    else:
        d["original"] = None
    d.update({"observations": openAsImage(string)})
    d['sources'] = []
    d['koefs'] = []
    d['indexes'] = []
    d["chosen_fits"] = []
    hdulist = fits.open(d["fname_points"])
    d["w"] = wcs.WCS(hdulist[0].header, hdulist)
    d["fin_fits"] = np.zeros(d["observations"].shape)
    d["src_points"] =  string_points[:points_option_position] + "_src_model_const.xml"
    x, y, koefs = get_original_points(d["src_points"], d["w"])
    d["original_coords"] = [[y[i], x[i]] for i in range(len(x))]
    d["original_brightness"] = koefs
    d["file_to_write"] =  "test_result_" + d["sky_number_points"]
    d["f"] = "likelihood**2"
    d["int_cr"] = 4
    #x, y, koefs = 
    #observations = (makeNoisy(obs))
    return d

files = (os.listdir("/media/mariia/Maxtor/database5/"))

points_models = []
back_models = []
for i in files:
    if i.find("points_model_map") != -1 and i.find("noisy") == -1 :
        points_models.append("/media/mariia/Maxtor/database5/"+i)
    if i.find("back_model_map") != -1 and i.find("noisy") == -1 :
        back_models.append("/media/mariia/Maxtor/database5/" + i)

#f = open("test2.fits", "w")
#f.write(point_model)
#f.close()
def test(fname_folder, ending):
    for i in os.listdir(fname_folder):
        if i[-len(ending):]==ending and i.find("real") == -1:
            d = extract_names_test(fname_folder+i)
            run_prog(d, True)

test(fname_folder="/media/mariia/Maxtor/database2/", ending="_model_map.fits.noisy")
#string = "/media/mariia/Maxtor/database2/sky4_coord_ra43.8245564807_dec77.6699953654_model_map.fits.noisy"
# = extract_names_test(string)
#run_prog(d, True)
#print(d)

"""for i in range(50):
    a, b = generate_linear_koefs()
    string_points = random.choice(points_models)
    string_back =random.choice(back_models)
    d = extract_names(string_points, string_back, a, b)
    run_prog(d)
#"""

