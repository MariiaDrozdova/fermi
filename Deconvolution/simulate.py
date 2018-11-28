import os
import random
import math
#import cv2 as cv2
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from ds9_cmap import *
from make_fermi_maps import *

def makeNoisy(model_map_name):
    os.system('cp '+model_map_name+' '+model_map_name+'.noisy')

    with pyfits.open(model_map_name+'.noisy','update') as f:
        print(f[0].data)
        f[0].data = numpy.random.poisson( lam=f[0].data)
        print(f[0].data)

def simulateObjects(sim=1):
    ra =random.random() * 360
    dec = random.random() * 180 - 90
    real_xml = simulateMaps(ra=ra, dec=dec, sim=sim, xml=1)
    if (real_xml):
        k = 0
        f_real = open(real_xml)
        lines = f_real.readlines()
        min_v = 1e4
        max_v = 1e-4
        max_scale = 0
        max_index= 0
        for i in lines:
            #print(i.find("degrees away from ROI center -->") >= 0.0)
            if (i.find("degrees away from ROI center -->") >= 0.0):
                #print(i)
                k = k + 1
            if (i.find('max="1e4" min="1e-4" name="Prefactor"') >= 0.0):
                value_str = i.split(" ")[-1]
                first = value_str.find('"')+1
                second = first + 15#value_str.rfind('"')+1
                value = float(value_str[first:second])                
                value_scale = i.split(" ")[-2]
                value_s = float(value_scale[value_scale.find('"')+1 : value_scale.rfind('"')])
                koef = value_s/(1e-13)
                print(value_s)
                if value*koef < min_v:
                    min_v = value*koef
                if value*koef > max_v:
                    max_v = value*koef
                    
            if (i.find('name="Scale" scale=') >= 0.0):
                value_scale_str = i.split(" ")[-1]
                value_scale = float(value_scale_str[value_scale_str.find('"')+1 : value_scale_str.rfind('"')])              

                if max_scale < value_scale:
                    max_scale = value_scale


            if (i.find(' name="Index" scale="') >= 0.0):
                value_scale_str = i.split(" ")[-1]
                value_scale = float(value_scale_str[value_scale_str.find('"')+1 : value_scale_str.rfind('"')])              

                if max_index < value_scale:
                    max_index = value_scale
    print(min_v)
    print(max_v)
    #print(k)
    i_range = random.randint(a=-2, b=2)
    res = '''<?xml version="1.0" ?>
        <source_library title="source library">

        <!-- Point Sources -->

        <!-- Sources between [0.0,3.4142135624) degrees of ROI center -->
        '''
    if (k+i_range > 0):
        k = k + i_range
    for i in range(k):
        r = random.random()#(np.exp(3.0*np.log(r)))
        r2 = random.random()#(np.exp(3.0*np.log(r)))
        val1 = np.exp(5.0*np.log(r2))*(max_v*1.15-min_v*0.1) + min_v*0.1
        val2 = random.random() * (max_index-0) + 0
        val3 = np.exp(5.0*np.log(r)) * (max_scale*1.15-30) + 30
        val4 = random.random() * (9.98-0)-4.99 + ra
        if val4>360:
            val4 = val4-360
        if val4<0:
            val4 = val4+360
        val5 = random.random() * (9.98-0)-4.99 + dec
        if val5>90:
            val5 = val5-90
        if val5 < -90:
            val5 = val5 + 90
        sc_val = len(str(int(math.floor(val1))))-1
        sc_val = 0
        val1 = val1*1e-13
        while not math.floor(val1):
            sc_val = sc_val + 1
            val1 = val1 * 10
        print(val1)
        print(sc_val)
        s = '''<source ROI_Center_Distance="8.583" name="obj_''' + str(i) + '''" type="PointSource">
            <spectrum apply_edisp="false" type="PowerLaw">
            <!-- Source is 8.582976793871993 degrees away from ROI center -->
            <!-- Source is outside ROI, all parameters should remain fixed -->
                <parameter free="0" max="1e4" min="1e-4" name="Prefactor" scale="1e-''' + str(sc_val) + '''" value="'''+str(val1)+'''"/>
                <parameter free="0" max="10.0" min="0.0" name="Index" scale="-1.0" value="'''+str(val2)+'''"/>
                <parameter free="0" max="5e5" min="30" name="Scale" scale="1.0" value="'''+str(val3)+'''"/>
            </spectrum>
            <spatialModel type="SkyDirFunction">
                <parameter free="0" max="360.0" min="-360.0" name="RA" scale="1.0" value="'''+str(val4)+'''"/>
                <parameter free="0" max="90" min="-90" name="DEC" scale="1.0" value="'''+str(val5)+'''"/>
            </spatialModel>
        </source>\n'''
        res = res + s
    end = '''
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
        </source_library>
        '''
    res = res + end
    xml_name = 'database/sky' + str(sim) + '_coord_ra' + str(ra)+'_dec' + str(dec) + '_src_model_const.xml'
    f = open(xml_name, 'w')
    f.write(res)
    f.close()
    simulateMaps(ra=ra, dec=dec, sim=sim, xml=0)
    makeNoisy('database/sky' + str(sim) + '_coord_ra' + str(ra)+'_dec' + str(dec) + '_model_map.fits')

for i in range(200, 1000):
	print(i)
	simulateObjects(i)
