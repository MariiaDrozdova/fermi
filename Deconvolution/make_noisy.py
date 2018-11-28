from astropy.io import fits as pyfits
import numpy
import os,sys

def makeNoisy(model_map_name):
    os.system('cp '+model_map_name+' '+model_map_name+'.noisy')

    with pyfits.open(model_map_name+'.noisy','update') as f:
        print(model_map_name)
        PEAK=1#np.mean(f[0].data)*0.00001
        f[0].data = numpy.random.poisson(f[0].data)

names = []
s = '/media/masha/Maxtor/database/'
for i in os.listdir(s):
    print(i)
    if (i[-15:] == "_model_map.fits"):
        names.append(s + i)
names.sort()

for i in names:
    print(i)
    makeNoisy(i)

