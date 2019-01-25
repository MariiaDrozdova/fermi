import numpy
from matplotlib import pyplot as plt
from ds9_cmap import *


fname = 'num_excl_mkn501.dat' #input data file
columns = [0,1,3] #columns to use in input file

fout = 'test.pdf' #file to store the output figure, will be overwritten


############## MAIN ###########################
x , y, color = numpy.loadtxt( fname,usecols=[0,1,3], unpack=True ) #read data

lenx = len( numpy.unique(x) ) #lengths of x,y axes
leny = len( numpy.unique(y) )

grid = color.reshape( (lenx, leny) ).T #reshape colors to rectangle instead of 1d array. Transpose is magic

cmap = plt.get_cmap('ds9b') # get ds9-equal palette

#actual plotting
plt.imshow(grid, extent=(x.min(), x.max(), y.min(), y.max()),
           origin='lower',vmin=-11.5, cmap=cmap, aspect=0.5) #may want to add interpolation='nearest'



#some plot settings
plt.xscale('log')
plt.yscale('log')
plt.xlabel('m ,eV')
plt.ylabel('g, GeV$^{-1}$')


plt.colorbar(shrink=0.7,label='$\Delta \chi^2$')
plt.savefig(fout)

plt.show()