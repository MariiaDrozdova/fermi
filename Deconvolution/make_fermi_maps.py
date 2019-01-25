#updated 05/09/2018 with a new gtifilter

from astropy.io import fits as pyfits
import numpy

import gt_apps as gt
from make3FGLxml import *

import re,os,sys
import glob

def simulateMaps(ra=193.98, dec=-5.82, side=10, sim=1, xml=1, flag = ''):
	#parameters to build map ; please change these for a new region on the sky
	#ra = 193.98 #coordinates of the source in J2000
	#dec = -5.82
	side = 10. #side of map , deg
	pixsize = 0.05 #deg size of the pixel

	prefix = 'sky' + str(sim) + '_coord_ra' + str(ra)+'_dec' + str(dec) + flag # all output files will start from prefix_
	prefold  = "/media/mariia/Maxtor/"


	emin = 1000 #minimal energy ; MeV
	emax = 10000 #maximal energy ; MeV



	### pleasem specify where fermi data is located/media/masha/Maxtor/
	data_path = 'lat_data/'#'/home/tu/tu_tu/tu_pside01/FERMI/Data/Variable/All/' #path to Fermi/LAT data
	catalogue = 'bright100mevsort.fit' # fermi catlogue file to use ;



	tmin = -1 # specify start and stop times for lightcurve here (in fermi-seconds)
	tmax = -1 # negative -- will use all available 

	gtifilter = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
	#gtifilter = '(DATA_QUAL==1) && ABS(ROCK_ANGLE)<90 && (LAT_CONFIG==1) && (angsep(RA_ZENITH,DEC_ZENITH,{:.3f},{:.3f})+1<105) && (angsep({:.3f},{:.3f},RA_SUN,DEC_SUN)>5+1) &&(angsep({:.3f}, {:.3f} ,RA_SCZ,DEC_SCZ)<180)'.format(ra,dec,ra,dec, ra, dec)

	irfs='P8R2_CLEAN_V6'
	zmax = 90
	database = 'database5/'

	#some file names which will be used through the code
	gti = prefold + 'others/' + prefix + '_events.fits' # event-file name
	sky_map = prefold + database +prefix+'_real_map.fits' # sky map
	expcube = prefold + 'others/' + prefix+'_expCube.fits'
	bexpmap = prefold + 'others/' + prefix+'_BexpMap.fits'
	ccubemap = prefold + 'others/' + prefix+'_ccube3D.fits'
	srcmap = prefold + 'others/' + prefix+'_SrcMap.fits'
	srcmdl = prefold + database + prefix+'_src_model_const.xml' # xml version of the  model map
	model_map = prefold + database+ prefix+'_model_map.fits' # model map
	deconv_map = prefold + database + prefix + '_deconvolved_map.fits' # "ideal deconvolution" map
	if (xml):
		srcmdl = prefold + database + prefix+'_real_src_model_const.xml' # xml version of the  model map
		model_map = prefold + database+ prefix+'_real_model_map.fits' # model map
		deconv_map =prefold + database + prefix + '_real_deconvolved_map.fits' # "ideal deconvolution" map

	tmp_gti = prefold + 'others/' + prefix+'_events.tmp' #temp event-file name
	tmp_evlist = prefold + 'others/' + prefix+'_evlist.tmp' #temp




	# get the data files names
	event_files = glob.glob(data_path+'/lat_photon*.fits') #all fermi-lat event files
	scfile = glob.glob(data_path+'/L*SC*.fits')[0] #fermi-lat spacecraft file

	#get tmin and tmax if not given
	with pyfits.open(scfile) as f:
		if(tmin<0):
			tmin = int( f[1].header['TSTART'] )
		if(tmax<0):
			tmax = int( f[1].header['TSTOP'] ) + 1



	#find event class for IRFs
	evc=-1
	if(re.findall(r'SOURCE',irfs)):evc=128
	if(re.findall(r'CLEAN',irfs)):evc=256
	if(re.findall(r'ULTRACLEAN',irfs)):evc=512
	if(re.findall(r'ULTRACLEANVETO',irfs)):evc=1024
	if(evc<0):
		print('Can not recognize irf '+irfs)
		sys.exit(0)

	convt=3
	if(re.findall(r'FRONT',irfs)):convt=1
	if(re.findall(r'BACK',irfs)):convt=2
	if(re.findall(r'PSF',irfs)):
		s1 = re.findall(r'PSF[0-9]+',irfs)[0]
		s1 = re.sub(r'PSF','',s1)
		convt = 4*2**int(s1)
	if(re.findall(r'EDISP',irfs)):
		s1 = re.findall(r'EDISP[0-9]+',irfs)[0]
		s1 = re.sub(r'EDISP','',s1)
		convt = 64*2**int(s1)


	#define selection radius from side
	roi = side * 0.5 * 2**0.5 # radius of circum-circle
	nxpix = int(side / pixsize ) #number of pixels
	nypix = nxpix # the map is a square nxpix*nypix pixels


	#selecting events
	print(gti)
	#if( not os.path.exists(gti) ): #if we do not have gti filtered file -> create it
        if True:
		numpy.savetxt( tmp_evlist ,   event_files, fmt='%s' )
		#gtselect
		print('GTSELECT started!')
		gt.filter['evclass'] = evc
		gt.filter['evtype']=convt
		gt.filter['ra'] = ra
		gt.filter['dec'] = dec
		gt.filter['rad'] = roi
		gt.filter['emin'] = emin
		gt.filter['emax'] = emax
		gt.filter['zmax'] = zmax
		gt.filter['tmin'] = tmin
		gt.filter['tmax'] = tmax
		gt.filter['infile'] = '@'+tmp_evlist
		gt.filter['outfile'] = tmp_gti
		print(evc)
		print(convt)
		print(gt.filter['infile'])
		print(gt.filter['outfile'])
		print(gt.filter)
		gt.filter.run() #run GTSELECT
		print('GTSELCT finished!')
		##############################
		print('GTMKTIME start')
		gt.maketime['scfile'] = scfile
		gt.maketime['filter'] = gtifilter
		gt.maketime['roicut'] = 'no'
		gt.maketime['evfile'] = tmp_gti
		gt.maketime['outfile'] = gti
		gt.maketime.run()
		
		try:
		        os.remove(tmp_gti)
		        os.remove(tmp_evlist)
		except:
			pass
		print('done!')
	else:
		print(gti+' file exist!')

	# building a sky map
	#if( not os.path.exists(sky_map) ):
        if True:
		print('Building sky map')
		gt.evtbin['evfile'] = gti
		gt.evtbin['scfile'] = scfile
		gt.evtbin['outfile'] = sky_map
		gt.evtbin['algorithm'] = 'CMAP'
		gt.evtbin['nxpix'] = nxpix
		gt.evtbin['nypix'] = nypix
		gt.evtbin['binsz'] = pixsize
		gt.evtbin['coordsys'] = 'CEL'
		gt.evtbin['xref'] = ra
		gt.evtbin['yref'] = dec
		gt.evtbin['axisrot'] = 0
		gt.evtbin['proj'] = 'CAR'
		gt.evtbin.run()
		print('done!')
	else:
		print(sky_map+' file exists!')

	#creating xml model of the region
	if( not os.path.exists( srcmdl ) ):
		mymodel = srcList(catalogue, gti , srcmdl)#
		try:
			print("ph")
			os.system( 'ln -s ${FERMI_DIR}/refdata/fermi/galdiffuse/gll_iem_v06.fits glprp.fits' )
			os.system( 'ln -s ${FERMI_DIR}/refdata/fermi/galdiffuse/iso_'+irfs+'_v06.txt .' ) #e.g. P8R2_CLEAN_V6
		except:
			print('Failed to create links. Files glprp.fits and iso_'+irfs+'_v06.txt has already been linked?')
			pass

		mymodel.makeModel('glprp.fits', 'gll_iem_v06', 'iso_'+irfs+'_v06.txt', 'iso_'+irfs+'_v06', sigFree=0)
	else:
		print('XML sky model '+ srcmdl+' exists!')

	#if(not os.path.exists(ccubemap)):
	if True:
		print('Creating CCube MAP')
		gt.evtbin['evfile'] = gti
		gt.evtbin['scfile'] = scfile
		gt.evtbin['outfile'] = ccubemap
		gt.evtbin['algorithm'] = 'CCUBE'
		gt.evtbin['nxpix'] = nxpix
		gt.evtbin['nypix'] = nypix
		gt.evtbin['binsz'] = pixsize
		gt.evtbin['coordsys'] = 'CEL'
		gt.evtbin['xref'] = ra
		gt.evtbin['yref'] = dec
		gt.evtbin['axisrot'] = 0
		gt.evtbin['proj'] = 'CAR'
		gt.evtbin['emin'] = emin
		gt.evtbin['emax'] = emax
		gt.evtbin['enumbins'] = 5
		gt.evtbin['ebinalg'] = 'LOG'
		gt.evtbin.run()
		print('done!')

	else: print('CCUBE 3D map '+ccubemap+' exists!')


	#Exposure cube create
	#if(not os.path.exists(expcube)):
	if True:
		print('Building Exposure cube ' + expcube +' ; this can take a while...')
		gt.expCube['evfile'] = gti
		gt.expCube['scfile'] = scfile
		gt.expCube['outfile'] = expcube
		gt.expCube['dcostheta'] = 0.025
		gt.expCube['binsz'] = 1.0
		gt.expCube['zmax'] = zmax
		gt.expCube.run() #run expCube
		print('done!')
	else: print('Exposure Cube '+expcube+' exists!')



	#if(not os.path.exists(bexpmap)):
	if True:
		print('Building binned exposure '+ bexpmap)
		gtexp = gt.GtApp('gtexpcube2')
		gtexp['infile']	= expcube
		gtexp['cmap'] = 'none'
		gtexp['outfile'] = bexpmap
		gtexp['irfs'] = irfs
		gtexp['nxpix'] = int(nxpix+40/pixsize)
		gtexp['nypix'] = int(nypix+40/pixsize)
		gtexp['binsz'] = pixsize
		gtexp['coordsys'] = 'CEL'
		gtexp['xref'] = ra
		gtexp['yref'] = dec
		gtexp['axisrot'] = 0.
		gtexp['proj'] = 'CAR'
		gtexp['ebinalg'] = 'LOG'
		gtexp['emin'] = emin
		gtexp['emax'] = emax
		gtexp['enumbins'] = 10
		gtexp['evtype'] = convt 
		gtexp.run()
		print('done!')

	else: print('Binned exposure map exists:'+bexpmap)



	#Exposure map create
	#if (not os.path.exists(srcmap)):
	if True:
		print('Calcualting Diffuse responses')
		gt.srcMaps['scfile'] =scfile
		gt.srcMaps['expcube'] =expcube
		gt.srcMaps['cmap'] =ccubemap
		print(scfile)
		gt.srcMaps['irfs'] = irfs
		gt.srcMaps['srcmdl'] =srcmdl
		gt.srcMaps['bexpmap'] =bexpmap
		gt.srcMaps['outfile'] =srcmap
		gt.srcMaps['minbinsz'] = 2*pixsize
		gt.srcMaps['evtype'] = convt
		gt.srcMaps['ptsrc'] = 'no'
		gt.srcMaps.run() #run expMap
		print('done!')

	else: print('SrcMap '+srcmap+' exists!')



	#if( not os.path.exists(model_map) ):
	if True:
		print('Building model map...')
		gtmdl = gt.GtApp('gtmodel')
		gtmdl['srcmaps'] = srcmap
		gtmdl['srcmdl'] = srcmdl
		gtmdl['outfile'] = model_map
		gtmdl['irfs'] = 'CALDB'
		gtmdl['expcube'] = expcube
		gtmdl['bexpmap'] = bexpmap
		gtmdl['convol'] = True
		print(srcmap)
		print(srcmdl)
		print(model_map)
		print(expcube)
		print(bexpmap)
		gtmdl.run()
		print('done!')

	else: print('Model map exists:'+model_map)


	#if( not os.path.exists( deconv_map ) ):
	if True:
		print('Building deconvoluted map...')
		gtmdl1 = gt.GtApp('gtmodel')
		gtmdl1['srcmaps'] = srcmap
		gtmdl1['srcmdl'] = srcmdl
		gtmdl1['outfile'] = deconv_map
		gtmdl1['irfs'] = 'CALDB'
		gtmdl1['expcube'] = expcube
		gtmdl1['bexpmap'] = bexpmap
		gtmdl1['convol'] = False #not to convolve with PSF. This should result in deconvolved map?
		gtmdl1.run()
		print('done!')

	else: print('Deconvoluted map exists:'+ deconv_map)

	if (xml):
		return srcmdl



