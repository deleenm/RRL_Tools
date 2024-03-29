#!/usr/bin/env python
'''
A period searching python wrapper for vartools

lcurve.py is a python wrapper for the vartools program and provides basic plotting infrastructure.

@package lcurve
@author deleenm
@version \e \$Revision$
@date \e \$Date$

Usage: lcurve.py
'''

# -----------------------------
# Standard library dependencies
# -----------------------------
import argparse
import os
import sys
import time
from subprocess import Popen
# -------------------
# Third-party imports
# -------------------
import numpy as np
from astropy.table import Table
from astropy.timeseries import LombScargle
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# -----------------
# Class Definitions
# -----------------

# --------------------
# Function Definitions
# --------------------
def plot_lc(curve,name,pp,clean=10,flux=False,tdict=False,ret_results=False,stype='ls',log=False
            ,manual=None,min=0.1, max=10.0):
    '''
    Plot up a phased light curve based on a raw light curve adds plot to pdf object
    
    Arguments:
        curve: A light curve file (date, mag, err)
        name: The Star Name.
        pp: PDF Object
    Keywords:
        clean: How many maximum and minimum points to remove
        flux: Use Flux units instead of Magnitude units
        log: Use a Log 10 days x-axis on the Periodogram
        manual: Use a manual period to plot light curve instead of searching for one.
        max: Maximum Period to search
        min: Minimum Period to search
        ret_results: If True, return the results of the period search
        tdict: A Dictionary of info to put in the title of the plots
        stype: Type of periodogram finder (ls or aov).
    Returns:
        None unless ret_results is True, then returns a tuple of results (Filename,Starname,Stype,Period1,Period2)
    '''

    retval = 0
    #Default String
    titlestring = "Name: {}".format(name)   
    if tdict != False and len(tdict) > 0:
        for key in tdict:
            titlestring = titlestring+" {}: {}".format(key,tdict[key])

    #Open file        
    try:
        date,mag,err = np.genfromtxt(curve,unpack=True)
    except:
        try:
            date,mag,err,phase = np.genfromtxt(curve,unpack=True)
        except:
            print("Cannot open: {}".format(curve) )
            return(1)

    #Clean up the data
    if(clean != 0):
        clean_idx = np.argsort(mag)
        clean_idx = clean_idx[clean:]
        clean_idx = clean_idx[:-clean]
        date = date[clean_idx]
        mag = mag[clean_idx]
        err = err[clean_idx]

    #Remove all points at date = 0
    good_date_idx = (date != 0)
    date = date[good_date_idx]
    mag = mag[good_date_idx]
    err = err[good_date_idx]
    
    #Remove all nan dates
    good_date_idx = (np.isfinite(date))
    date = date[good_date_idx]
    mag = mag[good_date_idx]
    err = err[good_date_idx]

    #Set the point size depeding on the number of data points
    if len(date) < 100:
        myptsize = 3
        myformat = 'o'
    else:
        myptsize = 1
        myformat = '.'   
    
    if len(date) == 0:
        plt.figure(figsize=(8.5,11))
        plt.title(titlestring)
        plt.plot(date,mag,myformat,ms=myptsize,c="blue",rasterized=True)
        plt.text(0,0,"No Lightcurves")
        plt.xlabel("Date")
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        if flux:
            plt.ylabel("Flux")
        else:
            plt.ylabel("Magnitude")
            plt.gca().invert_yaxis() 
        
        
        pp.savefig(dpi=200)
        plt.clf()
        print("No good data points: {}".format(curve) )
        return(1)
    
    #Write Out Temporary light curve
    tmpcur = open("../temp_lc.cur",'w')
    for i in range(len(date)):
        tmpcur.write("{} {} {}\n".format(date[i],mag[i],err[i]))
    tmpcur.close()
    
    #Determine Period
    if(stype == 'ls'):
        ttype = "Lomb-Scargle"
        #should rewrite to communicate directly
        lstool = Popen(["vartools -i ../temp_lc.cur -ascii -redirectstats ../ls.stat -header " + 
                        "-LS {} {} 0.1 2 1 ../ whiten clip 5. 1".format(min,max)],shell=True)
        lstool.wait()
        
        try:
            lsstat = Table.read("../ls.stat",format='ascii')
        except:
            lsstat = Table()
            lsstat['LS_Period_1_0'] = [1]
            lsstat['LS_Period_2_0'] = [1]
            print("Can't read ls.stat: {}".format(curve) )
            retval = 1
        
        try:
            lsperiod = Table.read("../temp_lc.cur.ls",format='ascii')
        except:
            lsperiod = Table()
            lsperiod['col1'] = []
            lsperiod['col2'] = []
            lsperiod['col3'] = []
            print("Can't read Periodogram: {}".format(curve) )
            retval = 1
            
        fperiod1 = (lsstat['LS_Period_1_0'])[0]
        fperiod2 = (lsstat['LS_Period_2_0'])[0]
    
    elif(stype == 'aov'):
        ttype = 'AOV'
        aovtool = Popen(["vartools -i ../temp_lc.cur -ascii -redirectstats ../aov.stat -header -aov Nbin 20 " +
                         "{} {} 0.1 0.01 2 1 ../ whiten clip 5. 1".format(min,max)],shell=True)
        aovtool.wait()
    
        try:
            lsstat = Table.read("../aov.stat",format='ascii')
        except:
            lsstat = Table()
            lsstat['Period_1_0'] = [1]
            lsstat['Period_2_0'] = [1]
            print("Can't read aov.stat: {}".format(curve) )
            retval = 1
        
        try:
            aovperiod = Table.read("../temp_lc.cur.aov",format='ascii')
                        
        except:
            aovperiod = Table()
            aovperiod['Period'] = []
            aovperiod['AOV_WhitenCycle_0'] = []
            aovperiod['AOV_WhitenCycle_1'] = []

            print("Can't read Periodogram: {}".format(curve) )
            retval = 1

        fperiod1 = (lsstat['Period_1_0'])[0]
        fperiod2 = (lsstat['Period_2_0'])[0]   
        
        
    elif(stype == 'fchi2'):
        ttype = 'FChi2'
        fchi2tool = Popen(["vartools -L /usr/local/share/vartools/USERLIBS/fastchi2.la -i ../temp_lc.cur " +
                         "-ascii -redirectstats ../fchi2.stat -header -fastchi2 " +
                         "Nharm fix 3 freqmax fix {} freqmin fix {} ".format(1/min, 1/max) +
                         "oversample fix 10 Npeak 2 oper ../"],shell=True)
        fchi2tool.wait()
    
        try:
            lsstat = Table.read("../fchi2.stat",format='ascii')
        except:
            lsstat = Table()
            lsstat['Period_1_0'] = [1]
            lsstat['Period_2_0'] = [1]
            print("Can't read fchi2.stat: {}".format(curve) )
            retval = 1
        
        try:
            fchi2period = Table.read("../temp_lc.cur.fastchi2_per",format='ascii')
                        
        except:
            fchi2period = Table()
            fchi2period['Frequency'] = []
            fchi2period['Chireduction'] = []
            
            print("Can't read Periodogram: {}".format(curve) )
            retval = 1

        fperiod1 = 1.0/(lsstat['Fastchi2_Frequency_1_0'])[0]
        fperiod2 = 1.0/(lsstat['Fastchi2_Frequency_2_0'])[0]   

    elif(stype == 'apyls'):
        ttype = 'Apy LS'
        
        apyls_freq,apyls_power = LombScargle(date, mag, err,nterms=1).autopower(
                                    minimum_frequency=1.0/max, maximum_frequency=1.0/min,nyquist_factor=10)
        
        frequency1 = apyls_freq[np.argmax(apyls_power)]
        fperiod1 = 1/frequency1
        #Search 
        fperiod2 = fperiod1/2 #Placeholder

    elif(stype == 'manual'):
        ttype = 'Manual'
        fperiod1 = manual
        fperiod2 = manual  
    
    else:
        print("Period Search type: {} is not a valid type.".format(stype))

    #Generate Phases for each date
    lsphase1 = date/fperiod1
    lsphase1 = (lsphase1 - np.floor(lsphase1))
    lsphase2 = date/fperiod2
    lsphase2 = (lsphase2 - np.floor(lsphase2))    
    
    #Write out a phase file (date, mag, err, phase)
    if os.path.exists(name+'.phase1'):
        os.remove(name+'.phase1')
    
    phasefile = open(name+'.phase1','w')
    for i in range(len(date)):
        phasefile.write("{} {} {} {:.5f}\n".format(date[i],mag[i],err[i],lsphase1[i]))
    phasefile.close()
    
    if os.path.exists(name+'.phase2'):
        os.remove(name+'.phase2')
        
    phasefile = open(name+'.phase2','w')
    for i in range(len(date)):
        phasefile.write("{} {} {} {:.5f}\n".format(date[i],mag[i],err[i],lsphase2[i]))
    phasefile.close()    
    
    lsphase1 = np.concatenate((lsphase1,lsphase1+1))
    lsphase2 = np.concatenate((lsphase2,lsphase2+1))
    

    nmag = np.concatenate((mag,mag))

    #Open Logfile
    try:
        filename = name + ".log"
        logfile = open(filename,'w')
    except IOError:
        print("{} could not be opened!".format(filename))

    #Write out header
    header = "#Filename,Starname,Stype,Min_Period,Max_Period,Period1,Period2\n"
    
    logfile.write(header)


    #Make plot
    plt.rcParams['mathtext.default'] = 'regular' 
    
    f,axarr = plt.subplots(2,2,figsize=(8.5,11))
    f.subplots_adjust(hspace=0.3,top=0.94,bottom=0.05)
    
    #Plot in MJD 
    if np.median(date) > 2400000:
        pdate = date - 2400000.5
    else:
        pdate = date
    plt.suptitle(titlestring)
    axarr[0,0].set_title("Data")
    axarr[0,0].plot(pdate,mag,myformat,ms=myptsize,c="blue",rasterized=True)
    axarr[0,0].set_xlabel("MJD Date")
    if flux:
        axarr[0,0].set_ylabel("Flux")
    else:
        axarr[0,0].set_ylabel("Magnitude")
        axarr[0,0].invert_yaxis() 

    #Periodogram
    axarr[0,1].set_title("Periodogram")
    if(stype == 'ls'):
        #Set power less than 0 t0 0.
        periodogram = lsperiod['col2']
        periodogram[periodogram < 0] = 0
        axarr[0,1].plot(1.0/lsperiod['col1'],periodogram,c="red",rasterized=True)
        axarr[0,1].set_ylabel("LS Power")
    elif(stype == 'aov'):
        #Set power less than 0 t0 0.
        periodogram = aovperiod['AOV_WhitenCycle_0']
        periodogram[periodogram < 0] = 0
        axarr[0,1].plot(aovperiod['Period'],periodogram,c="red",rasterized=True)
        axarr[0,1].set_ylabel("AOV Power")  
    elif(stype == 'fchi2'):
        #Set power less than 0 t0 0.
        periodogram = fchi2period['Chireduction']
        periodogram[periodogram < 0] = 0
        axarr[0,1].plot(1.0/fchi2period['Frequency'],periodogram,c="red",rasterized=True)
        axarr[0,1].set_ylabel("Chi Power")
    elif(stype == 'apyls'):
        #Set power less than 0 t0 0.
        periodogram = apyls_power
        periodogram[periodogram < 0] = 0
        axarr[0,1].plot(1.0/apyls_freq,periodogram,c="red",rasterized=True)
        axarr[0,1].set_ylabel("Astropy LS Power")
    else:
        axarr[0,1].plot(np.nan,np.nan)
        axarr[0,1].text(-2,0,"No Period Search")
        axarr[0,1].set_xlabel("Date")
        axarr[0,1].set_xlim(-5,5)
        axarr[0,1].set_ylim(-5,5)
    if log:
        axarr[0,1].set_xscale('log')    

    axarr[0,1].set_xlabel("Period (days)")

    #Period1
    axarr[1,0].set_title("{} Period1: {}".format(ttype,fperiod1))
    axarr[1,0].plot(lsphase1,nmag,myformat,ms=myptsize,c="green",rasterized=True)
    axarr[1,0].set_xlabel("Phase")
    if flux:
        axarr[1,0].set_ylabel("Flux")
    else:
        axarr[1,0].set_ylabel("Magnitude")
        axarr[1,0].invert_yaxis() 
    #Period1
    axarr[1,1].set_title("{} Period2: {}".format(ttype,fperiod2))
    axarr[1,1].plot(lsphase2,nmag,myformat,ms=myptsize,c="green",rasterized=True)
    axarr[1,1].set_xlabel("Phase")
    if not flux:
        axarr[1,1].invert_yaxis() 
      
    
    logfile.write("{},{},{},{},{},{},{}".format(curve,name,stype,min,max,fperiod1,fperiod2))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    pp.savefig(dpi=200)
    plt.clf()
    plt.close() #Frees up memory
    
    logfile.close()
    
    if ret_results == True:
        return((curve,name,stype,fperiod1,fperiod2))
    else:
        return(retval) 

# -------------
# Main Function
# -------------
def lcurve_main(filename,basename=None,flux=False,log=False,manual=None,max=10.0,min=0.1,stype='ls',ret_results=False,verb=False):
    '''
    Does a period search on a variable curve using the vartools package
    
    Arguments:
        filename: Curve file to be searched
    Keywords:
        basename: Give a basename for the output .pdf and log file.
        flux: Data is in Flux units not Magnitude units
        log: Use a Log 10 days x-axis on the Periodogram
        Manual: Use a manual period instead of searching for the period.
        max: Maximum Period to search
        min: Minimum Period to search
        stype: Type of period search aov,fchi2,ls,apyls default (ls).
        ret_results: If True, return the results that would normally go to a log file.
        verb: If True, increase verbosity
        
    Returns:
        0 for success. If ret_results is True, then returns a tuple of results.
    '''
    
    #Get New Filename
    if basename == None:
        (base,ext) = os.path.splitext(filename)
    else:
        base = basename
        
    if manual != None:
        stype = 'manual'
        
    newfilename = base+".pdf"
    if verb==True:
        print("Output Filename: {}".format(newfilename))
    
    pp = PdfPages('{}'.format(newfilename))
    results = plot_lc(filename,base,pp,clean=10,flux=flux,ret_results=ret_results,tdict=False,stype=stype,log=log,
                      manual=manual,min=min,max=max)
    
    pp.close()
    
    if ret_results == True:
        return results

if __name__ == '__main__':
    #Setup command line arguments

    parser = argparse.ArgumentParser(description='Convert KELT Terrestrial Time Light Curves to BJD_TBD Light Curves')
    parser.add_argument('Filename',help='Lightcurve file')
    parser.add_argument('-b', default=None,metavar="BASENAME",help="Give a basename for the output .pdf and log file.")
    parser.add_argument('-f', action='store_true',help="Data is in Flux not Magnitudes")
    parser.add_argument('-g', action='store_true',help="Use a Log 10 days x-axis on periodogram")
    parser.add_argument('-l', default=0.1,metavar="MIN",type=float ,help="Minimum Period to search in days (Default 0.1).")    
    parser.add_argument('-m', default=None,metavar="PERIOD",type=float ,help="Use a manual period to plot light curve.")
    parser.add_argument('-t', default='ls',metavar="STYPE" ,help="Type of period search aov,fchi2,ls,apyls (Default ls).")
    parser.add_argument('-u', default=10.0,metavar="MAX",type=float ,help="Maximum Period to search in days (Default 10.0).")
    parser.add_argument('-v', action='store_true',help="Increase verbosity")
    
#Put this in a dictionary    
    args = vars(parser.parse_args())
    ret = lcurve_main(args['Filename'],basename=args['b'],flux=args['f'],log=args['g'],manual=args['m'],min=args['l'],ret_results=False,
                      stype=args['t'],max=args['u'],verb=args['v'])
    sys.exit(0)
    
##
#@mainpage
 #@copydetails  lcurve
    