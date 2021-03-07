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
import os.path
import sys
import time
from subprocess import Popen
# -------------------
# Third-party imports
# -------------------
import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# -----------------
# Class Definitions
# -----------------

# --------------------
# Function Definitions
# --------------------
def plot_lc(curve,name,pp,clean=10,tdict=False,type='ls'):
    '''
    Plot up a phased light curve based on a raw light curve adds plot to pdf object
    
    Arguments:
        curve: A light curve file (date, mag, err)
        name: The Star Name.
        pp: PDF Object
    Keywords:
        clean: How many maximum and minimum points to remove
        tdict: A Dictionary of info to put in the title of the plots
        type: Type of periodogram finder (ls or aov).
    Returns:
        None
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
        plt.ylabel("Magnitude")
        plt.xlabel("Date")
        plt.xlim(-5,5)
        plt.ylim(-5,5)
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
    if(type == 'ls'):
        ttype = "Lomb-Scargle"
        #should rewrite to communicate directly
        lstool = Popen(["vartools -i ../temp_lc.cur -ascii -redirectstats ../ls.stat -header -LS 0.1 10. 0.1 2 1 ../ whiten clip 5. 1"],shell=True)
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
            
        lsperiod1 = (lsstat['LS_Period_1_0'])[0]
        lsperiod2 = (lsstat['LS_Period_2_0'])[0]
    
    if(type == 'aov'):
        ttype = 'AOV'
        aovtool = Popen(["vartools -i ../temp_lc.cur -ascii -redirectstats ../aov.stat -header -aov Nbin 20 0.1 10. 0.1 0.01 2 1 ../ whiten clip 5. 1"],shell=True)
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

        lsperiod1 = (lsstat['Period_1_0'])[0]
        lsperiod2 = (lsstat['Period_2_0'])[0]   
        
        
    if(type == 'fchi2'):
        ttype = 'FChi2'
        aovtool = Popen(["vartools -L /usr/local/share/vartools/USERLIBS/fastchi2.la -i ../temp_lc.cur -ascii -redirectstats ../fchi2.stat -header -fastchi2 Nharm fix 3 freqmax fix 10 freqmin fix .1 oversample fix 10 Npeak 2 oper ../"],shell=True)
        aovtool.wait()
    
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

        lsperiod1 = 1.0/(lsstat['Fastchi2_Frequency_1_0'])[0]
        lsperiod2 = 1.0/(lsstat['Fastchi2_Frequency_2_0'])[0]   

    lsphase1 = date/lsperiod1
    lsphase1 = (lsphase1 - np.floor(lsphase1))
    lsphase1 = np.concatenate((lsphase1,lsphase1+1))
    lsphase2 = date/lsperiod2
    lsphase2 = (lsphase2 - np.floor(lsphase2))
    lsphase2 = np.concatenate((lsphase2,lsphase2+1))

    nmag = np.concatenate((mag,mag))

    #Open Logfile
    try:
        filename = name + ".log"
        logfile = open(filename,'w')
    except IOError:
        print("{} could not be opened!".format(filename))

    #Write out header
    header = "#Filename,Starname,Type,Period1,Period2\n"
    
    logfile.write(header)


    #Make plot
    plt.rcParams['mathtext.default'] = 'regular' 
    
    f,axarr = plt.subplots(2,2,figsize=(8.5,11))
    f.subplots_adjust(hspace=0.3,top=0.94,bottom=0.05)
    plt.suptitle(titlestring)
    axarr[0,0].set_title("Data")
    axarr[0,0].plot(date,mag,myformat,ms=myptsize,c="blue",rasterized=True)
    axarr[0,0].set_ylabel("KELT Magnitude")
    axarr[0,0].set_xlabel("Date")
    axarr[0,0].invert_yaxis()
    #Periodogram
    axarr[0,1].set_title("Periodogram")
    if(type == 'ls'):
        axarr[0,1].plot(np.log10(1.0/lsperiod['col1']),lsperiod['col2'],c="red",rasterized=True)
        axarr[0,1].set_ylabel("LS Power")
        axarr[0,1].set_xlabel("Log10 Period (days)")
    if(type == 'aov'):
        axarr[0,1].plot(np.log10(aovperiod['Period']),aovperiod['AOV_WhitenCycle_0'],c="red",rasterized=True)
        axarr[0,1].set_ylabel("AOV Power")
        axarr[0,1].set_xlabel("Log10 Period (days)")
    if(type == 'fchi2'):
        axarr[0,1].plot(np.log10(1.0/fchi2period['Frequency']),fchi2period['Chireduction'],c="red",rasterized=True)
        axarr[0,1].set_ylabel("Chi Power")
        axarr[0,1].set_xlabel("Log10 Period (days)")

    #Period1
    axarr[1,0].set_title("{} Period1: {}".format(ttype,lsperiod1))
    axarr[1,0].plot(lsphase1,nmag,myformat,ms=myptsize,c="green",rasterized=True)
    axarr[1,0].set_xlabel("Phase")
    axarr[1,0].invert_yaxis()
    #Period1
    axarr[1,1].set_title("{} Period2: {}".format(ttype,lsperiod2))
    axarr[1,1].plot(lsphase2,nmag,myformat,ms=myptsize,c="green",rasterized=True)
    axarr[1,1].set_xlabel("Phase")
    axarr[1,1].invert_yaxis()  
    
    logfile.write("{},{},{},{},{}".format(curve,name,type,lsperiod1,lsperiod2))
    
    pp.savefig(dpi=200)
    plt.clf()
    plt.close() #Frees up memory
    
    logfile.close()
    
    return(retval) 

# -------------
# Main Function
# -------------
def lcurve_main(filename,basename=None,type='ls'):
    
    #Get New Filename
    if basename == None:
        (base,ext) = os.path.splitext(filename)
    else:
        base = basename
        
    newfilename = base+".pdf"
    print("Output Filename: {}".format(newfilename))
    
    pp = PdfPages('{}'.format(newfilename))
    plot_lc(filename,base,pp,clean=10,tdict=False,type=type)
    
    pp.close()

if __name__ == '__main__':
    #Setup command line arguments

    parser = argparse.ArgumentParser(description='Convert KELT Terrestrial Time Light Curves to BJD_TBD Light Curves')
    parser.add_argument('Filename',help='Lightcurve file')
    parser.add_argument('-b', default=None,metavar="BASENAME",help="Give a basename for the output .pdf and log file.")
    parser.add_argument('-t', default='ls',metavar="TYPE" ,help="Type of period search aov,fchi2,ls default (ls).")
    
    
#Put this in a dictionary    
    args = vars(parser.parse_args())
    ret = lcurve_main(args['Filename'],basename=args['b'],type=args['t'])
    sys.exit(0)
    
##
#@mainpage
 #@copydetails  lcurve
    