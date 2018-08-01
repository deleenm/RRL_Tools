#!/usr/bin/python
'''
This tool fits a spline to phased lightcurve data

Detailed Description

@package spline_fit
@author deleenm
@version \e \$Revision$
@date \e \$Date$

Usage: spline_fit.py
'''

# -----------------------------
# Standard library dependencies
# -----------------------------
import argparse
import os
import sys
# -------------------
# Third-party imports
# -------------------
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.interpolate as interp

# -----------------
# Class Definitions
# -----------------

# --------------------
# Function Definitions
# --------------------
def calc_aggregate(phase,mag,err,num=100,type='Median',sigclip=None):
    '''
    Creates aggregate points that can be used to remove outliers and improve errorbars
    
    Arguments:
        Phase: A numpy array of phased date values between 0 and 1
        mag: A numpy array of magnitudes for each phase
        err: A numpy array of magnitude errors for each phase
    Keywords:
        num: Number of aggregate points to divide the data into.
        sigclip: Set to a number to throw out all points more than that many sigma from the aggregation. Then recompute sigma
        type: Type of aggregation: Mean, Median, Weighted_mean
        
    Returns:
        A tuple of four numpy arrays giving the phase,mag,err,num for each aggregate point. Num is the number of data points 
        in each aggregate point. If sigclip is not None, then it returns a tuple of five numpy arrays with the first four the same as
        before, but with an additional one giving a boolean array with the length of phase with clipped data set to False.
    '''
    
    bins = np.linspace(0,1,num)
    final_phase = list()
    final_mag = list()
    final_err = list()
    final_num = list()
    
    for i in range(len(bins)-1):
        
        final_phase.append((bins[i+1]-bins[i])/2 + bins[i])
        idx = ((phase >= bins[i]) & (phase < bins[i+1]))
        sample_mag = mag[idx]
        sample_err = err[idx]
        
        final_num.append(len(idx))
        if len(idx) == 0:
            final_mag.append(np.nan)
            final_err.append(np.nan)
        else:
            if type=='Median':
                agg = np.nanmedian(sample_mag)
                agg_err = np.nanstd(sample_mag)
            if type=='Mean':
                agg = np.nanmean(sample_mag)
                agg_err = np.nanstd(sample_mag)   
            
            final_mag.append(agg)
            final_err.append(agg_err)
    
    #Return arrays coverting lists into numpy arrays as necessary
    if sigclip == None:
        return (np.array(final_phase),np.array(final_mag),np.array(final_err),final_num)
    else:
        clip_arr = None
        return (np.array(final_phase),np.array(final_mag),np.array(final_err),final_num,clip_arr)


def get_spline(phase,mag,error,order=3,extend_num=6):
    '''
    Fit a Bspline to data points
    
    Arguments:
        Phase: A numpy array of phased date values between 0 and 1
        mag: A numpy array of magnitudes for each phase
        err: A numpy array of magnitude errors for each phase
    Keywords:
        order: The order of the spline
        extend_num: The number of points to extend each side by to ensure matching on each end.
        
    Returns:
        A spline object
    '''
    #X values need to be sorted.
    sortind = np.argsort(phase)
    #Extend_num array by extend_num variable on 0.0 end and 1.1 end
    extind = np.concatenate((sortind[-extend_num:],sortind,sortind[0:extend_num]))
    phase = phase[extind]
    phase[0:extend_num] = phase[0:extend_num] - 1.0
    phase[-extend_num:] = phase[-extend_num:] + 1.0
    #Do this with rest of arrays
    mag = mag[extind]
    error = error[extind]
    #Interpolater doesn't do well with nan's
    goodind = np.isfinite(mag)
    phase = phase[goodind]
    mag = mag[goodind]
    error = error[goodind]
    #Do the spline interpolation
    phasemag_sp = interp.UnivariateSpline(phase,mag,k=order,w=1.0/error)
    return phasemag_sp #This is a spline object

def get_phase(jd,period,jdmax=None):
    '''
    Convert Julian dates and a period into phases with optionial JD of Maximum 
    
    Arguments:
        jd: The date of observations in days
        period: The period of the light curve in days
    Keywords:
        jdmax: Date of maximum light if known
        
    Returns:
        A numpy array of phases
    '''
    
    if jdmax == None:
        jdmax = 0
    
    phase = (jd - jdmax)/period
    phase = phase - np.floor(phase)
    indice = (phase < 0)
    phase[indice] = phase[indice] + 1
    return phase

def read_curve(filename):
    mytable = Table()
    date,mag,err = np.genfromtxt(filename,unpack=True)
    mytable['date'] = date
    mytable['mag'] = mag
    mytable['err'] = err
    #mytable.read(filename,format='ascii',names=('jd','mag','err'))
    return mytable


# -------------
# Main Function
# -------------
def spline_fit_main(filename,period,dates=None,base=None,plot=False,median=False,npts=10,verb=False):
    #Set base filename
    if base == None:
        base = os.path.splitext(filename)[0]
    
    curve_tab = read_curve(filename)
    #Start my pdf file
    pp = PdfPages('{}_sp.pdf'.format(base))
    
    curve_tab['phase'] = get_phase(curve_tab['date'],period)
    
    agg_tab = Table()
    
    #Calculate the aggregate point if requested
    (agg_tab['phase'],agg_tab['mag'], agg_tab['err'], agg_tab['num']) = calc_aggregate(curve_tab['phase'],curve_tab['mag'],
                                                                                       curve_tab['err'],num=npts,type='Mean')
 
    #Fit Spline
    extend_num = int(np.floor(len(agg_tab['phase'])/10))
    #Decreased errorbars by 90% to help the fit.
    myspline = get_spline(agg_tab['phase'],agg_tab['mag'], agg_tab['err']*0.1,order=3,extend_num=extend_num)
    spline_phase = np.linspace(0,1,npts*10)
    
    plt.plot(curve_tab['phase'],curve_tab['mag'],'.',ms=0.5,zorder=1)
    plt.errorbar(agg_tab['phase'],agg_tab['mag'],yerr=agg_tab['err'],marker='.',ls='None',lw=.75,ms=1.5,zorder=2)
    plt.plot(spline_phase,myspline(spline_phase),'--',zorder=3)
    plt.xlabel('Phase')
    plt.ylabel('Magnitude')
    plt.xlim(0,1)
    plt.gca().invert_yaxis()
    pp.savefig()



    pp.close()

if __name__ == '__main__':
       #Check to make sure we have 1 argument

    parser = argparse.ArgumentParser(description='Fits a Spline to a lightcurve.')
    parser.add_argument('filename',help='3 column light curve file JD Magnitude Error')
    parser.add_argument('period',type=float,help='The period to phase the light curve.')
    parser.add_argument('-b', default=None,metavar="BASENAME",help="Give a basename for the pdf and log files.")
    parser.add_argument('-d',type=float,default=None,metavar='DATE1,DATE2,etc',help='Comma separated list of Date(s) of interest. (Default None)')
    parser.add_argument('-m', action='store_true',help="Use median points instead of actual data.")
    parser.add_argument('-n', default=20,type=float,metavar="NUM",help="Number of median points (Default 20)")
    parser.add_argument('-p', action='store_true',help="Generate plots")
    parser.add_argument('-v', action='store_true',help="Be Verbose")

    
#Put this in a dictionary    
    args = vars(parser.parse_args())
    ret = spline_fit_main(args['filename'],args['period'],args['d'],plot=args['p'],median=args['m'],npts=args['n'],verb=args['v']
                           ,base=args['b'])
    sys.exit(0)
    
##
#@mainpage
 #@copydetails  spline_fit
    