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
import astropy.stats as astats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.interpolate as interp
from statsmodels.sandbox.nonparametric.tests.ex_smoothers import weights

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
        type: Type of aggregation: Mean, Median, and Weighted_Mean
        
    Returns:
        A tuple of five numpy arrays. The first four give the phase,mag,err,num for each aggregate point. Num is the number of data points 
        in each aggregate point. The final array is a boolean array that gives the masking for the phase array.
    '''
    
    bins = np.linspace(0,1,num)
    final_phase = list()
    final_mag = list()
    final_err = list()
    final_num = list()
    final_clip = np.full(len(mag), True)
    
    for i in range(len(bins)-1):
        
        final_phase.append((bins[i+1]-bins[i])/2 + bins[i])
        idx = ((phase >= bins[i]) & (phase < bins[i+1]))
        sample_mag = mag[idx]
        sample_err = err[idx]
        
        
        if len(idx) == 0:
            final_mag.append(np.nan)
            final_err.append(np.nan)
            final_num.append(len(idx))
        else:
            good_ind = np.full(len(sample_mag),True)
            if type=='Median':
                if sigclip != None:
                    #Note True values in masks are invalid data, so reverse the sense
                    good_ind = np.logical_not((astats.sigma_clip(sample_mag, sigclip,iters=5)).mask)
                    sample_mag = sample_mag[good_ind]
                    
                agg = np.nanmedian(sample_mag)
                agg_err = np.nanstd(sample_mag)
            if type=='Mean':
                if sigclip != None:
                    #Note True values in masks are invalid data, so reverse the sense
                    good_ind = np.logical_not((astats.sigma_clip(sample_mag, sigclip,iters=5)).mask)
                    sample_mag = sample_mag[good_ind]
                    
                agg = np.nanmean(sample_mag)
                agg_err = np.nanstd(sample_mag)   
                
            if type=='Weighted_Mean':
                if sigclip != None:
                    #Note True values in masks are invalid data, so reverse the sense
                    good_ind = np.logical_not((astats.sigma_clip(sample_mag, sigclip,iters=5)).mask)
                    sample_mag = sample_mag[good_ind] 
            
                (agg,agg_err) = calc_weighted_stat(sample_mag, sample_err)    
            
            final_clip[idx] = good_ind
            final_num.append(len(sample_mag))
            final_mag.append(agg)
            final_err.append(agg_err)
    
    #Return arrays coverting lists into numpy arrays as necessary
    return (np.array(final_phase),np.array(final_mag),np.array(final_err),final_num,final_clip)

def calc_props(curve_tab,myspline,spline_phase,period):
    '''
    Calculates lumin weighted magnitude, mag weighted magnitude, amplitude, Date of max from the Spline etc.
    
    Arguments:
        curve_tab: The lightcurve Table
        myspline: The Spline Function
        spline_phase: Phase array to go with spline
        period: period of lightcurve
    Keywords:
    
    Returns:
        Dictionary with these keys: Amplitude, Mag_Min, Mag_Max, Mag_Ave, Flux_Ave, Phase_Max, Epoch_Max
        
    '''
    prop_dict = dict()
    
    #Calculate Amplitude, Min, Max
    spline_mag = myspline(spline_phase)
    
    prop_dict['Amplitude'] = np.max(spline_mag) - np.min(spline_mag)
    prop_dict['Mag_Min'] = np.nanmin(spline_mag)
    prop_dict['Mag_Max'] = np.nanmax(spline_mag)
    
    #Calcluate Mag Weighted average
    prop_dict['Mag_Ave'] = np.nanmean(spline_mag)
    
    #Calculate Flux Weighted average
    spline_flux = -2.5*10**spline_mag
    prop_dict['Flux_Ave'] = np.nanmean(spline_flux)
    
    #Find Phase of Maximum (Mags work backwards)
    idx = np.nanargmin(spline_mag)
    prop_dict['Phase_Max'] = spline_phase[idx]
    
    #Date of Maximum
    #Find the middle date of my data and its corresponding cycle
    avg_date = np.nanmean(curve_tab['date'])
    cycle = np.floor(avg_date/period)
    prop_dict['Epoch_Max'] = period * (prop_dict['Phase_Max']+cycle)
    
    return prop_dict

def calc_weighted_stat(data,error):
    '''
    Calculated the weighted mean and standard deviation
    
    Arguments:
        data: A numpy array of data
        error: A numpy array of errors
    Keywords:
            
    Returns:
        A tuple (mean,std_dev)
        
    '''
    weight = 1/error**2
    weight_tot = np.nansum(weight)
    weight_sq_tot = np.nansum(weight**2)
    numerator = np.nansum(weight*data)
    
    weighted_mean = numerator/weight_tot
    
    weighted_std = 1/weight_sq_tot
    
    return (weighted_mean,weighted_std)


    
    
    
    
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
    
    #Remove any non-unique points by adding 1E-7 to the phase
    for i in range(len(phase)-1):
        if phase[i] == phase[i+1]:
            phase[i+1] = phase[i+1] + 1E-7
    
    #Do this with rest of arrays
    mag = mag[extind]
    error = error[extind]
    #Interpolater doesn't do well with nan's
    goodind = np.isfinite(mag)
    phase = phase[goodind]
    mag = mag[goodind]
    error = error[goodind]
    #Do the spline interpolation
    weights = list()
    for i in range(len(error)):
        if error[i] == 0:
            weights.append(1.0/0.000001)
        else:
            weights.append(1.0/error[i])
    weights = np.array(weights)
    
    phasemag_sp = interp.UnivariateSpline(phase,mag,k=order,w=weights)
    
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

def make_plot(curve_tab,myspline,spline_phase,prop_dict,pp,agg_tab=None,errorbars=False):
    '''
    Plot curves if requested
    
    Arguments:
        curve_tab: The lightcurve Table
        myspline: The Spline Function
        spline_phase: Phase array to go with spline
        prop_dict: Dictionary of spline properties
        pp: PDF pointer
    Keywords:
        agg_tab: The aggregate points Table
        Errorbars: Whether to plot errorbars on data
        
    Returns:
        None
    '''
    #Deal with large number of points
    if len(curve_tab['phase']) > 200:
        alpha = 0.2
    else:
        alpha = 1    
    
    if type(agg_tab) == type(None):
        #Make plots without aggregated points
        if errorbars:
            plt.errorbar(curve_tab['phase'],curve_tab['mag'],yerr=curve_tab['err'],marker='.',ls='None',lw=.75,ms=1.5,alpha=alpha,zorder=1)
        else:
            plt.plot(curve_tab['phase'],curve_tab['mag'],'.',ms=0.5,alpha=alpha,zorder=1)
    else:
        mask = curve_tab['mask']    
        if errorbars:
            plt.errorbar(curve_tab['phase'],curve_tab['mag'],yerr=curve_tab['err'],marker='.',ls='None',lw=.75,ms=1.5,alpha=alpha,zorder=1)
        else:
            plt.plot(curve_tab['phase'][mask],curve_tab['mag'][mask],'.',ms=0.5,alpha=alpha,zorder=1)
        plt.errorbar(agg_tab['phase'],agg_tab['mag'],yerr=agg_tab['err'],marker='.',ls='None',lw=.75,ms=1.5,zorder=2)
        
        
    plt.plot(spline_phase,myspline(spline_phase),'--',zorder=3,color='red')
    plt.xlabel('Phase')
    plt.ylabel('Magnitude')
    plt.axvline(prop_dict['Phase_Max'])
    plt.xlim(0,1)
    plt.gca().invert_yaxis()
    pp.savefig()

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
def spline_fit_main(filename,period,dates=None,factor=1,errorbars=False,base=None,plot=False,method="Median",npts=10
                    ,order=3,sigclip=None,verb=False):
    #Set base filename
    if base == None:
        base = os.path.splitext(filename)[0]
    
    curve_tab = read_curve(filename)
    #Check to make sure all dates are unique
    if len(curve_tab['date']) != len(np.unique(curve_tab['date'])):
        print("You have identical dates in your light curve!")
        sys.exit(1)
    
    #Start my pdf file
    if plot:
        pp = PdfPages('{}_sp.pdf'.format(base))
    
    curve_tab['phase'] = get_phase(curve_tab['date'],period)
    
    agg_tab = Table()
        
    #Calculate the aggregate point if requested
    if method != 'None':
        #Create the Aggregated points
        (agg_tab['phase'],agg_tab['mag'], agg_tab['err'], agg_tab['num'],curve_tab['mask']) = calc_aggregate(
            curve_tab['phase'],curve_tab['mag'],curve_tab['err'],num=npts,type='Median',sigclip=sigclip)
 
        #Fit Spline
        extend_num = int(np.floor(len(agg_tab['phase'])/10))
        myspline = get_spline(agg_tab['phase'],agg_tab['mag'], agg_tab['err']*factor,order=order,extend_num=extend_num)
        spline_phase = np.linspace(0,1,npts*10)
        prop_dict = calc_props(curve_tab, myspline, spline_phase, period)
        
        if plot:
            make_plot(curve_tab, myspline, spline_phase, prop_dict, pp, agg_tab=agg_tab, errorbars=errorbars) 
            
    else:
        #Fit Spline
        extend_num = int(np.floor(len(curve_tab['phase'])/10))
        myspline = get_spline(curve_tab['phase'],curve_tab['mag'], curve_tab['err'],order=order,extend_num=extend_num)
        spline_phase = np.linspace(0,1,len(curve_tab['phase'])*10)
        prop_dict = calc_props(curve_tab, myspline, spline_phase, period)
        if plot:
            make_plot(curve_tab, myspline, spline_phase, prop_dict, pp, errorbars=errorbars) 
    
    if verb:        
        print("{} Amplitude: {:.3f} {:.6f}".format(filename,prop_dict['Amplitude'],prop_dict['Epoch_Max']))
    

    if plot:
        pp.close()

if __name__ == '__main__':
    #Check to make sure we have 1 argument

    parser = argparse.ArgumentParser(description='Fits a Spline to a lightcurve.')
    parser.add_argument('filename',help='3 column light curve file JD Magnitude Error')
    parser.add_argument('period',type=float,help='The period to phase the light curve.')
    parser.add_argument('-b', default=None,metavar="BASENAME",help="Give a basename for the pdf and log files.")
    parser.add_argument('-d',type=float,default=None,metavar='DATE1,DATE2,etc'
                        ,help='Comma separated list of Date(s) of interest. (Default None)')
    parser.add_argument('-e', action='store_true',help="Show errorbars on Data")
    parser.add_argument('-f', default=1,type=float,metavar="FACTOR",help="Factor to tighten errorbars on aggregate 0 to 1 (Default 1)")
    parser.add_argument('-m', metavar='METHOD',default="None"
                        ,help="Method to aggregate points: Mean, Median, Weighted_Mean, or None (Default None)")
    parser.add_argument('-n', default=20,type=float,metavar="NUM",help="Number of aggregate points to break data into (Default 20)")
    parser.add_argument('-o', default=3,type=int,metavar="ORDER",help="Spline order betwee 1 and 5 (Default 3)")
    parser.add_argument('-s', default=None,type=float,metavar="SIGMA"
                        ,help="Sigmas used in sigma clipping. None for no sigma clipping (Default None)")
    parser.add_argument('-p', action='store_true',help="Generate plots")
    parser.add_argument('-v', action='store_true',help="Be Verbose")

    
#Put this in a dictionary    
    args = vars(parser.parse_args())
    ret = spline_fit_main(args['filename'],args['period'],dates=args['d'],factor=args['f'],errorbars=args['e']
                          ,plot=args['p'],method=args['m'],npts=args['n'],order=args['o'],sigclip=args['s'],verb=args['v']
                          ,base=args['b'])
    sys.exit(0)
    
##
#@mainpage
 #@copydetails  spline_fit
    