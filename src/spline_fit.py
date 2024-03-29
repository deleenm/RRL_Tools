#!/usr/bin/env python
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


# -----------------
# Class Definitions
# -----------------

# --------------------
# Function Definitions
# --------------------
def adjust_curve(curve_tab,prop_dict,agg_tab=None,flux=False,plot_clean_num=0):
    '''
    Removes high and low points and adjusts phase 0 to maximum
    
    Arguments:
        curve_tab: The lightcurve Table
        prop_dict: Dictionary of spline properties

    Keywords:
        agg_tab: The aggregate points Table
        flux: The points a in flux units not magnitudes
        plot_clean_num: Number of highest and lowest y value points to remove, to make the plot look cleaner.
                        Does not affect any calculation.
        
    Returns:
        None
    '''
    
    #Correct phases, so phase_max = 0
    curve_tab['phase_fixed'] = shift_phase(curve_tab['phase'],prop_dict['Phase_Max'])
    agg_tab['phase_fixed'] = shift_phase(agg_tab['phase'],prop_dict['Phase_Max'])
    
    #Clean up the data
    if(2*plot_clean_num >= len(curve_tab)):
        print("Lightcurve has only {} epochs. {} is too many points to clean!".format(len(curve_tab),2*plot_clean_num))
        print("Setting Plot Clean Num to 0.")
        plot_clean_num = 0
    
    if(plot_clean_num != 0):
        clean_idx = np.argsort(curve_tab['mag'])
        clean_idx = clean_idx[plot_clean_num:]
        clean_idx = clean_idx[:-plot_clean_num]
        curve_tab['clean_idx'] = np.zeros(len(curve_tab),dtype=bool)
        curve_tab['clean_idx'][clean_idx] = 1
    else:
        curve_tab['clean_idx'] = np.ones(len(curve_tab),dtype=bool)

def calc_aggregate(phase,mag,err,num=100,method='Median',sigclip=None):
    '''
    Creates aggregate points that can be used to remove outliers and improve errorbars
    
    Arguments:
        Phase: A numpy array of phased date values between 0 and 1
        mag: A numpy array of magnitudes for each phase
        err: A numpy array of magnitude errors for each phase
    Keywords:
        num: Number of aggregate points to divide the data into.
        sigclip: Set to a number to throw out all points more than that many sigma from the aggregation. Then recompute sigma
        method: Type of aggregation: Mean, Median, and Weighted_Mean
        
    Returns:
        A tuple of five numpy arrays. The first four give the phase,mag,err,num for each aggregate point. Num is the number of data points 
        in each aggregate point. The err depends on the method used. The final array is a boolean array that gives the masking for the phase array.
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
        #Select point per bin
        sample_mag = mag[idx]
        sample_err = err[idx]
        
        if np.sum(idx) == 0:
            final_mag.append(np.nan)
            final_err.append(np.nan)
            final_num.append(len(idx))
        else:
            good_ind = np.full(len(sample_mag),True)
            if method=='Median':
                if sigclip != None:
                    #Note True values in masks are invalid data, so reverse the sense
                    good_ind = np.logical_not((astats.sigma_clip(sample_mag, sigclip,maxiters=5)).mask)
                    sample_mag = sample_mag[good_ind]
                
                totalgood = np.sum(np.isfinite(sample_mag))   
                agg = np.nanmedian(sample_mag)
                #This is the standard error of the mean
                agg_err = np.nanstd(sample_mag) / np.sqrt(totalgood)
            elif method=='Mean':
                if sigclip != None:
                    #Note True values in masks are invalid data, so reverse the sense
                    good_ind = np.logical_not((astats.sigma_clip(sample_mag, sigclip,maxiters=5)).mask)
                    sample_mag = sample_mag[good_ind]
                
                totalgood = np.sum(np.isfinite(sample_mag))    
                agg = np.nanmean(sample_mag)
                #This is the standard error of the mean
                agg_err = np.nanstd(sample_mag) / np.sqrt(totalgood) 
                
            elif method=='Weighted_Mean':
                if sigclip != None:
                    #Note True values in masks are invalid data, so reverse the sense
                    good_ind = np.logical_not((astats.sigma_clip(sample_mag, sigclip,maxiters=5)).mask)
                    sample_mag = sample_mag[good_ind] 
                    sample_err = sample_err[good_ind] 
            
                (agg,agg_err) = calc_weighted_stat(sample_mag, sample_err)  
            
            else:
                print("Fatal: Aggregated Point Method: {} is not a valid method!".format(method))
                sys.exit(1)  
            
            final_clip[idx] = good_ind
            final_num.append(len(sample_mag))
            final_mag.append(agg)
            final_err.append(agg_err)
    
    #Return arrays coverting lists into numpy arrays as necessary
    return (np.array(final_phase),np.array(final_mag),np.array(final_err),final_num,final_clip)

def calc_props(curve_tab,myspline,spline_phase,prop_dict,flux=False,lower=None,upper=None,verb=False):
    '''
    Calculates lumin weighted magnitude, mag weighted magnitude, amplitude, Date of max from the Spline etc.
    
    Arguments:
        curve_tab: The lightcurve Table
        myspline: The Spline Function
        spline_phase: Phase array to go with spline
        prop_dict: Dictionary of stellar properties to add to
        
    Keywords:
        flux: Light curve is in flux units not magnitude units
        lower: A JD around which to find the Epoch of Minimum
        upper: A JD around which to find the Epoch of Maximum
        verb: Whether to be verbose
    Returns:
        Dictionary with these keys added: Amplitude, Mag_Min, Mag_Max, Mag_Ave, Flux_Ave, Phase_Max, Epoch_Max
        
    '''
    prop_dict
    
    #Calculate Amplitude, Min, Max
    spline_mag = myspline(spline_phase)
    
    prop_dict['Amplitude'] = np.max(spline_mag) - np.min(spline_mag)
    
    if flux:
        prop_dict['Flux_Min'] = np.nanmin(spline_mag)
        prop_dict['Flux_Max'] = np.nanmax(spline_mag)
    else:
        prop_dict['Mag_Min'] = np.nanmin(spline_mag)
        prop_dict['Mag_Max'] = np.nanmax(spline_mag)
    
    #Calcluate Mag Weighted average
    if flux:
        prop_dict['Mag_Ave'] = np.nanmean(-2.5*np.log10(spline_mag))
    else:    
        prop_dict['Mag_Ave'] = np.nanmean(spline_mag)
    
    #Calculate Flux Weighted average
    if flux:
        prop_dict['Flux_Ave'] = np.nanmean(spline_mag)
    else:    
        spline_flux = -2.5*10**spline_mag
        prop_dict['Flux_Ave'] = np.log10(np.nanmean(spline_flux)/-2.5)
    
    #Find Phase of Maximum (Mags work backwards)   
    if flux:
        idx = np.nanargmax(spline_mag)
        prop_dict['Phase_Max'] = spline_phase[idx]
    else:
        idx = np.nanargmin(spline_mag)
        prop_dict['Phase_Max'] = spline_phase[idx]
    
    #Find Phase of Minimum (Mags work backwards)
    if flux:
        idx = np.nanargmin(spline_mag)
        prop_dict['Phase_Min'] = spline_phase[idx] 
    else:
        idx = np.nanargmax(spline_mag)
        prop_dict['Phase_Min'] = spline_phase[idx] 
      
    #Date of Maximum
    if upper == None:
        #Find the middle date of my data and its corresponding cycle
        avg_date = np.nanmean(curve_tab['date'])
        cycle = np.floor(avg_date/prop_dict['Period'])
        prop_dict['Epoch_Max'] = prop_dict['Period'] * (prop_dict['Phase_Max']+cycle)
    else:
        #Find the cycle closest to my upper JD. Try the one less than my date and one above my date
        lcycle = np.floor(upper/prop_dict['Period'])
        hcycle = np.floor(upper/prop_dict['Period'])+1
        ljdmax = prop_dict['Period'] * (prop_dict['Phase_Max']+lcycle)
        hjdmax = prop_dict['Period'] * (prop_dict['Phase_Max']+hcycle)
        if np.abs(ljdmax - upper) < np.abs(hjdmax - upper):
            prop_dict['Epoch_Max'] = ljdmax
        else:
            prop_dict['Epoch_Max'] = hjdmax 
            
    #Date of Minimum
    if lower == None:
        #Find the middle date of my data and its corresponding cycle
        avg_date = np.nanmean(curve_tab['date'])
        cycle = np.floor(avg_date/prop_dict['Period'])
        prop_dict['Epoch_Min'] = prop_dict['Period'] * (prop_dict['Phase_Min']+cycle)
    else:
        #Find the cycle closest to my lower JD. Try the one less than my date and one above my date
        lcycle = np.floor(lower/prop_dict['Period'])
        hcycle = np.floor(lower/prop_dict['Period'])+1
        ljdmax = prop_dict['Period'] * (prop_dict['Phase_Min']+lcycle)
        hjdmax = prop_dict['Period'] * (prop_dict['Phase_Min']+hcycle)
        if np.abs(ljdmax - lower) < np.abs(hjdmax - lower):
            prop_dict['Epoch_Min'] = ljdmax
        else:
            prop_dict['Epoch_Min'] = hjdmax 
    
    #Create a table that contains the shifted spline values that are also sorted
    spline_tab = Table()
    spline_phase_fixed = shift_phase(spline_phase, prop_dict['Phase_Max'])
    idx = np.argsort(spline_phase_fixed)
    
    spline_tab['phase'] = np.concatenate((spline_phase_fixed[idx],spline_phase_fixed[idx] +1))
    spline_tab['mag'] = np.concatenate((spline_mag[idx],spline_mag[idx]))
    
    #Calc the phases for the given dates
    if prop_dict['Dates'] != None:
        dates_arr = np.array(prop_dict['Dates'].split(','),dtype='float64')
        
        phase_arr = get_phase(dates_arr, prop_dict['Period'], prop_dict['Epoch_Max'])
        prop_dict['Phases'] = phase_arr
    
    return (prop_dict,spline_tab)

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


def make_plot(curve_tab,spline_tab,prop_dict,pp,agg_tab=None,errorbars=False,flux=False,plot_clean_num=0,rast=False
              ,show_sigma=False):
    '''
    Plot curves if requested
    
    Arguments:
        curve_tab: The lightcurve Table
        spline_tab: The spline Table
        prop_dict: Dictionary of spline properties
        pp: PDF pointer
    Keywords:
        agg_tab: The aggregate points Table
        flux: The points a in flux units not magnitudes
        errorbars: Whether to plot errorbars on data
        plot_clean_num: Number of highest and lowest y value points to remove, to make the plot look cleaner.
                        Does not affect any calculation.
        rast: Rasterize the plots because there are too many points
        Show_sigma: Display the points that were removed due to sigma-clipping using purple x
        
    Returns:
        None
    '''
    #Deal with large number of points
    if len(curve_tab['phase']) > 200:
        alpha = 0.2
        psize = 0.5
        ptype = '.'
    else:
        alpha = 1
        psize = 2
        ptype = 'o'    
        
    ccurve_tab = curve_tab[curve_tab['clean_idx']]
        
    if type(agg_tab) == type(None):
        #Make plots without aggregated points
        
        if errorbars:
            plt.errorbar(ccurve_tab['phase_fixed'],ccurve_tab['mag'],yerr=ccurve_tab['err'],
                         marker=ptype,ls='None',lw=.75,ms=psize,alpha=alpha,zorder=1,color='C0',rasterized=rast)
            plt.errorbar(ccurve_tab['phase_fixed']+1,ccurve_tab['mag'],yerr=ccurve_tab['err'],
                         marker=ptype,ls='None',lw=.75,ms=psize,alpha=alpha,zorder=1,color='C0',rasterized=rast)
        else:
            plt.plot(ccurve_tab['phase_fixed'],ccurve_tab['mag'],ptype,ms=psize,alpha=alpha,
                     zorder=1,color='C0',rasterized=rast)
            plt.plot(ccurve_tab['phase_fixed']+1,ccurve_tab['mag'],ptype,ms=psize,alpha=alpha,
                     zorder=1,color='C0',rasterized=rast)
    else:
        mask = ccurve_tab['mask']
        bmask = np.logical_not(mask)

        if errorbars:
            plt.errorbar(ccurve_tab['phase_fixed'][mask],ccurve_tab['mag'][mask],yerr=ccurve_tab['err'][mask],
                         marker='.',ls='None',lw=.75,ms=1.5,alpha=alpha,zorder=1,color='C0',rasterized=rast)
            plt.errorbar(ccurve_tab['phase_fixed'][mask]+1,ccurve_tab['mag'][mask],yerr=ccurve_tab['err'][mask],
                         marker='.',ls='None',lw=.75,ms=1.5,alpha=alpha,zorder=1,color='C0',rasterized=rast)
            if show_sigma:
                plt.errorbar(ccurve_tab['phase_fixed'][bmask],ccurve_tab['mag'][bmask],yerr=ccurve_tab['err'][bmask],
                         marker='x',ls='None',lw=.75,ms=3,zorder=1,color='purple',rasterized=rast)
                plt.errorbar(ccurve_tab['phase_fixed'][bmask]+1,ccurve_tab['mag'][bmask],yerr=ccurve_tab['err'][bmask],
                         marker='x',ls='None',lw=.75,ms=3,zorder=1,color='purple',rasterized=rast)
                
        else:
            plt.plot(ccurve_tab['phase_fixed'][mask],ccurve_tab['mag'][mask],ptype,ms=psize,alpha=alpha,
                     zorder=1,color='C0',rasterized=rast)
            plt.plot(ccurve_tab['phase_fixed'][mask]+1,ccurve_tab['mag'][mask],ptype,ms=psize,alpha=alpha,
                     zorder=1,color='C0',rasterized=rast)
            if show_sigma:
                plt.plot(ccurve_tab['phase_fixed'][bmask],ccurve_tab['mag'][bmask],'x',ms=3,
                     zorder=1,color='purple',rasterized=rast)
                plt.plot(ccurve_tab['phase_fixed'][bmask]+1,ccurve_tab['mag'][bmask],'x',ms=3,
                     zorder=1,color='purple',rasterized=rast)
        
        plt.errorbar(agg_tab['phase_fixed'],agg_tab['mag'],yerr=agg_tab['err'],marker=ptype,
                     ls='None',lw=.75,ms=psize,zorder=2,color='C1',rasterized=rast)
        plt.errorbar(agg_tab['phase_fixed']+1,agg_tab['mag'],yerr=agg_tab['err'],marker=ptype,
                     ls='None',lw=.75,ms=psize,zorder=2,color='C1',rasterized=rast)
        
    
    plt.plot(spline_tab['phase'],spline_tab['mag'],'--',zorder=3,color='red')
    plt.xlabel('Phase')
    plt.axvline(1) #Epoch of Maximum
    plt.xlim(0,2)
    if flux:
        plt.ylabel('Flux')    
    else:
        plt.ylabel('Magnitude')
        plt.gca().invert_yaxis()
    plt.title('{} Period: {}\nAmp: {:.3f} Epoch_Max: {:6f}'.format(prop_dict['Starname'],prop_dict['Period'],
                                                       prop_dict['Amplitude'],prop_dict['Epoch_Max']))
    
    if prop_dict['Dates'] != None:
        for phase in prop_dict['Phases']:
            plt.axvline(phase,ls='--',c='k')
    
    if rast:
        pp.savefig(dpi=200)
    else:
        pp.savefig()

def read_curve(filename):
    mytable = Table()
    #Open file        
    try:
        date,mag,err = np.genfromtxt(filename,unpack=True)
    except:
        try:
            date,mag,err,phase = np.genfromtxt(filename,unpack=True)
        except:
            print("Cannot open: {}".format(filename) )
            return(1)

    mytable['date'] = date
    mytable['mag'] = mag
    mytable['err'] = err
    #mytable.read(filename,format='ascii',names=('jd','mag','err'))
    return mytable

def shift_phase(phase,phase_max):
    '''
    Shift a set of phases, so that the the maximum occurs at phase = 0
    
    Arguments:
        phase: Array of phases to be shifted
        phase_max: The phase of maxiumum in unshifted array
    Keywords:
        None
        
    Returns:
        A numpy array of shifted phases
    '''
    
    new_phase = phase - phase_max
    indice = (new_phase < 0)
    new_phase[indice] = new_phase[indice] + 1
    return new_phase


def write_log(base,prop_dict,flux=False,verb=False):
    #Open Logfile
    try:
        logname = base + ".slog"
        logfile = open(logname,'w')
    except IOError:
        print("{} could not be opened!".format(logname))

    #Write out header
    if flux:
        header = "#Name,Period,Flux_Max,Flux_Min,Amp,Ave(M),Ave(I),Epoch_Max,Epoch_Min,"
        header = header + "Factor,Method,Order,Npts,Sigmaclip"
    else:
        header = "#Name,Period,Mag_Max,Mag_Min,Amp,Ave(M),Ave(I),Epoch_Max,Epoch_Min,"
        header = header + "Factor,Method,Order,Npts,Sigmaclip"
    if prop_dict['Dates'] != None:
        header = header + ",Dates,Phases\n"
    else:
        header = header + "\n"
        
    logfile.write(header)
    if flux:
        data = "{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},{:.6f},{},{},{},{},{}".format(prop_dict['Starname'],
                                                        prop_dict['Period'],
                                                        prop_dict['Flux_Max'],prop_dict['Flux_Min'],
                                                        prop_dict['Amplitude'],prop_dict['Mag_Ave'],
                                                        prop_dict['Flux_Ave'], prop_dict['Epoch_Max'],
                                                        prop_dict['Epoch_Min'], prop_dict['Factor'],
                                                        prop_dict['Method'],prop_dict['Order'],
                                                        prop_dict['Npts'],prop_dict['Sigclip'])
    else:
        data = "{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},{:.6f},{},{},{},{},{}".format(prop_dict['Starname'],
                                                        prop_dict['Period'],
                                                        prop_dict['Mag_Max'],prop_dict['Mag_Min'],
                                                        prop_dict['Amplitude'],prop_dict['Mag_Ave'],
                                                        prop_dict['Flux_Ave'], prop_dict['Epoch_Max'],
                                                        prop_dict['Epoch_Min'], prop_dict['Factor'],
                                                        prop_dict['Method'],prop_dict['Order'],
                                                        prop_dict['Npts'],prop_dict['Sigclip'])
    
    
    
    if prop_dict['Dates'] != None:
        data = data + ",[{}],".format(prop_dict['Dates'].replace(',',';')) + np.array2string(prop_dict['Phases'],separator=';') + '\n'
    
        if verb:
            dates_arr = np.array(prop_dict['Dates'].split(','),dtype='float64')
            for i in range(len(dates_arr)):              
                print("{} {}".format(dates_arr[i],prop_dict['Phases'][i]))
        
    
    logfile.write(data)
    
    logfile.close()

def write_phase(base,curve_tab,agg_tab,flux=False):
    '''
    Helper function to write out phase file for the light curve and the aggregated curve if requested
    '''
    #Open Phase file
    try:
        filename = base + ".sp.phase"
        phfile = open(filename,'w')
    except IOError:
        print("{} could not be opened!".format(filename))
    
    
    #Write out header
    if flux:
        btype = 'Flux'
    else:
        btype = 'Mag'
    
    header = "#Date {} {}_err Phase Sigma_Clean Plot_Clean \n".format(btype,btype)
    
    phfile.write(header)
    
    for i in range(len(curve_tab)):
        phfile.write("{} {} {} {:.5f} {} {}\n".format(curve_tab['date'][i],curve_tab['mag'][i],curve_tab['err'][i]
                                              ,curve_tab['phase_fixed'][i],int(curve_tab['mask'][i])
                                              ,int(curve_tab['clean_idx'][i])))
    
    phfile.close()

    if type(agg_tab) != type(None):
    
        #Open Phase file
        try:
            filename = base + ".sp.aggphase"
            phfile = open(filename,'w')
        except IOError:
            print("{} could not be opened!".format(filename))
    
    
        #Write out header
        if flux:
            btype = 'Flux'
        else:
            btype = 'Mag'
    
        header = "#Phase {} {}_err \n".format(btype,btype)
    
        phfile.write(header)
    
        for i in range(len(agg_tab)):
            phfile.write("{:.5f} {:.5f} {:.5f} \n".format(agg_tab['phase_fixed'][i],agg_tab['mag'][i],agg_tab['err'][i]))
    
        phfile.close()


def write_spline(base,spline_tab):
    '''
    Helper function to write out spline file
    '''   
    
    #Open spline file
    try:
        filename = base + ".spl"
        splfile = open(filename,'w')
    except IOError:
        print("{} could not be opened!".format(filename))

    #Write out header
    header = "#Phase,Mag\n"
    
    splfile.write(header)
    
    for i in range(len(spline_tab)):
        splfile.write("{:.3f},{:.3f}\n".format(spline_tab['phase'][i],spline_tab['mag'][i]))
    
    splfile.close()


# -------------
# Main Function
# -------------
def spline_fit_main(filename,period,phfile=False,base=None,dates=None,errorbars=False,factor=1,flux=False,plot_clean_num=0,lower=None
                    ,method="Median",npts=10,order=3,plot=False,rast=False,ret_results=False,show_sigma=False
                    ,sigclip=None,upper=None,verb=False):
    #Set base filename
    if base == None:
        base = os.path.splitext(filename)[0]
    
    curve_tab = read_curve(filename)

    prop_dict = dict()
    prop_dict['Period'] = period
    prop_dict['Basename'] = base
    prop_dict['Starname'] = os.path.basename(base)
    prop_dict['Factor'] = factor
    prop_dict['Method'] = method
    prop_dict['Order'] = order
    prop_dict['Npts'] = npts
    prop_dict['Sigclip'] = sigclip
    prop_dict['Dates'] = dates

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
            curve_tab['phase'],curve_tab['mag'],curve_tab['err'],num=npts,method=method,sigclip=sigclip)
 
        #Fit Spline
        extend_num = int(np.floor(len(agg_tab['phase'])/10))
        myspline = get_spline(agg_tab['phase'],agg_tab['mag'], agg_tab['err']*factor,order=order,extend_num=extend_num)
        spline_phase = np.linspace(0,1,npts*10)
        (prop_dict,spline_tab) = calc_props(curve_tab, myspline, spline_phase, prop_dict,flux=flux,lower=lower,
                                            upper=upper,verb=verb)        
        
        #Create updated phases etc.
        adjust_curve(curve_tab,prop_dict,agg_tab=agg_tab,flux=flux,plot_clean_num=plot_clean_num)
        
        if plot:
            make_plot(curve_tab, spline_tab, prop_dict, pp, agg_tab=agg_tab, errorbars=errorbars,flux=flux,
                      plot_clean_num=plot_clean_num,rast=rast,show_sigma=show_sigma)
            
    else:
        #Fit Spline
        extend_num = int(np.floor(len(curve_tab['phase'])/10))
        myspline = get_spline(curve_tab['phase'],curve_tab['mag'], curve_tab['err'],order=order,extend_num=extend_num)
        spline_phase = np.linspace(0,1,len(curve_tab['phase'])*10)
        (prop_dict,spline_tab) = calc_props(curve_tab, myspline, spline_phase, prop_dict,flux=flux,lower=lower,
                                            upper=upper,verb=verb)
        
        #Create updated phases etc.
        adjust_curve(curve_tab,prop_dict,agg_tab=agg_tab,flux=flux,plot_clean_num=plot_clean_num)
        
        if plot:
            make_plot(curve_tab, spline_tab, prop_dict, pp, errorbars=errorbars, flux=flux,plot_clean_num=plot_clean_num,
                      rast=rast,show_sigma=show_sigma)
    
    if verb:        
        print("{} Amplitude: {:.3f} Epoch Maximum: {:.6f}".format(filename,
                                                                  prop_dict['Amplitude'],prop_dict['Epoch_Max']))
    
    write_log(base,prop_dict,flux=flux,verb=verb)
    write_spline(base,spline_tab)
    if phfile:
        write_phase(base,curve_tab,agg_tab,flux=flux)
    if plot:
        pp.close()
    if ret_results:
        return prop_dict

if __name__ == '__main__':
    #Check to make sure we have 1 argument
    parser = argparse.ArgumentParser(description='Fits a Spline to a lightcurve.')
    parser.add_argument('filename',help='3 column light curve file JD Magnitude Error')
    parser.add_argument('period',type=float,help='The period to phase the light curve.')
    parser.add_argument('-a', action='store_true',help="Leave the phased lightcurve")
    parser.add_argument('-b', default=None,metavar="BASENAME",help="Give a basename for the pdf and log files.")
    parser.add_argument('-d',type=str,default=None,metavar='DATE1,DATE2,etc'
                        ,help='Comma separated list of Date(s) to change into phases. (Default None)')
    parser.add_argument('-e', action='store_true',help="Show errorbars on Data")
    parser.add_argument('-f', default=1,type=float,metavar="FACTOR",help="Factor to tighten errorbars on aggregate 0 to 1 (Default 1)")
    parser.add_argument('-k', default=0,type=int,metavar="NUM",help="Number of highest and lowest points to remove to make plots look cleaner (Default 0)")
    parser.add_argument('-l',default=None,type=float,metavar='JD',help='Find the epoch of minimum closest to this JD.')
    parser.add_argument('-m', metavar='METHOD',default="None"
                        ,help="Method to aggregate points: Mean, Median, Weighted_Mean, or None (Default None (e.g. no binning)")
    parser.add_argument('-n', default=20,type=int,metavar="NUM",help="Number of aggregate points to break data into.(Default 20)")
    parser.add_argument('-o', default=3,type=int,metavar="ORDER",help="Spline order betwee 1 and 5 (Default 3)")
    parser.add_argument('-r', action='store_true',help="Rasterize plots")
    parser.add_argument('-s', default=None,type=float,metavar="SIGMA"
                        ,help="Sigmas used in sigma clipping. None for no sigma clipping (Default None)")
    parser.add_argument('-p', action='store_true',help="Generate plots")
    parser.add_argument('-u',default=None,type=float,metavar='JD',help='Find the epoch of maximum closest to this JD.')
    parser.add_argument('-v', action='store_true',help="Be Verbose")
    parser.add_argument('-w', action='store_true',help="Show Sigma Clipped points.")
    parser.add_argument('-x', action='store_true',help="Curves are in Flux not Magnitude units")
    
#Put this in a dictionary    
    args = vars(parser.parse_args())
    ret = spline_fit_main(args['filename'],args['period'],phfile=['a'],base=args['b'],dates=args['d'],errorbars=args['e']
                          ,factor=args['f'],flux=args['x'],plot_clean_num=args['k'],lower=args['l'],method=args['m']
                          ,npts=args['n'],order=args['o'],plot=args['p'],rast=args['r'],ret_results=False,sigclip=args['s']
                          ,show_sigma=args['w'],upper=args['u'],verb=args['v'])
    sys.exit(0)
    
##
#@mainpage
 #@copydetails  spline_fit
    
