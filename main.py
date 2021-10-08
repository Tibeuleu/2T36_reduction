#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
Main script where are progressively added the steps for the FOC pipeline reduction.
"""

#Project libraries
import sys
from glob import glob
import numpy as np
import copy
import lib.fits as proj_fits        #Functions to handle fits files
import lib.calibration as proj_cal  #Functions used in calibration pipeline
import lib.reduction as proj_red    #Functions used in reduction pipeline
import lib.plots as proj_plots      #Functions for plotting data
from lib.convex_hull import image_hull


def main():
    ##### User inputs
    ## Input and output locations
    globals()['target'] = "NGC7662"
    globals()['filters'] = ['I', 'B', 'V', 'R', 'H', 'O']
    globals()['filters_display'] = [['B', 'V', 'R']]
    globals()['data_folder'] = "../20210923/"
    globals()['calib_folder'] = "../Calib/"
    globals()['plots_folder'] = "../plots/"
    globals()['output_folder'] = "../reduced/"
    infiles = []
    for file in glob("{0:s}{1:s}*-????_?.fit".format(data_folder, target)):
        infiles.append(file[len(data_folder):])

    ## Image reduction
    # Deconvolution
    deconvolve = False
    if deconvolve:
        psf = 'gaussian'  #Can be user-defined as well
        psf_FWHM = 0.10
        psf_scale = 'arcsec'
        psf_shape=(9,9)
        iterations = 10
    # Cropping
    display_crop = True
    # Error estimation
    error_sub_shape = (100,100)
    display_error = True
    # Data binning
    rebin = False
    if rebin:
        pxsize = 0.10
        px_scale = 'arcsec'         #pixel or arcsec
        rebin_operation = 'sum'     #sum or average
    # Alignement
    align_center = 'image'          #If None will align image to image center
    display_data = True
    # Smoothing
    smoothing_function = 'combine'  #gaussian_after, gaussian or combine
    smoothing_FWHM = None           #If None, no smoothing is done
    smoothing_scale = 'arcsec'      #pixel or arcsec
    # Rotation
    rotate = False                  #rotation to North convention can give erroneous results
    # Image output
    figname = target                #target/intrument name
    figtype = '_reduced'            #additionnal informations

    ##### Pipeline start
    ## Step 1:
    # Get data from fits files and translate to flux in photons count.
    data_array, headers = proj_fits.get_obs_data(infiles, data_folder=data_folder, compute_flux=False)
    for i,head in enumerate(headers):
        filt = head['filter']
        if not(filt in filters):
            if filt.lower() in ['h', 'ha', 'h_a', 'h-a', 'halpha', 'h_alpha', 'h-alpha']:
                headers[i]['filter'] = 'H'
            elif filt.lower() in ['o', 'o3', 'o_3', 'o-3', 'oiii', 'o_iii', 'o-iii']:
                headers[i]['filter'] = 'O'
            else:
                raise ValueError("Filter {0:s} for {1:s} not valid.\
                        Possible value : {2}".format(filt, infiles[i], filters))
    for data in data_array:
        if (data < 0.).any():
            print("ETAPE 1 : ", data)
    # Crop data to remove outside blank margins.
    #data_array, error_array = proj_red.crop_array(data_array, headers, step=5, null_val=0., inside=True, display=display_crop, savename=figname, plots_folder=plots_folder)
    for data in data_array:
        if (data < 0.).any():
            print("ETAPE 2 : ", data)
    # Calibrate data using instrument calibration files and normalise obtained images.
    data_array, headers = proj_cal.image_calibration(data_array, headers, calib_folder=calib_folder)
    data_array = data_array/data_array.mean(axis=0)
    # Deconvolve data using Richardson-Lucy iterative algorithm with a gaussian PSF of given FWHM.
    if deconvolve:
        data_array = proj_red.deconvolve_array(data_array, headers, psf=psf, FWHM=psf_FWHM, scale=psf_scale, shape=psf_shape, iterations=iterations)
    # Estimate error from data background, estimated from sub-image of desired sub_shape.
    data_array, error_array = proj_red.get_error(data_array, headers=headers, sub_shape=error_sub_shape, display=display_error, savename=figname+"_errors", plots_folder=plots_folder)
    for data in data_array:
        if (data < 0.).any():
            print("ETAPE 3 : ", data)
    # Rebin data to desired pixel size.
    Dxy = np.array([np.min([header['xbinning'],header['ybinning']]) for header in headers])
    if rebin:
        data_array, error_array, headers, Dxy = proj_red.rebin_array(data_array, error_array, headers, pxsize=pxsize, scale=px_scale, operation=rebin_operation)
    for data in data_array:
        if (data < 0.).any():
            print("ETAPE 4 : ", data)
    # Align and rescale images with oversampling.
    data_array, error_array, data_mask = proj_red.align_data(data_array, headers, error_array, upsample_factor=int(Dxy.min()), ref_center=align_center, return_shifts=False)
    data_array, error_array = proj_red.crop_array(data_array, headers, step=5, null_val=0., inside=True, display=display_crop, savename=figname, plots_folder=plots_folder)
    for data in data_array:
        if (data < 0.).any():
            print("ETAPE 5 : ", data)

    #Plot array for checking output
    if display_data:
        proj_plots.plot_obs(data_array, headers, vmin=data_array.min(), vmax=data_array.max(), savename=figname+"_center_"+align_center, plots_folder=plots_folder)

    ## Step 2:
    # Compute avg on each image
    filter_array, filter_error, filter_headers = proj_red.filter_avg(data_array, error_array, headers, data_mask, filters=filters, FWHM=smoothing_FWHM, scale=smoothing_scale, smoothing=smoothing_function)

    ## Step 3:
    # Crop to desired Region of Interest (ROI)

    ## Step 5:
    # Save image to FITS.
    hdul = proj_fits.save_fits(filter_array, filter_error, filter_headers, filters=filters, filename=figname+figtype, data_folder=output_folder, return_hdul=True)

    # Display resulting image
    for filts in filters_display:
        proj_plots.display_image_RGB(hdul, filters=filts, savename=figname+figtype, plots_folder=plots_folder)

    return 0

if __name__ == "__main__":
    sys.exit(main())
