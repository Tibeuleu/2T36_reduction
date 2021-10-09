#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
Main script where are progressively added the steps for the FOC pipeline reduction.
"""

#Project libraries
from time import time
import sys
from glob import glob
import numpy as np
import copy
import lib.fits as proj_fits        #Functions to handle fits files
import lib.calibration as proj_cal  #Functions used in calibration pipeline
import lib.reduction as proj_red    #Functions used in reduction pipeline
import lib.plots as proj_plots      #Functions for plotting data


def main():
    ##### User inputs
    ## Input and output locations
    globals()['target'] = "NGC7662"
    globals()['filters'] = ['I', 'B', 'V', 'R', 'H', 'O']
    globals()['filters_display'] = [['R', 'V', 'B']]
    globals()['data_folder'] = "../20210923/"
    globals()['calib_folder'] = "../Calib/"
    globals()['plots_folder'] = "../plots/{0:s}/".format(target)
    globals()['output_folder'] = "../reduced/"
    infiles = []
    for file in glob("{0:s}{1:s}*-????_?.fit".format(data_folder, target)):
        infiles.append(file[len(data_folder):])

    ## Image reduction
    # Deconvolution
    deconvolve = False
    if deconvolve:
        psf = 'gaussian'  #Can be user-defined as well
        psf_FWHM = 1
        psf_scale = 'pixel'
        psf_shape=(9,9)
        iterations = 10
    # Cropping
    display_crop = True
    # Error estimation
    error_sub_shape = (50,50)
    display_error = True
    # Data binning
    rebin = False
    if rebin:
        pxsize = 2
        px_scale = 'pixel'         #pixel or arcsec
        rebin_operation = 'sum'     #sum or average
    # Alignement
    align_center = 'image'          #If None will align image to image center
    display_data = True
    # Smoothing
    smoothing_function = 'gaussian'  #gaussian_after, gaussian or combine
    smoothing_FWHM = 1           #If None, no smoothing is done
    smoothing_scale = 'pixel'      #pixel or arcsec
    # Image output
    figname = target                #target/intrument name
    figtype = '_reduced_smoothed'            #additionnal informations

    ##### Pipeline start
    ## Step 1:
    # Get data from fits files and translate to flux in photons count.
    print("Importing and calibrating data.")
    t0 = time()

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

    # Calibrate data using instrument calibration files and normalise obtained images.
    for i in range(len(infiles)):
        headers[i]['filename'] = infiles[i]
    data_array, headers = proj_cal.image_calibration(data_array, headers, calib_folder=calib_folder)

    center = np.ceil(np.array(data_array.shape[1:])/2.+np.array([25,50])).astype(int)
    extent = np.ceil((np.array(data_array.shape[1:]))/8.).astype(int)
    data_array = data_array[:,center[0]-extent[0]:center[0]+extent[0],center[1]-extent[1]:center[1]+extent[1]]

    # Get error and remove background
    data_array, error_array = proj_red.get_error(data_array, headers=headers, sub_shape=error_sub_shape, display=display_error, savename=figname+"_errors", plots_folder=plots_folder)

    # Sort images by filters
    filter_bool, filter_data, filter_error, filter_headers = dict(), dict(), dict(), dict()
    filter_data_array, filter_error_array, filter_headers_array = [], [], []

    t1 = time()
    print("Elapsed time {0:.2f} seconds.".format(t1-t0))
    print("Reducing and aligning images by filter.")

    for filt in filters:
        filter_bool[filt] = np.array([hdr['filter']==filt for hdr in headers])
        if np.max(filter_bool[filt]):
            filter_data[filt] = data_array[filter_bool[filt]]
            filter_error[filt] = error_array[filter_bool[filt]]
            filter_headers[filt] = [hdr for hdr in headers if hdr['filter']==filt]

            # Deconvolve data using Richardson-Lucy iterative algorithm with a gaussian PSF of given FWHM.
            if deconvolve:
                filter_data[filt] = proj_red.deconvolve_array(filter_data[filt], filter_headers[filt], psf=psf, FWHM=psf_FWHM, scale=psf_scale, shape=psf_shape, iterations=iterations)

            # Estimate error from data background, estimated from sub-image of desired sub_shape.
            filter_data[filt], filter_error[filt] = proj_red.get_error(filter_data[filt], headers=filter_headers[filt], sub_shape=error_sub_shape)

            # Rebin data to desired pixel size.
            Dxy = np.array([np.min([header['xbinning'],header['ybinning']]) for header in headers])
            if rebin:
                filter_data[filt], filter_error[filt], filter_headers[filt], Dxy = proj_red.rebin_array(filter_data[filt], filter_error[filt], filter_headers[filt], pxsize=pxsize, scale=px_scale, operation=rebin_operation)

            # Align and rescale images with oversampling.
            filter_data[filt], filter_error[filt], data_mask = proj_red.align_data(filter_data[filt], filter_headers[filt], filter_error[filt], upsample_factor=int(Dxy.min()), ref_data=filter_data[filt][-1], ref_center=align_center, return_shifts=False)
            proj_plots.plot_obs(filter_data[filt], filter_headers[filt], vmin=filter_data[filt].min(), vmax=filter_data[filt].max())
            filter_data_array += [value for value in filter_data[filt]]
            filter_error_array += [value for value in filter_error[filt]]
            filter_headers_array += [value for value in filter_headers[filt]]

    t2 = time()
    print("Elapsed time {0:.2f} seconds.".format(t2-t1))
    print("Combine filtered images and align different filters.")

    data_array = np.array(filter_data_array)
    error_array = np.array(filter_error_array)
    headers = filter_headers_array

    ## Step 2:
    # Compute avg on each image and align all images
    data_array, error_array, headers = proj_red.filter_avg(data_array, error_array, headers, data_mask, filters=filters, FWHM=smoothing_FWHM, scale=smoothing_scale, smoothing=smoothing_function)
    data_array, error_array, data_mask = proj_red.align_data(data_array, headers, error_array, upsample_factor=int(Dxy.min()), ref_data=data_array[-1], ref_center=align_center, return_shifts=False)

    #Plot array for checking output
    if display_data:
        proj_plots.plot_obs(data_array, headers, vmin=data_array.min(), vmax=data_array.max(), savename=figname+"_center_"+align_center, plots_folder=plots_folder)

    t3 = time()
    print("Elapsed time {0:.2f} seconds.".format(t3-t2))
    print("Save to fits and display images.")

    ## Step 3:
    # Save image to FITS.
    hdul = proj_fits.save_fits(data_array, error_array, headers, filters=filters, filename=figname+figtype, data_folder=output_folder, return_hdul=True)

    # Display resulting image
    for filts in filters_display:
        proj_plots.display_RGB_reduced(hdul, filters=filts, savename=figname+figtype, plots_folder=plots_folder)

    t4 = time()
    print("Total elapsed time {0:.2f} seconds.".format(t4-t0))
    return 0

if __name__ == "__main__":
    sys.exit(main())
