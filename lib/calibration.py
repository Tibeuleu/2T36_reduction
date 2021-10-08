"""
Library function for raw images calibration.

Prototypes :
    - get_master_bias(name_template, calib_folder) -> master_bias_data
        Retrieve the master bias from a list or an already computed fits file.

    - get_master_dark(exptime, name_template, calib_folder) -> master_dark_data
        Retrieve the master dark for a specific exposure time from a list or an already computed fits file.

    - get_master_flat(filt, name_template, calib_folder) -> master_flat_data
        Retrieve the master flat for a specific filter from a list or an already computed fits file.

    - image_calibration(data_array, headers, calib_folder, bias_files, dark_files, flat_files) -> data_calibrated, headers_calibrated
        Calibrate the images in data_array with corresponding calibration files.
"""
import numpy as np
from copy import copy, deepcopy
from glob import glob
from astropy.io import fits
from astropy import wcs
import lib.fits as proj_fits


def get_master_bias(infiles=None, name_template="*Bias.fit", calib_folder=""):
    """
    Compute or retrieve previously computed master bias data. If none was
    previously saved, save to master_bias.fits file in the calib_folder.
    ----------
    Inputs:
    infiles : strlist, optionnal
        List of the fits file names to compute the master bias from.
        If None, will look for previously computed master bias, or compute it
        using a list obtained from the name_template argument.
        Default to None
    name_template : str, optional
        String that should be matching the fits files name from which the
        master bias should be computed.
        Default to "*Bias.fit"
    calib_folder : str, optional
        Relative or absolute path to the folder containing the calibration
        fits file.
    ----------
    Returns:
    master_bias_data : numpy.ndarray
        2D array of float containing the master bias data.
    """
    test_presence = glob("{0:s}master_bias.fits".format(calib_folder))
    if (len(test_presence)>=1.) and (infiles is None):
        with fits.open(test_presence[0]) as f:
            master_bias_data = f[0].data
    else:
        if infiles is None:
            infiles = []
            for file in glob("{0:s}{1:s}".format(calib_folder, name_template)):
                infiles.append(file[len(calib_folder):])
        data_array, headers = proj_fits.get_obs_data(infiles, data_folder=calib_folder, compute_flux=False)
        master_bias_data = data_array.mean(axis=0)

        # Save to fits for next time
        master_bias_header = headers[0].copy()
        master_bias_header.remove('OBJECT')
        master_bias_header['CCD-TEMP'] = np.mean([hdr['CCD-TEMP'] for hdr in headers])
        master_bias_header['IMAGETYP'] = "Master Bias"
        master_bias_header.add_history("Cal Master Bias, {0:d} inputs".format(data_array.shape[0]))
        hdu = fits.PrimaryHDU(data=master_bias_data, header=master_bias_header)
        hdul = fits.HDUList([hdu])
        hdul.writeto("{0:s}master_bias.fits".format(calib_folder))

    return master_bias_data

def get_master_dark(exptime, infiles=None, name_template="*Dark[0-9]*", calib_folder=""):
    """
    Compute or retrieve previously computed master dark data for a given
    exposure time. If none was previously saved, save to
    master_dark_[exptime]s.fits file in the calib_folder.
    ----------
    Inputs:
    exptime : float, int
        The exposure time in seconds of the darks calibration.
    infiles : strlist, optionnal
        List of the fits file names to compute the master bias from.
        If None, will look for previously computed master bias, or compute it
        using a list obtained from the name_template argument.
        Default to None
    name_template : str, optional
        String that should be matching the fits files name from which the
        master bias should be computed.
        Default to "*Dark[0-9]*"
    calib_folder : str, optional
        Relative or absolute path to the folder containing the calibration
        fits file.
    ----------
    Returns:
    master_dark_data : numpy.ndarray
        2D array of float containing the master dark data.
    """
    test_presence = glob("{0:s}master_dark_{1:d}s.fits".format(calib_folder,int(exptime)))
    if (len(test_presence)>=1.) and (infiles is None):
        with fits.open(test_presence[0]) as f:
            master_dark_data = f[0].data
    else:
        if infiles is None:
            infiles = []
            for file in glob("{0:s}{1:s}".format(calib_folder, name_template)):
                infiles.append(file[len(calib_folder):])
        data_array, headers = proj_fits.get_obs_data(infiles, data_folder=calib_folder, compute_flux=False)
        dark_exp = np.array([header['exptime'] for header in headers])
        if not(float(exptime) in np.unique(dark_exp)):
            raise ValueError("Requested exptime has not been observed in dark setup.\
                    Possible values : {}".format(np.unique(dark_exp)))
        mask = (dark_exp == float(exptime))
        master_dark_data = data_array[mask].mean(axis=0)

        # Save to fits for next time
        master_dark_header = headers[np.argmax(mask)].copy()
        master_dark_header.remove('OBJECT')
        master_dark_header['CCD-TEMP'] = np.mean([hdr['CCD-TEMP'] for hdr in headers])
        master_dark_header['IMAGETYP'] = "Master Dark"
        master_dark_header.add_history("Cal Master Dark {0:d}s, {1:d} inputs".format(int(exptime), data_array[mask].shape[0]))
        hdu = fits.PrimaryHDU(data=master_dark_data, header=master_dark_header)
        hdul = fits.HDUList([hdu])
        hdul.writeto("{0:s}master_dark_{1:d}s.fits".format(calib_folder,int(exptime)))

    return master_dark_data

def get_master_flat(filt, infiles=None, name_template="Flat-????_{}.fit", calib_folder=""):
    """
    Compute or retrieve previously computed master flat data for a given
    filter. If none was previously saved, save to master_flat_[filt].fits
    file in the calib_folder.
    ----------
    Inputs:
    filt : str
        Name of the filter for the flat calibration
    infiles : strlist, optionnal
        List of the fits file names to compute the master bias from.
        If None, will look for previously computed master bias, or compute it
        using a list obtained from the name_template argument.
        Default to None
    name_template : str, optional
        String that should be matching the fits files name from which the
        master bias should be computed.
        Default to "Flat-????_[filt].fit"
    calib_folder : str, optional
        Relative or absolute path to the folder containing the calibration
        fits file.
    ----------
    Returns:
    master_flat_data : numpy.ndarray
        2D array of float containing the master flat data.
    """
    if filt.lower() in ['h', 'halpha', 'h_alpha', 'ha', 'h_a']:
        filt = 'H'
    elif filt.lower() in ['o', 'oiii', 'o3', 'o_iii', 'o_3']:
        filt = 'O'

    test_presence = glob("{0:s}master_flat_{1:s}.fits".format(calib_folder,filt))
    if (len(test_presence)>=1.) and (infiles is None):
        with fits.open(test_presence[0]) as f:
            master_flat_data = f[0].data
    else:
        if infiles is None:
            name_template = name_template.format(filt)
            infiles = []
            for file in glob("{0:s}{1:s}".format(calib_folder,name_template)):
                infiles.append(file[len(calib_folder):])
        data_array, headers = proj_fits.get_obs_data(infiles, data_folder=calib_folder, compute_flux=False)
        master_flat_data = data_array.mean(axis=0)

        # Save to fits for next time
        master_flat_header = headers[0].copy()
        master_flat_header.remove('OBJECT')
        master_flat_header['CCD-TEMP'] = np.mean([hdr['CCD-TEMP'] for hdr in headers])
        master_flat_header['IMAGETYP'] = "Master Flat"
        master_flat_header.add_history("Cal Master Flat {0:s}, {1:d} inputs".format(filt, data_array.shape[0]))
        hdu = fits.PrimaryHDU(data=master_flat_data, header=master_flat_header)
        hdul = fits.HDUList([hdu])
        hdul.writeto("{0:s}master_flat_{1:s}.fits".format(calib_folder, filt))

    return master_flat_data

def image_calibration(data_array, headers, calib_folder="", bias_files=None,
        dark_files=None, flat_files=None):
    """
    Calibrate a set of images on bias, dark currents and flatfield given the
    required calibration files.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats) containing the data to calibrate.
    headers : header list
        List of headers objects corresponding to each image in data_array.
    calib_folder : str, optional
        Relative or absolute path to the folder containing the calibration
        fits file.
    bias_files, dark_files, flat_files : strlist, optionnal
        List of the fits file names to compute the master bias/dark/flat from.
        If None, will look for previously computed master file, or compute it
        using a list obtained from a default name_template argument.
        Default to None.
    ----------
    Returns:
    data_calibrated : numpy.ndarray
        Array of images (2D floats) containing the calibrated data.
    headers_calibrated : header list
        List of headers objects corresponding to each image in data_calibrated.
    """

    filt_list, exp_list = [], []
    for hdr in headers:
        filt_list.append(hdr['filter'])
        exp_list.append(hdr['exptime'])

    # Get available dark exposure time for each image
    dark_exp = np.array([30., 60., 240., 480., 720., 1200.])
    for i in range(len(exp_list)):
        j=0
        while (j < len(dark_exp)) and (exp_list[i] > dark_exp[j]):
            j += 1
        if j == len(dark_exp):
            print("Warning : exposure time longer than longest dark")
            j -= 1
        exp_list[i] = dark_exp[j]

    master_bias = get_master_bias(infiles=bias_files, calib_folder=calib_folder)
    master_dark = dict([(time, get_master_dark(time, infiles=dark_files, calib_folder=calib_folder)) for time in np.unique(exp_list)])
    master_flat = dict([(filt, get_master_flat(filt, infiles=flat_files, calib_folder=calib_folder)) for filt in np.unique(filt_list)])

    data_calibrated = np.zeros(data_array.shape)
    headers_calibrated = deepcopy(headers)
    for i,data in enumerate(data_array):
        filt = filt_list[i]
        time = exp_list[i]
        flatfield = master_flat[filt]-master_bias-master_dark[time]
        flatfield = flatfield/flatfield.mean()

        data_calibrated[i] = (data-master_bias-master_dark[time])/flatfield
        headers_calibrated[i].add_history("Calibration using bias, dark and flatfield done.")

    return data_calibrated, headers_calibrated
