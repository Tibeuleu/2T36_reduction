"""
Library function for simplified fits handling.

prototypes :
    - get_obs_data(infiles, data_folder) -> data_array, headers
        Extract the observationnal data from fits files

    - save_fits(array, error, headers, filters, filename, data_folder, return_hdul) -> ( HDUL_data )
        Save computed filter parameters to a single fits file (and return HDUList)
"""

import numpy as np
from copy import deepcopy
from astropy.io import fits
from astropy import wcs


def get_obs_data(infiles, data_folder="", compute_flux=False):
    """
    Extract the observationnal data from the given fits files.
    ----------
    Inputs:
    infiles : strlist
        List of the fits file names to be added to the observation set.
    data_folder : str, optional
        Relative or absolute path to the folder containing the data.
    compute_flux : boolean, optional
        If True, return data_array will contain flux information, assuming
        raw data are counts and header have keywork EXPTIME and PHOTFLAM.
        Default to False.
    ----------
    Returns:
    data_array : numpy.ndarray
        Array of images (2D floats) containing the observation data.
    headers : header list
        List of headers objects corresponding to each image in data_array.
    """
    data_array, headers = [], []
    for i in range(len(infiles)):
        with fits.open(data_folder+infiles[i]) as f:
            headers.append(f[0].header)
            data_array.append(f[0].data)
    data_array = np.array(data_array)

    # Prevent negative count value in imported data
    for i in range(len(data_array)):
        data_array[i][data_array[i] < 0.] = 0.

    if compute_flux:
        for i in range(len(infiles)):
            # Compute the flux in counts/sec
            data_array[i] = data_array[i]/headers[i]['EXPTIME']

    return data_array, headers


def save_fits(array, error, headers, filters, filename, data_folder="",
        return_hdul=False):
    """
    Save computed filter parameters to a single fits file, updating header
    accordingly.
    ----------
    Inputs:
    array, error : numpy.ndarray
        Images (2D float arrays) containing the computed image for each filter
        and associated errors.
    headers : header list
        Headers of reference for each filter some keywords will be copied from
        (CDELT, INSTRUME, TARGNAME, EXPTOT).
    filter : strlist
        List containing the names of the available filters.
    filename : str
        Name that will be given to the file on writing (will appear in header).
    data_folder : str, optional
        Relative or absolute path to the folder the fits file will be saved to.
        Defaults to current folder.
    return_hdul : boolean, optional
        If True, the function will return the created HDUList from the
        input arrays.
        Defaults to False.
    ----------
    Return:
    hdul : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I filtered image in the PrimaryHDU, then B, V, R,
        Halpha, OIII in this order. Headers have been updated to relevant
        informations (data_type).
        Only returned if return_hdul is True.
    """
    header = deepcopy(headers[0])
    image_shape = array[0].shape
    # Sort images by filter : can be I, B, V, R, H, O
    filt_bool, filt_data, filt_error, filt_headers = dict(), dict(), dict(), dict()
    for filt in filters:
        filt_bool[filt] = np.array([hdr['filter']==filt for hdr in headers])
        if np.max(filt_bool[filt])==1.:
            filt_data[filt] = array[np.argmax(filt_bool[filt])]
            filt_error[filt] = error[np.argmax(filt_bool[filt])]
            filt_headers[filt] = [hdr for hdr in headers if hdr['filter']==filt][0]
        else:
            filt_data[filt] = np.zeros(image_shape)
            filt_error[filt] = np.zeros(image_shape)
            filt_headers[filt] = deepcopy(header)
            filt_headers[filt]['filter'] = filt
            filt_headers[filt]['exptime'] = 0.

    #Create new WCS object given the modified images
    exp_tot = np.array([hdr['exptime'] for hdr in headers]).sum()
    new_wcs = wcs.WCS(header).deepcopy()
    if new_wcs.wcs.has_cd():
        del new_wcs.wcs.cd
        keys = list(new_wcs.to_header().keys())+['CD1_1','CD1_2','CD2_1','CD2_2']
        for key in keys:
            header.remove(key, ignore_missing=True)
        new_wcs.wcs.cdelt = 3600.*np.sqrt(np.sum(new_wcs.wcs.get_pc()**2,axis=1))
    if (new_wcs.wcs.cdelt == np.array([1., 1.])).all():
        # Update WCS with relevant information
        aperture = header['aptdia']    # Telescope aperture in mm
        focal_len = header['focallen']
        px_dim = np.array([header['xpixsz'], header['xpixsz']])   # Pixel dimension in µm
        new_wcs.wcs.cdelt = 206.3*px_dim/(focal_len*aperture)
    new_wcs.wcs.crpix = [image_shape[0]/2, image_shape[1]/2]

    wcs_header = new_wcs.to_header()
    for key in wcs_header.keys():
        header[key] = wcs_header[key]

    header['exptot'] = (exp_tot, 'Total exposition time of the set of images')
    header['date-obs'] = str(np.min([np.datetime64(head['date-obs']) for head in headers]))
    header['jd'] = np.min([head['jd'] for head in headers])
    header['jd-helio'] = np.median([head['jd-helio'] for head in headers])
    header['airmass'] = np.mean([head['airmass'] for head in headers])

    header['filename'] = (filename, 'Original filename')
    header['filter'] = ('I','Filter used for image stored in the HDU')

    #Create HDUList object
    hdul = fits.HDUList([])

    #Add I filter as PrimaryHDU
    primary_hdu = fits.PrimaryHDU(data=filt_data['I'], header=header)
    hdul.append(primary_hdu)

    #Add B, V, R, Halpha, OIII to the HDUList
    for filt in ['B', 'V', 'R', 'H', 'O']:
        hdu_header = filt_headers[filt].copy()
        hdu_header['filter'] = filt
        hdu = fits.ImageHDU(data=filt_data[filt],header=hdu_header)
        hdul.append(hdu)
    #Add I_err, B_err, V_err, R_err, Halpha_err, OIII_err to the HDUList
    for filt in ['I','B', 'V', 'R', 'H', 'O']:
        hdu_header = filt_headers[filt].copy()
        hdu_header['filter'] = filt+"_err"
        hdu = fits.ImageHDU(data=filt_error[filt],header=hdu_header)
        hdul.append(hdu)


    #Save fits file to designated filepath
    hdul.writeto(data_folder+filename+".fits", overwrite=True)

    if return_hdul:
        return hdul
    else:
        return 0
