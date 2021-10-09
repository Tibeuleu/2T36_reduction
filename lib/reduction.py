"""
Library function computing various steps of the reduction pipeline.

prototypes :
    - bin_ndarray(ndarray, new_shape, operation) -> ndarray
        Bins an ndarray to new_shape.

    - crop_array(data_array, error_array, step, null_val, inside) -> crop_data_array, crop_error_array
        Homogeneously crop out null edges off a data array.

    - deconvolve_array(data_array, psf, FWHM, iterations) -> deconvolved_data_array
        Homogeneously deconvolve a data array using Richardson-Lucy iterative algorithm

    - get_error(data_array, headers, sub_shape, display, savename, plots_folder) -> data_array, error_array
        Compute the error (noise) on each image of the input array.

    - rebin_array(data_array, error_array, headers, pxsize, scale, operation) -> rebinned_data, rebinned_error, rebinned_headers, Dxy
        Homegeneously rebin a data array given a target pixel size in scale units.

    - align_data(data_array, error_array, upsample_factor, ref_data, ref_center, return_shifts) -> data_array, error_array, masked_array (, shifts, errors)
        Align data_array on ref_data by cross-correlation.

    - smooth_data(data_array, error_array, FWHM, scale, smoothing) -> smoothed_array
        Smooth data by convoluting with a gaussian or by combining weighted images

    - filter_avg(data_array, error_array, headers, data_mask, filters, FWHM, scale, smoothing) -> filter_array, filter_error, filter_headers
        Average images in data_array on each used filter.
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime
from scipy.ndimage import shift as sc_shift
from astropy.wcs import WCS
from lib.deconvolve import deconvolve_im, gaussian_psf
from lib.convex_hull import image_hull
from lib.plots import plot_obs
from lib.cross_correlation import phase_cross_correlation


# Useful tabulated values

def get_row_compressor(old_dimension, new_dimension, operation='sum'):
    """
    Return the matrix that allows to compress an array from an old dimension of
    rows to a new dimension of rows, can be done by summing the original
    components or averaging them.
    ----------
    Inputs:
    old_dimension, new_dimension : int
        Number of rows in the original and target matrices.
    operation : str, optional
        Set the way original components of the matrix are put together
        between summing ('sum') and averaging ('average', 'avg', 'mean') them.
        Defaults to 'sum'.
    ----------
    Returns:
    dim_compressor : numpy.ndarray
        2D matrix allowing row compression by matrix multiplication to the left
        of the original matrix.
    """
    dim_compressor = np.zeros((new_dimension, old_dimension))
    bin_size = float(old_dimension) / new_dimension
    next_bin_break = bin_size
    which_row, which_column = 0, 0

    while which_row < dim_compressor.shape[0] and which_column < dim_compressor.shape[1]:
        if round(next_bin_break - which_column, 10) >= 1:
            dim_compressor[which_row, which_column] = 1
            which_column += 1
        elif next_bin_break == which_column:

            which_row += 1
            next_bin_break += bin_size
        else:
            partial_credit = next_bin_break - which_column
            dim_compressor[which_row, which_column] = partial_credit
            which_row += 1
            dim_compressor[which_row, which_column] = 1 - partial_credit
            which_column += 1
            next_bin_break += bin_size

    if operation.lower() in ["mean", "average", "avg"]:
        dim_compressor /= bin_size

    return dim_compressor


def get_column_compressor(old_dimension, new_dimension, operation='sum'):
    """
    Return the matrix that allows to compress an array from an old dimension of
    columns to a new dimension of columns, can be done by summing the original
    components or averaging them.
    ----------
    Inputs:
    old_dimension, new_dimension : int
        Number of columns in the original and target matrices.
    operation : str, optional
        Set the way original components of the matrix are put together
        between summing ('sum') and averaging ('average', 'avg', 'mean') them.
        Defaults to 'sum'.
    ----------
    Returns:
    dim_compressor : numpy.ndarray
        2D matrix allowing columns compression by matrix multiplication to the
        right of the original matrix.
    """
    return get_row_compressor(old_dimension, new_dimension, operation).transpose()


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    if (np.array(ndarray.shape)%np.array(new_shape) == np.array([0.,0.])).all():
        compression_pairs = [(d, c//d) for d,c in zip(new_shape, ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)

        for i in range(len(new_shape)):
            if operation.lower() == "sum":
                ndarray = ndarray.sum(-1*(i+1))
            elif operation.lower() in ["mean", "average", "avg"]:
                ndarray = ndarray.mean(-1*(i+1))
    else:
        row_comp = np.mat(get_row_compressor(ndarray.shape[0], new_shape[0], operation))
        col_comp = np.mat(get_column_compressor(ndarray.shape[1], new_shape[1], operation))
        ndarray = np.array(row_comp * np.mat(ndarray) * col_comp)

    return ndarray


def crop_array(data_array, headers, error_array=None, step=5, null_val=None,
        inside=False, display=False, savename=None, plots_folder=""):
    """
    Homogeneously crop an array: all contained images will have the same shape.
    'inside' parameter will decide how much should be cropped.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the observation data (2D float arrays) to
        homogeneously crop.
    headers : header list
        Headers associated with the images in data_array.
    error_array : numpy.ndarray, optional
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
        If None, will be initialized to zeros.
        Defaults to None.
    step : int, optional
        For images with straight edges, not all lines and columns need to be
        browsed in order to have a good convex hull. Step value determine
        how many row/columns can be jumped. With step=2 every other line will
        be browsed.
        Defaults to 5.
    null_val : float or array-like, optional
        Pixel value determining the threshold for what is considered 'outside'
        the image. All border pixels below this value will be taken out.
        If None, will be put to 75% of the mean value of the associated error
        array.
        Defaults to None.
    inside : boolean, optional
        If True, the cropped image will be the maximum rectangle included
        inside the image. If False, the cropped image will be the minimum
        rectangle in which the whole image is included.
        Defaults to False.
    display : boolean, optional
        If True, data_array will be displayed with a rectangle around the
        sub-image selected for region of interest.
        Defaults to False.
    ----------
    Returns:
    cropped_array : numpy.ndarray
        Array containing the observationnal data homogeneously cropped.
    """
    if error_array is None:
        error_array = np.zeros(data_array.shape)
    if null_val is None:
        null_val = [1.00*error.mean() for error in error_array]
    elif type(null_val) is float:
        null_val = [null_val,]*error_array.shape[0]

    vertex = np.zeros((data_array.shape[0],4),dtype=int)
    for i,image in enumerate(data_array):
        vertex[i] = image_hull(image,step=step,null_val=null_val[i],inside=inside)
    v_array = np.zeros(4,dtype=int)
    if inside:
        v_array[0] = np.max(vertex[:,0]).astype(int)
        v_array[1] = np.min(vertex[:,1]).astype(int)
        v_array[2] = np.max(vertex[:,2]).astype(int)
        v_array[3] = np.min(vertex[:,3]).astype(int)
    else:
        v_array[0] = np.min(vertex[:,0]).astype(int)
        v_array[1] = np.max(vertex[:,1]).astype(int)
        v_array[2] = np.min(vertex[:,2]).astype(int)
        v_array[3] = np.max(vertex[:,3]).astype(int)

    new_shape = np.array([v_array[1]-v_array[0],v_array[3]-v_array[2]])
    rectangle = [v_array[2], v_array[0], new_shape[1], new_shape[0], 0., 'b']
    if display:
        fig, ax = plt.subplots()
        data = data_array[0]
        instr = headers[0]['telescop']
        exptime = headers[0]['exptime']
        filt = headers[0]['filter']
        #plots
        im = ax.imshow(data, vmin=data.min(), vmax=data.max(), origin='lower')
        x, y, width, height, angle, color = rectangle
        ax.add_patch(Rectangle((x, y),width,height,edgecolor=color,fill=False))
        #position of centroid
        ax.plot([data.shape[1]/2, data.shape[1]/2], [0,data.shape[0]-1], lw=1,
                color='black')
        ax.plot([0,data.shape[1]-1], [data.shape[0]/2, data.shape[0]/2], lw=1,
                color='black')
        ax.annotate(instr, color='white', fontsize=5,
                xy=(0.02, 0.95), xycoords='axes fraction')
        ax.annotate(filt, color='white', fontsize=10, xy=(0.02, 0.02),
                xycoords='axes fraction')
        ax.annotate(exptime, color='white', fontsize=5, xy=(0.80, 0.02),
                xycoords='axes fraction')
        ax.set(title="Location of cropped image.",
                xlabel='pixel offset',
                ylabel='pixel offset')

        fig.subplots_adjust(hspace=0, wspace=0, right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.75])
        fig.colorbar(im, cax=cbar_ax)

        if not(savename is None):
            fig.suptitle(savename+'_'+filt+'_crop_region')
            fig.savefig(plots_folder+savename+'_'+filt+'_crop_region.png',
                    bbox_inches='tight')
            plot_obs(data_array, headers, vmin=data_array.min(),
                    vmax=data_array.max(), rectangle=[rectangle,]*len(headers),
                    savename=savename+'_crop_region',plots_folder=plots_folder)
        plt.show()

    crop_array = np.zeros((data_array.shape[0],new_shape[0],new_shape[1]))
    crop_error_array = np.zeros((data_array.shape[0],new_shape[0],new_shape[1]))
    for i,image in enumerate(data_array):
        crop_array[i] = image[v_array[0]:v_array[1],v_array[2]:v_array[3]]
        crop_error_array[i] = error_array[i][v_array[0]:v_array[1],v_array[2]:v_array[3]]

    return crop_array, crop_error_array


def deconvolve_array(data_array, headers, psf='gaussian', FWHM=1., scale='px',
        shape=(9,9), iterations=20):
    """
    Homogeneously deconvolve a data array using Richardson-Lucy iterative algorithm.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the observation data (2D float arrays) to
        homogeneously deconvolve.
    headers : header list
        Headers associated with the images in data_array.
    psf : str or numpy.ndarray, optionnal
        String designing the type of desired Point Spread Function or array
        of dimension 2 corresponding to the weights of a PSF.
        Defaults to 'gaussian' type PSF.
    FWHM : float, optional
        Full Width at Half Maximum for desired PSF in 'scale units. Only used
        for relevant values of 'psf' variable.
        Defaults to 1.
    scale : str, optional
        Scale units for the FWHM of the PSF between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    shape : tuple, optional
        Shape of the kernel of the PSF. Must be of dimension 2. Only used for
        relevant values of 'psf' variable.
        Defaults to (9,9).
    iterations : int, optional
        Number of iterations of Richardson-Lucy deconvolution algorithm. Act as
        as a regulation of the process.
        Defaults to 20.
    ----------
    Returns:
    deconv_array : numpy.ndarray
        Array containing the deconvolved data (2D float arrays) using given
        point spread function.
    """
    # If chosen FWHM scale is 'arcsec', compute FWHM in pixel scale
    if scale.lower() in ['arcsec','arcseconds']:
        pxsize = np.zeros((data_array.shape[0],2))
        for i,header in enumerate(headers):
            # Get current pixel size
            w = WCS(header).deepcopy()
            if w.wcs.has_cd():
                del w.wcs.cd
                keys = list(w.to_header().keys())+['CD1_1','CD1_2','CD2_1','CD2_2']
                for key in keys:
                    header.remove(key, ignore_missing=True)
                w.wcs.cdelt = 3600.*np.sqrt(np.sum(w.wcs.get_pc()**2,axis=1))
            if (w.wcs.cdelt == np.array([1., 1.])).all():
                # Update WCS with relevant information
                aperture = header['aptdia']    # Telescope aperture in mm
                focallen = header['focallen']
                px_dim = np.array([header['xpixsz'], header['ypixsz']])   # Pixel dimension in µm
                w.wcs.cdelt = 206.3*px_dim/(focallen*aperture)
            header.update(w.to_header())
            pxsize[i] = np.round(w.wcs.cdelt,5)
        if (pxsize != pxsize[0]).any():
            raise ValueError("Not all images in array have same pixel size")
        FWHM /= pxsize[0].min()

    # Define Point-Spread-Function kernel
    if psf.lower() in ['gauss','gaussian']:
        kernel = gaussian_psf(FWHM=FWHM, shape=shape)
    elif (type(psf) == np.ndarray) and (len(psf.shape) == 2):
        kernel = psf
    else:
        raise ValueError("{} is not a valid value for 'psf'".format(psf))

    # Deconvolve images in the array using given PSF
    deconv_array = np.zeros(data_array.shape)
    for i,image in enumerate(data_array):
        deconv_array[i] = deconvolve_im(image, kernel, iterations=iterations,
                clip=True, filter_epsilon=None)

    return deconv_array


def get_error(data_array, headers=None, sub_shape=(15,15), display=False,
        savename=None, plots_folder="", return_background=False):
    """
    Look for sub-image of shape sub_shape that have the smallest integrated
    flux (no source assumption) and define the background on the image by the
    standard deviation on this sub-image.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to study (2D float arrays).
    headers : header list, optional
        Headers associated with the images in data_array. Will only be used if
        display is True.
        Defaults to None.
    sub_shape : tuple, optional
        Shape of the sub-image to look for. Must be odd.
        Defaults to (15,15).
    display : boolean, optional
        If True, data_array will be displayed with a rectangle around the
        sub-image selected for background computation.
        Defaults to False.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed). Only used if display is True.
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    return_background : boolean, optional
        If True, the pixel background value for each image in data_array is
        returned.
        Defaults to False.
    ----------
    Returns:
    data_array : numpy.ndarray
        Array containing the data to study minus the background.
    error_array : numpy.ndarray
        Array containing the background values associated to the images in
        data_array.
    background : numpy.ndarray
        Array containing the pixel background value for each image in
        data_array.
        Only returned if return_background is True.
    """
    # Crop out any null edges
    data = data_array#, error = crop_array(data_array, headers, step=5, null_val=0., inside=False)

    sub_shape = np.array(sub_shape, dtype=int)
    # Make sub_shape of odd values
    if not(np.all(sub_shape%2)):
        sub_shape += 1-sub_shape%2

    shape = np.array(data.shape)
    diff = (sub_shape-1).astype(int)
    temp = np.zeros((shape[0],shape[1]-diff[0],shape[2]-diff[1]))
    error_array = np.ones(data_array.shape)
    rectangle = []
    background = np.zeros((shape[0]))

    for i,image in enumerate(data):
        # Find the sub-image of smallest integrated flux (suppose no source)
        #sub-image dominated by background
        for r in range(temp.shape[1]):
            for c in range(temp.shape[2]):
                temp[i][r,c] = image[r:r+diff[0],c:c+diff[1]].sum()

    minima = np.unravel_index(np.argmin(temp.sum(axis=0)),temp.shape[1:])

    for i, image in enumerate(data):
        rectangle.append([minima[1], minima[0], sub_shape[1], sub_shape[0], 0., 'r'])
        # Compute error : root mean square of the background
        sub_image = image[minima[0]:minima[0]+sub_shape[0],minima[1]:minima[1]+sub_shape[1]]
        #error =  np.std(sub_image)    # Previously computed using standard deviation over the background
        error = np.sqrt(np.sum(sub_image**2)/sub_image.size)
        error_array[i] *= error

        # Quadratically add uncertainties in the "correction factors" (see Kishimoto 1999)
        #wavelength dependence of the polariser filters
        #estimated to less than 1%
        err_wav = data_array[i]*0.01
        #difference in PSFs through each polarizers
        #estimated to less than 3%
        err_psf = data_array[i]*0.03
        #flatfielding uncertainties
        #estimated to less than 3%
        err_flat = data_array[i]*0.03
        #error_array[i] = np.sqrt(error_array[i]**2 + err_wav**2 + err_psf**2 + err_flat**2)

        background[i] = sub_image.sum()
        data_array[i] = data_array[i] - sub_image.mean()
        data_array[i][data_array[i] <= 0.] = np.min(data_array[i][data_array[i] > 0.])
        if (data_array[i] < 0.).any():
            print(data_array[i])

    if display:

        date_time = np.array([headers[i]['date-obs'] for i in range(len(headers))])
        date_time = np.array([datetime.strptime(d,'%Y-%m-%dT%H:%M:%S')
            for d in date_time])
        filt = np.array([headers[i]['filter'] for i in range(len(headers))])
        dict_filt = {"R":'r', "V":'g', "B":'b', "I":'k', "O":'c', "H":'y'}
        c_filt = np.array([dict_filt[f] for f in filt])

        fig,ax = plt.subplots(figsize=(10,6), constrained_layout=True)
        for f in np.unique(filt):
            mask = [fil==f for fil in filt]
            ax.scatter(date_time[mask], background[mask], color=dict_filt[f],
                    label="Filter : {0:s}".format(f))
        ax.errorbar(date_time, background, yerr=error_array[:,0,0], fmt='+k',
                markersize=0, ecolor=c_filt)
        # Date handling
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Observation date and time")
        ax.set_ylabel(r"Count [$photons$]")
        ax.set_title("Background flux and error computed for each image")
        plt.legend()

        if not(savename is None):
            fig.suptitle(savename+"_background_flux")
            fig.savefig(plots_folder+savename+"_background_flux.png",
                    bbox_inches='tight')
            vmin = np.min(np.log10(data[data > 0.]))
            vmax = np.max(np.log10(data[data > 0.]))
            plot_obs(data, headers, vmin=data.min(), vmax=data.max(),
                    rectangle=rectangle,
                    savename=savename+"_background_location",
                    plots_folder=plots_folder)

        else:
            vmin = np.min(np.log10(data[data > 0.]))
            vmax = np.max(np.log10(data[data > 0.]))
            plot_obs(np.log10(data), headers, vmin=vmin, vmax=vmax,
                    rectangle=rectangle)

        plt.show()

    if return_background:
        return data_array, error_array, np.array([error_array[i][0,0] for i in range(error_array.shape[0])])
    else:
        return data_array, error_array


def rebin_array(data_array, error_array, headers, pxsize, scale,
        operation='sum'):
    """
    Homogeneously rebin a data array to get a new pixel size equal to pxsize
    where pxsize is given in arcsec.
    ----------
    Inputs:
    data_array, error_array : numpy.ndarray
        Arrays containing the images (2D float arrays) and their associated
        errors that will be rebinned.
    headers : header list
        List of headers corresponding to the images in data_array.
    pxsize : float
        Physical size of the pixel in arcseconds that should be obtain with
        the rebinning.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    operation : str, optional
        Set the way original components of the matrix are put together
        between summing ('sum') and averaging ('average', 'avg', 'mean') them.
        Defaults to 'sum'.
    ----------
    Returns:
    rebinned_data, rebinned_error : numpy.ndarray
        Rebinned arrays containing the images and associated errors.
    rebinned_headers : header list
        Updated headers corresponding to the images in rebinned_data.
    Dxy : numpy.ndarray
        Array containing the rebinning factor in each direction of the image.
    """
    # Check that all images are from the same instrument
    ref_header = headers[0]
    instr = ref_header['instrume']
    same_instr = np.array([instr == header['instrume'] for header in headers]).all()
    if not same_instr:
        raise ValueError("All images in data_array are not from the same\
                instrument, cannot proceed.")

    rebinned_data, rebinned_error, rebinned_headers = [], [], []
    Dxy = np.array([1, 1],dtype=int)

    aperture = ref_header['aptdia']    # Telescope aperture in mm
    for i, enum in enumerate(list(zip(data_array, error_array, headers))):
        image, error, header = enum
        # Get current pixel size
        w = WCS(header).deepcopy()
        if w.wcs.has_cd():
            del w.wcs.cd
            keys = list(w.to_header().keys())+['CD1_1','CD1_2','CD2_1','CD2_2']
            for key in keys:
                header.remove(key, ignore_missing=True)
            w.wcs.cdelt = 3600.*np.sqrt(np.sum(w.wcs.get_pc()**2,axis=1))
        if (w.wcs.cdelt == np.array([1., 1.])).all():
            # Update WCS with relevant information
            focallen = header['focallen']
            px_dim = np.array([header['xpixsz'], header['ypixsz']])   # Pixel dimension in µm
            w.wcs.cdelt = 206.3*px_dim/(focallen*aperture)
        header.update(w.to_header())

        # Compute binning ratio
        if scale.lower() in ['px', 'pixel']:
            Dxy = np.array([pxsize,]*2)
        elif scale.lower() in ['arcsec','arcseconds']:
            Dxy = np.floor(pxsize/w.wcs.cdelt).astype(int)
        else:
            raise ValueError("'{0:s}' invalid scale for binning.".format(scale))

        if (Dxy <= 1.).any():
            raise ValueError("Requested pixel size is below resolution.")
        new_shape = (image.shape//Dxy).astype(int)

        # Rebin data
        rebinned_data.append(bin_ndarray(image, new_shape=new_shape,
            operation=operation))

        # Propagate error
        rms_image = np.sqrt(bin_ndarray(image**2, new_shape=new_shape,
            operation='average'))
        #std_image = np.sqrt(bin_ndarray(image**2, new_shape=new_shape,
        #    operation='average') - bin_ndarray(image, new_shape=new_shape,
        #        operation='average')**2)
        new_error = np.sqrt(Dxy[0]*Dxy[1])*bin_ndarray(error,
                new_shape=new_shape, operation='average')
        rebinned_error.append(np.sqrt(rms_image**2 + new_error**2))

        # Update header
        w = w.slice((np.s_[::Dxy[0]], np.s_[::Dxy[1]]))
        header['NAXIS1'],header['NAXIS2'] = w.array_shape
        header.update(w.to_header())
        rebinned_headers.append(header)


    rebinned_data = np.array(rebinned_data)
    rebinned_error = np.array(rebinned_error)

    return rebinned_data, rebinned_error, rebinned_headers, Dxy


def align_data(data_array, headers, error_array=None, upsample_factor=1.,
        ref_data=None, ref_center=None, return_shifts=True):
    """
    Align images in data_array using cross correlation, and rescale them to
    wider images able to contain any rotation of the reference image.
    All images in data_array must have the same shape.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to align (2D float arrays).
    headers : header list
        List of headers corresponding to the images in data_array.
    error_array : numpy.ndarray, optional
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
        If None, will be initialized to zeros.
        Defaults to None.
    upsample_factor : float, optional
        Oversampling factor for the cross-correlation, will allow sub-
        pixel alignement as small as one over the factor of a pixel.
        Defaults to one (no over-sampling).
    ref_data : numpy.ndarray, optional
        Reference image (2D float array) the data_array should be
        aligned to. If "None", the ref_data will be the first image
        of the data_array.
        Defaults to None.
    ref_center : numpy.ndarray, optional
        Array containing the coordinates of the center of the reference
        image or a string in 'max', 'flux', 'maxflux', 'max_flux'. If None,
        will fallback to the center of the image.
        Defaults to None.
    return_shifts : boolean, optional
        If False, calling the function will only return the array of
        rescaled images. If True, will also return the shifts and
        errors.
        Defaults to True.
    ----------
    Returns:
    rescaled : numpy.ndarray
        Array containing the aligned data from data_array, rescaled to wider
        image with margins of value 0.
    rescaled_error : numpy.ndarray
        Array containing the errors on the aligned images in the rescaled array.
    shifts : numpy.ndarray
        Array containing the pixel shifts on the x and y directions from
        the reference image.
        Only returned if return_shifts is True.
    errors : numpy.ndarray
        Array containing the relative error computed on every shift value.
        Only returned if return_shifts is True.
    """
    if ref_data is None:
        # Define the reference to be the first image of the inputed array
        #if None have been specified
        ref_data = data_array[0]
    same = 1
    for array in data_array:
        # Check if all images have the same shape. If not, cross-correlation
        #cannot be computed.
        same *= (array.shape == ref_data.shape)
    if not same:
        raise ValueError("All images in data_array must have same shape as\
            ref_data")
    if error_array is None:
        _, error_array, background = get_error(data_array, return_background=True)
    else:
        _, _, background = get_error(data_array, return_background=True)

    # Crop out any null edges
    #(ref_data must be cropped as well)
    full_array = np.concatenate((data_array,[ref_data]),axis=0)
    err_array = np.concatenate((error_array,[np.zeros(ref_data.shape)]),axis=0)

    #full_array, err_array = crop_array(full_array, headers, step=5, null_val=0.,
    #        inside=False)

    data_array, ref_data = full_array[:-1], full_array[-1]
    error_array = err_array[:-1]
    background = np.zeros(data_array.shape[0])

    if ref_center is None:
        # Define the center of the reference image to be the center pixel
        #if None have been specified
        ref_center = (np.array(ref_data.shape)/2).astype(int)
    elif ref_center.lower() in ['max', 'flux', 'maxflux', 'max_flux']:
        # Define the center of the reference image to be the pixel of max flux.
        ref_center = np.unravel_index(np.argmax(ref_data),ref_data.shape)
    else:
        # Default to image center.
        ref_center = (np.array(ref_data.shape)/2).astype(int)

    # Create a rescaled null array that can contain any rotation of the
    #original image (and shifted images)
    shape = data_array.shape
    res_shape = (np.ceil(np.sqrt(1.1)*np.array(shape[1:]))).astype(int)
    rescaled_image = np.zeros((shape[0],res_shape[0],res_shape[1]))
    rescaled_error = np.ones((shape[0],res_shape[0],res_shape[1]))
    rescaled_mask = np.ones((shape[0],res_shape[0],res_shape[1]),dtype=bool)
    res_center = (np.array(rescaled_image.shape[1:])/2).astype(int)

    shifts, errors = [], []
    for i,image in enumerate(data_array):
        # Initialize rescaled images to background values
        #rescaled_error[i] *= background[i]
        # Get shifts and error by cross-correlation to ref_data
        shift, error, phase_diff = phase_cross_correlation(ref_data, image,
                upsample_factor=upsample_factor)
        # Rescale image to requested output
        center = np.fix(ref_center-shift).astype(int)
        res_shift = res_center-ref_center
        rescaled_image[i,res_shift[0]:res_shift[0]+shape[1],
                res_shift[1]:res_shift[1]+shape[2]] = deepcopy(image)
        rescaled_error[i,res_shift[0]:res_shift[0]+shape[1],
                res_shift[1]:res_shift[1]+shape[2]] = deepcopy(error_array[i])
        rescaled_mask[i,res_shift[0]:res_shift[0]+shape[1],
                res_shift[1]:res_shift[1]+shape[2]] = False
        # Shift images to align
        rescaled_image[i] = sc_shift(rescaled_image[i], shift, cval=0.)
        rescaled_error[i] = sc_shift(rescaled_error[i], shift, cval=background[i])
        rescaled_mask[i] = sc_shift(rescaled_mask[i], shift, cval=True)

        rescaled_image[i][rescaled_image[i] < 0.] = 0.

        # Uncertainties from shifting
        prec_shift = np.array([1.,1.])/upsample_factor
        shifted_image = sc_shift(rescaled_image[i], prec_shift, cval=0.)
        error_shift = np.abs(rescaled_image[i] - shifted_image)/2.
        #sum quadratically the errors
        rescaled_error[i] = np.sqrt(rescaled_error[i]**2 + error_shift**2)

        shifts.append(shift)
        errors.append(error)

    shifts = np.array(shifts)
    errors = np.array(errors)

    if return_shifts:
        return rescaled_image, rescaled_error, rescaled_mask.any(axis=0), shifts, errors
    else:
        return rescaled_image, rescaled_error, rescaled_mask.any(axis=0)


def smooth_data(data_array, error_array, data_mask, headers, FWHM=1.,
        scale='pixel', smoothing='gaussian'):
    """
    Smooth a data_array using selected function.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to smooth (2D float arrays).
    error_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    headers : header list
        List of headers corresponding to the images in data_array.
    FWHM : float, optional
        Full Width at Half Maximum for desired smoothing in 'scale' units.
        Defaults to 1.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    smoothing : str, optional
        Smoothing algorithm to be used on the input data array.
        -'combine','combining' use the N images combining algorithm with
          weighted pixels (inverse square of errors).
        -'gauss','gaussian' convolve any input image with a gaussian of
          standard deviation stdev = FWHM/(2*sqrt(2*log(2))).
        Defaults to 'gaussian'. Won't be used if FWHM is None.
    ----------
    Returns:
    smoothed_array : numpy.ndarray
        Array containing the smoothed images.
    error_array : numpy.ndarray
        Array containing the error images corresponding to the images in
        smoothed_array.
    """
    # If chosen FWHM scale is 'arcsec', compute FWHM in pixel scale
    if scale.lower() in ['arcsec','arcseconds']:
        pxsize = np.zeros((data_array.shape[0],2))
        for i,header in enumerate(headers):
            # Get current pixel size
            w = WCS(header).deepcopy()
            if w.wcs.has_cd():
                del w.wcs.cd
                keys = list(w.to_header().keys())+['CD1_1','CD1_2','CD2_1','CD2_2']
                for key in keys:
                    header.remove(key, ignore_missing=True)
                w.wcs.cdelt = 3600.*np.sqrt(np.sum(w.wcs.get_pc()**2,axis=1))
            if (w.wcs.cdelt == np.array([1., 1.])).all():
                # Update WCS with relevant information
                aperture = header['aptdia']    # Telescope aperture in mm
                focal_len = header['focallen']
                px_dim = np.array([header['xpixsz'], header['ypixsz']])   # Pixel dimension in µm
                w.wcs.cdelt = 206.3*px_dim/(focal_len*aperture)
            header.update(w.to_header())
            pxsize[i] = np.round(w.wcs.cdelt,4)
        if (pxsize != pxsize[0]).any():
            raise ValueError("Not all images in array have same pixel size")
        FWHM /= pxsize[0].min()

    # Define gaussian stdev
    stdev = FWHM/(2.*np.sqrt(2.*np.log(2)))
    fmax = np.finfo(np.float64).max

    if smoothing.lower() in ['combine','combining']:
        # Smooth using N images combination algorithm
        # Weight array
        weight = 1./error_array**2
        # Prepare pixel distance matrix
        xx, yy = np.indices(data_array[0].shape)
        # Initialize smoothed image and error arrays
        smoothed = np.zeros(data_array[0].shape)
        error = np.zeros(data_array[0].shape)

        # Combination smoothing algorithm
        for r in range(smoothed.shape[0]):
            for c in range(smoothed.shape[1]):
                # Compute distance from current pixel
                dist_rc = np.where(data_mask, fmax, np.sqrt((r-xx)**2+(c-yy)**2))
                g_rc = np.array([np.exp(-0.5*(dist_rc/stdev)**2),]*data_array.shape[0])
                # Apply weighted combination
                smoothed[r,c] = (1.-data_mask[r,c])*np.sum(data_array*weight*g_rc)/np.sum(weight*g_rc)
                error[r,c] = np.sqrt(np.sum(weight*g_rc**2))/np.sum(weight*g_rc)

        # Nan handling
        error[np.isnan(smoothed)] = 0.
        smoothed[np.isnan(smoothed)] = 0.
        error[np.isnan(error)] = 0.

    elif smoothing.lower() in ['gauss','gaussian']:
        # Convolution with gaussian function
        smoothed = np.zeros(data_array.shape)
        error = np.zeros(error_array.shape)
        for i,image in enumerate(data_array):
            xx, yy = np.indices(image.shape)
            for r in range(image.shape[0]):
                for c in range(image.shape[1]):
                    dist_rc = np.where(data_mask, fmax, np.sqrt((r-xx)**2+(c-yy)**2))
                    g_rc = np.exp(-0.5*(dist_rc/stdev)**2)/(2.*np.pi*stdev**2)
                    smoothed[i][r,c] = (1.-data_mask[r,c])*np.sum(data_array*weight*g_rc)/np.sum(weight*g_rc)
                    error[i][r,c] = np.sqrt(np.sum(weight*g_rc**2))/np.sum(weight*g_rc)

            # Nan handling
            error[i][np.isnan(smoothed)] = 0.
            smoothed[i][np.isnan(smoothed)] = 0.
            error[i][np.isnan(error)] = 0.

    else:
        raise ValueError("{} is not a valid smoothing option".format(smoothing))

    return smoothed, error


def filter_avg(data_array, error_array, headers, data_mask, filters, FWHM=None,
        scale='pixel', smoothing='gaussian'):
    """
    Make the average image from a single filter for a given instrument.
    -----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) of a
        single observation with multiple polarizers of an instrument.
    error_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    headers : header list
        List of headers corresponding to the images in data_array.
    filter : strlist
        List containing the names of the available filters.
    FWHM : float, optional
        Full Width at Half Maximum of the detector for smoothing of the
        data on each polarizer filter in 'scale' units. If None, no smoothing
        is done.
        Defaults to None.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    smoothing : str, optional
        Smoothing algorithm to be used on the input data array.
        -'combine','combining' use the N images combining algorithm with
          weighted pixels (inverse square of errors).
        -'gaussian' convolve any input image with a gaussian of standard
          deviation stdev = FWHM/(2*sqrt(2*log(2))).
        Defaults to 'gaussian'. Won't be used if FWHM is None.
    ----------
    Returns:
    filter_array : numpy.ndarray
        Array of images averaged on each filter of the instrument
    filter_error : numpy.ndarray
        Array of errors associated to each image in filter_array
    filter_headers : header list
        List of headers corresponding to the images in filter_array.
    """
    # Check that all images are from the same instrument
    instr = headers[0]['instrume']
    same_instr = np.array([instr == hdr['instrume'] for hdr in headers]).all()
    if not same_instr:
        raise ValueError("All images in data_array are not from the same\
                instrument, cannot proceed.")

    # Sort images by filter : can be B, V, R, I, H, O
    filt_bool, filt_data, filt_error, filt_headers, filt_avg, filt_err = dict(), dict(), dict(), dict(), dict(), dict()
    for filt in filters:
        filt_bool[filt] = np.array([hdr['filter']==filt for hdr in headers])
        if np.max(filt_bool[filt]):
            filt_data[filt] = data_array[filt_bool[filt]]
            filt_error[filt] = error_array[filt_bool[filt]]
            filt_headers[filt] = [hdr for hdr in headers if hdr['filter']==filt]

            if not(FWHM is None) and (smoothing.lower() in ['combine','combining']):
                # Smooth by combining each polarizer images
                filt_avg[filt], filt_err[filt] = smooth_data(filt_data[filt],
                        filt_error[filt], data_mask, filt_headers[filt], FWHM=FWHM,
                        scale=scale, smoothing=smoothing)

            else:
                # Sum on each polarization filter.
                filt_avg[filt] = filt_data[filt].sum(axis=0)
                filt_err[filt] = np.sum(filt_error[filt],axis=0)*np.sqrt(filt_error[filt].shape[0])

                if not(FWHM is None) and (smoothing.lower() in ['gaussian','gauss']):
                    # Smooth by convoluting with a gaussian each polX image.
                        filt_avg[filt], filt_err[filt] = smooth_data(filt_avg[filt],
                                filt_err[filt], data_mask, filt_headers[filt],
                                FWHM=FWHM, scale=scale, smoothing=smoothing)

    # Construct the filter array and filter error array
    for header in headers:
        filt = header['filter']
        if np.max(filt_bool[filt]):
            list_head = filt_headers[filt]
            filt_headers[filt][0]['exptime'] = np.sum([head['exptime'] for head in list_head])/len(list_head)
            filt_headers[filt][0]['date-obs'] = str(np.min([np.datetime64(head['date-obs']) for head in list_head]))
            filt_headers[filt][0]['jd'] = np.min([head['jd'] for head in list_head])
            filt_headers[filt][0]['jd-helio'] = np.median([head['jd-helio'] for head in list_head])
            filt_headers[filt][0]['airmass'] = np.mean([head['airmass'] for head in list_head])

    filter_array = np.array([value for value in filt_avg.values()])
    filter_error = np.array([value for value in filt_err.values()])
    filter_headers = [filt_headers[filt][0] for filt in filters if np.max(filt_bool[filt])]

    return filter_array, filter_error, filter_headers
