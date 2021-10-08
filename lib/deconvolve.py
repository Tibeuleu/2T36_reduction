"""
Library function for the implementation of Richardson-Lucy deconvolution algorithm.
"""

import numpy as np
from scipy.signal import convolve


def richardson_lucy(image, psf, iterations=20, clip=True, filter_epsilon=None):
    """
    Richardson-Lucy deconvolution algorithm.
    ----------
    Inputs:
    image : numpy.darray
       Input degraded image (can be N dimensional) of floats between 0 and 1.
    psf : numpy.darray
       The point spread function.
    iterations : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    filter_epsilon: float, optional
       Value below which intermediate results become 0 to avoid division
       by small numbers.
    ----------
    Returns:
    im_deconv : ndarray
       The deconvolved image.
    ----------
    References
    [1] https://doi.org/10.1364/JOSA.62.000055
    [2] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    float_type = np.promote_types(image.dtype, np.float32)
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    im_deconv = np.full(image.shape, 0.5, dtype=float_type)
    psf_mirror = np.flip(psf)

    for _ in range(iterations):
        conv = convolve(im_deconv, psf, mode='same')
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
        else:
            relative_blur = image / conv
        im_deconv *= convolve(relative_blur, psf_mirror, mode='same')

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv


def deconvolve_im(image, psf, iterations=20, clip=True, filter_epsilon=None):
    """
    Prepare an image for deconvolution using Richardson-Lucy algorithm and
    return results.
    ----------
    Inputs:
    image : numpy.darray
       Input degraded image (can be N dimensional) of floats between 0 and 1.
    psf : numpy.darray
       The point spread function.
    iterations : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    filter_epsilon: float, optional
       Value below which intermediate results become 0 to avoid division
       by small numbers.
    ----------
    Returns:
    im_deconv : ndarray
       The deconvolved image.
    """
    # Normalize image to highest pixel value
    pxmax = image[np.isfinite(image)].max()
    if pxmax == 0.:
        raise ValueError("Invalid image")
    norm_image = image/pxmax

    # Deconvolve normalized image
    norm_deconv = richardson_lucy(image=norm_image, psf=psf, iterations=iterations,
            clip=clip, filter_epsilon=filter_epsilon)

    # Output deconvolved image with original pxmax value
    im_deconv = pxmax*norm_deconv

    return im_deconv


def gaussian_psf(FWHM=1., shape=(5,5)):
    """
    Define the gaussian Point-Spread-Function of chosen shape and FWHM.
    ----------
    Inputs:
    FWHM : float, optional
        The Full Width at Half Maximum of the desired gaussian function for the
        PSF in pixel increments.
        Defaults to 1.
    shape : tuple, optional
        The shape of the PSF kernel. Must be of dimension 2.
        Defaults to (5,5).
    ----------
    Returns:
    kernel : numpy.ndarray
        Kernel containing the weights of the desired gaussian PSF.
    """
    # Compute standard deviation from FWHM
    stdev = FWHM/(2.*np.sqrt(2.*np.log(2.)))

    # Create kernel of desired shape
    xx, yy = np.indices(shape)
    x0, y0 = (np.array(shape)-1.)/2.
    kernel = np.exp(-0.5*((xx-x0)**2+(yy-y0)**2)/stdev**2)

    return kernel
