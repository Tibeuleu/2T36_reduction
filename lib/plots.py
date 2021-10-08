"""
Library functions for displaying  informations using matplotlib

prototypes :
    - plot_obs(data_array, headers, shape, vmin, vmax, savename, plots_folder) -> 0
        Plots whole observation raw data in given display shape.

    - display_RGB_reduced(hdul, filters, savename, plots_folder) -> 0
        Display requested filtered images as a RGB stacked image.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar, AnchoredDirectionArrows
from astropy.wcs import WCS


def princ_angle(ang):
    """
    Return the principal angle in the 0-180° quadrant.
    """
    while ang < 0.:
        ang += 180.
    while ang > 180.:
        ang -= 180.
    return ang


def sci_not(v,err,rnd=1):
    """
    Return the scientifque error notation as a string.
    """
    power = - int(('%E' % v)[-3:])+1
    output = r"({0}".format(round(v*10**power,rnd))
    if type(err) == list:
        for error in err:
            output += r" $\pm$ {0}".format(round(error*10**power,rnd))
    else:
        output += r" $\pm$ {0}".format(round(err*10**power,rnd))
    return output+r")e{0}".format(-power)


def plot_obs(data_array, headers, shape=None, vmin=0., vmax=6., rectangle=None,
        savename=None, plots_folder=""):
    """
    Plots raw observation imagery with some information on the instrument and
    filters.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) of a
        single observation with multiple polarizers of an instrument
    headers : header list
        List of headers corresponding to the images in data_array
    shape : array-like of length 2, optional
        Shape of the display, with shape = [#row, #columns]. If None, defaults
        to the optimal square.
        Defaults to None.
    vmin : float, optional
        Min pixel value that should be displayed.
        Defaults to 0.
    vmax : float, optional
        Max pixel value that should be displayed.
        Defaults to 6.
    rectangle : numpy.ndarray, optional
        Array of parameters for matplotlib.patches.Rectangle objects that will
        be displayed on each output image. If None, no rectangle displayed.
        Defaults to None.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    """
    if shape is None:
        shape = np.array([np.ceil(np.sqrt(data_array.shape[0])).astype(int),]*2)
    fig, ax = plt.subplots(shape[0], shape[1], figsize=(10,10), dpi=200,
            sharex=True, sharey=True)

    for i, enum in enumerate(list(zip(ax.flatten(),data_array))):
        ax = enum[0]
        data = enum[1]
        instr = headers[i]['telescop']
        exptime = headers[i]['exptime']
        filt = headers[i]['filter']
        #plots
        im = ax.imshow(data, vmin=vmin, vmax=vmax, origin='lower')
        if not(rectangle is None):
            x, y, width, height, angle, color = rectangle[i]
            ax.add_patch(Rectangle((x, y), width, height, angle=angle,
                edgecolor=color, fill=False))
        #position of centroid
        ax.plot([data.shape[1]/2, data.shape[1]/2], [0,data.shape[0]-1], lw=1,
                color='black')
        ax.plot([0,data.shape[1]-1], [data.shape[0]/2, data.shape[0]/2], lw=1,
                color='black')
        ax.annotate(instr,color='white',fontsize=5,xy=(0.02, 0.95),
                xycoords='axes fraction')
        ax.annotate(filt,color='white',fontsize=10,xy=(0.02, 0.02),
                xycoords='axes fraction')
        ax.annotate(exptime,color='white',fontsize=5,xy=(0.80, 0.02),
                xycoords='axes fraction')

    fig.subplots_adjust(hspace=0, wspace=0, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.75])
    fig.colorbar(im, cax=cbar_ax)

    if not (savename is None):
        fig.suptitle(savename)
        fig.savefig(plots_folder+savename+".png",bbox_inches='tight')
    plt.show()
    return 0


def display_RGB_reduced(hdul, filters, savename=None, plots_folder=""):
    """
    Plots reduced filtered images.
    ----------
    Inputs:
    hdul : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I, B, V, R, H, O filters images (in this particular
        order) for one observation
    filters : strlist
        Names of the filters to be shown as RGB image.
    savename : str, optional
        Name of the figure the image should be saved to. If None, the image
        won't be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the image will
        be saved. Not used if savename is None.
        Defaults to current folder.
    """
    # Get data
    R = hdul[np.argmax([hdul[i].header['filter']==filters[0] for i in range(len(hdul))])].data
    G = hdul[np.argmax([hdul[i].header['filter']==filters[1] for i in range(len(hdul))])].data
    B = hdul[np.argmax([hdul[i].header['filter']==filters[2] for i in range(len(hdul))])].data
    # Normalise images
    R = R/R.max()
    G = G/G.max()
    B = B/B.max()

    image = np.dstack((R, G, B))

    wcs = WCS(hdul[0]).deepcopy()

    # Display the 3 filters separately
    fig1 = plt.figure(figsize=(30,10))

    ax1 = fig1.add_subplot(131)#, projection=wcs)
    im = ax1.imshow(R, origin='lower')
    #plt.colorbar(im)
    ax1.set(xlabel="pixel offset", ylabel="pixel offset", title="{0:s} filter".format(filters[0]))

    ax1 = fig1.add_subplot(132)#, projection=wcs)
    im = ax1.imshow(G, origin='lower')
    #plt.colorbar(im)
    ax1.set(xlabel="pixel offset", ylabel="pixel offset", title="{0:s} filter".format(filters[1]))

    ax1 = fig1.add_subplot(133)#, projection=wcs)
    im = ax1.imshow(B, origin='lower')
    #plt.colorbar(im)
    ax1.set(xlabel="pixel offset", ylabel="pixel offset", title="{0:s} filter".format(filters[2]))

    fig2 = plt.figure(figsize=(30,10))

    ax2 = fig2.add_subplot(111)#, projection=wcs)
    ax2.imshow(image, origin='lower')
    ax2.set(xlabel="pixel offset", ylabel="pixel offset", title="RGB image using {0:s}, {1:s}, {2:s} filters respectively.".format(*filters))

    if not (savename is None):
        fig1.suptitle(savename+" {0:s} {1:s} {2:s} filters".format(*filters))
        fig1.savefig(plots_folder+savename+"_{0:s}{1:s}{2:s}.png".format(*filters),bbox_inches='tight')
        fig2.suptitle(savename+" {0:s} {1:s} {2:s} filters as RGB".format(*filters))
        fig2.savefig(plots_folder+savename+"_{0:s}{1:s}{2:s}_RGB.png".format(*filters),bbox_inches='tight')
    plt.show()
    return 0
