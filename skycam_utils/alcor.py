import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits

import astropy.visualization as viz


def fits_to_fig(filename, outimage=None, outfig=None, rotation=3, xcen=1003, ycen=707, radius=680,
                powerstretch=0.75, contrast=0.35, gscale=0.7, bscale=1.7):
    """
    Take a FITS file as produced by the alcor OMEA 8C and create a trimmed, rotated, and annotated figure
    file appropriate for display

    Parameters
    ----------
    filename : str
        FITS filename of image. Uses astropy.io.fits so gz and bz2 extentions are allowed.
    outimage : str (default=None)
        If not None, write out raw, unannotated image
    outfig : str (default=None)
        If not None, write out annotated image as produced by matplotlib
    rotation : float (default=3)
        Camera rotation w.r.t. true north. Default is empirically from alcor software.
    xcen : int (default=1003)
        X center of illuminated region in original image coordinates
    ycen : int (default=709)
        Y center of illuminated region in original image coordinates
    radius : float (default=578)
        Radius of illuminated region
    powerstretch : float (default=0.75)
        Power of the stretch function to use
    contrast : float (default=0.35)
        ZScale contrast factor
    gscale : float (default=0.7)
        Scale factor to apply to green channel
    bscale : float (default=1.7)
        Scale factor to apply to blue channel
    """
    hdul = fits.open(filename)
    im = np.transpose(hdul[0].data, axes=(1, 2, 0)) - 2000  # 2000 is a bit above the normal bias level of the camera.
    im[im < 0] = 0  # remove negative values
    im = im * 1.0  # hacky way to cast to float
    im[:, :, 1] *= gscale  # the factors to scale the green and blue channels were determined empirically and provide a
    im[:, :, 2] *= bscale  # reasonably good white/color balance for both day and night images.
    xl = xcen - radius
    xu = xcen + radius
    yl = ycen - radius
    yu = ycen + radius
    im = im[yl:yu, xl:xu, :]  # trim image to be square and centered on the illuminated region
    im = np.flipud(rotate(im, rotation, reshape=False))  # flip image to put north at top and tweak rotation
                                                         # to align N-E with X-Y
    stretch = viz.PowerStretch(powerstretch) + viz.ZScaleInterval(contrast=contrast)  # apply a power-law stretch and
                                                                                      # zscale interval to the image data
    im = stretch(im)

    if outimage is not None:
        plt.imsave(outimage, im)

    fig, ax = plt.subplots(figsize=(8, 8))
    circle = Circle((radius, radius), radius, facecolor='none', edgecolor=(0, 0, 0), linewidth=1, alpha=0.5)
    ax.add_patch(circle)
    ax.axis("off")

    im_plot = plt.imshow(im)
    im_plot.set_clip_path(circle)

    pax = fig.add_subplot(111, polar=True, label='polar')
    pax.set_facecolor("None")
    pax.set_theta_zero_location("N")
    yticks = np.array([15, 30, 45, 60, 75]) / 90.0
    ylabels = [" 75°", " 60°", " 45°", " 30°", " 15°"]
    pax.set_yticks(yticks, labels=ylabels, color="white", alpha=0.5)
    pax.set_rlabel_position(90)
    pax.tick_params(grid_alpha=0.5)

    if outfig is not None:
        plt.savefig(outfig)

    return fig
