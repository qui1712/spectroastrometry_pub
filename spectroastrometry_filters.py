"""
Created on Sat Feb 25 17:26:09 2023.

@author: Quirijn B. van Woerkom
Utility classes for filters used in spectroastrometry; also contains noise
analysis functions, as those are instrument-property dependent.
"""
# Standard imports
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
import time
import tqdm

# Specific imports
from scipy.interpolate import interpn, interp1d, PchipInterpolator
from matplotlib.ticker import ScalarFormatter
import os
import pandas as pd
from matplotlib.colors import Normalize, Colormap, LogNorm

# Plotstyle changes
# Increase the matplotlib font size
plt.rcParams.update({"font.size": 22})
# Set layout to tight
plt.rcParams.update({"figure.autolayout": True})
# Set grid to true
plt.rcParams.update({"axes.grid": True})
# Set the plotting style to interactive (autoshows plots)
plt.ion()

# %%
# Define the supported filters as global variables; also save some telescope
# and instrument properties in this manner

# Save the supported filters as well as the METIS filter properties in global
# variables.
supported_filters = {}
supported_filters["JWST"] = {}
supported_filters["JWST"]["NIRCam_HCI"] = [
    "F322W2",
    "F200W", "F277W", "F356W", "F444W",
    "F182M", "F210M", "F250M", "F300M", "F335M", "F360M", "F410M", "F430M",
    "F460M", "F480M", "F187N", "F212N"]

supported_filters["VLT"] = {}
supported_filters["VLT"]["SPHERE"] = [
    "B_H", "B_J", "B_Ks", "B_Y",
    "D_H23", "D_J23", "D_K12", "D_ND-H23", "D_Y23",
    "N_BrG", "N_CntH", "N_CntJ", "N_CntK1", "N_CntK2", "N_CO", "N_FeII",
    "N_H2", "N_HeI", "N_PaB"]

supported_filters["ELT"] = {}
supported_filters["ELT"]["METIS"] = [
    "L'", "short-L", "HCI-L short", "HCI-L long", "M'", "N1", "N2",
    "H2O-ice", "PAH 3.3", "Br-alpha", "CO(1-0)-ice", "PAH 8.6", "PAH 11.25",
    "[S IV]", "[Ne II]"]

METIS_lam_pivot = np.array([
    3.79, 3.31, 3.60, 3.82, 4.80, 8.65, 11.20, 3.10, 3.30, 4.05, 4.66, 8.60,
    11.20,
    12.82, 10.50])*u.um
METIS_delta_lam = np.array([
    .63, .43, .22, .27, .60, 1.16, 2.36, .22, .07, .03, .22, .45, .35, .23,
    .19])*u.um

supported_filters["JWST"]["MIRI_HCI_rect"] = [
    "F1065C", "F1140C", "F1550C", "F2300C"]

supported_filters["JWST"]["NIRCam_HCI/MIRI_HCI_rect"] = [
    "F322W2",
    "F200W", "F277W", "F356W", "F444W",
    "F182M", "F210M", "F250M", "F300M", "F335M", "F360M", "F410M", "F430M",
    "F460M", "F480M", "F187N", "F212N",
    "F1065C", "F1140C", "F1550C", "F2300C"]

MIRI_rect_lam_pivot = np.array([
    10.575, 11.30, 15.50, 22.75])*u.um
MIRI_rect_delta_lam = np.array([
    .75, .8, .9, 5.5])*u.um

# Also save a dictionary containing the diameters of each telescope for future
# reference
diameter = {}
diameter["JWST"] = 6.5*u.m
diameter["VLT"] = 8.2*u.m
# diameter["ELT"] = 39.3*u.m
diameter["ELT"] = 37*u.m

# Define the photon flux density for speed-up purposes
ph_flux_dens = u.photon/u.s/u.um/u.m**2
# Similarly, define the intensity in units of ph/s/m^2/mas^2
intens_unit = u.ph/u.s/u.m**2/u.mas**2
# %% Function definitions


@u.quantity_input
def calc_sigma_pixel_pphoton(pixel_size: u.mas = 63*u.mas, k_px=1/2):
    """Calculate the centroid standard deviation per photon due to pixel size.

    Calculates the centroid standard deviation per photon due to the use of
    pixels on the detector rather than precise centroid measurements per
    incoming photon.

    Parameters
    ----------
    k_px (float):
        Pixel constant. Between 1/6 and 1/2; set to 1/2 for the loosest
        but most conservative constraint on pixel noise.
    pixel_size (astropy.Quantity):
        Pixel-size of the used telescope. Default is the long-wavelength
        channel of JWST/NIRCam (63 mas). ELT is 5.5 mas in the L/M-bands,
        and 6.8 mas in the N-band (i.e. above 7.5 um).

    Returns
    -------
    out (astropy.Quantity):
        Pixel noise per photon.
    """
    return pixel_size*np.sqrt(k_px)


@u.quantity_input
def calc_sigma_noise_pphoton(sigma_PSF: u.mas, k_n=2/3, SNR_flux=5,
                             n_sigma_PSF=3, pixel_size: u.mas = 63*u.mas):
    """Calculate the centroid standard deviation per photon due to flux noise.

    Calculates the centroid standard deviation per photon due to noise in the
    flux measurements (which will affect the position of the centroid).

    Parameters
    ----------
    sigma_PSF (astropy.Quantity):
        Standard deviation of the PSF.
    k_n (float):
        Noise constant. Between 1/2 and 2/3 for convex regions; depends on the
        region used to sample the PSF. Set to 2/3 for the loosest but most
        conservative constraint. By using pre-defined convex regions, we do not
        run the risk of affecting the reliability of our results by choosing
        regions that will yield a "detection" after the fact. Allowing concave
        regions may be useful for well-known concave PSF shapes, but for now
        we shall restrict ourselves explicitly to convex regions. Set to 1/2
        for a circle, and 2/3 for a square region.
    SNR_flux (float):
        Signal-to-noise ratio in flux. Set to 5 to align with detection
        criteria.
    n_sigma_PSF (float):
        Characteristic dimension of the region over which to determine the
        centroid i.e. radius for a circle, width for a square. Expressed in
        multiples of sigma_PSF.
    pixel_size (astropy.Quantity):
        Pixel-size of the used telescope. Default is the long-wavelength
        channel of JWST/NIRCam (63 mas). ELT is 5.5 mas in the L/M-bands,
        and 6.8 mas in the N-band (i.e. above 7.5 um). Used to round the extent
        of the region to a whole-integer number of pixels.

    Returns
    -------
    out (astropy.Quantity):
        Noise per photon due to flux noise.
    """
    return (np.sqrt(k_n/SNR_flux)*pixel_size*np.ceil(
        n_sigma_PSF*sigma_PSF/pixel_size))


@u.quantity_input
def calc_sigma_pphoton(sigma_PSF: u.mas, k_px=1/2, k_n=2/3,
                       SNR_flux=5, pixel_size: u.mas = 63*u.mas,
                       sigma_PO: u.mas = 1*u.mas, n_sigma_PSF=3,
                       return_components=False):
    """Calculate the centroid standard deviation per photon.

    Calculates the centroid standard deviation per photon in a filter.
    Taken into account are photon noise, pixel noise, flux noise and pointing
    noise.

    Parameters
    ----------
    sigma_PSF (astropy.Quantity):
        Standard deviation of the PSF.
    k_px (float):
        Pixel constant. Between 1/6 and 1/2; set to 1/2 for the loosest
        but most conservative constraint on pixel noise.
    k_n (float):
        Noise constant. Between 1/2 and 2/3 for convex regions; depends on the
        region used to sample the PSF. Set to 2/3 for the loosest but most
        conservative constraint. By using pre-defined convex regions, we do not
        run the risk of affecting the reliability of our results by choosing
        regions that will yield a "detection" after the fact. Allowing concave
        regions may be useful for well-known concave PSF shapes, but for now
        we shall restrict ourselves explicitly to convex regions.
    SNR_flux (float):
        Signal-to-noise ratio in flux. Set to 5 to align with detection
        criteria.
    pixel_size (astropy.Quantity):
        Pixel-size of the used telescope. Default is the long-wavelength
        channel of JWST/NIRCam (63 mas). ELT is 5.5 mas in the L/M-bands,
        and 6.8 mas in the N-band (i.e. above 7.5 um).
    sigma_PO (astropy.Quantity):
        Pointing noise (offset) of the used telescope. Equal to ~1 mas for
        JWST and ELT; significantly higher for VLT at 3 as.
    n_sigma_PSF (float):
        Characteristic dimension of the region over which to determine the
        centroid i.e. radius for a circle, width for a square. Expressed in
        multiples of sigma_PSF. Default is three, as that gives a relatively
        high encircled energy.
    return_components (bool):
        Whether to return the components of the noise.

    Returns
    -------
    sigma_pphoton (astropy.Quantity):
        Centroid noise per photon expressed in mas.
    sigma_pixel (astropy.Quantity):
        Pixel noise. Only returned if return_components=True.
    sigma_noise (astropy.Quantity):
        Centroid variation due to flux noise. Only returned if
        return_components=True.
    """
    sigma_pixel = calc_sigma_pixel_pphoton(
        pixel_size=pixel_size,
        k_px=k_px)
    sigma_noise = calc_sigma_noise_pphoton(
        sigma_PSF=sigma_PSF,
        k_n=k_n,
        SNR_flux=SNR_flux,
        n_sigma_PSF=n_sigma_PSF,
        pixel_size=pixel_size)
    sigma_pphoton = np.zeros((4,))*u.mas
    sigma_pphoton[0] = sigma_PSF
    sigma_pphoton[1] = sigma_pixel
    sigma_pphoton[2] = sigma_noise
    sigma_pphoton[3] = sigma_PO
    sigma_pphoton = np.linalg.norm(sigma_pphoton)
    if return_components:
        return sigma_pphoton, sigma_pixel, sigma_noise
    else:
        return sigma_pphoton


@u.quantity_input
def pixel_size(telescope, lam: u.um, instrument=None):
    """Return the pixel size for the given telescope at the given wavelength.

    Give the pixel size for the given telescope at a certain wavelength.

    Parameters
    ----------
    telescope (str):
        Name of the telescope. In principle, if instruments do not overlap in
        wavelength, this should be sufficient to identify the proper pixel size
        to use, but the instrument may have to be specified in some cases.
    lam (astropy.Quantity):
        Wavelength at which the pixel size is to be given: it is not strictly
        a function of wavelength, but the instrument is in general tied to
        wavelength.
    instrument (str or NoneType):
        Name of the instrument. In principle not necessary, but we will allow
        this such that in the future this function remains robust if we expand
        it to also work for non-coronagraphic instruments that may therefore
        overlap in wavelength. None by default.

    Returns
    -------
    pixel_size (astropy.Quantity):
        The pixel size expressed in mas.
    """
    # Set pixel_size to nan if it is not within a wavelength range of an
    # instrument
    if telescope not in ["JWST", "ELT", "VLT"]:
        raise ValueError("Please choose a supported telescope!")
    pixel_size = np.nan*u.mas
    if telescope == "JWST":
        # Short wavelengths on NIRCam:
        # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview#:~:text=Each%20detector%20contains%202048%20%C3%97,bias%20voltage%20drifts%20during%20exposures.
        # Long wavelengths on MIRI:
        # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-coronagraphic-imaging
        if (1.8*u.um < lam < 2.3*u.um):
            pixel_size = 31*u.mas
        elif (2.3*u.um < lam < 5.1*u.um):
            pixel_size = 63*u.mas
        elif ((10*u.um < lam < 12*u.um) or (15*u.um < lam < 16*u.um) or
              (20*u.um < lam < 26*u.um)):
            pixel_size = 110*u.mas
    elif telescope == "ELT":
        # Based on Brandl et al. 2021
        # The LM-arm has a pixel size of 5.5 mas, the N-arm of 6.8 mas
        if (2.9*u.um < lam < 5.5*u.um):
            pixel_size = 5.5*u.mas
        elif ((8*u.um < lam < 9.3*u.um) or (10*u.um < lam < 13*u.um)):
            pixel_size = 6.8*u.mas
    elif telescope == "VLT":
        pixel_size = 12.25*u.mas
        # https://www.eso.org/sci/facilities/paranal/instruments/sphere/doc/VLT-MAN-SPH-14690-0430_v96.pdf
    return pixel_size


def pointing_noise(telescope, lam=None):
    """Calculate the pointing noise for the given telescope.

    Calculate the pointing noise for a given telescope: where applicable,
    will calculate it based off the wavelength, otherwise returns the
    worst-case value.

    Parameters
    ----------
    telescope (str):
        Which telescope the pointing noise is to be calculated for.
    lam (Nonetype or astropy.Quantity):
        The wavelength for which the pointing noise is to be calculated. If set
        to None, the worst-case pointing noise will be given.

    Returns
    -------
    pointing_offset (astropy.Quantity):
        The pointing noise in mas.
    """
    if telescope not in ["JWST", "ELT", "VLT"]:
        raise ValueError("Please choose a supported telescope!")
    elif telescope == "VLT":
        # https://www.eso.org/sci/facilities/paranal/telescopes/ut/utperformance.html
        return 3000*u.mas
    elif telescope == "JWST":
        # Based off Hartig & Lallo, 2022
        return 1*u.mas
    elif telescope == "ELT":
        # Basesd off Brandl et al., 2021
        if lam is None:
            # If wavelength is not specified, return the N-band value
            return 1*u.mas
        else:
            # Return the value of .02 lam/D
            return (.02*lam/diameter[telescope]*u.rad).to(u.mas)


@u.quantity_input
def calc_sigma_pphoton_telescope(lams: u.um, telescope, k_px=1/2, k_n=2/3,
                                 SNR_flux=5, sigma_PO=1*u.mas, n_sigma_PSF=3,
                                 PSF_multiplier=1):
    """Calculate the centroid standard deviation per photon at a wavelength.

    Calculates the centroid standard deviation per photon at a given
    wavelength. Taken into account are photon noise, pixel noise, flux noise
    and pointing noise.

    Parameters
    ----------
    lams (array_like of astropy.Quantity):
        Wavelengths at which to evaluate the noise.
    telescope (str):
        Telescope for which the noise is to be calculated.
    k_px (float):
        Pixel constant. Between 1/6 and 1/2; set to 1/2 for the loosest
        but most conservative constraint on pixel noise.
    k_n (float):
        Noise constant. Between 1/2 and 2/3 for convex regions; depends on the
        region used to sample the PSF. Set to 2/3 for the loosest but most
        conservative constraint. By using pre-defined convex regions, we do not
        run the risk of affecting the reliability of our results by choosing
        regions that will yield a "detection" after the fact. Allowing concave
        regions may be useful for well-known concave PSF shapes, but for now
        we shall restrict ourselves explicitly to convex regions.
    SNR_flux (float):
        Signal-to-noise ratio in flux. Set to 5 to align with detection
        criteria.
    sigma_PO (astropy.Quantity):
        Pointing noise (offset) of the used telescope. Equal to ~1 mas for
        JWST and ELT; significantly higher for VLT at 3 as.
    n_sigma_PSF (float):
        Characteristic dimension of the region over which to determine the
        centroid i.e. radius for a circle, width for a square. Expressed in
        multiples of sigma_PSF. Default is three, as that gives a relatively
        high encircled energy.
    PSF_multiplier (float):
        Multiplicative factor by which to exaggerate the PSF width compared
        to the Airy PSF.

    Returns
    -------
    sigmas (array of astropy.Quantity):
        Total centroid noise per photon as function of wavelength.
    sigmas_PSF (array of astropy.Quantity):
        Centroid noise per photon due to photon (PSF) noise as function of
        wavelength.
    sigmas_pixel (array of astropy.Quantity):
        Pixel noise per photon as function of wavelength.
    sigmas_noise (array of astropy.Quantity):
        Centroid noise due to flux noise per photon as function of wavelength.
    sigmas_PO (array of astropy.Quantity):
        Pointing offset per photon as a function of wavelength.
    """
    if telescope not in ["JWST", "ELT", "VLT"]:
        raise ValueError("Please choose a supported telescope!")
    diameter_tel = diameter[telescope]
    sigmas = np.zeros_like(lams.value)*u.mas
    sigmas_PSF = np.zeros_like(lams.value)*u.mas
    sigmas_pixel = np.zeros_like(lams.value)*u.mas
    sigmas_noise = np.zeros_like(lams.value)*u.mas
    sigmas_PO = np.zeros_like(lams.value)*u.mas
    for idx, lam in enumerate(lams):
        pixel_size_tel = pixel_size(telescope, lam)
        sigma_PSF = PSF_multiplier*(.45*lam/diameter_tel*u.rad).to(u.mas)
        sigma_PO = pointing_noise(telescope, lam)
        sigma, sigma_pixel, sigma_noise = calc_sigma_pphoton(
            sigma_PSF,
            k_px=k_px, k_n=k_n,
            SNR_flux=SNR_flux,
            pixel_size=pixel_size_tel,
            sigma_PO=sigma_PO,
            n_sigma_PSF=n_sigma_PSF,
            return_components=True)
        sigmas[idx] = sigma.to(u.mas)
        sigmas_PSF[idx] = sigma_PSF.to(u.mas)
        sigmas_pixel[idx] = sigma_pixel.to(u.mas)
        sigmas_noise[idx] = sigma_noise.to(u.mas)
        sigmas_PO[idx] = sigma_PO.to(u.mas)
    return sigmas, sigmas_PSF, sigmas_pixel, sigmas_noise, sigmas_PO


@u.quantity_input
def noise_overview(telescope, lam_low: u.um = .5*u.um, lam_up: u.um = 25*u.um,
                   n_lam=1000, xlims=(0, 25), log=True):
    """Produce a plot detailing the noise behaviour for a telescope.

    Produce a plot detailing the noise behaviour over wavelength for a given
    telescope.

    Parameters
    ----------
    telescope (str):
        Telescope for which to perform the analysis.
    lam_low (astropy.Quantity):
        Lowest wavelength in the range.
    lam_up (astropy.Quantity):
        Highest wavelength in the range.
    n_lam (int):
        Number of wavelengths to sample over the interval [lam_low, lam_up].
    xlims (2-tuple of floats):
        x-limits for which to plot, expressed in microns.
    log (bool):
        Whether to plot the noise overview logarithmically.
    """
    lams = np.linspace(lam_low.to(u.um).value, lam_up.to(u.um).value,
                       num=n_lam, endpoint=True)*u.um
    # First calculate the benchmark/reference case with the loosest constraints
    # on the analytical parameters k_n and k_px
    sigmas, sigmas_PSF, sigmas_pixel, sigmas_noise, sigmas_PO = (
        calc_sigma_pphoton_telescope(lams, telescope))
    # Also calculate the best-case value
    sigmas_constr, _, _, _, _ = calc_sigma_pphoton_telescope(lams, telescope,
                                                             k_px=1/6, k_n=1/2)
    # And the 10x Airy PSF case:
    sigmas_large_PSF, sigmas_PSF_large, _, sigmas_noise_large, _ = (
        calc_sigma_pphoton_telescope(lams, telescope,
                                     PSF_multiplier=3))
    # Create the figure
    fig = plt.figure()
    # Add an axis
    ax = fig.add_subplot(111)
    # Plot the benchmark case
    ax.plot(lams, sigmas, linestyle="solid", color="k")
    # Plot the benchmark case
    ax.plot(lams, sigmas, linestyle="solid", color="k",
            label="Most conservative constraints")
    # Plot the photon noise
    ax.plot(lams, sigmas_PSF, linestyle="solid", color="orange", alpha=0.5,
            label="Photon noise")
    # Plot the 10x PSF noise
    # ax.plot(lams, sigmas_PSF_large, linestyle="solid", color="orange",
    #         label="3x PSF photon noise")
    # Plot the pixel noise
    ax.plot(lams, sigmas_pixel, linestyle="solid", color="gray",
            label="Pixel noise")
    # Plot the flux noise
    ax.plot(lams, sigmas_noise, linestyle="solid", color="green",
            label="Flux noise")
    # Plot the pointing noise
    ax.plot(lams, sigmas_PO, linestyle="solid", color="blue",
            label="Pointing noise")
    # Plot the best-constrained case
    ax.plot(lams, sigmas_constr, linestyle="dashed", color="k",
            label="Tightest constraints")
    # Plot the 10x PSF case
    # ax.plot(lams, sigmas_large_PSF, linestyle="dotted", color="k",
    #         label="3x PSF total noise")
    # Plot the 10x PSF flux noise
    # ax.plot(lams, sigmas_noise_large, linestyle="dotted", color="green",
    #         label="3x PSF flux noise")
    ax.set_xlabel("Wavelength [$\\mu$m]")
    ax.set_ylabel("Noise [mas]")
    if log:
        ax.set_yscale("log")
    # max_val_bench = np.nanmax(sigmas_large_PSF)
    # ylims = (.5, max_val_bench.to(u.mas).value)
    # ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.legend(loc="lower right")

    # %% Class definitions: Filter and FilterLibrary


class Filter:
    """Generalised filter-saving class.

    Class that contains the transmission curve, an interpolant thereof and some
    useful properties of the filter. As it is intended for use with
    spectroastrometry, only gives coronagraphic filters (for now).

    Attributes
    ----------
    transmission_curve (array of floats):
        Transmission values (expressed as fraction) per wavelength.
    wavs (array of astropy.Quantity):
        Wavelengths at which the transmission curve is sampled.
    trans_interp (PchipInterpolator or interp1d):
        Interpolant of the transmission curve. Takes a dimensionless wavelength
        (expressed in microns) as input, returns the transmission coefficient
        at that wavelength.
    name (str):
        Name of the filter.
    lam_pivot (astropy.Quantity):
        Pivot wavelength of the filter (sometimes denoted as central
        wavelength instead).
    delta_lam (astropy.Quantity):
        Bandwidth of the filter.
    R (float):
        Spectral resolution of the filter.
    square (bool):
        Whether the filter is square or not i.e. it is True for those filters
        where no numerical transmission curves are available (those on METIS),
        and False for those where a numerical curve is in fact available.
    instrument (str):
        The instrument the filter belongs to.
    telescope (str):
        The telescope the filter belongs to.
    PSF_sigma (astropy.Quantity):
        The standard deviation of the PSF. Expressed in mas.
    diameter (astropy.Quantity):
        Diameter of the telescope the filter is used on.
    pixel_size (astropy.Quantity):
        Pixel-size of the detector of the instrument the filter belongs to.
        Expressed in mas.
    PO_sigma (astropy.Quantity):
        Pointing inaccuracy of the telescope, expressed as standard deviation
        of the on-sky position.
    sigma_totpp (astropy.Quantity):
        Total noise per photon of the filter.
    bb_temp_interp (func):
        Interpolant for the blackbody temperature as function of the intensity
        through this filter.
    bb_temp_acc (astropy.Quantity):
        Accuracy of the interpolant, expressed as the distance between sampled
        gridpoints.

    Methods
    -------
    plot:
        Plot the transmission curve of the filter.
    blackbody_intensity:
        Calculate the blackbody intensity through the filter for a temperature.
    """

    def __init__(self, instrument, filter_name, interp_method="PCHIP",
                 square_transmission=1.):
        """Read the transmission curve of a filter and creates an interpolant.

        Read the transmission curve from a .txt file (or equivalent) and
        create an interpolant for the curve.

        Parameters
        ----------
        instrument (str):
            Which instrument/subsystem the filter belongs to. Currently
            allowed arguments are:
            NIRCam_HCI, SPHERE, METIS, MIRI_HCI_rect
        filter_name (str):
            Name of the filter.
        interp_method (str):
            Which interpolation method to use. Supported are PCHIP (for
            PchipInterpolator) and all methods that interp1d supports.
        square_transmission (float):
            Which value to set square filters to (i.e. those for which no
            numerical transmission curves are available). Default is 1., but
            one might want to set ~.8 for ground-based telescopes, based
            off of VLT values, and for space telescopes one might prefer to
            use values akin to JWST (~.5).
        """
        allowed_instruments = ["NIRCam_HCI", "SPHERE", "METIS",
                               "MIRI_HCI_rect",
                               "NIRCam_HCI/MIRI_HCI_rect"]
        if instrument not in allowed_instruments:
            raise ValueError(f"{instrument} is not a supported instrument.")
        self.instrument = instrument

        # As transmission curves for METIS are not provided, we shall define
        # it slightly differently than the others. Hence, split into the
        # non-METIS and METIS cases; same for MIRI since we do not have the
        # necessary packages installed at the moment.
        if instrument not in ["METIS", "MIRI_HCI_rect"]:
            # Split into the different instruments
            if instrument == "NIRCam_HCI":
                self.telescope = "JWST"
                if filter_name not in supported_filters["JWST"]["NIRCam_HCI"]:
                    raise ValueError(
                        f"{filter_name} is not a supported filter.")
                file_path = "Filter transmission curves\\JWST" + \
                    "\\nircam_throughputs\\mean_throughputs" + \
                    f"\\{filter_name}_mean_system_throughput.txt"
                # Load the data. Because of the formatting, we skip the
                # first row
                data = np.loadtxt(file_path, skiprows=1)
                # Assign attributes
                self.wavs = data[:, 0]*u.um
                self.transmission_curve = data[:, 1]
                self.name = filter_name

            if instrument == "SPHERE":
                self.telescope = "VLT"
                if filter_name not in supported_filters["VLT"]["SPHERE"]:
                    raise ValueError(
                        f"{filter_name} is not a supported filter.")
                file_path = "Filter transmission curves\\VLT SPHERE IRDIS" + \
                    f"\\SPHERE_IRDIS_{filter_name}.dat"
                data = np.loadtxt(file_path)
                self.wavs = (data[:, 0]*u.nm).to(u.um)
                self.transmission_curve = data[:, 1]
                self.name = filter_name

            # Create an interpolant for the data using the relevant method
            # Outside the interpolation interval, we can of course always set
            # the result to zero.
            if interp_method == "PCHIP":
                interp = PchipInterpolator(self.wavs.value,
                                           self.transmission_curve,
                                           extrapolate=False)

                def transmission_func(lam):
                    trans = interp(lam)
                    # Set NaN-values to be zero and return
                    return np.nan_to_num(trans)
                self.trans_interp = transmission_func
            else:
                self.trans_interp = interp1d(self.wavs.value,
                                             self.transmission_curve,
                                             kind=interp_method,
                                             bounds_error=False,
                                             fill_value=0.)
            # Set square to False
            self.square = False
            # Calculate the central (i.e. pivot) wavelength and bandwidth
            # Do this in line with the definitions used by the STScI for JWST
            # For simplicity, let us use Simpson's rule as implemented in scipy
            self.delta_lam = sp.integrate.simpson(self.transmission_curve,
                                                  self.wavs.to(
                                                      u.um).value)*u.um
            self.lam_pivot = np.sqrt(
                (sp.integrate.simpson(self.transmission_curve*self.wavs,
                                      self.wavs.value)) /
                (sp.integrate.simpson(self.transmission_curve/self.wavs,
                                      self.wavs.value)))*self.wavs.unit

        if instrument == "METIS":
            """
            Based on Brandl, B. R., Bettonvil, F. C. M., Boekel, R. van,
            Glauser, A., Quanz, S., Absil, O., … Winckel, H. van. (2021).
            METIS: The Mid-infrared ELT Imager and Spectrograph. The Messenger
            (Eso), 182, 22-26. doi:10.18727/0722-6691/5218
            """
            self.telescope = "ELT"
            if filter_name not in supported_filters["ELT"]["METIS"]:
                raise ValueError(f"{filter_name} is not a supported filter.")
            self.name = filter_name
            self.square = True
            # Find which index corresponds to the filter
            filter_idx = supported_filters["ELT"]["METIS"].index(filter_name)
            # Set the pivot wavelength and bandwidth
            self.lam_pivot = METIS_lam_pivot[filter_idx]
            self.delta_lam = METIS_delta_lam[filter_idx]
            # Set the "interpolant" (for a square filter we may as well use an
            # exact function):

            def transmission_func(lam):
                # Within the filter, set the value to square_transmission:
                # outside, set it to zero.
                trans = np.where(np.abs(
                    self.lam_pivot - lam*u.um) < self.delta_lam/2,
                    square_transmission, 0.)
                return trans
            self.trans_interp = transmission_func
        if instrument == "MIRI_HCI_rect":
            """
            Square filter-approximation of MIRI filters, as transmission curves
            require the use of Pandeia. Values taken from
            https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-coronagraphic-imaging
            """
            self.telescope = "JWST"
            if filter_name not in supported_filters["JWST"]["MIRI_HCI_rect"]:
                raise ValueError(f"{filter_name} is not a supported filter.")
            self.name = filter_name
            self.square = True
            # Find which index corresponds to the filter
            filter_idx = supported_filters["JWST"]["MIRI_HCI_rect"].index(
                filter_name)
            # Set the pivot wavelength and bandwidth
            self.lam_pivot = MIRI_rect_lam_pivot[filter_idx]
            self.delta_lam = MIRI_rect_delta_lam[filter_idx]
            # Set the "interpolant" (for a square filter we may as well use an
            # exact function):

            def transmission_func(lam):
                # Within the filter, set the value to square_transmission:
                # outside, set it to zero. For MIRI_rect, we shall always
                # use a transmission value of .25.
                trans = np.where(np.abs(
                    self.lam_pivot - lam*u.um) < self.delta_lam/2,
                    .25, 0.)
                return trans
            self.trans_interp = transmission_func

        # Set the PSF. While this is somewhat laborious, we shall set it
        # explicitly per instrument such that we may eventually input
        # simulated PSFs if necessary. Also set the pixel size
        if instrument == "NIRCam_HCI":
            self.PSF_sigma = (.45*self.lam_pivot/diameter["JWST"]*u.rad).to(
                u.mas)
            self.diameter = diameter["JWST"]
            self.PO_sigma = 1*u.mas
        if instrument == "MIRI_HCI_rect":
            self.PSF_sigma = (.45*self.lam_pivot/diameter["JWST"]*u.rad).to(
                u.mas)
            self.diameter = diameter["JWST"]
            self.PO_sigma = 1*u.mas
            # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-imaging
        if instrument == "SPHERE":
            self.PSF_sigma = (.45*self.lam_pivot/diameter["VLT"]*u.rad).to(
                u.mas)
            self.diameter = diameter["VLT"]
            self.PO_sigma = 1*u.arcsec  # Tentative; cannot yet find source
            # https://www.eso.org/sci/facilities/paranal/instruments/sphere/doc/VLT-MAN-SPH-14690-0430_v96.pdf
        if instrument == "METIS":
            self.PSF_sigma = (.45*self.lam_pivot/diameter["ELT"]*u.rad).to(
                u.mas)
            self.diameter = diameter["ELT"]
            self.PO_sigma = 1*u.mas
        # Set the pixel size of the filter using pixel_size
        self.pixel_size = pixel_size(self.telescope, self.lam_pivot,
                                     instrument=instrument)
        # Calculate the spectral resolution of the filter
        self.R = float(self.lam_pivot/self.delta_lam)
        # Calculate the total noise per photon for the filter
        self.sigma_totpp = calc_sigma_pphoton_telescope(
            np.array([self.lam_pivot.value])*self.lam_pivot.unit,
            self.telescope)[0][0]
        #######################################################################
        # Create an interpolant for the blackbody temperature: if the file
        # "filter_name.txt" already exists in the folder
        # "Filter blackbody intensities" in the working directory, load that;
        # otherwise, create it
        temp_acc = 1 * u.K
        min_temp = 10 * u.K
        max_temp = 1000 * u.K
        bb_filter_directory = "Filter blackbody intensities\\"
        file_path = bb_filter_directory + f"{filter_name}.txt"
        if not os.path.exists(file_path):
            print(f"Creating {file_path}...")
            # Create the interpolant
            temp_arr = np.arange(min_temp.value,
                                 max_temp.value, temp_acc.value)*u.K
            intens_arr = np.zeros(temp_arr.shape)*u.ph/u.s/u.m**2/u.mas**2
            for temp_idx, temp in enumerate(tqdm.tqdm(temp_arr)):
                intens_arr[temp_idx] = self.blackbody_intensity(temp)
            # Ensure that temp_arr is in units of K, intens_arr in
            # u.ph/u.s/u.m**2/u.mas**2:
            temp_arr = temp_arr.to(u.K)
            intens_arr = intens_arr.to(u.ph/u.s/u.m**2/u.mas**2)
            # Save the temperature and corresponding intensities in a .txt-file
            # The first column will contain the temperatures and the second
            # column the corresponding intensities
            save_arr = np.zeros((temp_arr.shape[0], 2))
            save_arr[:, 0] = temp_arr.value
            save_arr[:, 1] = intens_arr.value
            # Write a header for the file
            header = ("BB temperature (K);" +
                      " Intensity through filter (ph/s/m^2/mas^2)")
            np.savetxt(file_path, save_arr, header=header)
        else:
            # Load the interpolating values
            save_arr = np.loadtxt(file_path)
            temp_arr = save_arr[:, 0]*u.K
            intens_arr = save_arr[:, 1]*intens_unit
        # Create an interpolant for temperature as function of intensity
        interp = sp.interpolate.interp1d(intens_arr, temp_arr,
                                         kind="linear", bounds_error=False)

        @u.quantity_input
        def interp_func(intens: intens_unit):
            intens_unitless = intens.to(intens_unit).value
            temp = interp(intens_unitless)*u.K
            return temp
        self.bb_temp_interp = interp_func
        self.bb_temp_acc = temp_acc

    def plot(self, xwindow=None, n_samples=1000):
        """Plot the transmission curve of the filter.

        Plots the transmission curve of the filter over the specified window.
        To do this, it samples evenly over the interval xwindow.

        Parameters
        ----------
        xwindow (list of floats or NoneType):
            Wavelength window over which to plot the transmission curve. Must
            be expressed in microns.The default is None. If set to None, use
            the extent of self.wavs as window or in the case of a square
            filter, set it to [.5, 25.]
        n_samples (int):
            Number of places at which to sample the transmission.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if xwindow is None:
            if self.square:
                xwindow = [.5, 25.]
            else:
                xwindow = [self.wavs.min().value, self.wavs.max().value]
        sample_wavs = np.linspace(xwindow[0], xwindow[1], n_samples, True)*u.um
        ax.plot(sample_wavs.value, self.trans_interp(sample_wavs.value),
                color="k")
        ax.set_xlabel("Wavelength [$\\mathrm{\\mu m}$]")
        ax.set_ylabel("Transmission [-]")
        ax.set_xlim(xwindow)
        ax.set_ylim((0., 1.))

    @u.quantity_input
    def blackbody_intensity(self, temp: u.K, square_samples=100):
        """
        Calculate the blackbody intensity through the filter for a temperature.

        Calculate the blackbody intensity through the filter for a given
        temperature. BB calculation based on code provided by Elina Kleisioti
        validated against http://www.sun.org/encyclopedia/black-body-radiation
        [16 Nov 2022].

        Parameters
        ----------
        temp (astropy.Quantity):
            Temperature of the blackbody in question.
        square_samples (int):
            If the filter is square, the number of samples to use. Otherwise
            not used.

        Returns
        -------
        bb_intens (astropy.Quantity):
            Blackbody intensity at the given temperature, expressed in terms
            of photons per second per m^2 per mas^2.
        """
        try:
            bb_intens = np.zeros(temp.shape)*intens_unit
            for temp_idx, T in enumerate(temp):
                # Check if the input is close to 0 K; then return 0.
                # The value of 20 K is based off the fact that we should never
                # reach this low a temperature, and 1 K yields overflow errors
                # in expm1.
                if np.abs(T) <= 20*u.K:
                    bb_intens[temp_idx] = .0001*intens_unit
                # For square filters, do not use self.wavs but instead sample
                # a number of points over the range of the filter.
                if self.square:
                    lams = np.linspace(self.lam_pivot - self.delta_lam,
                                       self.lam_pivot + self.delta_lam,
                                       square_samples)
                    trans_curve = np.ones((lams.shape)) *\
                        self.trans_interp(self.lam_pivot.to(u.um).value)
                else:
                    lams = self.wavs
                    trans_curve = self.transmission_curve
                # First calculate the blackbody intensity at each value and
                # correct it by the transmission value
                constant = 2 * const.c / np.power(lams, 4) / (1.0*u.sr)*u.ph
                # The above line is changed (div. by hc/lam) such that it now
                # corresponds to a photon flux.
                exponent = const.h * const.c / (lams * const.k_B * T)
                planck_val = constant/np.expm1(exponent)
                corr_vals = (planck_val*trans_curve).to(
                    intens_unit/u.um)
                # Integrate over wavelength
                bb_intens[temp_idx] = np.trapz(corr_vals, lams).to(intens_unit)
        except TypeError:
            # Check if the input is close to 0 K; then immediately return 0.
            # The value of 20 K is based off the fact that we should never
            # reach this low a temperature, and 1 K yields overflow errors in
            # expm1.
            if np.abs(temp) <= 20*u.K:
                return 0.0001*intens_unit
            # For square filters, do not use self.wavs but instead sample
            # a number of points over the range of the filter.
            if self.square:
                lams = np.linspace(self.lam_pivot - self.delta_lam,
                                   self.lam_pivot + self.delta_lam,
                                   square_samples)
                trans_curve = np.ones((lams.shape)) *\
                    self.trans_interp(self.lam_pivot.to(u.um).value)
            else:
                lams = self.wavs
                trans_curve = self.transmission_curve
            # First calculate the blackbody intensity at each value and
            # correct it by the transmission value
            constant = 2 * const.c / np.power(lams, 4) / (1.0*u.sr)*u.ph
            # The above line is changed (div. by hc/lam) such that it now
            # corresponds to a photon flux.
            exponent = const.h * const.c / (lams * const.k_B * temp)
            planck_val = constant/np.expm1(exponent)
            corr_vals = (planck_val*trans_curve).to(
                intens_unit/u.um)
            # Integrate over wavelength
            bb_intens = np.trapz(corr_vals, lams).to(intens_unit)
        return bb_intens

    @u.quantity_input
    def find_blackbody_temp(self, intensity: intens_unit,
                            temp_acc=.1*u.K, max_loops=100):
        """
        Calculate the blackbody temperature corresponding to an intensity.

        Calculate the temperature a black body must have to have the given
        intensity. Uses a bisection algorithm with the initial interval
        based on a guess from the precomputed interpolant.

        Parameters
        ----------
        intensity (astropy.Quantity):
            Intensity for which to calculate the corresponding blackbody
            temperature.
        temp_acc (astropy.Quantity):
            Accuraccy with which to determine the temperature.

        Returns
        -------
        temp_est (astropy.Quantity):
            Temperature estimate (accurate to within temp_acc).
        """
        guess_temp = self.bb_temp_interp(intensity)
        # Retrieve the interval in which the guessed temperature lay
        low_temp = guess_temp - guess_temp % self.bb_temp_acc
        high_temp = low_temp + self.bb_temp_acc
        # Initialise the bisection algorithm
        diff = 1e3*u.K
        num_loops = 0
        while (diff > temp_acc) and (num_loops < max_loops):
            # Calculate the midpoint values for this interval
            mid_temp = (low_temp + high_temp)/2
            mid_intens = self.blackbody_intensity(mid_temp)
            # And the corresponding difference value
            diff = np.abs(high_temp - low_temp)
            # As the intensity is strictly increasing, if the intensity is
            # greater than the one we're looking for, mid_temp was too high
            # and so the desired temperature lies in the interval
            # [low_temp, mid_temp]
            if mid_intens > intensity:
                high_temp = mid_temp
            else:
                low_temp = mid_temp
            num_loops += 1
        if num_loops >= max_loops:
            print("Maximum number of bisection steps exceeded!")
        temp_est = (low_temp + high_temp)/2
        return temp_est


# Class definition: FilterLibrary


class FilterLibrary:
    """Collection of Filters for easy access.

    A custom storage-class for objects of the Filter class. Allows easy reading
    of filters into a compatible format, for ease of use.

    Attributes
    ----------
    filters (dict):
        Dictionary of Filter objects, each accessible by their name.
    supported_filters (dict of dict):
        Dictionary of dictionaries, each corresponding to a telescope
        respectively an instrument, containing the names of the filters on
        that instrument on that telescope that FilterLibrary contains e.g.
        supported_filters[Telescope][Instrument] yields the filters on
        instrument Instrument of telescope Telescope.

    Methods
    -------
    telescope_from_instrument:
        Return on which telescope an instrument is located.

    """

    def __init__(self, interp_method="PCHIP", square_transmission=.8):
        """Initialise the FilterLibrary object.

        Create a FilterLibrary object by reading all filters. Create
        interpolants according to interp_method.

        Parameters
        ----------
        interp_method (str):
            Interpolation method to use for the filters. Default is "PCHIP".
        square_transmission (float):
            Transmission value to set for square filters.
        """
        self.filters = {}
        # Suggestion for the future: convert this to a more general form using
        # the outline given in
        # https://stackoverflow.com/questions/18262293/how-to-open-every-file-in-a-folder
        # Suggestion for the future: add non-coronagraph compatible filters!
        # Create a dictionary of dictionaries to go through
        # supported_filters = {}
        # supported_filters["JWST"] = {}
        # supported_filters["JWST"]["NIRCam_HCI"] = [
        #     "F322W2",
        #     "F200W", "F277W", "F356W", "F444W",
        #     "F182M", "F210M", "F250M", "F300M", "F335M", "F360M", "F410M",
        #     "F430M", "F460M", "F480M",
        #     "F187N", "F212N"]

        # supported_filters["VLT"] = {}
        # supported_filters["VLT"]["SPHERE"] = [
        #     "B_H", "B_J", "B_Ks", "B_Y",
        #     "D_H23", "D_J23", "D_K12", "D_ND-H23", "D_Y23",
        #     "N_BrG", "N_CntH", "N_CntJ", "N_CntK1", "N_CntK2", "N_CO",
        #     "N_FeII", "N_H2", "N_HeI", "N_PaB"]

        # supported_filters["ELT"] = {}
        # supported_filters["ELT"]["METIS"] = [
        #     "L'", "short-L", "HCI-L short", "HCI-L long", "M'", "N1", "N2",
        #     "H2O-ice", "PAH 3.3", "Br-alpha", "CO(1-0)-ice", "PAH 8.6",
        #     "PAH 11.25", "[S IV]", "[Ne II]"]
        # Save each filter in the right location
        self.supported_filters = supported_filters
        for telescope in supported_filters.keys():
            for instrument in supported_filters[telescope].keys():
                for filter_name in supported_filters[telescope][instrument]:
                    # If the instrument is the "virtual instrument" of the
                    # combination of NIRCam and MIRI, split into two cases
                    # and set the instrument manually; otherwise, proceed
                    # as normal
                    if instrument == "NIRCam_HCI/MIRI_HCI_rect":
                        if filter_name in supported_filters[telescope][
                                "NIRCam_HCI"]:
                            self.filters[filter_name] = Filter(
                                "NIRCam_HCI", filter_name,
                                interp_method=interp_method,
                                square_transmission=square_transmission)
                        else:
                            self.filters[filter_name] = Filter(
                                "MIRI_HCI_rect", filter_name,
                                interp_method=interp_method,
                                square_transmission=square_transmission)
                    else:
                        self.filters[filter_name] = Filter(
                            instrument, filter_name,
                            interp_method=interp_method,
                            square_transmission=square_transmission)

    def telescope_from_instrument(self, instrument):
        """Return on which telescope an instrument is located.

        Utility function that yields on which telescope an instrument is
        located.

        Parameters
        ----------
        instrument (str):
            The instrument in question.

        Returns
        -------
        telescope (str):
            The telescope on which the instrument is located.
        """
        # Go through all telescopes in the keys of self.supported_filters, and
        # return the first telescope that has an instrument called instrument
        for telescope in self.supported_filters.keys():
            if instrument in self.supported_filters[telescope].keys():
                return telescope


# Instantiate an object of the FilterLibrary class; this will always be
# available when this toolkit is loaded
filters = FilterLibrary()
