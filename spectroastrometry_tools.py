"""
Created on Sun Jan 29 13:22:01 2023.

@author: Quirijn B. van Woerkom
A Class and corresponding methods for spectroastrometric analysis
and simulations of hypothetical tidally heated exomoons around nearby stars.

Note: due to notation changes in the paper, the conventions used in this
code do not entirely align with the notation used there. The parameter
gamma is called p here, and instead of M the moon filter is called F.
"""
# Standard imports
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy.table import Table, Column
import time
import tqdm

# Specific imports
from scipy.interpolate import interpn, interp1d, PchipInterpolator
from matplotlib.ticker import ScalarFormatter
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
import os
# import pandas as pd
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm
from matplotlib.patches import Patch
from spectroastrometry_filters import (supported_filters, diameter,
                                       calc_sigma_pphoton, FilterLibrary)
from spectroastrometry_orbit import (Orbit, E0_guess, solve_Kepler,
                                     theta_from_E, xi_hat)
import datetime
# Plotstyle changes
# Increase the matplotlib font size
plt.rcParams.update({'font.size': 22})
# Set layout to tight
plt.rcParams.update({'figure.autolayout': True})
# Set grid to true
plt.rcParams.update({'axes.grid': True})

# %% Useful definitions
# Standard values
RIo = 1821.49*u.km  # Radius of Io ( https://ssd.jpl.nasa.gov/sats/phys_par/)
# Define the photon flux density a priori for speed-up purposes
ph_flux_dens = u.photon/u.s/u.um/u.m**2
ph_flux = u.photon/u.s/u.m**2
# Similarly, define the intensity in units of ph/s/m^2/mas^2
intens_unit = u.ph/u.s/u.m**2/u.mas**2
# Define the normalisation value p for the inclination for the lower, middle
# and upper bounds
K2 = sp.special.ellipk(1/2)**2
p_i = np.array([2/np.pi, 2*K2/np.pi**2 + 1/(2*K2), 1.])

# %% Auxiliary function definitions


@u.quantity_input
def bb_atdist(d: u.pc, r: u.km, lam: u.um, T: u.K):
    """Calculate the blackbody flux for an object at a given distance.

    Gives the result of the Planck function for (a) given wavelength(s) and
    blackbody temperature, and evaluates what the resulting flux density would
    be at a given distance from the Earth for a given size body.
    Taken from code provided by Elina Kleisioti validated against
    http://www.sun.org/encyclopedia/black-body-radiation [16 Nov 2022].

    Parameters
    ----------
    lam: array_like of scalars
        Wavelength(s) at which the Planck function is to be evaluated.
    T: scalar
        Temperature of the blackbody.
    d: scalar
        Distance of the blackbody from Earth.
    r: scalar
        Radius of the blackbody.

    Returns
    -------
    val: scalar
        Flux of the blackbody in W m^-2 um^-1.

    """
    # np.seterr(all='ignore')

    constant = 2 * const.h * const.c * const.c / np.power(lam, 5) / (1.0*u.sr)
    exponent = const.h * const.c / (lam * const.k_B * T)
    planck_val = constant/np.expm1(exponent)
    val = (planck_val * np.pi * u.sr * (r/d)**2).to(u.W/u.m**2/u.um)

    return val


@u.quantity_input
def star_from_temp(Teff: u.K, calc_luminosity=False):
    """Generate main sequence-star properties based off its spectral type.

    Interpolates from the data in "An Introduction to Stellar Astrophysics" by
    Francis LeBlanc (Appendix G) to predict stellar mass and radius. In case
    real properties of a star are available, one should of course prefer to
    use those. As performance should not be a concern here, use PCHIP
    interpolation.

    Parameters
    ----------
    Teff (astropy.Quantity):
        Effective temperature of the star. Should be between an M5 and O5 star
        i.e. 3170 K < Teff < 42000 K.
    calc_luminosity (bool):
        Whether to calculate (and return) the luminosity, too. Implemented
        in this manner to provide backward compatibility.

    Returns
    -------
    mass (astropy.Quantity):
        Mass of the star.
    radius (astropy.Quantity):
        Radius of the star.
    luminosity (astropy.Quantity):
        Luminosity of the star.
    """
    if not (3170 < Teff.value < 42000):
        raise ValueError(
            "Please provide a temperature within the interpolation range "
            "[3170, 42000] K.")
    T_arr = np.array([3170, 3520, 3840, 4410, 5150, 5560, 5940, 6650, 7300,
                      8180, 9790, 11400, 15200, 30000, 42000])  # K
    M_arr = np.array([.21, .40, .51, .67, .79, .92, 1.05, 1.4, 1.6, 2.0, 2.9,
                      3.8, 5.9, 17.5, 60.])  # Msun
    R_arr = np.array([.27, .50, .60, .72, .85, .92, 1.1, 1.3, 1.5, 1.7, 2.4,
                      3.0, 3.9, 7.4, 12.])  # Rsun
    L_arr = np.array([0.0066, 0.034, 0.070, 0.18, 0.46, 0.73, 1.4, 3.0, 5.7,
                      12, 48, 140, 730, 40000, 400000])  # Lsun
    results_arr = np.zeros((3, T_arr.shape[0]))
    results_arr[0, :] = M_arr
    results_arr[1, :] = R_arr
    results_arr[2, :] = L_arr
    interp = PchipInterpolator(T_arr, results_arr, axis=1)
    results = interp(Teff.value)
    if calc_luminosity:
        return results[0]*u.Msun, results[1]*u.Rsun, results[2]*u.Lsun
    else:
        return results[0]*u.Msun, results[1]*u.Rsun


@u.quantity_input
def planet_radius_from_mass(mass: u.Mearth, m1: u.Mearth = 10.55*u.Mearth,
                            r1: u.Rearth = 3.90*u.Rearth, k1=-.209594,
                            k2=.0799, k3=.413):
    """Estimate a planet(-like body)'s radius from its mass.

    Estimatea planet-like body's radius from it's mass, based on the relation
    given in Eq. 23 of Seager et al., 2007. Default values for the constants
    m1, r1, k1, k2 and k3 are those for MgSiO3 (perovskite) with mod. EOS;
    other compositions can be modelled by taking other values for these
    constants from Seager at al.'s Table 4. A valid approximation up until the
    mass of the planet in question is greater than ~40*m1.

    Parameters
    ----------
    mass (astropy.Quantity):
        Mass of the planet in question.
    m1 (astropy.Quantity):
        Scaling mass. The default is 7.38*u.Mearth.
    r1 (astropy.Quantity):
        Scaling radius. The default is 3.58*u.Rearth.
    k1 (float):
        Dimensionless constant k_1. The default is -.20945.
    k2 (float):
        Dimensionless constant k_2. The default is .0804.
    k3 (float):
        Dimensionless constant k_3. The default is .394.

    Returns
    -------
    radius (astropy.Quantity):
        Radius of the object in question, expressed in Earth radii.

    """
    Ms = mass/m1
    radius = (r1*10**(k1 + 1/3*np.log10(Ms) - k2*(Ms)**(k3))).to(u.Rearth)
    return radius


# Instantiate an object of the FilterLibrary class; this will always be
# available when this toolkit is loaded and provides useful tools!
filters = FilterLibrary()


# %% LuminousObject class


class LuminousObject:
    """Abstraction of any luminous celestial object.

    Class that represents any luminous celestial object of interest in our
    system e.g. the star, planet or moon.

    Attributes
    ----------
    dist (astropy.Quantity):
        Distance from the observer.
    flux_density (array_like of astropy.Quantity):
        Integrated (over angular size) flux density
        of the object; shape of the array is equal to that of wavs, as that is
        where these values are sampled.
        N.B.: distance-dependent!
    wavs (astropy.Quantity):
        Wavelengths at which the flux density is provided.
    radius (astropy.Quantity):
        Radius of the body.
    Teff (astropy.Quantity):
        Effective temperature of the body (if applicable).
    age (astropy.Quantity):
        Age of the body, if applicable.
    mass (astropy.Quantity):
        Mass of the body, if applicable.
    grid (str):
        Which grid/library the spectrum is read from, if applicable.
    clouds (str):
        Cloud model of the spectrum, if applicable.
    metallicity (str):
        Metallicity setting of the spectrum, if applicable.
    equil_chem (str):
        Vertical mixing model of the spectrum, if applicable.
    logg (float):
        Surface gravity of the spectrum, expressed in
        log([cm/s^2]), if applicable.
    name (str):
        Name of the body. Purely for accounting/plotting purposes.
    flux_density_interp (func):
        Interpolator for the flux density of the body. Always float -> float to
        forego issues with non-astropy compatible scipy modules; input must be
        expressed in um, output is given in photon flux density per wavelength.
        N.B.: distance-dependent!
    interpolate (bool):
        Whether an interpolant has been created for the body or not.
    fluxes_filter (dict of astropy.Quantity):
        Fluxes through a filter are saved in this dict once they have been
        calculated, so as to forego double calculations.
        N.B.: distance-dependent!
    r_mp (astropy.Quantity):
        Distance between moon and planet in the current epoch.
    orbit (Orbit):
        Object from a custom Orbit class that contains the orbit of the object
        about its host body.

    Methods
    -------
    set_blackbody:
        Sets the flux density of the body equal to that
        dictated by a black body.
    read_gas_giant:
        Reads the flux density from a precomputed gas giant
        grid such as Spiegel & Burrows 2012 or ATMO2020 by interpolation.
    create_interpolant:
        Creates an interpolant for the body using its flux density.
    through_square_filter:
        Calculates the photon flux through a square filter.
    through_filter:
        Calculates the photon flux through a filter.
    set_orbit:
        Sets the orbit of the object about its host body.
    calc_Roche_limit:
        Calculate the Roche limit for the object given a lunar density.
    calc_Hill_radius:
        Calculate the Hill radius for the object, if it has an orbit.
    """

    @u.quantity_input
    def __init__(self, dist: u.pc = 10*u.pc):
        """Initialise the LuminousObject.

        Initialise the LuminousObject class by specifiying its distance. If
        the distance is not provided, set it at a distance of 10 pc, such that
        all fluxes and magnitudes become absolute (identical to the
        MoonPlanetSystem default). Sets all other attributes to None (they will
        be changed later, but this eases error handling).
        """
        # Set attributes to None initially
        self.dist = dist.to(u.pc)
        self.flux_density = None
        self.wavs = None
        self.radius = None
        self.Teff = None
        self.age = None
        self.mass = None
        self.grid = None
        self.clouds = None
        self.metallicity = None
        self.equil_chem = None
        self.logg = None
        self.name = None
        self.flux_density_interp = None
        self.interpolate = False
        self.fluxes_filter = {}
        self.orbit = None

    @u.quantity_input
    def set_blackbody(self, Teff: u.K, radius: u.m, wavs: u.um, name="Unnamed",
                      interpolate=True, reflected_light_temp=None,
                      host_dist: u.AU = 10*u.AU, reflected_light_power=-20.5):
        """Give the object blackbody properties, potentially with reflection.

        Set the flux of the body as though it were a black body of temperature
        Teff and with the given radius. Sample the Planck curve at wavelengths
        set by wavs. Optionally adds a reflected-light correction.

        Parameters
        ----------
        Teff (astropy.Quantity):
            Effective temperature of the black body.
        radius (astropy.Quantity):
            Radius of the body.
        wavs (array_like of astropy.Quantity):
            Wavelengths at which the Planck curve is to be sampled.
        name (str):
            Name of the body. Purely for accounting/plotting purposes.
        interpolate (bool):
            Whether to create an interpolator for the flux density or not.
            Can be turned off for performance purposes.
        reflected_light_temp (NoneType or astropy.Quantity):
            Sets whether to include reflected light, too. Currently only
            implemented by simply taking Agol's moon spectrum as a lower
            bound. If None, do not include reflected light. If set to a
            temperature, set the reflected light to be that of a star with
            that temperature.
        host_dist (astropy.Quantity):
            If a reflected-light correction is added, this parameter must be
            the distance between host planet and star.
        reflected_light_power (astropy.Quantity):
            If a reflected-light correction is added, this parameter determines
            how great it is. Default (modelled as a lower bound off Agol's
            work) is -20.5.
        """
        self.radius = radius
        self.wavs = wavs
        self.Teff = Teff
        self.name = name
        # Add reflected light if necessary
        if reflected_light_temp is not None:
            R_moon = 0.2725 * u.Rearth  # (from
            # https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html)
            # Correct for the distance to the system, for the size of the moon,
            # for the position of the moon in the system and for the luminosity
            # of the host star
            refl_light_corr = ((
                np.ones(self.wavs.shape)*(10**(reflected_light_power
                                               ))*u.W/u.m**(2)/u.um *
                (1.34*u.pc/self.dist)**2*(self.radius/R_moon)**2)/(
                    host_dist.to(u.AU).value)**2 * star_from_temp(
                        reflected_light_temp, calc_luminosity=True)[2].to(
                            u.Lsun).value).to(u.W/u.um/u.m**2)
        else:
            refl_light_corr = np.array([0.])*u.W/u.um/u.m**2

        self.flux_density = (bb_atdist(self.dist, self.radius, self.wavs,
                                       self.Teff) + refl_light_corr).to(
            ph_flux_dens,
            equivalencies=u.spectral_density(
                self.wavs))
        if interpolate:
            # For a black body, of course, it would be a waste to calculate
            # an interpolant!
            self.flux_density_interp = lambda wav: bb_atdist(
                self.dist,
                self.radius, wav*u.um,
                self.Teff).to(
                    ph_flux_dens,
                    equivalencies=u.spectral_density(
                        wav*u.um)).value + refl_light_corr[0].to(
                            ph_flux_dens,
                            equivalencies=u.spectral_density(wav*u.um)
            ).value
        self.interpolate = interpolate

    ###########################################################
    # ADD ANY DISTANCE-DEPENDENT ATTRIBUTES TO THIS FUNCTION! #
    ###########################################################

    @u.quantity_input
    def reset_distance(self, new_dist: u.pc):
        """Reset the body to be at a different distance.

        Reset the distance to the body and scale all quantities appropriately.
        To avoid unintended behaviour, preferably avoid using this.

        Parameters
        ----------
        new_dist (astropy.Quantity):
            New distance at which to place the body.
        """
        old_dist = self.dist
        self.dist = new_dist
        # Check whether the relevant attributes have already been assigned
        # a value and recalculate quantities.
        if self.flux_density is not None:
            self.flux_density *= (old_dist/new_dist)*(old_dist/new_dist)
        if self.flux_density_interp is not None:
            dummy_func = self.flux_density_interp
            self.flux_density_interp = lambda wav: dummy_func(wav) * \
                (old_dist/new_dist)*(old_dist/new_dist)
        for key in self.fluxes_filter.keys():
            self.fluxes_filter[key] = self.fluxes_filter[key] * \
                (old_dist/new_dist)*(old_dist/new_dist)
        if self.orbit is not None:
            self.orbit.dist = new_dist

    @u.quantity_input
    def read_gas_giant(self, age: u.Gyr, mass: u.Mjup, grid="ATMO2020",
                       clouds="cf", metallicity="1s", equil_chem="CEQ",
                       temp_warning=True, name="Unnamed", interpolate=True,
                       interp_method="PCHIP"):
        """Interpolate a gas giant spectrum from a precomputed grid.

        Set the flux of the body by reading them from either the
        Spiegel & Burrows 2012 grid or from the ATMO2020 grid by Phillips et
        al.; for a Spiegel-Burrows grid, one can moreover choose the clouds to
        be a hybrid model or cloud-free, and the metallicity to be solar or
        three times solar (as there's only two datapoints for metallicity,
        interpolation is not supported). For an ATMO2020 grid, one can choose
        whether to have chemical equilibrium, strong mixing or weak mixing.

        Now also supports the Solar System calibration spectra from the STScI.
        In that case, set grid='STScI' and name to the giant you wish to
        model.

        Parameters
        ----------
        age (astropy.Quantity):
            Age of the gas giant. N.B.: ATMO2020 provides
            ages up to 10 Gyr or until the planet cools below 200K, whichever
            occurs first. SB2012 provides ages between 1 Myr and 1 Gyr.
        mass (astropy.Quantity):
            Mass of the gas giant. SB2012 provides masses
            between 1 and 15 Mjup; ATMO2020 goes from .001 Msun to .075 Msun,
            equivalent to roughly 1.05 Mjup to 78.57 Mjup.
        grid (str):
            The grid to use; choices are SB2012 for Spiegel & Burrows'
            2012 grid, or ATMO2020 for that of Phillips et al.; SB2012 is
            likely better suited for various scenarios of younger planets
            (<1 Gyr), whereas ATMO2020 seems better suited for planets > 1 Gyr.
            For ATMO2020, atmospheres of Teff > 2000 K are not valid and so
            should not be used, even if they are included as a possibility.
        clouds (str):
            If one uses SB2012, this can be set to cf for a cloud-free
            or to hy for a hybrid model.
        metallicity (str):
            For SB2012, set this to 1s for solar metallicity
            or to 3s for three times solar metallicity.
        equil_chem (str):
            For ATMO2020, set this to CEQ_new or CEQ for chemical equilibrium,
            NEQ_strong for strong vertical mixing and NEQ_weak for weak
            vertical mixing. Additionally, set this to CEQ_old to use the grid
            values with the old equation of state (only for comparison
            purposes).
        temp_warning (bool):
            sets whether a warning must be printed for effective temperatures
            exceeding 2000K. Default is true.
        name (str):
            Name of the object.
        interpolate (bool):
            Whether to also create an interpolant for the flux density. This
            does not affect interpolation in the grid interpolations through
            the precomputed grid; those will always occur and be linear.
        interp_method (str):
            Which method to use to interpolate if interpolate is set to True.
            Allowed methods are simply those for scipy.interpolate but also
            PCHIP, which uses scipy.interpolate.PchipInterpolator. This is
            a monotonic interpolator, which will therefore perform best and so
            is set to default.
        """
        # Save parameters
        self.age = age
        self.mass = mass
        self.grid = grid
        self.name = name
        # Check which grid we use
        if self.grid == "SB2012":
            # For SB2012, we shall also save the relevant parameters
            self.clouds = clouds
            self.metallicity = metallicity

            # Determine which grid values bound our chosen point in parameter
            # space
            mass_grid = np.arange(1., 16., 1.)*u.Mjup
            age_grid = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 12, 15, 20, 25, 30, 40, 50, 75,
                                 100, 125, 150, 200, 250, 300, 400, 500, 750,
                                 1000])*u.Myr

            if self.mass < np.min(mass_grid) or self.mass > np.max(mass_grid):
                raise ValueError(
                    "Please input a mass value within the interpolation"
                    " bounds")
            if self.age < np.min(age_grid) or self.age > np.max(age_grid):
                raise ValueError(
                    "Please input an age value within the interpolation"
                    " bounds")
            # Do this first for the mass
            for mass_idx, mass_val in enumerate(mass_grid[:-1]):
                if mass_val < self.mass <= mass_grid[mass_idx+1]:
                    mass_bounds = [mass_val.value, mass_grid[mass_idx+1].value]
                    break
            # Then for the age
            for age_idx, age_val in enumerate(age_grid[:-1]):
                if age_val < self.age <= age_grid[age_idx+1]:
                    age_bounds = [age_val.value, age_grid[age_idx+1].value]
                    break

            # Load each of the bounding spectra
            # Preallocate an array of shape (2, 2, 600)
            # N.B.: 600 is the number of wavelengths sampled in SB2012
            bound_spectra = np.zeros((2, 2, 600))
            # The first index shall be the mass, the second the age
            # Both in ascending order
            ############# Must be manually set to the correct path ###########
            spectra_path = "Gas giant spectrum libraries\\Spiegel-Burrows" + \
                "\\spectra\\spectra"
            ##################################################################
            for mass_idx, mass_val in enumerate(mass_bounds):
                # Format the mass into a string compatible with the notation
                # used in the grid files
                mass_str = str(int(mass_val.round(0)))
                while len(mass_str) < 3:
                    mass_str = "0" + mass_str
                for age_idx, age_val in enumerate(age_bounds):
                    # Similarly format the age
                    age_str = str(int(age_val.round(0)))
                    while len(age_str) < 4:
                        age_str = "0" + age_str
                    file_name = f"\\spec_{self.clouds}{self.metallicity}_" + \
                        f"mass_{mass_str}_age_{age_str}.txt"
                    file_path = spectra_path + file_name
                    data_table = np.loadtxt(file_path)
                    if (mass_idx == 0) and (age_idx == 0):
                        self.wavs = data_table[0, 1:]*u.um
                    bound_spectra[mass_idx, age_idx] = (
                        data_table[1, 1:]*u.mJy).to(
                            ph_flux_dens,
                            equivalencies=u.spectral_density(self.wavs))

            # Now use this to interpolate the data; correct the result for
            # distance, as SB2012 provides fluxes at distances of 10 pc
            self.flux_density = ((interpn((mass_bounds, age_bounds),
                                          bound_spectra,
                                          np.array([self.mass.to(u.Mjup).value,
                                                    self.age.to(u.Myr).value]))
                                 * u.W/u.m**2/u.um*(10*u.pc/self.dist)**2)[0]
                                 ).to(ph_flux_dens,
                                      equivalencies=u.spectral_density(
                                          self.wavs))
        elif self.grid == "ATMO2020":
            self.equil_chem = equil_chem
            # First check the evolutionary grid to find the effective
            # temperature, radius, and surface gravity corresponding to this
            # mass and age by interpolation.
            mass_grid = np.arange(.001, 0.076, 0.001)*u.Msun
            # Find the two bounding masses
            if self.mass < np.min(mass_grid) or self.mass > np.max(mass_grid):
                raise ValueError(
                    "Please input a mass value within the "
                    "interpolation bounds")
            for mass_idx, mass_val in enumerate(mass_grid[:-1]):
                if mass_val < self.mass <= mass_grid[mass_idx+1]:
                    mass_bounds = [mass_val.value, mass_grid[mass_idx+1].value]
                    break

            # Load the two evolutionary tracks
            if self.equil_chem in ["CEQ_new", "CEQ"]:
                track_path = ("Gas giant spectrum libraries\\ATMO2020 models"
                              "\\evolutionary_tracks\\evolutionary_tracks"
                              "\\ATMO_2020\\ATMO_CEQ_new"
                              "\\JWST_coron_NIRCAM_MASK430R")
                file_path_low = track_path + \
                    f"\\{mass_bounds[0].round(3)}_ATMO_CEQ_vega_NIRCAM.txt"
                file_path_high = track_path + \
                    f"\\{mass_bounds[1].round(3)}_ATMO_CEQ_vega_NIRCAM.txt"
            elif self.equil_chem in ["CEQ_old"]:
                track_path = ("Gas giant spectrum libraries"
                              "\\ATMO2020 models\\evolutionary_tracks"
                              "\\evolutionary_tracks\\ATMO_2020\\ATMO_CEQ_old"
                              "\\JWST_coron_NIRCAM_MASK430R")
                file_path_low = track_path + \
                    f"\\{mass_bounds[0].round(3)}_" + \
                    "ATMO_CEQ_vega_NIRCAM.txt"
                file_path_high = track_path + \
                    f"\\{mass_bounds[1].round(3)}_" + \
                    "ATMO_CEQ_vega_NIRCAM.txt"
            elif self.equil_chem in ["NEQ_strong"]:
                track_path = ("Gas giant spectrum libraries\\ATMO2020 models"
                              "\\evolutionary_tracks\\evolutionary_tracks"
                              "\\ATMO_2020\\ATMO_NEQ_strong\\JWST_coronagraphy"
                              "\\JWST_coron_NIRCAM_MASK430R")
                file_path_low = track_path + \
                    f"\\{mass_bounds[0].round(3)}_" + \
                    "ATMO_NEQ_strong_vega_NIRCAM.txt"
                file_path_high = track_path + \
                    f"\\{mass_bounds[1].round(3)}_" + \
                    "ATMO_NEQ_strong_vega_NIRCAM.txt"
            elif self.equil_chem in ["NEQ_weak"]:
                track_path = ("Gas giant spectrum libraries\\ATMO2020 models"
                              "\\evolutionary_tracks\\evolutionary_tracks"
                              "\\ATMO_2020\\ATMO_NEQ_weak\\JWST_coronagraphy"
                              "\\JWST_coron_NIRCAM_MASK430R")
                file_path_low = track_path + \
                    f"\\{mass_bounds[0].round(3)}_" + \
                    "ATMO_NEQ_weak_vega_NIRCAM.txt"
                file_path_high = track_path + \
                    f"\\{mass_bounds[1].round(3)}_" + \
                    "ATMO_NEQ_weak_vega_NIRCAM.txt"
            else:
                raise ValueError(
                    "Please choose a valid vertical mixing setting.")
            # Load the two bounding mass-files
            low_mass_data = np.loadtxt(file_path_low)
            high_mass_data = np.loadtxt(file_path_high)
            # Cut them to the age range corresponding to the most restrictive
            # data;
            min_rows = np.min(
                (low_mass_data.shape[0], high_mass_data.shape[0]))
            low_mass_data = low_mass_data[:min_rows]
            high_mass_data = high_mass_data[:min_rows]
            # Set up the data values (that we want to interpolate)
            # in a format consistent with scipy.interpolate.interpn
            ages = high_mass_data[:, 1]*u.Gyr
            ages_points = ages.value
            values_arr = np.zeros((2, min_rows, low_mass_data.shape[1]-2))
            for mass_idx in range(2):
                for age_idx in range(min_rows):
                    if mass_idx == 0:
                        mass_data = low_mass_data
                    else:
                        mass_data = high_mass_data
                    values_arr[mass_idx, age_idx, :] = mass_data[age_idx, 2:]
            # Perform the interpolation
            interp_values = interpn((mass_bounds, ages_points), values_arr,
                                    (self.mass.to(u.Msun).value,
                                     self.age.to(u.Gyr).value))[0]
            # Save all relevant values
            self.Teff = interp_values[0]*u.K
            # Check whether Teff exceeds 2000K and print a warning if it does
            if (self.Teff > 2000*u.K) and (temp_warning):
                print(
                    "Warning: effective temperatures greater than 2000 K are "
                    "not accurate!")
            self.radius = interp_values[2]*u.Rsun
            self.logg = interp_values[3]  # in units of log([cm/s^2])!!!

            # Now use these values to find the corresponding spectrum
            # This is done by interpolating, but this time in temperature and
            # log(g). Aside from this, it will all be analogous to the
            # procedure for SB2012 again;

            # Set up grid values
            logg_grid = np.arange(2.5, 6., .5)
            temp_grid_1 = np.arange(200., 600., 50.)*u.K  # The first part is
            # spaced per 50
            temp_grid_2 = np.arange(600., 3100., 100.) * \
                u.K  # the second per 100
            temp_grid = np.zeros(
                (temp_grid_1.shape[0]+temp_grid_2.shape[0]))*u.K
            temp_grid[:temp_grid_1.shape[0]] = temp_grid_1
            temp_grid[temp_grid_1.shape[0]:] = temp_grid_2
            # Do this first for the temperature
            for temp_idx, temp_val in enumerate(temp_grid[:-1]):
                if temp_val < self.Teff.to(u.K) <= temp_grid[temp_idx+1]:
                    temp_bounds = [temp_val.value, temp_grid[temp_idx+1].value]
                    break
            # Then for logg
            for logg_idx, logg_val in enumerate(logg_grid[:-1]):
                if logg_val < self.logg <= logg_grid[logg_idx+1]:
                    logg_bounds = [logg_val, logg_grid[logg_idx+1]]
                    break
            # Prepare the file locations
            if self.equil_chem in ["CEQ_new", "CEQ", "CEQ_old"]:
                ########################### Set file path ###################
                spectra_path = ("Gas giant spectrum libraries\\ATMO2020 models"
                                "\\atmosphere_models\\atmosphere_models"
                                "\\CEQ_spectra")
                #############################################################
                file_path_suffix = "CEQ.txt"
            elif self.equil_chem in ["NEQ_strong"]:
                ########################## Set file path #####################
                spectra_path = (
                    "Gas giant spectrum libraries\\ATMO2020 models"
                    "\\atmosphere_models\\atmosphere_models"
                    "\\NEQ_strong_spectra")
                ##############################################################
                file_path_suffix = "NEQ_strong.txt"
            elif self.equil_chem in ["NEQ_weak"]:
                ########################## Set file path #####################
                spectra_path = (
                    "Gas giant spectrum libraries\\ATMO2020 models"
                    "\\atmosphere_models\\atmosphere_models\\NEQ_weak_spectra")
                ##############################################################
                file_path_suffix = "NEQ_weak.txt"

            for temp_idx, temp_val in enumerate(temp_bounds):
                # Format the temperature into a string compatible with the
                # notation
                # used in the grid files
                temp_str = str(int(round(temp_val, -1)))
                for logg_idx, logg_val in enumerate(logg_bounds):
                    # Similarly format the age
                    logg_str = str(round(logg_val, 1))
                    file_name = f"\\spec_T{temp_str}_lg{logg_str}_" +\
                        file_path_suffix
                    file_path = spectra_path + file_name
                    data_table = np.loadtxt(file_path)
                    if (temp_idx == 0) and (logg_idx == 0):
                        bound_spectra = np.zeros((2, 2, data_table.shape[0]))
                        self.wavs = data_table[:, 0]*u.um
                    bound_spectra[temp_idx, logg_idx] = data_table[:, 1]
            # Now use this to interpolate the data; note that the resulting
            # flux is the flux at the surface of the giant, and so should
            # be multiplied by (radius/distance)**2 to yield the true flux
            # density
            self.flux_density = ((interpn((temp_bounds, logg_bounds),
                                          bound_spectra,
                                          np.array([self.Teff.to(u.K).value,
                                                    self.logg])) *
                                  u.W/u.m**2/u.um*(
                                      self.radius/self.dist)**2)[0]).to(
                                          ph_flux_dens,
                                          equivalencies=u.spectral_density(
                                              self.wavs))
        if grid == "STScI":
            # Use the STScI calibration spectra for the Solar System
            # gas giants
            # https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/solar-system-objects-spectra
            name_lowcase = self.name.lower()
            # Take radii from https://ssd.jpl.nasa.gov/planets/phys_par.html
            if name_lowcase == "jupiter":
                self.radius = 1*u.Rjup
                self.mass = 1*u.Mjup
            elif name_lowcase == "saturn":
                self.radius = (58232*u.km).to(u.Rjup)
                self.mass = (568.317e24*u.kg).to(u.Mjup)
            elif name_lowcase == "neptune":
                self.radius = (24622*u.km).to(u.Rjup)
                self.mass = (102.4092e24*u.kg).to(u.Mjup)
            elif name_lowcase == "uranus":
                self.radius = (25362*u.km).to(u.Rjup)
                self.mass = (86.8099e24*u.kg).to(u.Mjup)
            sol_ang = (np.pi*self.radius**2/self.dist**2*u.rad**2).to(
                u.arcsec**2)
            file_loc = r"Gas giant spectrum libraries\SS giants"
            filename = file_loc + \
                fr'\{name_lowcase}_solsys_surfbright_001.fits'
            with fits.open(filename) as hdul:
                data = hdul[1].data
                n_vals = hdul[1].header['NAXIS2']
                self.wavs = np.zeros((n_vals,))*u.um
                self.flux_density = np.zeros((n_vals,))*ph_flux_dens
                for idx in range(n_vals):
                    self.wavs[idx] = (data[idx][0]*u.angstrom).to(u.um)
                    self.flux_density[idx] = (data[idx][1]*u.erg/u.s/u.cm**2
                                              / u.angstrom/u.arcsec**2*sol_ang
                                              ).to(ph_flux_dens,
                                                   equivalencies=u.spectral_density(
                                                       self.wavs[idx]))

        # Create an interpolant if we set interpolate to True
        if interpolate:
            if interp_method != "PCHIP":
                interp_dummy = interp1d(self.wavs.value,
                                        self.flux_density.to(
                                            ph_flux_dens,
                                            equivalencies=u.spectral_density(
                                                self.wavs)).value,
                                        kind=interp_method)
                self.flux_density_interp = interp_dummy
            else:
                interp_dummy = PchipInterpolator(
                    self.wavs.value,
                    self.flux_density.to(
                        ph_flux_dens,
                        equivalencies=u.spectral_density(
                            self.wavs)).value)
                self.flux_density_interp = interp_dummy
        self.interpolate = interpolate

    def create_interpolant(self, interp_method="PCHIP"):
        """Create an interpolant for the body.

        Create an interpolant for the body in question. A flux density must
        already have been read/calculated!

        Parameters
        ----------
        interp_method:
            Determines the interpolation method; available choices are those
            used in interp1d and additionally PCHIP.
        """
        if self.flux_density is None:
            raise AttributeError("A flux density must be read or calculated"
                                 " before an interpolant can be created!")
        if interp_method != "PCHIP":
            interp_dummy = interp1d(self.wavs.value, self.flux_density.to(
                ph_flux_dens,
                equivalencies=u.spectral_density(
                    self.wavs)).value,
                kind=interp_method)
            self.flux_density_interp = interp_dummy
        else:
            interp_dummy = PchipInterpolator(
                self.wavs.value,
                self.flux_density.to(
                    ph_flux_dens,
                    equivalencies=u.spectral_density(self.wavs)).value
            )
            self.flux_density_interp = interp_dummy
        self.interpolate = True

    @u.quantity_input
    def through_square_filter(self, lambda_c: u.um, R_f, const_trans=1.,
                              integ_method="PCHIP"):
        """Calculate filter flux in photons for a square filter.

        Calculate the total photon flux through a filter
        centred on lambda_c with spectral resolution R_f, expressed in units of
        photons/m^2/um/s.
        OPTIMIZE: It may be useful to implement sp.integrate.simpson
        in the future if speed-up is required.

        Attributes
        ----------
        lambda_c (astropy.Quantity):
            Pivot wavelength of the filter.
        R_f (float):
            Spectral resolution of the filter.
        const_trans (float):
            Transmission value for those regions where the filter transmits.
        integ_method (str):
            Which integration method to use. Currently supports "Simpson" and
            "PCHIP"; "PCHIP" uses "PCHIP" interpolation followed by analytic
            evaluation of the integral of the resulting spline, whereas
            "Simpson" simply applies Simpson's rule.

        Returns
        -------
        ph_flux (astropy.Quantity):
            Photon flux through the filter.
        """
        if self.flux_density is None:
            raise AttributeError("A flux density must be read or calculated"
                                 " before the flux through a filter can be"
                                 " calculated!")
        if integ_method == "PCHIP":
            # Create an interpolant for the integrand wavelength*flux
            integrand = (
                PchipInterpolator(self.wavs.value,
                                  (const_trans*(
                                      self.flux_density.to(
                                          ph_flux_dens,
                                          equivalencies=u.spectral_density(
                                              self.wavs))).value)))
            anti_deriv = integrand.antiderivative()
            ph_flux = (anti_deriv(lambda_c.to(u.um).value*(1+.5/R_f)/u.um) -
                       anti_deriv(lambda_c.to(u.um).value*(1-.5/R_f))) * \
                ph_flux_dens*u.um
        elif integ_method == "Simpson":
            # Define an exact function for the filter:
            # this function will take a wavelength expressed as float in
            # wavelengths (explicitly not an astropy.Quantity for speed) and
            # return the transmission value of the square filter at that
            # wavelength
            def filter_func(lam):
                if (lambda_c.to(u.um).value*(1-.5/R_f) <
                        lam < lambda_c.to(u.um).value*(1+.5/R_f)):
                    return const_trans
                else:
                    return 0.
            # Use this to sample the flux at each wavelength
            flux_through_filter = np.zeros_like(self.flux_density.value)
            lams = self.wavs.to(u.um).value
            for idx, flux in enumerate(self.flux_density.to(
                ph_flux_dens,
                equivalencies=u.spectral_density(
                    self.wavs)).value):
                lam = lams[idx]
                flux_through_filter[idx] = flux*filter_func(lam)
            # Create a mask for all values where the flux is zero
            mask = np.where(flux_through_filter == 0., False, True)
            # Calculate the resulting total flux using Simpson's rule
            ph_flux = sp.integrate.simpson(flux_through_filter[mask],
                                           lams[mask])
        else:
            raise ValueError("Please input a supported integration method!")
        return ph_flux

    @u.quantity_input
    def through_filter(self, filter_name):
        """Calculate the photon flux through a given filter.

        Calculate the total photon flux through a filter. The
        filter must be one of those supported by the FilterLibrary class.
        Integration performed using sp.integrate.simpson.

        Parameters
        ----------
        filter_name (str):
            Filter through which to calculate the flux.

        Returns
        -------
        ph_flux (astropy.Quantity):
            Photon flux through the filter.
        """
        # Check whether the given filter is an allowed filter
        if filter_name not in filters.filters.keys():
            raise ValueError(f"Filter {filter_name} is not supported!")
        # If it is, we can proceed without error
        # Check whether the flux has already been calculated
        # if it has been, immediately return it to save calculation
        if filter_name in self.fluxes_filter.keys():
            return np.copy(self.fluxes_filter[filter_name])
        # extract the Filter object from filters
        filter_obj = filters.filters[filter_name]
        # Calculate an array with values for flux
        # Save this without units such that we will not experience any problems
        # when using scipy functions that ambiguously support astropy.Quantity
        wavs_unitless = self.wavs.to(u.um).value
        ph_fluxes = filter_obj.trans_interp(
            wavs_unitless)*self.flux_density.value
        ph_flux = sp.integrate.simpson(ph_fluxes,
                                       wavs_unitless)*ph_flux_dens*u.um
        # Save the flux for later reference
        self.fluxes_filter[filter_name] = ph_flux
        return ph_flux

    @u.quantity_input
    def set_orbit(self, M0, a=421800.*u.km, e=0.004, i=0.*u.deg, Omg=0.*u.deg,
                  omg=0.*u.deg, tau=0.*u.yr):
        """Set the orbit of the object.

        Sets the orbit of the object about its parent body.
        Defaults to the orbital elements of Io as given by
        https://ssd.jpl.nasa.gov/sats/elem/ (for a, e and i; Omg, omg and tau
        are set to zero, as those determine the orientation of the orbit with
        respect to the reference frame).

        Parameters
        ----------
        M0 (astropy.Quantity):
            Mass of the central body the object orbits.
        a (astropy.Quantity):
            Semi-major axis of the orbit.
        e (float):
            Eccentricity of the orbit.
        i (astropy.Quantity):
            Inclination of the orbit.
        Omg (astropy.Quantity):
            Right ascension of the ascending node.
        omg (astropy.Quantity):
            Argument of periapsis.
        tau (astropy.Quantity):
            Time of pericenter passage.
        """
        self.orbit = Orbit(a=a, e=e, i=i, Omg=Omg, omg=omg, M0=M0, tau=tau,
                           dist=self.dist)
        # Set the distance from the object to its host
        self.r_mp = np.linalg.norm(self.orbit.pos()[:2])

    @u.quantity_input
    def calc_Roche_limit(self, dens: u.g/u.cm**3 = 3530*u.kg/u.m**3):
        """
        Calculate the Roche limit for a given density. Default is Io's.

        Calculate the Roche limit for a given density. Default is Io's. Based
        on the expression given in Lissauer & de Pater's Fundamentals of
        Planetary Science.

        Parameters
        ----------
        dens (astropy.Quantity):
            Density of the moon for which to calculate the Roche limit.
            Default is Io's, from
            https://nssdc.gsfc.nasa.gov/planetary/factsheet/galileanfact_table.html
        """
        return ((9/(4*np.pi)*self.mass/dens)**(1/3)).to(u.km)

    def calc_Hill_radius(self):
        """
        Calculate the Hill radius for the object, if it has an orbit.

        Calculate the Hill radius for the object, if it has an orbit.
        Otherwise, return np.nan and print a warning.

        Returns
        -------
        R_H (astropy.Quantity):
            Hill radius of the object.
        """
        if self.orbit is not None:
            m2 = self.mass
            m1 = self.orbit.M0
            a = self.orbit.a
            return ((m2/(3*(m1+m2)))**(1/3)*a).to(a.unit)
        else:
            print("This object has no orbit, and therefore cannot have"
                  "a Hill radius!")
            return np.nan

    @u.quantity_input
    def period(self, a: u.km):
        """
        Calculate the period of an orbit around the object at distance a.

        Parameters
        ----------
        a : u.km
            Semi-major axis of the orbit around the object.

        Returns
        -------
        P : u.h
            Period of the orbit.
        """
        GM = const.G*self.mass
        return (2*np.pi*np.sqrt(a*a*a/GM)).to(u.h)


# %% class MoonPlanetSystem


class MoonPlanetSystem:
    """
    Representation of a satellite system around a planet.

    Class that represents a moon-planet system for use in spectroastrometric
    simulations/analysis.

    Attributes
    ----------
    dist (astropy.Quantity):
        Distance to the moon-planet system as measured from the observer.
    Moons (list of LuminousObject):
        The moons of the system, represented as a list of objects in the
        custom-created LuminousObject class, which allows
        creation of black bodies (or precomputed gas giants of mass 1 Mjup
        or greater, which is excessive for moons).
        May later add the possibility of reading precomputed moon spectra.
    Planet (LuminousObject):
        The host planet, represented as an object in the LuminousObject
        class. Can both be instantiated as a blackbody or read from
        one of several precomputed grids based on age and mass.
    Star (LuminousObject):
        The star of the system. Has no particular role to play but in
        contrast calculations. Nonetheless added for convenience.
    signals (list of dict of dict):
        A structure in which computed signal values are stored so as to forego
        expensive recalculation. Currently not used, but maintained to exist
        if necessary in the future.
    noises (list of dict of dict):
        A structure in which computed photon noise values are stored so as to
        forego expensive recalculation. If two identical keys are given,
        returns the noise in an individual filter instead. Currently not used,
        but maintained if necessary in the future.
    t (astropy.Quantity):
        Current time. Always set to zero initially.




    Methods
    -------
    reset_distance:
        Move the system to a new distance, without the necessity of redoing
        all calculations (i.e. it simply scales distance-sensitive quantities)
    create_blackbody_planet:
        Create a planet with a blackbody spectrum.
    create_gas_giant_from_grid:
        Read/interpolate a gas giant/brown dwarf spectrum from one of a
        variety of precomputed grids.
    create_moon_from_grid:
        Create a moon from a precomputed grid.
    create_blackbody_moon:
        Create a moon with a blackbody spectrum.
    create_blackbody_star:
        Create a star with a blackbody spectrum.
    create_blackbody_star_from_temp:
        Create a blackbody star solely from its temperature.
    create_epsIndiA:
        Create a star in the system analogous to epsilon Indi A.
    plot_spectra:
        Plot the spectra of all bodies in the system. Provided as a
        convenience function; publishable figures need some polishing.
    quick_create:
        Quickly create a reference system. Mostly based off of eps Indi A(b)
        with a moon added with the specified temperature and radius.
    through_filter:
        Calculate the total photon flux of the planet and its moon(s) through
        a specified filter.
    through_square_filter:
        Calculate the total photon flux of the planet and its moon(s) through
        a square filter.
    max_signal:
        Calculate the maximum possible spectroastrometric signal for a moon.
    clear:
        Clear all previously calculated and stored values for signals and
        noises. Currently not used, but maintained if necessary in the future.
    plot_orbits:
        Plots the orbits of all objects.
    min_bb_temp:
        Calculate the minimum blackbody temperature required to observe a moon.
    plot_min_T_R:
        Plot the minimum temperatures for given radii moons.
    signal_over_orbit:
        Calculate the signal of the moon over an orbit. Expressed in p*a/d.
    plot_signal_over_orbit:
        Plot the signal of the moon over an orbit, expressed in p*a/d.
    detectability_over_ecc:
        Calculate the detectability of a moon as function of eccentricity.
    calc_detectability_ae:
        Calculate the detectability of a moon as function of a and e.
    load_detectability_ae:
        Load and plot a previously generated detectability array from the disk.
    min_bb_temp_e_t0P:
        Compute the min. temperature as a function of eccentricity and phase.
    plot_SNR_T_d:
        Plot the SNR for moons as function of temperature and distance.
    plot_SNR_a_d:
        Plot the SNR for moons as function of semi-major axis and distance.
    noise_radius_temp:
        Plot the absolute noise and the part of it due to the moon filter.
    """

    @u.quantity_input
    def __init__(self, dist: u.pc = 10*u.pc):
        """Initialise the MoonPlanetSystem class.

        Initialise the MoonPlanetSystem class by specifiying its distance. If
        the distance is not provided, set it at a distance of 10 pc, such that
        all fluxes and magnitudes become absolute. Also instantiate the Planet
        and Star attributes, and create an empty list for Moons.
        """
        self.dist = dist.to(u.pc)
        self.Planet = LuminousObject(dist=dist)
        self.Star = LuminousObject(dist=dist)
        self.Moons = []
        self.signals = []
        self.noises = []
        self.t = 0.*u.yr

    @u.quantity_input
    def reset_distance(self, new_dist: u.pc):
        """Reset the system to be at a new distance.

        Resets the system to a new distance, scaling all distance-sensitive
        attributes in the process. Allows moving of systems without requiring
        expensive recalculations. Avoid relying on this excessively, as it may
        produce unintended behaviour.

        Parameters
        ----------
        new_dist (astropy.Quantity):
            New distance at which to place the system.
        """
        old_dist = self.dist
        self.dist = new_dist
        self.Star.reset_distance(new_dist)
        self.Planet.reset_distance(new_dist)
        for moon_idx, Moon in enumerate(self.Moons):
            Moon.reset_distance(new_dist)
            for filter_F in self.signals[moon_idx].keys():
                for filter_P in self.signals[moon_idx][filter_F].keys():
                    self.signals[moon_idx][filter_F][filter_P] = \
                        self.signals[moon_idx][filter_F][filter_P] * \
                        old_dist/new_dist
            for filter_F in self.SNRs[moon_idx].keys():
                for filter_P in self.SNRs[moon_idx][filter_F].keys():
                    self.noises[moon_idx][filter_F][filter_P] = \
                        self.noises[moon_idx][filter_F][filter_P] * \
                        (new_dist/old_dist)

    @u.quantity_input
    def create_blackbody_planet(self, Teff: u.K, radius: u.m, wavs: u.um,
                                name="Planet", mass=1.*u.Mjup):
        """Create a planet with blackbody properties.

        Creates a blackbody planet; uses the same inputs as the set_blackbody
        method for LuminousObjects.

        Parameters
        ----------
            Teff (astropy.Quantity):
                Effective temperature of the black body.
            radius (astropy.Quantity):
                Radius of the body.
            wavs (array_like of astropy.Quantity):
                Wavelengths at which the Planck curve is to be sampled.
            mass (astropy.Quantity):
                Mass of the planet. Default is 1 Mjup.
        """
        self.Planet.set_blackbody(Teff, radius, wavs, interpolate=True)
        self.Planet.mass = mass
        # Set the mass of the central body of the moons if their orbit had
        # already been set
        for Moon in self.Moons:
            if Moon.orbit is not None:
                Moon.orbit.mass = mass

    @u.quantity_input
    def create_gas_giant_from_grid(self, age: u.Gyr, mass: u.Mjup,
                                   grid="ATMO2020", clouds="cf",
                                   metallicity="1s", equil_chem="CEQ",
                                   temp_warning=True, name="Planet",
                                   interpolate=True,
                                   interp_method="PCHIP"):
        """Create a gas giant planet from a precomputed grid.

        Reads the gas giant spectrum from a precomputed grid by interpolation.
        In essence no more than a wrapper of LuminousObject.read_gas_giant.

        Parameters
        ----------
        age (astropy.Quantity):
            Age of the gas giant. N.B.: ATMO2020 provides
            ages up to 10 Gyr or until the planet cools below 200K, whichever
            occurs first. SB2012 provides ages between 1 Myr and 1 Gyr.
        mass (astropy.Quantity):
            Mass of the gas giant. SB2012 provides masses
            between 1 and 15 Mjup; ATMO2020 goes from .001 Msun to .075 Msun,
            equivalent to roughly 1.05 Mjup to 78.57 Mjup.
        grid (str):
            The grid to use; choices are SB2012 for Spiegel & Burrows'
            2012 grid, or ATMO2020 for that of Phillips et al.; SB2012 is
            likely better suited for various scenarios of younger planets
            (<1 Gyr), whereas ATMO2020 seems better suited for planets > 1 Gyr.
            For ATMO2020, atmospheres of Teff > 2000 K are not valid and so
            should not be used, even if they are included as a possibility.
        clouds (str):
            If one uses SB2012, this can be set to cf for a cloud-free
            or to hy for a hybrid model.
        metallicity (str):
            For SB2012, set this to 1s for solar metallicity
            or to 3s for three times solar metallicity.
        equil_chem (str):
            For ATMO2020, set this to CEQ_new or CEQ for chemical equilibrium,
            NEQ_strong for strong vertical mixing and NEQ_weak for weak
            vertical mixing. Additionally, set this to CEQ_old to use the grid
            values with the old equation of state (only for comparison
            purposes).
        temp_warning (bool):
            Sets whether a warning must be printed for effective temperatures
            exceeding 2000K. Default is True; it is recommended to only turn
            this off if exceeding 2000K is desired behaviour or unlikely to
            affect results.
        interpolate (bool):
            Whether to create an interpolator for the flux density or not.
            Can be turned off for performance purposes.
        interp_method (str):
            Which method to use to interpolate if interpolate is set to True.
            Allowed methods are simply those for scipy.interpolate, but best
            used with linear or cubic.
         name (str):
            Name of the planet; only for bookkeeping and plotting.
        """
        self.Planet = LuminousObject(dist=self.dist)
        self.Planet.read_gas_giant(age, mass, grid=grid, clouds=clouds,
                                   metallicity=metallicity,
                                   equil_chem=equil_chem,
                                   temp_warning=temp_warning,
                                   interpolate=interpolate,
                                   interp_method=interp_method, name=name)
        # Set the mass of the central body of the moons if their orbit had
        # already been set
        for Moon in self.Moons:
            if Moon.orbit is not None:
                Moon.orbit.mass = mass

    @u.quantity_input
    def create_moon_from_grid(self, age: u.Gyr, mass: u.Mjup,
                              grid="ATMO2020", clouds="cf",
                              metallicity="1s", equil_chem="CEQ",
                              temp_warning=True, name="Planet",
                              interpolate=True,
                              interp_method="PCHIP", append=False):
        """Create a moon from a precomputed grid.

        Reads the moon spectrum from a precomputed grid by interpolation.
        In essence no more than a wrapper of LuminousObject.read_gas_giant.

        Parameters
        ----------
        age (astropy.Quantity):
            Age of the gas giant. N.B.: ATMO2020 provides
            ages up to 10 Gyr or until the planet cools below 200K, whichever
            occurs first. SB2012 provides ages between 1 Myr and 1 Gyr.
        mass (astropy.Quantity):
            Mass of the gas giant. SB2012 provides masses
            between 1 and 15 Mjup; ATMO2020 goes from .001 Msun to .075 Msun,
            equivalent to roughly 1.05 Mjup to 78.57 Mjup.
        grid (str):
            The grid to use; choices are SB2012 for Spiegel & Burrows'
            2012 grid, or ATMO2020 for that of Phillips et al.; SB2012 is
            likely better suited for various scenarios of younger planets
            (<1 Gyr), whereas ATMO2020 seems better suited for planets > 1 Gyr.
            For ATMO2020, atmospheres of Teff > 2000 K are not valid and so
            should not be used, even if they are included as a possibility.
        clouds (str):
            If one uses SB2012, this can be set to cf for a cloud-free
            or to hy for a hybrid model.
        metallicity (str):
            For SB2012, set this to 1s for solar metallicity
            or to 3s for three times solar metallicity.
        equil_chem (str):
            For ATMO2020, set this to CEQ_new or CEQ for chemical equilibrium,
            NEQ_strong for strong vertical mixing and NEQ_weak for weak
            vertical mixing. Additionally, set this to CEQ_old to use the grid
            values with the old equation of state (only for comparison
            purposes).
        temp_warning (bool):
            Sets whether a warning must be printed for effective temperatures
            exceeding 2000K. Default is True; it is recommended to only turn
            this off if exceeding 2000K is desired behaviour or unlikely to
            affect results.
        interpolate (bool):
            Whether to create an interpolator for the flux density or not.
            Can be turned off for performance purposes.
        interp_method (str):
            Which method to use to interpolate if interpolate is set to True.
            Allowed methods are simply those for scipy.interpolate, but best
            used with linear or cubic.
         name (str):
            Name of the planet; only for bookkeeping and plotting.
        append (bool):
            Whether to add the moon to the list of moons or reset the list
            of moons. By default, we will reset the list to avoid adding
            a multitude of moons by accident.
        """
        NewMoon = LuminousObject(dist=self.dist)
        NewMoon.read_gas_giant(age, mass, grid=grid, clouds=clouds,
                               metallicity=metallicity,
                               equil_chem=equil_chem,
                               temp_warning=temp_warning,
                               interpolate=interpolate,
                               interp_method=interp_method, name=name)
        if append:
            self.Moons.append(NewMoon)
            # Create a dictionary in which the dictionaries are stored for
            # each of the filters, such that we can initialise the signals
            # attribute corresponding to NewMoon
            new_signals = {}
            new_noises = {}
            for filter_name in filters.filters.keys():
                new_signals[filter_name] = {}
                new_noises[filter_name] = {}
            self.signals.append(new_signals)
            self.noises.append(new_noises)
        else:
            self.Moons = [NewMoon]
            # Create a dictionary in which the dictionaries are stored for
            # each of the filters, such that we can initialise the signals
            # attribute corresponding to NewMoon
            new_signals = {}
            new_noises = {}
            for filter_name in filters.filters.keys():
                new_signals[filter_name] = {}
                new_noises[filter_name] = {}
            self.signals = [new_signals]
            self.noises = [new_noises]

    @u.quantity_input
    def create_blackbody_moon(self, Teff: u.K, radius: u.m, wavs: u.um,
                              name="Moon", append=False,
                              reflected_light=False, reflected_light_amp=0,
                              host_dist: u.AU = 10*u.AU):
        """Create a blackbody moon.

        Creates a blackbody moon; uses the same inputs as the set_blackbody
        method for LuminousObjects. It is recommended that this be run
        after a host planet has already been created such that the wavelength
        samples thereof can be used here for consistency.

        Parameters
        ----------
            Teff (astropy.Quantity):
                Effective temperature of the black body.
            radius (astropy.Quantity):
                Radius of the body.
            wavs (array_like of astropy.Quantity):
                Wavelengths at which the Planck curve is to be sampled.
            append (bool):
                Whether to add the moon to the list of moons or reset the list
                of moons. By default, we will reset the list to avoid adding
                a multitude of moons by accident.
            reflected_light (bool):
                Whether to include reflected light, too. Currently only
                implemented by simply taking Agol's moon spectrum as a lower
                bound.
            reflected_light_amp (float):
                By how much to amplify reflected light. Added as an exploratory
                factor, and should not be used in actual simulations.
                Determines by how many magnitudes to increase the reflected
                light.
            host_dist (astropy.Quantity):
                Set the distance between the host planet and its star. Only
                required to include reflection.
        """
        NewMoon = LuminousObject(dist=self.dist)
        if reflected_light:
            reflected_light_temp = self.Star.Teff
        else:
            reflected_light_temp = None
        NewMoon.set_blackbody(Teff, radius, wavs, interpolate=True, name=name,
                              reflected_light_temp=reflected_light_temp,
                              reflected_light_power=-20.5+reflected_light_amp,
                              host_dist=host_dist)
        if append:
            self.Moons.append(NewMoon)
            # Create a dictionary in which the dictionaries are stored for
            # each of the filters, such that we can initialise the signals
            # attribute corresponding to NewMoon
            new_signals = {}
            new_noises = {}
            for filter_name in filters.filters.keys():
                new_signals[filter_name] = {}
                new_noises[filter_name] = {}
            self.signals.append(new_signals)
            self.noises.append(new_noises)
        else:
            self.Moons = [NewMoon]
            # Create a dictionary in which the dictionaries are stored for
            # each of the filters, such that we can initialise the signals
            # attribute corresponding to NewMoon
            new_signals = {}
            new_noises = {}
            for filter_name in filters.filters.keys():
                new_signals[filter_name] = {}
                new_noises[filter_name] = {}
            self.signals = [new_signals]
            self.noises = [new_noises]

    @u.quantity_input
    def create_blackbody_star(self, Teff: u.K, radius: u.m, wavs: u.um,
                              name="Star", mass=1*u.Msun):
        """Create a blackbody star.

        Creates a blackbody star; uses the same inputs as the set_blackbody
        method for LuminousObjects. It is recommended that this be run
        after a host planet has already been created such that the wavelength
        samples thereof can be used here for consistency.

        Parameters
        ----------
            Teff (astropy.Quantity):
                Effective temperature of the black body.
            radius (astropy.Quantity):
                Radius of the body.
            wavs (array_like of astropy.Quantity):
                Wavelengths at which the Planck curve is to be sampled.
            name (str):
                Name of the star.
            mass (astropy.Quantity):
                Mass of the star.
        """
        self.Star.set_blackbody(Teff, radius, wavs)
        self.Star.mass = mass
        # Set the mass of the central body of the Planet if its orbit had
        # already been set
        if self.Planet.orbit is not None:
            self.Planet.orbit.mass = mass

    @u.quantity_input
    def create_blackbody_star_from_temp(self, Teff: u.K, wavs, name="Star"):
        """Create a blackbody star based off solely a temperature.

        Creates a blackbody star from a temperature by interpolating from the
        data provided in Francis LeBlanc's "An Introduction to Stellar
        Astrophysics", yielding mass and radius. It is recommended that this be
        run after a host planet has already been created such that the
        wavelength samples thereof can be used here for consistency.

        Parameters
        ----------
            Teff (astropy.Quantity):
                Effective temperature of the black body.
            radius (astropy.Quantity):
                Radius of the body.
            wavs (array_like of astropy.Quantity):
                Wavelengths at which the Planck curve is to be sampled.
            name (str):
                Name of the star.
        """
        mass, radius = star_from_temp(Teff)
        self.Star.set_blackbody(Teff, radius, wavs, name=name)
        self.Star.mass = mass
        # Set the mass of the central body of the Planet if its orbit had
        # already been set
        if self.Planet.orbit is not None:
            self.Planet.orbit.mass = mass

    @u.quantity_input
    def create_epsIndiA(self, wavs: u.um):
        """Create a blackbody star with eps Indi A-properties.

        Create a star that is analogous to epsilon Indi A; properties are taken
        from Demory et al. (2009), and the star is assumed to be a black body.
        Later iterations may want to instead use an empirical spectrum. Note
        that this is no different from calling create_blackbody_star with
        the properties of eps indi A and setting the mass manually. N.B.: this
        does not automatically set the distance to eps Indi A (3.6384 pc
        according to GAIA EDR3).

        Parameters
        ----------
            wavs (array_like of astropy.Quantity):
                Wavelengths at which the spectrum is to be sampled.
        """
        Teff = 4560*u.K  # From GAIA DR3
        mass = .762*u.Msun  # Demory et al. 2018
        radius = .732*u.Rsun  # Demory et al. 2018

        self.Star = LuminousObject(dist=self.dist)
        self.Star.set_blackbody(Teff, radius, wavs, name="$\\epsilon$ Indi A")
        self.Star.mass = mass
        # Set the mass of the central body of the Planet if its orbit had
        # already been set
        if self.Planet.orbit is not None:
            self.Planet.orbit.mass = mass

    def plot_spectra(self, wavelength_window=[2.5, 26], include_star=True,
                     include_tot=True,
                     interp_nodes=1000, interp_method="PCHIP", ywindow=None,
                     spectral_unit=u.uJy, contrast=1e-6, xlog=False,
                     ylog=True, filter_plot=None):
        """Plot the spectra of all bodies in the system.

        Plots the spectra of all bodies in the system. Added as a
        quality-of-life feature, and for general plotting purposes probably
        not very convenient. Uses an interpolant to reconcile all spectra to
        the same wavelengths.

        Parameters
        ----------
            wavelength_window (length-2 array_like):
                Wavelengths between which to plot, expressed in microns.
            include_star (bool):
                Determines whether the star is included in the plot for
                reference or not.
            include_tot (bool):
                Determines whether the total flux of the planet and moon is
                included in the plot for reference or not.
            ywindow (None or length-2 array_like):
                y-values to limit the plot by.
            spectral_unit (astropy.Unit):
                Unit of spectral density to use.
            contrast (float or NoneType):
                If set to None, contrast is not drawn. If set to a float,
                the (constant) contrast curve for the star is drawn at that
                value i.e. for 1e-6 a curve is drawn at
                1e-6*self.Star.flux_density.
            xlog (bool):
                Whether to draw the x-axis logarithmically or not.
            ylog (bool):
                Whether to draw the y-axis logarithmically or not.
            filter_plot : list of str or NoneType
                Filters to plot: must be supplied as a list.

        Returns
        -------
            figure (matplotlib.Figure):
                The figure in which the plot is drawn.
            ax (matplotlib.Axes):
                The axis in which the plot is drawn.
        """
        # Create a new figure
        figure = plt.figure()
        # figure.set_size_inches(32, 18)

        # Add an axis
        ax = figure.add_subplot(111)

        plot_wavs = np.linspace(wavelength_window[0], wavelength_window[1],
                                interp_nodes, endpoint=True)*u.um
        # Plot the star
        if include_star:
            if not self.Star.interpolate:
                self.Star.create_interpolant(interp_method=interp_method)
            ax.plot(plot_wavs,
                    (self.Star.flux_density_interp(plot_wavs.value)
                     * ph_flux_dens).to(spectral_unit,
                                        equivalencies=u.spectral_density(
                                            plot_wavs)),
                    linestyle="--", color="orange", label=self.Star.name)

        if contrast is not None:
            if not self.Star.interpolate:
                self.Star.create_interpolant(interp_method=interp_method)
            ax.plot(plot_wavs,
                    contrast*(self.Star.flux_density_interp(plot_wavs.value)
                              * ph_flux_dens).to(
                                  spectral_unit,
                                  equivalencies=u.spectral_density(plot_wavs)),
                    linestyle="solid", color="orange",
                    label="Limiting contrast of " + str(float(contrast)),
                    alpha=.3)

        # Plot the planet
        if not self.Planet.interpolate:
            self.Planet.create_interpolant(interp_method=interp_method)
        ax.plot(plot_wavs, (self.Planet.flux_density_interp(
            plot_wavs)*ph_flux_dens).to(spectral_unit,
                                        equivalencies=u.spectral_density(
                                            plot_wavs)),
            linestyle="solid", color="k", label=self.Planet.name)

        # Plot the moons
        for Moon in self.Moons:
            if not Moon.interpolate:
                Moon.create_interpolant(interp_method=interp_method)
            ax.plot(plot_wavs, (Moon.flux_density_interp(
                plot_wavs.value)*ph_flux_dens).to(
                    spectral_unit, equivalencies=u.spectral_density(
                        plot_wavs)),
                linestyle="solid", color="red", label=Moon.name)

        # Plot the total flux
        if include_tot:
            tot_flux = (self.Planet.flux_density_interp(
                plot_wavs.value)*ph_flux_dens).to(
                    spectral_unit, equivalencies=u.spectral_density(plot_wavs))
            for Moon in self.Moons:
                tot_flux += (Moon.flux_density_interp(
                    plot_wavs.value)*ph_flux_dens).to(
                        spectral_unit,
                        equivalencies=u.spectral_density(plot_wavs))
            ax.plot(plot_wavs, tot_flux, linestyle="--",
                    color="k", label="Total")

        # Plot the filters
        filter_colors = ["cadetblue", "indianred"]
        if filter_plot is not None:
            for idx, filter_name in enumerate(filter_plot):
                filter_obj = filters.filters[filter_name]
                filter_lam_c = filter_obj.lam_pivot
                filter_bw = filter_obj.delta_lam
                low_lam = filter_lam_c - filter_bw/2
                high_lam = filter_lam_c + filter_bw/2
                ax.axvspan(low_lam.to(u.um).value, high_lam.to(u.um).value,
                           color=filter_colors[idx], alpha=.5,
                           label=filter_name)
        if ylog:
            ax.set_yscale('log')
        if ywindow is not None:
            ax.set_ylim(ywindow)
        if xlog:
            ax.set_xscale('log')
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlim(wavelength_window)
        ax.set_xlabel("Wavelength [$\\mu$m]")
        if spectral_unit == u.uJy:
            ax.set_ylabel("$F_{\\nu}$ [$\\mu$Jy]")
        else:
            spectral_unit_draw = (spectral_unit/u.ph).to_string(
                'latex_inline')
            ax.set_ylabel(
                f"Photon flux [{spectral_unit_draw}]")
        ax.legend(frameon=False)
        return figure, ax

    @u.quantity_input
    def quick_create(self, T_moon: u.K = 350*u.K,
                     R_moon: u.Rearth = .5*u.Rearth,
                     age_p: u.Gyr = 3*u.Gyr, mass_p: u.Mjup = 3.25*u.Mjup,
                     grid_p="ATMO2020", name_p="$\\epsilon$ Indi Ab",
                     name_s="Star", Teff_s: u.K = None,
                     dist: u.pc = 3.6384*u.pc, reflected_light=False):
        """Create a pre-set system or one with specified parameters quickly.

        Auxiliary function that quickly creates a fairly representative system
        for quality-of-life purposes. Default modelled off of eps Indi Ab.

        Parameters
        ----------
        T_moon (astropy.Quantity):
            Temperature of the moon.
        R_moon (astropy.Quantity):
            Radius of the moon.
        age_p (astropy.Quantity):
            Age of the planet.
        mass_p (astropy.Quantity):
            Mass of the planet.
        grid_p (str):
            Grid to read the planet spectrum from.
        name_p (str):
            Name of the planet.
        name_s (str):
            Name of the star. If Teff_s is set to None, automatically set to
            eps Indi A.
        Teff_s (NoneType or astropy.Quantity):
            Effective temperature of the star. If set to None, use eps Indi A.
        dist (astropy.Quantity):
            Distance of the system.
        reflected_light (bool):
            Whether to include reflected light or not.
        """
        # Set the distance to be that of eps Indi by default
        self.dist = dist
        # First create the planet
        self.create_gas_giant_from_grid(age_p, mass_p, grid=grid_p,
                                        name=name_p)
        # Then the star
        if Teff_s is None:
            self.create_epsIndiA(self.Planet.wavs)
        else:
            self.create_blackbody_star_from_temp(Teff_s, self.Planet.wavs,
                                                 name=name_s)
        # Set the orbit of the planet
        # By default, set this to be equal to that for eps Indi Ab that
        # Feng et al. found at the reference epoch BJD 2448929.56 (which is
        # therefore set to t=0).
        self.Planet.orbit = Orbit(a=11.55*u.AU,
                                  e=.26,
                                  i=64.25*u.deg,
                                  Omg=(250.20+90)*u.deg,
                                  omg=77.83*u.deg,
                                  Mref=143.8*u.deg,
                                  M0=self.Star.mass)
        # And set the parameters for the moon
        self.create_blackbody_moon(T_moon, R_moon, self.Planet.wavs,
                                   reflected_light=reflected_light,
                                   host_dist=self.Planet.orbit.r)
        # By default, set the parameters to be those of Io, but align
        # the plane of the orbit with that of its host planet.
        self.Moons[0].set_orbit(self.Planet.mass, i=self.Planet.orbit.i,
                                Omg=self.Planet.orbit.Omg,
                                omg=self.Planet.orbit.omg)

    def through_filter(self, filter_name, moon_idx=0):
        """Calculate the wavelength-weighted flux through a given filter.

        Calculate the total wavelength-weighted flux through a filter. The
        filter must be one of those supported by the FilterLibrary class.
        Integration performed using sp.integrate.simpson. Analogous to the
        method through_filter for the LuminousObject class.

        Parameters
        ----------
        filter_name (str):
            Filter through which to calculate the flux.
        moon_idx (int or str):
            The moon for which to calculate the flux, as denoted by its index
            in the list self.Moons. If not set, takes the first moon in the
            system. If set to "all" will calculate the total flux for all
            moons.

        Returns
        -------
        lambda_flux (astropy.Quantity):
            Wavelength-weighted flux through the filter.
        """
        total_flux = np.copy(self.Planet.through_filter(filter_name))
        if moon_idx == "all":
            for Moon in self.Moons:
                total_flux += Moon.through_filter(filter_name)
        else:
            total_flux += self.Moons[moon_idx].through_filter(filter_name)
        return total_flux

    def through_square_filter(self, lambda_c: u.um, R_f, const_trans=1.,
                              integ_method="PCHIP", moon_idx=0):
        """Calculate the wavelength-weighted flux through a square filter.

        Calculate the total wavelength-weighted flux through a square filter.
        Analogous to the method through_square_filter for the LuminousObject
        class.

        Parameters
        ----------
        lambda_c (astropy.Quantity):
            Pivot wavelength of the filter.
        R_f (float):
            Spectral resolution of the filter.
        const_trans (float):
            Transmission value for those regions where the filter transmits.
        integ_method (str):
            Which integration method to use. Currently supports "Simpson" and
            "PCHIP"; "PCHIP" uses "PCHIP" interpolation followed by analytic
            evaluation of the integral of the resulting spline, whereas
            "Simpson" simply applies Simpson's rule.
        moon_idx (int or str):
            The moon for which to calculate the flux, as denoted by its index
            in the list self.Moons. If not set, takes the first moon in the
            system. If set to "all" will calculate the total flux for all
            moons.

        Returns
        -------
        lambda_flux (astropy.Quantity):
            Wavelength-weighted flux through the filter.
        """
        total_flux = np.copy(self.Planet.through_square_filter(lambda_c, R_f,
                                                               const_trans,
                                                               integ_method))
        if moon_idx == "all":
            for Moon in self.Moons:
                total_flux += Moon.through_square_filter(lambda_c, R_f,
                                                         const_trans,
                                                         integ_method)
        else:
            total_flux += self.Moons[moon_idx].through_square_filter(
                lambda_c, R_f, const_trans, integ_method)
        return total_flux

    def max_signal(self, moon_idx=0):
        """Calculate the maximum possible spectroastrometric signal for a moon.

        Quality-of-life function that calculates the maximum possible signal
        that can be obtained from a given moon, assuming it is the only
        moon in the system. Corresponds simply to the angular separation
        of the planet and moon on-sky.

        Parameters
        ----------
        moon_idx (int):
            The moon for which to calculate the signal, as denoted by its index
            in the list self.Moons. If not set, takes the first moon in the
            system.

        Returns
        -------
        max_signal_value (astropy.Quantity):
            Maximum possible value of the spectroastrometric signal.
        """
        max_signal_value = (self.Moons[moon_idx].r_mp/self.dist*u.rad
                            ).to(u.mas)
        return max_signal_value

    def clear(self):
        """Clear all previously calculated and stored for signals and noises.

        Clear all previously calculated and stored values for signals and
        noises. Use this when trying new exposure times or time divisions.
        """
        # Create a dictionary in which the dictionaries are stored for
        # each of the filters, such that we can initialise the signals
        # attribute corresponding to each moon
        for moon_idx, Moon in enumerate(self.Moons):
            new_signals = {}
            new_noises = {}
            for filter_name in filters.filters.keys():
                new_signals[filter_name] = {}
                new_noises[filter_name] = {}
                self.signals[moon_idx] = new_signals
                self.noises[moon_idx] = new_noises

    def plot_orbits(self, n_planet=1000, n_moons=100, dist_label="mas",
                    view_3D=False, zoom_to_moon=False):
        """Plot the planet and its moons.

        Plot the planet and its moons, each over one full orbit. Centred on the
        current position of the planet. Supports both 3D and 2D views.

        Parameters
        ----------
        n_planet (int):
            Number of samples to take for the planet's orbit.
        n_moons (int):
            Number of samples to take for the moon orbits.
        dist_label (str):
            Set to "AU" to label the axes in AU; set to "mas" to use
            angular distances instead. Always remains centred on the planet,
            though.
        view_3D (bool):
            Whether to view the orbits in 3D (True) or 2D (False).
        """
        # First plot the planet
        xvec_arr = self.Planet.orbit.calc_orbit_pos(n_planet)
        # Shift the planet positions such that the plot is centred about
        # its current position.
        xvec_planet_current = self.Planet.orbit.pos(self.t)
        if dist_label == "AU":
            plot_star_vec = -xvec_planet_current
        elif dist_label == "mas":
            plot_star_vec = -((xvec_planet_current/self.dist).to(
                u.dimensionless_unscaled)*u.rad).to(u.mas)
        xvec_arr = xvec_arr - xvec_planet_current.reshape((3, 1))
        xvec_arr[:, -1] = np.array([0., 0., 0.])*u.AU
        # Set up the figure
        fig = plt.figure()
        # Calculate the angular distances if necessary
        if dist_label == "mas":
            plot_arr = ((xvec_arr/self.dist).to(
                u.dimensionless_unscaled)*u.rad).to(u.mas)
        if dist_label == "AU":
            plot_arr = xvec_arr
        # Determine whether to plot a 3D or 2D figure
        if view_3D:
            ax = fig.add_subplot(projection="3d")
            ax.plot(plot_arr[0, :], plot_arr[1, :], plot_arr[2, :],
                    linestyle="solid", alpha=.3,
                    color="grey",
                    label=self.Planet.name + " Orbit")
            ax.plot(plot_arr[0, 0], plot_arr[1, 0], plot_arr[2, 0],
                    marker="o", color="grey",
                    label=self.Planet.name,
                    linestyle="None")
            # Also draw the star
            ax.plot(plot_star_vec[0],
                    plot_star_vec[1],
                    plot_star_vec[2],
                    marker="*", color="orange",
                    linestyle="None", markersize=20)
        else:
            ax = fig.add_subplot()
            ax.plot(plot_arr[0, :], plot_arr[1, :],
                    linestyle="solid", alpha=.3,
                    color="grey",
                    label=self.Planet.name + " orbit")
            ax.plot(plot_arr[0, 0], plot_arr[1, 0],
                    marker="o", color="grey",
                    label=self.Planet.name,
                    linestyle="None")
            ax.plot(plot_star_vec[0],
                    plot_star_vec[1],
                    marker="*", color="orange",
                    linestyle="None", markersize=20)

        # Now plot its moons
        moon_colours = ["orange", "blue", "red", "green"]
        # Create a dummy variable that will keep track of the greatest
        # apocentre that any moon will achieve.
        if dist_label == "mas":
            apocentre_max = 0.*u.mas
        if dist_label == "AU":
            apocentre_max = 0.*u.AU
        for moon_idx, Moon in enumerate(self.Moons):
            xvec_arr = Moon.orbit.calc_orbit_pos(n_moons)
            # Calculate the angular distances if necessary
            if dist_label == "mas":
                plot_arr = ((xvec_arr/self.dist).to(
                    u.dimensionless_unscaled)*u.rad).to(u.mas)
            if dist_label == "AU":
                plot_arr = xvec_arr
            # Determine whether to plot a 3D or 2D figure
            if view_3D:
                ax.plot(plot_arr[0, :], plot_arr[1, :], plot_arr[2, :],
                        linestyle="solid", alpha=.3,
                        color=moon_colours[moon_idx],
                        label=Moon.name + " orbit")
                ax.plot(plot_arr[0, 0], plot_arr[1, 0], plot_arr[2, 0],
                        marker="o", color=moon_colours[moon_idx],
                        label=Moon.name,
                        linestyle="None")
                ax.set_zlabel(f"$\\Delta z$ [{dist_label}]")
            else:
                ax.plot(plot_arr[0, :], plot_arr[1, :],
                        linestyle="solid", alpha=.3,
                        color=moon_colours[moon_idx],
                        label=Moon.name + " Orbit")
                ax.plot(plot_arr[0, 0], plot_arr[1, 0],
                        marker="o", color=moon_colours[moon_idx],
                        label=Moon.name,
                        linestyle="None")
            # Keep track of the moon with the greatest apocentre
            if dist_label == "mas":
                apocentre_max = np.max([apocentre_max.value,
                                       ((Moon.orbit.a*(
                                           1 + Moon.orbit.e)/self.dist).to(
                                           u.dimensionless_unscaled)*u.rad).to(
                                               u.mas).value])*u.mas
            if dist_label == "AU":
                apocentre_max = np.max([apocentre_max.value,
                                       (Moon.orbit.a*(
                                           1 + Moon.orbit.e)).to(
                                               u.AU).value])*u.AU
        if view_3D:
            ax.set_zlabel(f"$\\Delta z$ [{dist_label}]")
        ax.set_aspect('equal', 'box')
        if zoom_to_moon:
            ax.set_xbound(-apocentre_max.value, +apocentre_max.value)
            ax.set_ybound(-apocentre_max.value, +apocentre_max.value)
            if view_3D:
                ax.set_zbound(-apocentre_max.value, +apocentre_max.value)
        ax.set_ylabel(f"$\\Delta y$ [{dist_label}]")
        ax.set_xlabel(f"$\\Delta x$ [{dist_label}]")
        ax.legend(loc="upper right")

    @u.quantity_input
    def min_bb_temp(self, instrument, t: u.h, eps, planet_filter, moon_filter,
                    r_moon: u.Rearth, tPtF=None, SNR=5, i="bounds",
                    nvals_a=1000, bisec_buffer=10000*u.km, moon_idx=0,
                    flux_ratio=False):
        """
        Calculate the minimum blackbody temperature required to observe a moon.

        Calculate the minimum blackbody temperature required to observe a moon
        at the given SNR at the given distance to its host, with an assumed
        value of the moon flux in the planet band as fraction of the total flux
        in that same band, f_P. If tPtF is set to None, calculates the optimal
        time allocation per moon-planet distance; if tPtF is set, uses that
        time allocation throughout the entire range of semi-major axes.

        Parameters
        ----------
        instrument : str
            Instrument on which planet_filter and moon_filter are found.
        t : astropy.Quantity
            Total exposure time.
        eps : float
            Achromatic efficiency of the telescope, expressed as fraction.
        planet_filter : str
            Name of the filter to take as planet filter.
        moon_filter : str
            Name of the filter to take as moon filter.
        SNR : float
            Required signal-to-noise ratio. Default is 5.
        tPtF : float or NoneType
            Time allocation to use. If set to None, use ideal time allocation.
        i : NoneType or astropy.Quantity or str
            Inclination of the moon. If set to None, uses the unconstrained
            value of p; otherwise, calculates the corresponding value of the
            coefficient p. If set to "bounds", plots the i=0(or 180), i=90 and
            i=U(0, 180) limits.
        nvals_a : int
            Number of values to sample between the internal and Hill radii.
            Default is 1000.
        bisec_buffer : astropy.Quantity
            Buffer value to add to the bisection search result to make sure
            we do not run into divide-by-zero errors.
        r_moon : astropy.Quantity or NoneType
            Radius of the moon. If set to None, uses the moon corresponding
            to moon_idx.
        moon_idx : int
            Moon for which to calculate the minimum blackbody temperature.
            Only the radius of the moon is used; if r_moon is supplied, use
            that value instead.
        flux_ratio : bool
            Whether to return also the flux ratio, signal and noise for the
            given temperature (True) or not (False). Default is False.

        Returns
        -------
        min_T : astropy.Quantity
            Required moon blackbody temperature in the moon band for detection,
            for each value of the semi-major axis that was sampled.
            If i is set to "bounds", returns a Nx3 array (with N the number of
            semi-major axis points that were sampled).
        a_vals : astropy.Quantity
            Semi-major axis values corresponding to each row in min_T.
        """
        # First, find the telescope to which the instrument belongs
        telescope = filters.telescope_from_instrument(instrument)
        S = np.pi*(diameter[telescope]/2)**2
        # And calculate some quantities that depend on this
        sigma_P = filters.filters[planet_filter].sigma_totpp
        sigma_F = filters.filters[moon_filter].sigma_totpp
        FpP = self.Planet.through_filter(planet_filter)
        FpP_norm = FpP/sigma_P**2
        FpF = self.Planet.through_filter(moon_filter)
        FpF_norm = FpF/sigma_F**2
        if tPtF is None:
            tPtF = (np.sqrt(FpF_norm/FpP_norm)).to(u.dimensionless_unscaled
                                                   ).value
        # moreover define shorthands for ubiquitous quantities
        d = self.dist
        T_F = t/(1+tPtF)
        T_P = t*tPtF/(1+tPtF)
        mu = self.Planet.mass*const.G
        intens_func = filters.filters[planet_filter].blackbody_intensity
        intens_func2 = filters.filters[moon_filter].bb_temp_interp
        # And calculate quantities that will appear throughout
        if r_moon is not None:
            moon_solid_angle = np.pi*(r_moon/self.dist*u.rad)**2
        else:
            moon_solid_angle = np.pi*(
                self.Moons[moon_idx].radius/self.dist*u.rad)**2
        moon_solid_angle = moon_solid_angle.to(u.mas**2)
        # Calculate the value of the inclination-correction p (2/pi<p<1)
        if i is None:
            K2 = sp.special.ellipk(1/2)**2
            p = np.array([2*K2/np.pi**2 + 1/(2*K2)])
            p_num = 1
        elif i != "bounds":
            m = np.sin(i)**2
            p = np.array([sp.special.ellipe(m)])
            p_num = 1
        else:
            K2 = sp.special.ellipk(1/2)**2
            p = np.array([2/np.pi, 2*K2/np.pi**2 + 1/(2*K2), 1.])
            p_num = 3
        # Set up the grid
        # Calculate the Hill radius
        r_Hill = self.Planet.calc_Hill_radius().to(u.km).value
        # Compute the innermost semi-major axis that will always allow for
        # a positive numerator in the minimum-flux expression
        # (this guarantees that we do not try to compute absurdly high
        # temperatures)
        # Do this using a bisection search over an interval that is guaranteed
        # to contain the value (the why and how is not particularly
        # interesting)
        # We can show that a positive numerator will always occur regardless
        # of the value of f_P so long as a*sinc(T_F/P) > K
        # with K a value as defined below
        K = (SNR*d/(p[0]*np.sqrt(eps*S*FpP_norm*T_P/u.ph))/u.rad).to(
            u.km)
        a0 = ((mu*T_F*T_F/(4*np.pi*np.pi))**(1/3)).to(u.km)
        corr = ((2*np.pi*np.sqrt((K)**3/mu)/T_F)**(2/3))
        P1 = T_F*(1+corr)**(3/2)
        a1 = ((mu*P1*P1/(4*np.pi*np.pi))**(1/3)).to(u.km)
        # Set up the bisection interval
        bisec_interval = np.array([a0.value, a1.value])

        def f(a):
            a = a*u.km
            val = 2*a**(5/2)/(np.sqrt(mu)*T_F) *\
                np.sin((T_F*np.sqrt(mu)/(2*a**(3/2))).to(
                    u.dimensionless_unscaled).value) - K
            return val.to(u.km).value

        # Perform the bisection search
        a_min = sp.optimize.bisect(f, bisec_interval[0], bisec_interval[1]) + \
            bisec_buffer.to(u.km).value
        # Set up the semi-major axis interval
        a_vals = np.logspace(np.log10(a_min), np.log10(r_Hill),
                             num=nvals_a)[1:]*u.km
        # Compute from this the period interval
        P_vals = (2*np.pi*np.sqrt(a_vals**3/mu.to(u.km**3/u.h**2)))

        # Set up the iterative temperature calculation
        # Compute all quantities to do with the orbit; we use a circular orbit,
        # but structure the code such that we can change it to an elliptical
        # orbit later if desired
        t0 = 0.*u.h
        e = 0
        t0P = (t0/P_vals).to(u.dimensionless_unscaled).value
        T_FP = (T_F/P_vals).to(u.dimensionless_unscaled).value
        T_PP = (T_P/P_vals).to(u.dimensionless_unscaled).value
        # Calculate all quantities to do with xi
        xi_F2 = np.zeros((a_vals.shape[0],))
        xi_P2 = np.zeros((a_vals.shape[0],))
        xi_prod = np.zeros((a_vals.shape[0]))
        xi_F_arr = np.zeros((a_vals.shape[0], 2))
        xi_P_arr = np.zeros((a_vals.shape[0], 2))
        for idx, t0P_val in enumerate(t0P):
            T_FP_val = T_FP[idx]
            T_PP_val = T_PP[idx]
            xi_F = xi_hat(t0P_val, T_FP_val, e)
            xi_P = xi_hat(t0P_val + T_FP_val, T_PP_val, e)
            xi_F_arr[idx, :] = xi_F
            xi_P_arr[idx, :] = xi_P
            xi_F2[idx] = np.dot(xi_F, xi_F)
            xi_P2[idx] = np.dot(xi_P, xi_P)
            xi_prod[idx] = np.dot(xi_P, xi_F)

        # Set the initial temperature values
        T_init = np.zeros(a_vals.shape)
        # For what follows, we must cycle over values of p
        # Preallocate arrays for the minimum temperature, and if applicable,
        # the flux ratio
        min_T = np.zeros((a_vals.shape[0], p_num))*u.K
        if flux_ratio:
            min_flux_ratio_M = np.zeros((a_vals.shape[0], p_num))
            min_flux_ratio_P = np.zeros((a_vals.shape[0], p_num))
        for p_idx, p_val in enumerate(p):
            CP_arr = ((SNR*d/(a_vals*p_val))**2/(FpP_norm*eps*S*T_P
                                                 )*u.ph/u.rad**2).to(
                u.dimensionless_unscaled).value
            CF_arr = ((SNR*d/(a_vals*p_val))**2/(FpF_norm*eps*S*T_F
                                                 )*u.ph/u.rad**2).to(
                u.dimensionless_unscaled).value
            # Define a function that can calculate the min values of the moon
            # flux given the fractional moon flux in the planet band

            def FFmFFp(f_P_prime):
                f_P = 1/(1+1/f_P_prime)
                CP = CP_arr*(1-f_P)
                CF = CF_arr
                sqrt = np.sqrt((CF/2 - f_P*xi_prod)**2 - xi_F2*(
                    f_P*f_P*xi_P2-CF-CP))
                num = CF/2 + CP + f_P*xi_prod - f_P*f_P*(xi_P2) + sqrt
                denom = xi_F2 - 2*f_P*xi_prod + f_P*f_P*xi_P2 - CP
                return num/denom

            def iterative_func(T_vec):
                T_vec = T_vec*u.K
                # Calculate f_P using these temperatures
                f_P_prime = intens_func(T_vec)*moon_solid_angle/FpP
                # Calculate the minimum FmF corresponding to this f_P
                FmF = FFmFFp(f_P_prime)*FpF
                # and calculate the intensity FmF corresponds to
                intens_F = FmF/moon_solid_angle
                # which in turn should correspond to our new estimate of the
                # temperature
                new_T_vec = intens_func2(intens_F).to(u.K).value
                return new_T_vec

            min_T[:, p_idx] = sp.optimize.fixed_point(iterative_func,
                                                      x0=T_init,
                                                      xtol=1e-4)*u.K
            if flux_ratio:
                # Determine the flux ratios required
                T_vec = min_T[:, p_idx]
                f_P_prime = intens_func(T_vec)*moon_solid_angle/FpP
                FmF = FFmFFp(f_P_prime)*FpF
                min_flux_ratio_M[:, p_idx] = FmF/(FmF+FpF)
                min_flux_ratio_P[:, p_idx] = 1/(1+1/f_P_prime)

        if flux_ratio:
            # Compute the signal and noise
            dimless_sig = np.linalg.norm(
                min_flux_ratio_M[:, :, None]*xi_F_arr[:, None, :] -
                min_flux_ratio_P[:, :, None]*xi_P_arr[:, None, :],
                axis=2)
            signal = (p[None, :]*a_vals[:, None]/d*dimless_sig*u.rad).to(
                u.mas)
            noise = (1/np.sqrt(eps*S)*np.sqrt(
                (1-min_flux_ratio_M)/(FpF_norm*T_F)*u.ph +
                (1-min_flux_ratio_P)/(FpP_norm*T_P)*u.ph)).to(u.mas)
            # Compute the moonless noise and corresponding min. signal,
            # print them
            moonless_noise = (1/np.sqrt(eps*S)*np.sqrt(
                1/(FpF_norm*T_F)*u.ph +
                1/(FpP_norm*T_P)*u.ph)).to(u.mas)
            min_signal = SNR*moonless_noise
            print(f"Moonless noise: {moonless_noise}")
            print(f"Corresponding min. signal: {min_signal}")
            return (min_T, a_vals, min_flux_ratio_M, min_flux_ratio_P,
                    signal, noise)
        else:
            return min_T, a_vals

    @u.quantity_input
    def plot_min_T_R(self, instrument, t: u.h, eps, planet_filter, moon_filter,
                     r_moon_arr: u.Rearth, tPtF=None, SNR=5, nvals_a=100,
                     bisec_buffer=30000*u.km, plot_tides=True):
        """
        Plot the minimum temperatures for given radii moons.

        Calculate the minimum blackbody temperature required to observe a moon
        at the given SNR at the given distance to its host, with an assumed
        value of the moon flux in the planet band as fraction of the total flux
        in that same band, f_P. If tPtF is set to None, calculates the optimal
        time allocation per moon-planet distance; if tPtF is set, uses that
        time allocation throughout the entire range of semi-major axes.

        Parameters
        ----------
        instrument : str
            Instrument on which planet_filter and moon_filter are found.
        t : astropy.Quantity
            Total exposure time.
        eps : float
            Achromatic efficiency of the telescope, expressed as fraction.
        planet_filter : str
            Name of the filter to take as planet filter.
        moon_filter : str
            Name of the filter to take as moon filter.
        SNR : float
            Required signal-to-noise ratio. Default is 5.
        tPtF : float or NoneType
            Time allocation to use. If set to NoneType, use ideal time
            allocation for negligible moon flux.
        nvals_a : int
            Number of values to sample between the internal and Hill radii.
            Default is 1000.
        bisec_buffer : astropy.Quantity
            Buffer value to add to the bisection search result to make sure
            we do not run into divide-by-zero errors.
        r_moon_arr : astropy.Quantity
            Radii of the moon for which to plot the minimum temperature.
        plot_tides : bool
            Whether to plot the tidal heating temperatures for an Io-like,
            Mars-like and Earth-like moon with eccentricities of .005,
            .01, .1, .5.
        """
        # Find whether r_moon_arr is iterable or not: if it is not, assume it
        # is a single object and store that in an array to make the remaining
        # code run smoothly
        try:
            iter(r_moon_arr)
        except(TypeError):
            r_moon_arr = np.array([r_moon_arr.value])*r_moon_arr.unit
        # Initialise necessities for plot
        r_linestyle = ["solid", "dashed", "dashdot", "dotted"]
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_xscale('log')
        ax_twin = ax.twinx()
        ax_e = fig.add_subplot(212)
        ax_e.set_xscale('log')
        ax_e.set_yscale('log')
        handles, labels = ax.get_legend_handles_labels()
        handles_e, labels_e = ax_e.get_legend_handles_labels()
        handles_e.append(Line2D([0], [0], color='w', alpha=.0))
        # Set bounds
        T_plot_min = 100
        T_plot_max = 350
        for r_idx, r_moon in enumerate(r_moon_arr):
            T_min, a_vals, min_ratio_M, min_ratio_P, signal, noise = \
                self.min_bb_temp(
                instrument, t, eps, planet_filter, moon_filter, r_moon,
                tPtF=tPtF, nvals_a=nvals_a, bisec_buffer=bisec_buffer,
                flux_ratio=True
                )
            ls = r_linestyle[r_idx]
            # Plot the min. flux ratio only for the first moon
            if r_idx == 0:
                # Plot the min. flux ratios
                ax_twin.plot(a_vals, min_ratio_M[:, ::2], linestyle='solid',
                             color="slategrey")
                ax_twin.plot(a_vals, min_ratio_M[:, 1], linestyle='solid',
                             color="orange")
                ax_twin.fill_between(a_vals.value, min_ratio_M[:, 0],
                                     min_ratio_M[:, 2],
                                     alpha=.5, color="grey")
            # # Plot the min. flux ratio
            # ax_twin.plot(a_vals, min_ratio_P[:, ::2], linestyle=ls,
            #              color="slategrey")
            # ax_twin.plot(a_vals, min_ratio_P[:, 1], linestyle=ls,
            #              color="orange")
            # ax_twin.fill_between(a_vals.value, min_ratio_P[:, 0],
            #                      min_ratio_P[:, 2],
            #                      alpha=.5, color="grey")
            # Plot the required temperature
            ax.plot(a_vals, T_min[:, ::2], linestyle=ls, color="slategrey")
            Rearth = u.Rearth.to_string('latex')
            ax.plot(a_vals, T_min[:, 1], linestyle=ls, color="r")
            ax.fill_between(a_vals.value, T_min[:, 0].value,
                            T_min[:, 2].value,
                            alpha=.5, color="grey")
            # Compute the eccentricity needed to heat a tidally heated moon
            # to the required temperature
            Imk2 = .015  # For Io: taken from Lainey et al. (2016)
            P_vals = self.Planet.period(a_vals)
            e_req = np.zeros(T_min.shape)
            e_req[:, 0] = ((const.sigma_sb*const.G*P_vals**5*T_min[:, 0]**4 /
                            (84*Imk2*np.pi**4*r_moon**3))**(1/2)).to(
                                u.dimensionless_unscaled).value
            e_req[:, 1] = ((const.sigma_sb*const.G*P_vals**5*T_min[:, 1]**4 /
                            (84*Imk2*np.pi**4*r_moon**3))**(1/2)).to(
                                u.dimensionless_unscaled).value
            e_req[:, 2] = ((const.sigma_sb*const.G*P_vals**5*T_min[:, 2]**4 /
                            (84*Imk2*np.pi**4*r_moon**3))**(1/2)).to(
                                u.dimensionless_unscaled).value
            ax_e.plot(a_vals, e_req[:, 1], c='royalblue', linestyle=ls)
            ax_e.plot(a_vals, e_req[:, ::2], c='slategrey', linestyle=ls)
            ax_e.fill_between(a_vals.value, e_req[:, 0],
                              e_req[:, 2], alpha=.5, color="grey")
            label = f"{r_moon.to(u.Rearth).round(3).value} {Rearth}"
            handles.append(Line2D([0], [0], color='grey', linestyle=ls,
                                  label=label))
            handles_e.append(Line2D([0], [0], color='grey', linestyle=ls,
                                    label=label))
            # Mark the locations of Io and Callisto and the temperatures that
            # an Io-sized (respectively Earth-sized) moon must have at those
            # locations
            a_mark = np.array([421.7e3, 0, 1882.7e3])
            if r_idx in [0, 2]:
                ls = 'dashed'
                color = 'grey'
                # color = 'green'
                alpha = .4
                color_mark = 'green'
                T_min_mark = sp.interpolate.interp1d(a_vals.value,
                                                     T_min[:, 1].value
                                                     )(a_mark[r_idx])
                ax.vlines(a_mark[r_idx], T_plot_min, T_min_mark,
                          linestyles=ls, color=color, alpha=alpha)
                ax.hlines(T_min_mark, a_vals.min().value, a_mark[r_idx],
                          linestyles=ls, color=color, alpha=alpha)
                ax.scatter(a_mark[r_idx], T_min_mark, color=color_mark,
                           marker='o', zorder=3)
        # Determine order of the twinx, disable background
        ax.set_zorder(ax_twin.get_zorder()+1)
        ax.patch.set_visible(False)
        # Set labels
        ax_twin.set_ylabel("Moon flux fraction [-]")
        ax_e.set_xlabel(
            f"Semi-major axis [{a_vals.unit.to_string('latex')}]")
        ax_e.set_ylabel('Req. eccentricity [-]')
        ax.set_ylabel(f"Min. temperature [{T_min.unit.to_string('latex')}]")
        # ax.axvline((421700*u.km).to(a_vals.unit).value, color="purple",
        #            alpha=.5, label="$a_{\\mathrm{Io}}$")
        # sigma_P_a = filters.filters[planet_filter].PSF_sigma.to(u.rad
        #                                                         ).value * \
        #     self.dist
        # sigma_F_a = filters.filters[moon_filter].PSF_sigma.to(u.rad
        #                                                       ).value * \
        #     self.dist
        # ax.axvline(sigma_P_a.to(a_vals.unit).value, color="cyan", alpha=.5,
        #            linestyle='dashed',
        #            label="Resolution limit in P")
        # ax.axvline(sigma_F_a.to(a_vals.unit).value, color="green", alpha=.5,
        #            linestyle='dashed',
        #            label="Resolution limit in F")
        SS_moons = ["M", "I", "E", "G", "C", "Ti", "Tr"]
        SS_moons_a = np.array([384.4, 421.8, 671.1, 1070.4, 1882.7, 1221.9,
                               354.8])*1e3*u.km
        SS_moons_e = np.array([.0554, .004, .009, .001, .007, .029, 0.000016])
        for SS_moon_idx, SS_moon in enumerate(SS_moons):
            a = SS_moons_a[SS_moon_idx].to(u.km).value
            e = SS_moons_e[SS_moon_idx]
            ax_e.scatter(a, e, c='mediumseagreen')
            ax_e.annotate(SS_moon, xy=(a, e), xycoords='data',
                          xytext=(2, 2), textcoords='offset pixels',
                          fontsize=26)
        # Also add the migration line for angular momentum-conserved Triton
        a_Tr0 = 354.8e3
        e_Tr0 = 0.000016
        e_Tr_vals = np.linspace(e_Tr0, 1, num=100, endpoint=False)
        a_Tr_vals = a_Tr0*(1-e_Tr0*e_Tr0)/(1-e_Tr_vals*e_Tr_vals)
        ax_e.plot(a_Tr_vals, e_Tr_vals, c='mediumseagreen', alpha=.5)
        if plot_tides:
            # Plot the tidal heating temperatures for an Io-like, Mars-like
            # and Earth-like moon (with -Imk2 from Lainey et al. 2016),
            # respectively:
            # -Imk2 = .015, .00165, .00107
            # for eccentricities of .005, .01, .1 and .5
            P_vals = self.Planet.period(a_vals)
            e_vals = np.array([.0041, .1])
            Imk2vals = np.array([.015])
            colors = ['lightcoral', 'brown', 'maroon', 'indigo']
            # Pre-calculate the constant component of the temperature
            T_coeff = (84*np.pi**4/(const.G*const.sigma_sb))**(1/4)
            for idx, (Imk2, radius) in enumerate(zip(Imk2vals, r_moon_arr)):
                ls = r_linestyle[idx]
                for e, color in zip(e_vals, colors):
                    T_surf = (T_coeff*(radius**3*e*e/P_vals**5*Imk2)**(1/4)
                              ).to(u.K)
                    ax.plot(a_vals, T_surf, color=color, linestyle=ls)

            for e, color in zip(e_vals, colors):
                handles.append(Line2D([0], [0], color=color,
                                      label=f'TH for e={e}'))
        ax_e.set_ylim(e_Tr0, 1)
        ax_e.set_ylabel('Required eccentricity [-]')
        ax.set_ylim((T_plot_min, T_plot_max))
        ax_twin.set_ylim((0., .2))
        # ax_twin.set_yscale('log')
        ax.set_xlim((a_vals.min().value, a_vals.max().value))
        ax.set_xlim((a_vals.min().value, 5e6))
        ax_e.set_xlim((a_vals.min().value, 5e6))
        ticks = [2e5, 4e5, 6e5, 8e5, 1e6, 2e6, 4e6]
        labels = ["$2\cdot10^5$", "$4\cdot10^5$", "$6\cdot10^5$",
                  "$8\cdot10^5$", "$1\cdot10^6$",
                  "$2\cdot10^6$", "$4\cdot10^6$"]
        ax.set_xticks(ticks=ticks, labels=labels)
        ax_e.set_xticks(ticks=ticks, labels=labels)
        # ax.set_yscale('log')
        handles.append(Patch(facecolor='r', edgecolor='r',
                             label='Min. temperature'))
        handles.append(Patch(facecolor='orange', edgecolor='orange',
                             label='Min. $f_{Mm}$'))
        handles_e.append(Patch(facecolor='royalblue',
                               edgecolor='royalblue',
                               label='Req. eccentricity'))
        handles_e.append(Patch(facecolor='mediumseagreen',
                               edgecolor='mediumseagreen',
                               label='Solar System moons'))
        ax.legend(handles=handles, loc='upper right', frameon=False,
                  ncols=5)
        ax_e.legend(handles=handles_e, loc='lower right', frameon=False,
                    ncols=3)

    @u.quantity_input
    def signal_over_orbit(self, instrument, planet_filter, moon_filter,
                          TP_vals, r_moon: u.Rearth, T_moon: u.K, e_arr,
                          n=100):
        """
        Calculate the signal of the moon over an orbit. Expressed in p*a/d.

        Parameters
        ----------
        TP_vals : array of floats
            Observation times as a fraction of the orbital period of the moon
            for which the values must be computed.
        r_moon : u.Rearth
            Radius of the moon.
        T_moon : u.K
            Blackbody temperature of the moon.
        e_arr : float
            Eccentricities of the moon to compute the values for.
        n : int
            Number of samples over the orbit to take.

        Returns
        -------
        sig_arr : array of floats
            Signals for the various quantities expressed as multiple of
            p*a/d.
        t0P_vals : array of floats
            Orbital phase values at which the signal was sampled.
        """
        # Calculate some filter-dependent quantities
        sigma_P = filters.filters[planet_filter].sigma_totpp
        sigma_F = filters.filters[moon_filter].sigma_totpp
        FpP = self.Planet.through_filter(planet_filter)
        FpP_norm = FpP/sigma_P**2
        FpF = self.Planet.through_filter(moon_filter)
        FpF_norm = FpF/sigma_F**2
        intens_funcP = filters.filters[planet_filter].blackbody_intensity
        intens_funcF = filters.filters[moon_filter].blackbody_intensity
        d = self.dist
        FmP = intens_funcP(T_moon)*np.pi*(r_moon/d*u.rad)**2
        FmF = intens_funcF(T_moon)*np.pi*(r_moon/d*u.rad)**2
        fP = (FmP/(FpP + FmP)).to(u.dimensionless_unscaled).value
        fF = (FmF/(FpF + FmF)).to(u.dimensionless_unscaled).value
        fPfF = fP/fF
        tPtF = (np.sqrt(FpF_norm/FpP_norm)).to(u.dimensionless_unscaled
                                               ).value
        # Preallocate sig_arr
        sig_arr = np.zeros((TP_vals.shape[0], e_arr.shape[0], n))
        t0P_vals = np.linspace(0, 1, n, endpoint=True)
        for TP_idx, TP in enumerate(TP_vals):
            T_FP = TP/(1+tPtF)
            T_PP = TP/(1+1/tPtF)
            t0PP_vals = t0P_vals + T_PP
            for e_idx, e in enumerate(e_arr):
                for t0P_idx, t0P in enumerate(t0P_vals):
                    t0PP = t0PP_vals[t0P_idx]
                    xi_F = xi_hat(t0P - T_FP/2, T_FP, e)
                    xi_P = xi_hat(t0PP - T_FP/2, T_PP, e)
                    vec = xi_F - fPfF*xi_P
                    sig_arr[TP_idx, e_idx, t0P_idx] = np.linalg.norm(vec)*fF
        return sig_arr, t0P_vals

    @u.quantity_input
    def plot_signal_over_orbit(self, instrument, planet_filter, moon_filter,
                               TP_vals, r_moon: u.Rearth, T_moon: u.K, e_arr,
                               n=100):
        """
        Plot the signal of the moon over an orbit, expressed in p*a/d.

        Parameters
        ----------
        instrument : str
            Instrument on which planet_filter and moon_filter are found.
        planet_filter : str
            Name of the filter to take as planet filter.
        moon_filter : str
            Name of the filter to take as moon filter.
        TP_vals : array of floats
            Observation times as a fraction of the orbital period of the moon
            for which the values must be computed.
        r_moon : u.Rearth
            Radius of the moon.
        T_moon : u.K
            Blackbody temperature of the moon.
        e_arr : float
            Eccentricities of the moon to compute the values for.
        n : int
            Number of samples over the orbit to take.

        Returns
        -------
        None.
        """
        sig_arr, t0P_vals = self.signal_over_orbit(instrument, planet_filter,
                                                   moon_filter, TP_vals,
                                                   r_moon, T_moon, e_arr, n=n)
        # Initialise the necessicities for the plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        e_cols = ['k', 'blue', 'red', 'yellowgreen']
        TP_ls = ['solid', 'dashed', 'dashdot', 'dotted']
        legend_elements = []
        for e_idx, (col, e) in enumerate(zip(e_cols, e_arr)):
            legend_elements.append(Patch(facecolor=col,
                                         label=f'e={e}'))
            for TP_idx, (ls, TP) in enumerate(zip(TP_ls, TP_vals)):
                if e_idx == e_arr.shape[0]-1:
                    legend_elements.append(Line2D([0], [0], color='grey',
                                                  alpha=.7,
                                                  label=f'$T/P={TP}$',
                                                  linestyle=ls))
                ax.plot(t0P_vals, sig_arr[TP_idx, e_idx, :], linestyle=ls,
                        color=col)
        ax.set_xlim((0, 1))
        ax.set_xlabel('Phase [-]')
        ax.set_ylabel('Normalised signal $\\left[\\frac{pa}{d}\\right]$')
        # Create custom legend
        ax.legend(handles=legend_elements)

    @u.quantity_input
    def detectability_over_ecc(self, instrument, planet_filter, moon_filter,
                               eps, t: u.h, r_moon: u.Rearth, T_moon: u.K,
                               a_moon: u.km, SNR=5, n_e=100, n_samples=100,
                               plot=False):
        """
        Calculate the detectability of a moon as function of eccentricity.

        Calculate the detectability of a moon (as percentage of time) as a
        function of eccentricity for a given moon radius, semi-major axis and
        temperature.

        Parameters
        ----------
        instrument : str
            Instrument on which planet_filter and moon_filter are found.
        planet_filter : str
            Name of the filter to take as planet filter.
        moon_filter : str
            Name of the filter to take as moon filter.
        eps : float
            Achromatic efficiency of the telescope, expressed as float between
            0 and 1.
        t : astropy.Quantity
            Observation time in both filters combined.
        r_moon : u.Rearth
            Radius of the moon.
        T_moon : u.K
            Blackbody temperature of the moon.
        a_moon : u.km
            Semi-major axis of the moon.
        SNR : float
            Signal-to-noise ratio we wish to achieve. Default is 5.
        n_e : int
            Number of samples of the eccentricty to take over the interval
            [0, 1). The default is 100.
        n_samples : int
            Number of samples of the orbit for each eccentricty sample to take
            (i.e. number of samples along the orbital phase). The default is
            100. Determines the accuracy of the percentage.
        plot : bool
            If True, plot the result.

        Returns
        -------
        detectability : array of floats
            Array containing the detectability at each eccentricity sample and
            for each of the three bounding values for p.
        e_arr : array of floats
            Array containing the eccentricities at which the samples were
            taken.
        """
        # First, find the telescope to which the instrument belongs
        telescope = filters.telescope_from_instrument(instrument)
        S = np.pi*(diameter[telescope]/2)**2
        # Calculate system-dependent quantities
        d = self.dist
        P = self.Planet.period(a_moon)
        # Calculate some filter-dependent quantities
        sigma_P = filters.filters[planet_filter].sigma_totpp
        sigma_F = filters.filters[moon_filter].sigma_totpp
        FpP = self.Planet.through_filter(planet_filter)/u.ph
        FpP_norm = FpP/sigma_P**2
        FpF = self.Planet.through_filter(moon_filter)/u.ph
        FpF_norm = FpF/sigma_F**2
        intens_funcP = filters.filters[planet_filter].blackbody_intensity
        intens_funcF = filters.filters[moon_filter].blackbody_intensity
        FmP = intens_funcP(T_moon)*np.pi*(r_moon/d*u.rad)**2/u.ph
        FmF = intens_funcF(T_moon)*np.pi*(r_moon/d*u.rad)**2/u.ph
        fP = (FmP/(FpP + FmP)).to(u.dimensionless_unscaled).value
        fF = (FmF/(FpF + FmF)).to(u.dimensionless_unscaled).value
        tPtF = (np.sqrt(FpF_norm/FpP_norm)).to(u.dimensionless_unscaled
                                               ).value
        T_F = t/(1+tPtF)
        T_P = t/(1+1/tPtF)
        TP_vals = np.array([(t/P).to(u.dimensionless_unscaled).value])
        # Calculate the total noise
        noise_F_comp = sigma_F*sigma_F*(1-fF)/(FpF*T_F)
        noise_P_comp = sigma_P*sigma_P*(1-fP)/(FpP*T_P)
        noise = (np.sqrt((noise_F_comp + noise_P_comp)/(S*eps))).to(u.mas)
        five_sig = SNR*noise
        # Set up the calculation loop
        e_arr = np.linspace(0, 1, n_e, endpoint=False)
        # Calculate the required values
        sig_arr, t0P_vals = self.signal_over_orbit(instrument, planet_filter,
                                                   moon_filter, TP_vals,
                                                   r_moon, T_moon, e_arr,
                                                   n=n_samples)
        # Set up the detectability array
        detectability = np.zeros((3, n_e))
        for e_idx, e_val in enumerate(e_arr):
            sigs = (sig_arr[0, e_idx, :]*(a_moon/d*u.rad)).to(u.mas)
            for p_idx, p_val in enumerate(p_i):
                count = np.sum(np.where(sigs*p_val > five_sig, True, False))
                detectability[p_idx, e_idx] = count/sigs.shape[0]*100
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(e_arr, detectability[1, :], color='k')
            ax.plot(e_arr, detectability[0, :], color='r')
            ax.plot(e_arr, detectability[2, :], color='r')
            ax.fill_between(e_arr, detectability[0, :], detectability[2, :],
                            color='grey', alpha=.5)
            ax.set_xlabel('Eccentricity [-]')
            ax.set_ylabel('Detectability [%]')
            ax.set_ylim((0, 101))
            ax.set_xlim((0, 1))
        return detectability, e_arr

    @u.quantity_input
    def calc_detectability_ae(self, instrument, planet_filter, moon_filter,
                              eps, t: u.h, r_moon: u.Rearth, T_moon: u.K,
                              SNR=5, n_e=100, n_a=100, n_samples=100, p=None,
                              save=True, plot=True, filename=None):
        """
        Calculate the detectability of a moon as function of a and e.

        Calculate the detectability of a moon (as percentage of time) as a
        function of eccentricity and semi-major axis for a given moon radius
        and temperature.

        Parameters
        ----------
        instrument : str
            Instrument on which planet_filter and moon_filter are found.
        planet_filter : str
            Name of the filter to take as planet filter.
        moon_filter : str
            Name of the filter to take as moon filter.
        eps : float
            Achromatic efficiency of the telescope, expressed as float between
            0 and 1.
        t : astropy.Quantity
            Observation time in both filters combined.
        r_moon : u.Rearth
            Radius of the moon.
        T_moon : u.K
            Blackbody temperature of the moon.
        SNR : float
            Signal-to-noise ratio we wish to achieve. Default is 5.
        n_e : int
            Number of samples of the eccentricty to take over the interval
            [0, 1). The default is 100.
        n_a : int
            Number of samples of the semi-major axis to take over the range
            [r_Roche, 1/3r_Hill]
        n_samples : int
            Number of samples of the orbit for each eccentricty sample to take
            (i.e. number of samples along the orbital phase). The default is
            100. Determines the accuracy of the percentage.
        plot : bool
            If True, plot the result.
        p : float or NoneType
            Value of p to use. If set to None, uses the mean value, which is
            approximately equal to 0.842; the lower bound is 2/pi and the
            upper bound is 1.
        save : bool, optional
            Whether to save the produced array or not. If True, saves the
            array into a file named either filename or if filename is None,
            detect_{T_moon}_{r_moon}_{YYYYMMDD_HHMM}.txt in a folder
            called 'Detectability'.
        filename : str or NoneType, optional
            Filename to use; if set to None, use the naming format as described
            in save instead.

        Returns
        -------
        detectability : array of floats
            Array containing the detectability at each eccentricity sample and
            for each of the three bounding values for p.
        e_arr : array of floats
            Array containing the eccentricities at which the samples were
            taken.
        """
        # First, find the telescope to which the instrument belongs
        telescope = filters.telescope_from_instrument(instrument)
        S = np.pi*(diameter[telescope]/2)**2
        # Calculate system-dependent quantities
        d = self.dist
        # Calculate some filter-dependent quantities
        sigma_P = filters.filters[planet_filter].sigma_totpp
        sigma_F = filters.filters[moon_filter].sigma_totpp
        FpP = self.Planet.through_filter(planet_filter)/u.ph
        FpP_norm = FpP/sigma_P**2
        FpF = self.Planet.through_filter(moon_filter)/u.ph
        FpF_norm = FpF/sigma_F**2
        intens_funcP = filters.filters[planet_filter].blackbody_intensity
        intens_funcF = filters.filters[moon_filter].blackbody_intensity
        FmP = intens_funcP(T_moon)*np.pi*(r_moon/d*u.rad)**2/u.ph
        FmF = intens_funcF(T_moon)*np.pi*(r_moon/d*u.rad)**2/u.ph
        fP = (FmP/(FpP + FmP)).to(u.dimensionless_unscaled).value
        fF = (FmF/(FpF + FmF)).to(u.dimensionless_unscaled).value
        fPfF = fP/fF
        tPtF = (np.sqrt(FpF_norm/FpP_norm)).to(u.dimensionless_unscaled
                                               ).value
        T_F = t/(1+tPtF)
        T_P = t/(1+1/tPtF)
        # Calculate the total noise
        noise_F_comp = sigma_F*sigma_F*(1-fF)/(FpF*T_F)
        noise_P_comp = sigma_P*sigma_P*(1-fP)/(FpP*T_P)
        noise = (np.sqrt((noise_F_comp + noise_P_comp)/(S*eps))).to(u.mas)
        five_sig = SNR*noise

        # Calculate the Roche and Hill limits
        r_Roche = (self.Planet.calc_Roche_limit()).to(u.km)
        r_Hill = (self.Planet.calc_Hill_radius()).to(u.km)
        a_vals = np.logspace(np.log10(r_Roche.value),
                             np.log10(1/3*r_Hill.value),
                             num=n_a, endpoint=False)*u.km
        P_vals = self.Planet.period(a_vals)
        TP_vals = (t/P_vals).to(u.dimensionless_unscaled).value
        # Compute the dimensionless value of the noise
        if p is None:
            K2 = sp.special.ellipk(1/2)**2
            p = np.array([2*K2/np.pi**2 + 1/(2*K2)])
        five_sig_dimless_arr = (five_sig/(p*a_vals/d*u.rad)).to(
            u.dimensionless_unscaled).value
        # Compute the values of e to test
        e_vals = np.linspace(0, 1, num=n_e, endpoint=False)

        # Preallocate an array for the detectability
        detec_arr = np.zeros((a_vals.shape[0], e_vals.shape[0]))
        sig_arr = np.zeros((TP_vals.shape[0], e_vals.shape[0], n_samples))
        t0P_vals = np.linspace(0, 1, n_samples, endpoint=True)
        for TP_idx, TP in enumerate(tqdm.tqdm(TP_vals, desc='a-loop',
                                              position=0, ascii=True,
                                              leave=True)):
            T_FP = TP/(1+tPtF)
            T_PP = TP/(1+1/tPtF)
            t0PP_vals = t0P_vals + T_PP
            five_sig_dl = five_sig_dimless_arr[TP_idx]
            for e_idx, e in enumerate(e_vals):
                for t0P_idx, t0P in enumerate(t0P_vals):
                    t0PP = t0PP_vals[t0P_idx]
                    xi_F = xi_hat(t0P, T_FP, e)
                    xi_P = xi_hat(t0PP, T_PP, e)
                    vec = xi_F - fPfF*xi_P
                    sig_arr[TP_idx, e_idx, t0P_idx] = np.linalg.norm(vec)*fF
                count = np.sum(np.where(
                    sig_arr[TP_idx, e_idx, :] > five_sig_dl, True, False))
                detec_arr[TP_idx, e_idx] = count/n_samples*100
        # As this is a very expensive function, we can optionally save the
        # result
        if save:
            save_arr = np.zeros((detec_arr.shape[0]+1, detec_arr.shape[1]+1))
            save_arr[1:, 0] = e_vals
            save_arr[0, 1:] = a_vals.to(u.km).value
            save_arr[0, 0] = r_Roche.to(u.km).value
            save_arr[1:, 1:] = detec_arr
            if filename is None:
                date_str = time.strftime("%Y%m%d_%H%M")
                T_str = str(int(round(T_moon.to(u.K).value, 0)))
                r_str = str(round(r_moon.to(u.Rearth).value, 3))
                r_str = r_str.replace('.', '')
                fname = f'Detectability/detect_{T_str}_{r_str}_{date_str}.txt'
            else:
                fname = 'Detectability/' + filename
            np.savetxt(fname, save_arr)
        # Finally, plot the result
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            contourf = ax.contourf(e_vals, a_vals.to(u.km), detec_arr,
                                   vmin=0, vmax=100, cmap='viridis',
                                   levels=np.arange(0, 100 + 1, 1))
            contour = ax.contour(e_vals, a_vals.to(u.km), detec_arr,
                                 colors="r", levels=[5, 50, 95])
            ax.clabel(contour, inline=True, fontsize=10)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_xlabel('Eccentricity [-]')
            ax.set_ylabel('Semi-major axis [km]')
            ax.set_yscale('log')
            cbar = fig.colorbar(mappable=contourf, label="Detectability [%]")
            cbar.set_ticks(np.arange(0, 100 + 10, 10))
            e_Roche_vals = np.linspace(0, 1, 1000, endpoint=False)
            a_Roche = r_Roche.to(u.km)/(1 - e_Roche_vals)
            ax.fill_between(e_Roche_vals, a_Roche.value, a_vals.min().value,
                            hatch='/', edgecolor='purple', alpha=.5,
                            facecolor='purple')
            SS_moons = ["M", "I", "E", "G", "C", "Ti", "Tr"]
            SS_moons_a = np.array([384.4, 421.8, 671.1, 1070.4, 1882.7, 1221.9,
                                   354.8])*1e3*u.km
            SS_moons_e = np.array([.0554, .004, .009, .001, .007, .029, 0.])
            for SS_moon_idx, SS_moon in enumerate(SS_moons):
                a = SS_moons_a[SS_moon_idx].to(u.km).value
                e = SS_moons_e[SS_moon_idx]
                ax.scatter(e, a, c='cyan')
                ax.annotate(SS_moon, xy=(e, a), xycoords='data',
                            xytext=(2, 2), textcoords='offset pixels',
                            fontsize=16)
            # Also add the migration line for angular momentum-conserved Triton
            a_Tr0 = 354.8e3
            e_Tr0 = 0.
            e_Tr_vals = np.linspace(0, 1, num=100, endpoint=False)
            a_Tr_vals = a_Tr0*(1-e_Tr0*e_Tr0)/(1-e_Tr_vals*e_Tr_vals)
            ax.plot(e_Tr_vals, a_Tr_vals)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        return detec_arr

    @u.quantity_input
    def load_detectability_ae(self, filepath=None, T_moon: u.K = None,
                              r_moon: u.Rearth = None, timestamp=None,
                              day=None, month=None, year=None):
        """
        Load and plot a previously generated detectability array from the disk.

        Parameters
        ----------
        filepath : str
            File path to use. If not provided, check for the timestamp.
        T_moon : u.K
            Temperature of the moon in the previous computation.
        r_moon : u.Rearth
            Radius of the moon.
        timestamp : int
            Time stamp in HHMM format.
        day : int or NoneType, optional
            Day of the month. If set to None, use current day.
        month : int or NoneType, optional
            Month, as number. If set to None, use current day
        year : int or NoneType, optional
            Year. The default is 2023. If set to None, use current year.

        Returns
        -------
        None.
        """
        if filepath is not None:
            fname = filepath
        else:
            # Recreate the filename
            T_str = str(int(round(T_moon.to(u.K).value, 0)))
            r_str = str(round(r_moon.to(u.Rearth).value, 3))
            r_str = r_str.replace('.', '')
            # If day, month or year are set to None, set them to the current
            # value
            today = datetime.date.today()
            if year is None:
                year = today.year
            if month is None:
                month = today.month
            if day is None:
                day = today.day
            day_str = str(day).zfill(2)
            month_str = str(month).zfill(2)
            date_str = f'{year}{month_str}{day_str}_{timestamp}'
            fname = f'Detectability/detect_{T_str}_{r_str}_{date_str}.txt'
        save_arr = np.loadtxt(fname)
        e_vals = save_arr[1:, 0]
        a_vals = save_arr[0, 1:]*u.km
        detec_arr = save_arr[1:, 1:]
        r_Roche = save_arr[0, 0]*u.km
        fig = plt.figure()
        ax = fig.add_subplot(111)
        contourf = ax.contourf(e_vals, a_vals.to(u.km), detec_arr,
                               vmin=0, vmax=100, cmap='viridis',
                               levels=np.arange(0, 100 + 1, 1))
        contour = ax.contour(e_vals, a_vals.to(u.km), detec_arr,
                             colors="r", levels=[5, 50, 95])
        ax.clabel(contour, inline=True, fontsize=10)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlabel('Eccentricity [-]')
        # ax.set_xscale('log')
        ax.set_ylabel('Semi-major axis [km]')
        ax.set_yscale('log')
        cbar = fig.colorbar(mappable=contourf, label="Detectability [%]")
        cbar.set_ticks(np.arange(0, 100 + 10, 10))
        e_Roche_vals = np.linspace(0, 1, 1000, endpoint=False)
        a_Roche = r_Roche.to(u.km)/(1 - e_Roche_vals)
        ax.fill_between(e_Roche_vals, a_Roche.value, a_vals.min().value,
                        hatch='/', edgecolor='purple', alpha=.5,
                        facecolor='purple')
        SS_moons = ["M", "I", "E", "G", "C", "Ti", "Tr"]
        SS_moons_a = np.array([384.4, 421.8, 671.1, 1070.4, 1882.7, 1221.9,
                               354.8])*1e3*u.km
        SS_moons_e = np.array([.0554, .004, .009, .001, .007, .029, 0.])
        for SS_moon_idx, SS_moon in enumerate(SS_moons):
            a = SS_moons_a[SS_moon_idx].to(u.km).value
            e = SS_moons_e[SS_moon_idx]
            ax.scatter(e, a, c='cyan')
            ax.annotate(SS_moon, xy=(e, a), xycoords='data',
                        xytext=(2, 2), textcoords='offset pixels',
                        fontsize=16)
        # Also add the migration line for angular momentum-conserved Triton
        a_Tr0 = 354.8e3
        e_Tr0 = 0.
        e_Tr_vals = np.linspace(0, 1, num=100, endpoint=False)
        a_Tr_vals = a_Tr0*(1-e_Tr0*e_Tr0)/(1-e_Tr_vals*e_Tr_vals)
        ax.plot(e_Tr_vals, a_Tr_vals, c='cyan')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    @u.quantity_input
    def min_bb_temp_e_t0P(self, instrument, t: u.h, eps, planet_filter,
                          moon_filter, r_moons: u.Rearth, a_moons: u.km, SNR=5,
                          nvals_e=100, nvals_t0P=100, min_Ts=None,
                          max_Ts=None):
        """
        Compute the min. temperature as a function of eccentricity and phase.

        Compute the minimum temperature as a function of eccentricity and
        orbital phase, and plot the result.

        Parameters
        ----------
        instrument : str
            Instrument on which planet_filter and moon_filter are found.
        t : u.h, astropy.Quantity
            Total exposure time.
        eps : float
            Achromatic efficiency of the telescope, expressed as fraction.
        planet_filter : str
            Name of the filter to take as planet filter.
        moon_filter : str
            Name of the filter to take as moon filter.
        r_moons : u.Rearth, astropy.Quantity
            Moon radius. Multiple cases can be provided, in which case the
            length of the array must equal that of the array provided for
            a_moons.
        a_moons : u.km, astropy.Quantity
            Semi-major axis of the moon.
        SNR : float, optional
            Required signal-to-noise ratio. Default is 5.
        nvals_e : int, optional
            Number of samples to take in the eccentricity range [0, 1]. The
            default is 100.
        nvals_t0P : int, optional
            Number of samples to take in the phase range [0, 1]. The
            default is 100.
        max_Ts : float or NoneType, optional
            Maximum temperature to pass to the colorbar for each moon. Must
            be of equal length to r_moons and a_moons. If set to None,
            simply use the maximum value calculated.
        max_Ts : float or NoneType, optional
            Minimum temperature to pass to the colorbar for each moon. Must
            be of equal length to r_moons and a_moons. If set to None,
            simply use the minimum value calculated.

        Returns
        -------
        Tmin_arr : u.K, astropy.Quantity
            Minimum temperature for SNR-sigma detectability as function of
            eccentricity and phase.
        """
        try:
            r_moons.shape[0]
        except IndexError:
            r_moons = np.array([r_moons.value])*r_moons.unit
            a_moons = np.array([a_moons.value])*a_moons.unit
        if a_moons.shape[0] != r_moons.shape[0]:
            raise ValueError("a_moons and r_moons must be of equal length")
        n_graphs = a_moons.shape[0]
        # First, find the telescope to which the instrument belongs
        telescope = filters.telescope_from_instrument(instrument)
        S = np.pi*(diameter[telescope]/2)**2
        # And calculate some quantities that depend on this
        sigma_P = filters.filters[planet_filter].sigma_totpp
        sigma_F = filters.filters[moon_filter].sigma_totpp
        FpP = self.Planet.through_filter(planet_filter)
        FpP_norm = FpP/sigma_P**2
        FpF = self.Planet.through_filter(moon_filter)
        FpF_norm = FpF/sigma_F**2
        tPtF = (np.sqrt(FpF_norm/FpP_norm)).to(u.dimensionless_unscaled
                                               ).value
        # Define useful shorthands
        d = self.dist
        T_F = t/(1+tPtF)
        T_P = t*tPtF/(1+tPtF)
        intens_func = filters.filters[planet_filter].blackbody_intensity
        intens_func2 = filters.filters[moon_filter].bb_temp_interp
        # Set up the figure
        fig, axs = plt.subplots(nrows=n_graphs, ncols=1, squeeze=False)
        for i, (a_moon, r_moon) in enumerate(zip(a_moons, r_moons)):
            P = self.Planet.period(a_moon)
            T_FP = (T_F/P).to(u.dimensionless_unscaled).value
            T_PP = (T_P/P).to(u.dimensionless_unscaled).value
            # And calculate quantities that will appear throughout
            moon_solid_angle = np.pi*(r_moon/self.dist*u.rad)**2
            moon_solid_angle = moon_solid_angle.to(u.mas**2)
            # Compute the inclination/orientation-correction
            K2 = sp.special.ellipk(1/2)**2
            p = 2*K2/np.pi**2 + 1/(2*K2)
            # Compute CF and CP
            CP = ((SNR*d/(a_moon*p))**2/(FpP_norm*eps*S*T_P)*u.ph/u.rad**2).to(
                u.dimensionless_unscaled).value
            CF = ((SNR*d/(a_moon*p))**2/(FpF_norm*eps*S*T_F)*u.ph/u.rad**2).to(
                u.dimensionless_unscaled).value
            # N.B.: CP here differs from the definition used in the paper by
            # a factor (1-f_Pm)

            # Set up the eccentricity and phase grids
            e_arr = np.logspace(-5, 0, nvals_e, endpoint=False)
            t0P_arr = np.linspace(0, 1, nvals_t0P, endpoint=True)
            xi_F2_grid = np.zeros((nvals_e, nvals_t0P))
            xi_P2_grid = np.zeros((nvals_e, nvals_t0P))
            xi_prod_grid = np.zeros((nvals_e, nvals_t0P))
            min_T_grid = np.zeros((nvals_e, nvals_t0P))
            for e_idx, e in enumerate(tqdm.tqdm(e_arr)):
                for t0P_idx, t0P in enumerate(t0P_arr):
                    xi_F = xi_hat(t0P - T_FP/2, T_FP, e)
                    xi_P = xi_hat(t0P+T_FP - T_FP/2, T_PP, e)
                    xi_F2_grid[e_idx, t0P_idx] = np.dot(xi_F, xi_F)
                    xi_P2_grid[e_idx, t0P_idx] = np.dot(xi_P, xi_P)
                    xi_prod_grid[e_idx, t0P_idx] = np.dot(xi_P, xi_F)
                    # Compute the corresponding minimum temperature for each
                    # gridpoint
                    # We reuse the code from MoonPlanetSystem.min_bb_temp here
                    CP_arr = CP
                    CF_arr = CF
                    xi_F2 = xi_F2_grid[e_idx, t0P_idx]
                    xi_P2 = xi_P2_grid[e_idx, t0P_idx]
                    xi_prod = xi_prod_grid[e_idx, t0P_idx]

                    def FFmFFp(f_P_prime):
                        f_P = 1/(1+1/f_P_prime)
                        CP = CP_arr*(1-f_P)
                        CF = CF_arr
                        sqrt = np.sqrt((CF/2 - f_P*xi_prod)**2 - xi_F2*(
                            f_P*f_P*xi_P2-CF-CP))
                        num = CF/2 + CP + f_P*xi_prod - f_P*f_P*(xi_P2) + sqrt
                        denom = xi_F2 - 2*f_P*xi_prod + f_P*f_P*xi_P2 - CP
                        return num/denom

                    def iterative_func(T_vec):
                        T_vec = T_vec*u.K
                        # Calculate f_P using these temperatures
                        # f_P = np.zeros((T_vec.shape))
                        f_P_prime = intens_func(T_vec)*moon_solid_angle/FpP
                        # for T_idx, T in enumerate(T_vec):
                        #     f_P[T_idx] = intens_func(T)*moon_solid_angle/FpP
                        # Calculate the minimum FmF corresponding to this f_P
                        FmF = FFmFFp(f_P_prime)*FpF
                        # and calculate the intensity FmF corresponds to
                        intens_F = FmF/moon_solid_angle
                        # which in turn should correspond to our new estimate
                        # of the temperature
                        new_T_vec = intens_func2(intens_F).to(u.K).value
                        return new_T_vec
                    T_init = 0
                    try:
                        min_T_grid[e_idx, t0P_idx] = sp.optimize.fixed_point(
                            iterative_func, x0=T_init, xtol=1e-3, maxiter=500)
                    except RuntimeError:
                        min_T_grid[e_idx, t0P_idx] = np.nan
            # Calculate the zero-eccentricity value
            xi_F = xi_hat(-T_FP/2, T_FP, 0)
            xi_P = xi_hat(T_FP - T_FP/2, T_PP, 0)
            xi_F20 = np.dot(xi_F, xi_F)
            xi_P20 = np.dot(xi_P, xi_P)
            xi_prod0 = np.dot(xi_P, xi_F)

            xi_F2 = xi_F20
            xi_P2 = xi_P20
            xi_prod = xi_prod0
            # print('correct:')
            # print(xi_F20)
            # print(xi_P20)
            # print(xi_prod0)
            # print(CF_arr)
            # print(CP_arr)

            def FFmFFp(f_P_prime):
                f_P = 1/(1+1/f_P_prime)
                CP = CP_arr*(1-f_P)
                CF = CF_arr
                sqrt = np.sqrt((CF/2 - f_P*xi_prod)**2 - xi_F2*(
                    f_P*f_P*xi_P2-CF-CP))
                num = CF/2 + CP + f_P*xi_prod - f_P*f_P*(xi_P2) + sqrt
                denom = xi_F2 - 2*f_P*xi_prod + f_P*f_P*xi_P2 - CP
                return num/denom

            def iterative_func(T_vec):
                T_vec = T_vec*u.K
                # Calculate f_P using these temperatures
                # f_P = np.zeros((T_vec.shape))
                f_P_prime = intens_func(T_vec)*moon_solid_angle/FpP
                # for T_idx, T in enumerate(T_vec):
                #     f_P[T_idx] = intens_func(T)*moon_solid_angle/FpP
                # Calculate the minimum FmF corresponding to this f_P
                FmF = FFmFFp(f_P_prime)*FpF
                # and calculate the intensity FmF corresponds to
                intens_F = FmF/moon_solid_angle
                # which in turn should correspond to our new estimate
                # of the temperature
                new_T_vec = intens_func2(intens_F).to(u.K).value
                return new_T_vec
            # Compute the corresponding minimum temperature for each
            # gridpoint
            # We reuse the code from MoonPlanetSystem.min_bb_temp here
            T_init = 0
            min_T0 = sp.optimize.fixed_point(
                iterative_func, x0=T_init, xtol=1e-4, maxiter=500)
            if max_Ts is None:
                max_T = np.max(min_T_grid)
            else:
                max_T = max_Ts[i]
            if min_Ts is None:
                min_T = np.min(min_T_grid)
            else:
                min_T = min_Ts[i]
            min_T_grid[np.isnan(min_T_grid)] = np.max(min_T_grid) + 100
            # min_T_grid = np.clip(min_T_grid, a_min=min_T, a_max=max_T+1)
            bounds = np.arange(min_T, max_T+2, 2)
            cmap = cm.afmhot
            norm = BoundaryNorm(bounds, cmap.N, extend='max')
            contourf = axs[i, 0].contourf(t0P_arr, e_arr, min_T_grid,
                                          norm=norm,
                                          cmap=cmap,
                                          levels=bounds,
                                          extend='max')
            contour = axs[i, 0].contour(t0P_arr, e_arr, min_T_grid,
                                        levels=[min_T0-10, min_T0, min_T0+10],
                                        colors=['dodgerblue', 'limegreen',
                                                'dodgerblue'])
            cbar = fig.colorbar(mappable=contourf,
                                # mappable=ScalarMappable(norm=norm,
                                # cmap=cmap),
                                # ticks=bounds,
                                # spacing='proportional',
                                label="Min. temperature [K]",
                                extend='max',
                                ax=axs[i, 0])
            cbar.add_lines(contour)
            axs[i, 0].set_xticks(ticks=[0, .25, .5, .75, 1.],
                                 labels=['0.0', '0.25', '0.5', '0.75', '1.0'])
            if i == n_graphs-1:
                axs[i, 0].set_xlabel('Observed phase [-]')
            axs[i, 0].set_ylabel('Eccentricity [-]')
            axs[i, 0].set_yscale('log')
            r_moon_val = r_moon.to(u.Rearth).value
            r_unit_str = u.Rearth.to_string("latex")
            a_moon_val = a_moon.to(u.km).value/1e3
            a_unit_str = u.km.to_string("latex")
            title_text = f'$R={r_moon_val}$ {r_unit_str}, $a={a_moon_val}$' + \
                f'$\cdot 10^3$ {a_unit_str}'
            axs[i, 0].set_title(title_text)
            # cbar.set_ticks(np.arange(0, 100 + 10, 10))

    @u.quantity_input
    def plot_SNR_T_d(self, instrument, t: u.h, eps, planet_filter,
                     moon_filter, r_moons: u.Rearth, a_moons: u.km,
                     T_vals: u.K = np.array([[150, 350], [100, 300]])*u.K,
                     n_T=100,
                     d_vals: u.pc = np.array([[1, 10], [1, 30]])*u.pc,
                     n_d=100):
        """
        Plot the SNR for moons as function of temperature and distance.

        Plot the SNR for a set of moons as function of their temperature and
        distance; the size of the moons and their semi-major axes must be
        provided. Plots the results on a single row of subplots.

        Parameters
        ----------
        instrument : str
            Instrument on which planet_filter and moon_filter are found.
        t : u.h, astropy.Quantity
            Total exposure time.
        eps : float
            Achromatic efficiency of the telescope, expressed as fraction.
        planet_filter : str
            Name of the filter to take as planet filter.
        moon_filter : str
            Name of the filter to take as moon filter.
        r_moons : u.Rearth, astropy.Quantity
            Moon radius. Multiple cases can be provided, in which case the
            length of the array must equal that of the array provided for
            a_moons.
        a_moons : u.km, astropy.Quantity
            Semi-major axis of the moon.
        T_vals : u.K, optional
            Temperature range over which to explore the SNR, with each row
            corresponding to a moon.
        n_T : int, optional
            Number of samples to take over T_vals. Default is 100.
        d_vals : u.pc, optional
            Distance range over which to explore the SNR, with each row
            corresponding to a moon.
        n_d : int, optional
            Number of samples to take over d_vals. Default is 100.
        """
        # Note that the SNR scales as d^-2, such that we need only calculate
        # it for a single distance. For this we will use the distance as set
        # for the MoonPlanetSystem, so as to forego order-of-magnitude issues
        # if we had simply used a unit value.
        try:
            r_moons.shape[0]
        except IndexError:
            r_moons = np.array([r_moons.value])*r_moons.unit
            a_moons = np.array([a_moons.value])*a_moons.unit
        if a_moons.shape[0] != r_moons.shape[0]:
            raise ValueError("a_moons and r_moons must be of equal length")
        n_graphs = a_moons.shape[0]
        # First, find the telescope to which the instrument belongs
        telescope = filters.telescope_from_instrument(instrument)
        S = np.pi*(diameter[telescope]/2)**2
        # And calculate some quantities that depend on this
        sigma_P = filters.filters[planet_filter].sigma_totpp
        sigma_F = filters.filters[moon_filter].sigma_totpp
        FpP = self.Planet.through_filter(planet_filter)
        FpP_norm = FpP/sigma_P**2
        FpF = self.Planet.through_filter(moon_filter)
        FpF_norm = FpF/sigma_F**2
        tPtF = (np.sqrt(FpF_norm/FpP_norm)).to(u.dimensionless_unscaled
                                               ).value
        # Define useful shorthands
        d = self.dist
        T_F = t/(1+tPtF)
        T_P = t*tPtF/(1+tPtF)
        intens_func_P = filters.filters[planet_filter].blackbody_intensity
        intens_func_F = filters.filters[moon_filter].blackbody_intensity
        # Set up the figure
        fig, axs = plt.subplots(nrows=1, ncols=n_graphs, squeeze=False)
        for i, (a_moon, r_moon) in enumerate(zip(a_moons, r_moons)):
            # For each moon, compute the SNR along a single distance for all
            # values of T:
            # First compute some useful quantities
            P = self.Planet.period(a_moon)
            T_FP = (T_F/P).to(u.dimensionless_unscaled).value
            T_PP = (T_P/P).to(u.dimensionless_unscaled).value
            # And calculate quantities that will appear throughout
            moon_solid_angle = np.pi*(r_moon/self.dist*u.rad)**2
            moon_solid_angle = moon_solid_angle.to(u.mas**2)
            # Compute the inclination/orientation-correction
            K2 = sp.special.ellipk(1/2)**2
            p = 2*K2/np.pi**2 + 1/(2*K2)
            # Compute the grid of values of T and d
            T_arr = np.linspace(T_vals[i, 0].value, T_vals[i, 1].value,
                                num=n_T, endpoint=True)*T_vals.unit
            d_arr = np.linspace(d_vals[i, 0].value, d_vals[i, 1].value,
                                num=n_d, endpoint=True)*d_vals.unit
            # Compute the values of f_Pm and f_Fm for each temperature
            f_prime_Pm = intens_func_P(T_arr)*moon_solid_angle/FpP
            f_prime_Fm = intens_func_F(T_arr)*moon_solid_angle/FpF
            f_Pm = 1/(1+1/f_prime_Pm)
            f_Fm = 1/(1+1/f_prime_Fm)
            # print(f_Pm)
            # print(f_Fm)
            # Compute xi_F and xi_P (which do not depend on T or d)
            xi_F = xi_hat(0, T_FP, 0)
            xi_P = xi_hat(T_FP, T_PP, 0)
            # xi_F_arr = np.zeros((2, n_T))
            # xi_P_arr = np.zeros((2, n_T))
            # xi_F_arr[0, :] = xi_F[0]
            # xi_F_arr[1, :] = xi_F[1]
            # xi_P_arr[0, :] = xi_P[0]
            # xi_P_arr[1, :] = xi_P[1]
            # Compute the corresponding signal
            ref_signal = (p*a_moon/d*np.linalg.norm(
                f_Fm[:, None]*xi_F[None, :] - f_Pm[:, None]*xi_P[None, :],
                axis=1)
            ).to(u.dimensionless_unscaled).value*u.rad
            ref_sigma_F2 = (1-f_Fm)/(FpF_norm*T_F*S*eps)*u.ph
            ref_sigma_P2 = (1-f_Pm)/(FpP_norm*T_P*S*eps)*u.ph
            ref_noise_tot = np.sqrt(ref_sigma_F2 + ref_sigma_P2)
            # print((np.sqrt(ref_sigma_F2)/ref_noise_tot).to(
            #     u.dimensionless_unscaled))
            SNR_ref = (ref_signal/ref_noise_tot).to(
                u.dimensionless_unscaled).value
            # Reshape SNR_ref to a column vector to prepare for the matrix
            # product:
            SNR_ref = SNR_ref.reshape((SNR_ref.shape[0], 1))
            d_mult_arr = (d*d*(d_arr**(-2)).reshape((1, d_arr.shape[0]))).to(
                u.dimensionless_unscaled).value
            # Multiply SNR by d_ref^2/d^2 to sample the grid of values of d
            # for each value of T
            SNR_grid = SNR_ref @ d_mult_arr
            # Plot the result
            cmap = cm.plasma
            # bounds=np.arange(1, np.ceil(np.max(SNR_grid)) + 1, 1)
            # norm = BoundaryNorm(bounds, cmap.N, extend='both')
            # norm = LogNorm(vmin=.1, vmax=100)
            norm = Normalize(vmin=.1, vmax=50)
            levels = np.arange(0, 51, 1)
            # levels = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
            #           25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
            contourf = axs[0, i].contourf(d_arr.value, T_arr.value, SNR_grid,
                                          norm=norm, cmap=cmap, levels=levels,
                                          extend='max')
            axs[0, i].set_xlabel('Distance [pc]')
            axs[0, i].set_ylabel('Temperature [K]')
            contour = axs[0, i].contour(d_arr.value, T_arr.value, SNR_grid,
                                        levels=[1, 3, 5],
                                        colors=['grey', 'dodgerblue',
                                                'r'])
            # axs[0, i].set_xscale('log')
            # axs[0, i].set_yscale('log')
            if i == 1:
                cbar = fig.colorbar(mappable=contourf,
                                    # mappable=ScalarMappable(norm=norm,
                                    # cmap=cmap),
                                    # ticks=bounds,
                                    # spacing='proportional',
                                    label="Signal-to-noise ratio [-]",
                                    ax=axs[0, i])
                cbar.add_lines(contour)

    @u.quantity_input
    def plot_SNR_a_d(self, instrument, t: u.h, eps, planet_filter,
                     moon_filter, moon_idx=0,
                     a_vals: u.km = np.array([1.5e5, 2e6])*u.km,
                     n_a=100,
                     d_vals: u.pc = np.array([1, 30])*u.pc,
                     n_d=100):
        """
        Plot the SNR for moons as function of semi-major axis and distance.

        Plot the SNR for a set of moons as function of their semi-major axis
        and distance, for a fixed set of spectra. Plots the results on a
        single row of subplots.

        Parameters
        ----------
        instrument : str
            Instrument on which planet_filter and moon_filter are found.
        t : u.h, astropy.Quantity
            Total exposure time.
        eps : float
            Achromatic efficiency of the telescope, expressed as fraction.
        planet_filter : str
            Name of the filter to take as planet filter.
        moon_filter : str
            Name of the filter to take as moon filter.
        moon_idx : int
            Index of the moon for which the spectrum is to be used.
        a_vals : u.km, optional
            Semi-major axis range over which to explore the SNR.
        n_a : int, optional
            Number of samples to take over a_vals. Default is 100.
        d_vals : u.pc, optional
            Distance range over which to explore the SNR.
        n_d : int, optional
            Number of samples to take over d_vals. Default is 100.
        """
        # Note that the SNR scales as d^-2, such that we need only calculate
        # it for a single distance. For this we will use the distance as set
        # for the MoonPlanetSystem, so as to forego order-of-magnitude issues
        # if we had simply used a unit value.
        # First, find the telescope to which the instrument belongs
        telescope = filters.telescope_from_instrument(instrument)
        S = np.pi*(diameter[telescope]/2)**2
        # And calculate some quantities that depend on this
        sigma_P = filters.filters[planet_filter].sigma_totpp
        sigma_F = filters.filters[moon_filter].sigma_totpp
        # Planet fluxes
        FpP = self.Planet.through_filter(planet_filter)
        FpP_norm = FpP/sigma_P**2
        FpF = self.Planet.through_filter(moon_filter)
        FpF_norm = FpF/sigma_F**2
        # Moon fluxes
        FmP = self.Moons[moon_idx].through_filter(planet_filter)
        FmP_norm = FmP/sigma_P**2
        FmF = self.Moons[moon_idx].through_filter(moon_filter)
        FmF_norm = FmF/sigma_F**2

        tPtF = (np.sqrt(FpF_norm/FpP_norm)).to(u.dimensionless_unscaled
                                               ).value
        # Define useful shorthands
        d = self.dist
        T_F = t/(1+tPtF)
        T_P = t*tPtF/(1+tPtF)
        r_moon = self.Moons[moon_idx].radius
        # Set up the figure
        fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False)
        # For each moon, compute the SNR along a single distance for all
        # values of a:
        a_arr = np.linspace(a_vals[0].value, a_vals[1].value,
                            num=n_a, endpoint=True)*a_vals.unit
        ref_signal = np.zeros(a_arr.shape)*u.rad
        for a_idx, a_val in enumerate(a_arr):
            # First compute some useful quantities
            P = self.Planet.period(a_val)
            T_FP = (T_F/P).to(u.dimensionless_unscaled).value
            T_PP = (T_P/P).to(u.dimensionless_unscaled).value
            # And calculate quantities that will appear throughout
            moon_solid_angle = np.pi*(r_moon/self.dist*u.rad)**2
            moon_solid_angle = moon_solid_angle.to(u.mas**2)
            # Compute the inclination/orientation-correction
            K2 = sp.special.ellipk(1/2)**2
            p = 2*K2/np.pi**2 + 1/(2*K2)
            # Compute the grid of values of a and d
            d_arr = np.linspace(d_vals[0].value, d_vals[1].value,
                                num=n_d, endpoint=True)*d_vals.unit
            # Compute the values of f_Pm and f_Fm for each temperature
            f_prime_Pm = (FmP/FpP).to(u.dimensionless_unscaled).value
            f_prime_Fm = (FmF/FpF).to(u.dimensionless_unscaled).value
            f_Pm = 1/(1+1/f_prime_Pm)
            f_Fm = 1/(1+1/f_prime_Fm)
            # print(f_Pm)
            # print(f_Fm)
            # Compute xi_F and xi_P (which do not depend on T or d)
            xi_F = xi_hat(0, T_FP, 0)
            xi_P = xi_hat(T_FP, T_PP, 0)
            # xi_F_arr = np.zeros((2, n_T))
            # xi_P_arr = np.zeros((2, n_T))
            # xi_F_arr[0, :] = xi_F[0]
            # xi_F_arr[1, :] = xi_F[1]
            # xi_P_arr[0, :] = xi_P[0]
            # xi_P_arr[1, :] = xi_P[1]
            # Compute the corresponding signal
            ref_signal[a_idx] = (p*a_val/d*np.linalg.norm(
                f_Fm*xi_F[None, :] - f_Pm*xi_P[None, :],
                axis=1)
            ).to(u.dimensionless_unscaled).value*u.rad
        ref_sigma_F2 = (1-f_Fm)/(FpF_norm*T_F*S*eps)*u.ph
        ref_sigma_P2 = (1-f_Pm)/(FpP_norm*T_P*S*eps)*u.ph
        ref_noise_tot = np.sqrt(ref_sigma_F2 + ref_sigma_P2)
        # print((np.sqrt(ref_sigma_F2)/ref_noise_tot).to(
        #     u.dimensionless_unscaled))
        SNR_ref = (ref_signal/ref_noise_tot).to(
            u.dimensionless_unscaled).value
        # Reshape SNR_ref to a column vector to prepare for the matrix
        # product:
        SNR_ref = SNR_ref.reshape((SNR_ref.shape[0], 1))
        d_mult_arr = (d*d*(d_arr**(-2)).reshape((1, d_arr.shape[0]))).to(
            u.dimensionless_unscaled).value
        # Multiply SNR by d_ref^2/d^2 to sample the grid of values of d
        # for each value of a
        SNR_grid = SNR_ref @ d_mult_arr
        # Plot the result
        cmap = cm.plasma
        # bounds=np.arange(1, np.ceil(np.max(SNR_grid)) + 1, 1)
        # norm = BoundaryNorm(bounds, cmap.N, extend='both')
        # norm = LogNorm(vmin=.1, vmax=100)
        norm = Normalize(vmin=.1, vmax=50)
        levels = np.arange(0, 51, 1)
        # levels = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
        #           25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
        contourf = axs[0, 0].contourf(d_arr.value, a_arr.value, SNR_grid,
                                      norm=norm, cmap=cmap, levels=levels,
                                      extend='max')
        axs[0, 0].set_xlabel('Distance [pc]')
        axs[0, 0].set_ylabel('Semi-major axis [km]')
        contour = axs[0, 0].contour(d_arr.value, a_arr.value, SNR_grid,
                                    levels=[1, 3, 5],
                                    colors=['grey', 'dodgerblue',
                                            'r'])
        # axs[0, i].set_xscale('log')
        # axs[0, i].set_yscale('log')
        cbar = fig.colorbar(mappable=contourf,
                            # mappable=ScalarMappable(norm=norm,
                            # cmap=cmap),
                            # ticks=bounds,
                            # spacing='proportional',
                            label="Signal-to-noise ratio [-]",
                            ax=axs[0, 0])
        cbar.add_lines(contour)

    @u.quantity_input
    def noise_radius_temp(self, instrument, t: u.h, eps, planet_filter,
                          moon_filter, T_bounds: u.K, R_bounds: u.Rearth,
                          n_T=100, n_R=100):
        """
        Plot the absolute noise and the part of it due to the moon filter.

        Parameters
        ----------
        instrument : str
            Instrument on which planet_filter and moon_filter are found.
        t : u.h, astropy.Quantity
            Total exposure time.
        eps : float
            Achromatic efficiency of the telescope, expressed as fraction.
        planet_filter : str
            Name of the filter to take as planet filter.
        moon_filter : str
            Name of the filter to take as moon filter.
        T_bounds : u.K
            Temperature range to explore.
        R_bounds : u.Rearth
            Radius range to explore.
        n_T : int, optional
            Number of grid points over T_bounds. The default is 100.
        n_R : int, optional
            Number of grid points over R_bounds. The default is 100.

        Returns
        -------
        None.

        """
        # First, find the telescope to which the instrument belongs
        telescope = filters.telescope_from_instrument(instrument)
        S = np.pi*(diameter[telescope]/2)**2
        # And calculate some quantities that depend on this
        sigma_P = filters.filters[planet_filter].sigma_totpp
        sigma_F = filters.filters[moon_filter].sigma_totpp
        FpP = self.Planet.through_filter(planet_filter)
        FpP_norm = FpP/sigma_P**2
        FpF = self.Planet.through_filter(moon_filter)
        FpF_norm = FpF/sigma_F**2
        tPtF = (np.sqrt(FpF_norm/FpP_norm)).to(u.dimensionless_unscaled
                                               ).value
        # Define useful shorthands
        d = self.dist
        T_F = t/(1+tPtF)
        T_P = t*tPtF/(1+tPtF)
        intens_func_P = filters.filters[planet_filter].blackbody_intensity
        intens_func_F = filters.filters[moon_filter].blackbody_intensity
        # And calculate quantities that will appear throughout
        R_arr = np.linspace(R_bounds[0], R_bounds[1], num=n_R)
        T_arr = np.linspace(T_bounds[0], T_bounds[1], num=n_T)
        # Compute the solid angle as a function of R
        solid_angle_vals = (np.pi*(R_arr/self.dist*u.rad)**2).to(u.mas**2)
        solid_angle_vals = solid_angle_vals.reshape((1, n_R))
        # Compute the intensity as a function of T
        intens_P = intens_func_P(T_arr).reshape((n_T, 1))
        intens_F = intens_func_F(T_arr).reshape((n_T, 1))
        f_prime_Pm = (intens_P @ solid_angle_vals)/FpP
        f_prime_Fm = (intens_F @ solid_angle_vals)/FpF
        f_Pm = (1/(1+1/f_prime_Pm)).to(u.dimensionless_unscaled).value
        f_Fm = (1/(1+1/f_prime_Fm)).to(u.dimensionless_unscaled).value
        sigma_F = (np.sqrt((1-f_Fm)/(FpF_norm*T_F*S*eps/u.ph))).to(u.mas)
        sigma_P = (np.sqrt((1-f_Pm)/(FpP_norm*T_P*S*eps/u.ph))).to(u.mas)
        sigma_tot = (np.sqrt(sigma_F**2 + sigma_P**2)).to(u.mas)
        fig, axs = plt.subplots(2, 2, squeeze=True)

        # Plot the results
        cmap = cm.plasma
        contour1 = axs[0, 0].contourf(R_arr, T_arr, sigma_tot.value,
                                      cmap=cmap)
        cbar1 = fig.colorbar(mappable=contour1, label="Total noise [mas]")
        axs[0, 0].set_ylabel('Temperature [K]')
        axs[0, 0].set_xlabel('Radius [$R_{\oplus}$]')
        contour2 = axs[0, 1].contourf(R_arr, T_arr,
                                      (sigma_F/sigma_tot).value*100,
                                      cmap=cmap)
        axs[0, 1].set_ylabel('Temperature [K]')
        axs[0, 1].set_xlabel('Radius [$R_{\oplus}$]')
        cbar2 = fig.colorbar(mappable=contour2,
                             label="$\sigma_{tot,F}/\sigma_S$ [%]")
        contour3 = axs[1, 0].contourf(R_arr, T_arr, f_Pm,
                                      cmap=cmap)
        cbar3 = fig.colorbar(mappable=contour3,
                             label="$f_{Pm}$ [-]")
        axs[1, 0].set_ylabel('Temperature [K]')
        axs[1, 0].set_xlabel('Radius [$R_{\oplus}$]')
        contour4 = axs[1, 1].contourf(R_arr, T_arr, f_Fm,
                                      cmap=cmap)
        cbar4 = fig.colorbar(mappable=contour4,
                             label="$f_{Fm}$ [-]")
        axs[1, 1].set_xlabel('Radius [$R_{\oplus}$]')
        axs[1, 1].set_ylabel('Temperature [K]')
